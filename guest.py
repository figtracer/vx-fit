"""Runs on the rented GPU: load each model with vLLM and record what fits."""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('gpu')
parser.add_argument('models', nargs='+')
parser.add_argument('--utils', nargs='+', type=float, default=[0.9])  # vLLM gpu_memory_utilization; 0.9 is its default.
parser.add_argument('--tag', default='')
parser.add_argument('--skip-native', action='store_true')  # repeats only: no 32k-prompt launches
parser.add_argument('--attention-backend', default='')  # Blackwell's default FlashInfer backend JIT-compiles with nvcc.
args = parser.parse_args()
out = Path('/workspace/out'); out.mkdir(parents=True, exist_ok=True)
SHORT, NATIVE = 4096, 32768
# The image has no nvcc; FlashInfer's sampler would JIT-compile with it. The sampler choice does not change memory use.
OVERRIDES = {'VLLM_USE_FLASHINFER_SAMPLER': '0'}
PATTERNS = {
    'kv_tokens': r'GPU KV cache size: ([\d,]+) tokens',
    'kv_gib': r'Available KV cache memory: ([\d.]+) GiB',
    'weights_gib': r'Model loading took ([\d.]+) GiB',
    'kv_limit_tokens': r'maximum number of tokens that can be stored in KV cache \(([\d,]+)\)',
    'estimated_max_len': r'estimated maximum model length is ([\d,]+)',
}
# A long launch fills the whole context with one real prompt, so the KV cache is actually written, not just reserved.
PROBE = '''
import json, sys, torch
from vllm import LLM, SamplingParams, TokensPrompt
model, length, util, long, backend = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
llm = LLM(model=model, max_model_len=length, gpu_memory_utilization=util, seed=0,
          **({"attention_config": {"backend": backend}} if backend else {}))
prompt = TokensPrompt(prompt_token_ids=[1000 + i % 20000 for i in range(length - 64)]) if long else "Once upon a time"
result = llm.generate([prompt], SamplingParams(max_tokens=16, temperature=0))[0]
free, total = torch.cuda.mem_get_info()
print("FIT_RESULT " + json.dumps({"free_bytes": free, "total_bytes": total, "prompt_tokens": len(result.prompt_token_ids),
                                   "generated_tokens": len(result.outputs[0].token_ids), "text": result.outputs[0].text}), flush=True)
'''


def smi():
    return subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True).stdout.strip()


def launch(model, length, util, long=False):
    start = time.time()
    run = subprocess.run([sys.executable, '-c', PROBE, model, str(length), str(util), '1' if long else '0', args.attention_backend],
                         capture_output=True, text=True, timeout=1800, env={**os.environ, **OVERRIDES})
    log = run.stdout + run.stderr
    record = {'max_model_len': length, 'util': util, 'long_prompt': long, 'exit': run.returncode,
              'seconds': round(time.time() - start, 1), 'loaded': 'FIT_RESULT ' in log, 'log_tail': log.splitlines()[-60:],
              'errors': [line for line in log.splitlines() if 'Error' in line or 'error:' in line][:40]}
    for key, pattern in PATTERNS.items():
        found = re.findall(pattern, log)
        if found:
            record[key] = float(found[-1].replace(',', ''))
    if record['loaded']:
        record.update(json.loads(log.split('FIT_RESULT ', 1)[1].splitlines()[0]))
    record['oom'] = 'CUDA out of memory' in log or 'OutOfMemoryError' in log
    print(json.dumps({k: record.get(k) for k in ['max_model_len', 'util', 'long_prompt', 'loaded', 'kv_tokens', 'seconds']}), flush=True)
    return record


memory = Path('/sys/fs/cgroup/memory.max')
report = {'gpu': args.gpu, 'nvidia_smi': smi(), 'overrides': OVERRIDES, 'attention_backend': args.attention_backend or 'auto', 'cc': shutil.which('cc'),
          'cpus': len(os.sched_getaffinity(0)), 'meminfo': Path('/proc/meminfo').read_text().splitlines()[:3],
          'cgroup_memory_max': memory.read_text().strip() if memory.exists() else None,
          'disk_free_bytes': shutil.disk_usage('/root').free,
          'vllm': subprocess.run([sys.executable, '-c', 'import vllm; print(vllm.__version__)'], capture_output=True, text=True).stdout.strip(),
          'models': {}}
for model in args.models:
    start = time.time()
    path = subprocess.run([sys.executable, '-c', 'import sys; from huggingface_hub import snapshot_download as s; '
                           'print(s(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt"]))', model],
                          capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    entry = {'download_seconds': round(time.time() - start, 1), 'idle_smi': smi(), 'launches': []}
    for util in args.utils:
        entry['launches'].append(launch(model, SHORT, util))
    if not args.skip_native and entry['launches'][0].get('kv_tokens', 0) < NATIVE:
        # Boundary model: try the native context with a full-length prompt wherever the short launch loaded.
        for short in list(entry['launches']):
            if short['loaded']:
                entry['launches'].append(launch(model, NATIVE, short['util'], long=True))
    report['models'][model] = entry
    (out / f'{args.gpu}{args.tag}.json').write_text(json.dumps(report, indent=1))
    shutil.rmtree(Path(path).parents[1], ignore_errors=True)  # One checkpoint on disk at a time.
