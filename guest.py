"""Runs on the rented GPU: load each model with vLLM defaults and record what fits."""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

gpu, models = sys.argv[1], sys.argv[2:]
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
PROBE = '''
import json, sys, torch
from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], max_model_len=int(sys.argv[2]), seed=0)
text = llm.generate(["Once upon a time"], SamplingParams(max_tokens=16, temperature=0))[0].outputs[0].text
free, total = torch.cuda.mem_get_info()
print("FIT_RESULT " + json.dumps({"free_bytes": free, "total_bytes": total, "text": text}), flush=True)
'''


def smi():
    return subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True).stdout.strip()


def launch(model, length):
    start = time.time()
    run = subprocess.run([sys.executable, '-c', PROBE, model, str(length)], capture_output=True, text=True, timeout=1800,
                         env={**os.environ, **OVERRIDES})
    log = run.stdout + run.stderr
    record = {'max_model_len': length, 'exit': run.returncode, 'seconds': round(time.time() - start, 1),
              'loaded': 'FIT_RESULT ' in log, 'log_tail': log.splitlines()[-60:],
              'errors': [line for line in log.splitlines() if 'Error' in line or 'error:' in line][:40]}
    for key, pattern in PATTERNS.items():
        found = re.findall(pattern, log)
        if found:
            record[key] = float(found[-1].replace(',', ''))
    if record['loaded']:
        record.update(json.loads(log.split('FIT_RESULT ', 1)[1].splitlines()[0]))
    record['oom'] = 'CUDA out of memory' in log or 'OutOfMemoryError' in log
    return record


memory = Path('/sys/fs/cgroup/memory.max')
report = {'gpu': gpu, 'nvidia_smi': smi(), 'overrides': OVERRIDES, 'cc': shutil.which('cc'), 'cpus': len(os.sched_getaffinity(0)),
          'meminfo': Path('/proc/meminfo').read_text().splitlines()[:3],
          'cgroup_memory_max': memory.read_text().strip() if memory.exists() else None,
          'disk_free_bytes': shutil.disk_usage('/root').free,
          'vllm': subprocess.run([sys.executable, '-c', 'import vllm; print(vllm.__version__)'], capture_output=True, text=True).stdout.strip(),
          'models': {}}
for model in models:
    start = time.time()
    path = subprocess.run([sys.executable, '-c', 'import sys; from huggingface_hub import snapshot_download as s; '
                           'print(s(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt"]))', model],
                          capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    entry = {'download_seconds': round(time.time() - start, 1), 'idle_smi': smi(), 'launches': [launch(model, SHORT)]}
    if entry['launches'][0].get('kv_tokens', NATIVE) < NATIVE:
        entry['launches'].append(launch(model, NATIVE))  # Observe vLLM refusing the native context directly.
    report['models'][model] = entry
    (out / f'{gpu}.json').write_text(json.dumps(report, indent=1))
    shutil.rmtree(Path(path).parents[1], ignore_errors=True)  # One checkpoint on disk at a time.
print(json.dumps({m: [(l['max_model_len'], l['loaded'], l.get('kv_tokens')) for l in e['launches']] for m, e in report['models'].items()}))
