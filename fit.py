"""Compare Vx admission verdicts with what vLLM actually fits on rented GPUs."""
import argparse, json, math, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct']
MACHINES = {'H100': 'fleet/h100-sxm.vx', 'A100-80GB': 'fleet/a100-80.vx'}
CONTEXTS, BATCHES = [4096, 8192, 16384, 32768], [1, 4, 16, 64]
VLLM = '0.31.0'


def fission(*args, check=True):
    result = subprocess.run(['fission', 'advanced', *args], capture_output=True, text=True)
    print('$ fission advanced', ' '.join(args[:3]), '->', result.returncode, flush=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout) if result.stdout.strip().startswith('{') else result.stdout


def measure(args):
    name = args.name or f'vxfit-{args.gpu.lower()}-{time.strftime("%m%d%H%M")}'
    plan = fission('plan', name, '--provider', 'modal-tempo', '--gpu', args.gpu,
                   '--duration', args.duration, '--budget', args.budget)
    print(json.dumps({k: plan.get(k) for k in ['id', 'body', 'creationQuote', 'totalCap']}))
    if not args.approve:
        return print('Preview only; repeat with --approve to pay.')
    try:
        state = fission('open', name, '--plan', plan['id'], '--approve')
        if not (state.get('guestGpu') or {}).get('verified'):
            raise RuntimeError(f"GPU check failed: {state.get('preparationError')}")
        print(json.dumps(state['guestGpu']['devices']))
        fission('wait', name, 'bootstrap', '--duration', '10m', '--max-spend', '0.01')
        fission('upload', name, str(ROOT / 'guest.py'), '/workspace/guest.py')
        fission('run', name, 'fit', '--duration', args.work, '--', 'sh', '-c',
                'command -v cc >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc g++ >/dev/null); '
                f'pip install -q vllm=={VLLM} && python3 /workspace/guest.py "$0" "$@"', args.gpu, *args.models)
        job = fission('wait', name, 'fit', '--duration', args.work, '--max-spend', '0.05', check=False)
        print(json.dumps(job if isinstance(job, str) else {k: job.get(k) for k in ['phase', 'waitingStopped']}))
        target = ROOT / 'measurements'
        target.mkdir(exist_ok=True)
        for remote, local in [(f'/workspace/out/{args.gpu}.json', f'{args.gpu}.json'),
                              ('/workspace/.fission/jobs/fit/output.log', f'{args.gpu}.log')]:
            (target / local).unlink(missing_ok=True)
            fission('download', name, remote, str(target / local), check=False)
    finally:
        fission('close', name, '--discard-output', check=False)
        for _ in range(8):
            status = fission('status', name, '--refresh', '--json', check=False)
            if isinstance(status, dict) and status.get('phase') == 'terminated':
                break
            time.sleep(20)
        print('cleanup:', status.get('phase') if isinstance(status, dict) else status)


def shape(model):
    base = f'https://huggingface.co/{model}/resolve/main/'
    config = json.load(urllib.request.urlopen(base + 'config.json'))
    index = json.load(urllib.request.urlopen(base + 'model.safetensors.index.json'))
    head = config['hidden_size'] // config['num_attention_heads']
    return {'d': config['hidden_size'], 'layers': config['num_hidden_layers'], 'heads': config['num_attention_heads'],
            'kv_heads': config['num_key_value_heads'], 'head_dim': head, 'native_context': config['max_position_embeddings'],
            'weight_bytes': index['metadata']['total_size']}


def exact_program(s, context, batch):
    # Same three residents as fleet/admit.vx, with weights taken from the real checkpoint size.
    rows = math.ceil(s['weight_bytes'] / 2 / s['d'])
    tensors = [('weights', [rows, s['d']]), ('kv_cache', [2 * s['layers'] * batch * context, s['kv_heads'] * s['head_dim']]),
               ('activations', [batch * context, s['d']])]
    lines = ['fn main() -> i32 {']
    for name, (a, b) in tensors:
        lines += [f'  let {name}_host : Tensor<f16, [{a}, {b}]> = Tensor<f16>([{a}, {b}]);',
                  f'  let {name} = transfer({name}_host, Memory::HBM);']
    lines += [f'  let _{name}_back = transfer({name}, Memory::CPU_DRAM);' for name, _ in tensors]
    return '\n'.join(lines + ['  return 0;', '}']) + '\n'


def reference_program(vx, s, context, batch):
    source = (vx / 'fleet/admit.vx').read_text()
    call = 'return admit<80, 64, 8, 128, 4096, 1, 1>();'
    assert call in source, 'fleet/admit.vx changed; update the reference call'
    return source.replace(call, f"return admit<{s['layers']}, {s['heads']}, {s['kv_heads']}, {s['head_dim']}, {context}, {batch}, 1>();")


def admit(vx, machine, source, path):
    path.write_text(source)
    diagnostics = path.with_suffix('.json')
    diagnostics.unlink(missing_ok=True)
    run = subprocess.run([str(vx / 'target/debug/vxc'), '--machine', str(vx / machine), '--host', str(vx / 'fleet/host-x86-e5-2666v3.vx'),
                          str(path), '--action', 'emit-mlir', '--diagnostics-json', str(diagnostics)], capture_output=True, text=True)
    record = json.loads(diagnostics.read_text())
    if record['verdict'] not in ['admitted', 'rejected'] or (run.returncode == 0) != (record['verdict'] == 'admitted'):
        raise RuntimeError(f'{path.name}: {run.stderr[-2000:]}')
    required = [d['capacity']['required_bytes'] for d in record['diagnostics'] if d.get('capacity')]
    required += [r['total_bytes'] for r in record.get('resident_sets', []) if r['space'] == 'HBM']
    if record['verdict'] == 'rejected' and not any(d['code'] in ['E6009', 'E6010'] for d in record['diagnostics']):
        raise RuntimeError(f'{path.name} rejected for a reason other than capacity: {record["diagnostics"]}')
    return record['verdict'] == 'admitted', max(required)


def table(args):
    vx = Path(args.vx).resolve()
    out = ROOT / 'results'
    (out / 'vx').mkdir(parents=True, exist_ok=True)
    shapes = {model: shape(model) for model in MODELS}
    cells, summary = [], []
    for gpu, machine in MACHINES.items():
        measured = ROOT / 'measurements' / f'{gpu}.json'
        if not measured.exists():
            print(f'skip {gpu}: no {measured.name}')
            continue
        report = json.loads(measured.read_text())
        for model, s in shapes.items():
            entry = report['models'].get(model)
            if entry is None:
                continue
            short = entry['launches'][0]
            loaded, kv_tokens = short['loaded'], short.get('kv_tokens', 0) if short['loaded'] else 0
            kv_per_token = 2 * s['layers'] * s['kv_heads'] * s['head_dim'] * 2
            capacity = 80 * 2**30
            implied = max(0, (capacity - s['weight_bytes']) // (kv_per_token + 2 * s['d']))
            summary.append({'gpu': gpu, 'model': model, 'weight_bytes': s['weight_bytes'], 'vllm_loaded': loaded,
                            'vllm_kv_tokens': kv_tokens, 'vllm_weights_gib': short.get('weights_gib'), 'vllm_kv_gib': short.get('kv_gib'),
                            'vx_implied_tokens': implied, 'device_total_bytes': short.get('total_bytes'),
                            'native_launch': [(l['max_model_len'], l['loaded']) for l in entry['launches'][1:]]})
            for context in CONTEXTS:
                for batch in BATCHES:
                    tag = f"{gpu}-{model.split('/')[1]}-{context}-{batch}"
                    exact, exact_bytes = admit(vx, machine, exact_program(s, context, batch), out / 'vx' / f'{tag}-exact.vx')
                    ref, ref_bytes = admit(vx, machine, reference_program(vx, s, context, batch), out / 'vx' / f'{tag}-admit.vx')
                    fits = loaded and context <= s['native_context'] and batch * context <= kv_tokens
                    cells.append({'gpu': gpu, 'model': model, 'context': context, 'batch': batch, 'vllm_fits': fits,
                                  'vx_exact': exact, 'vx_exact_bytes': exact_bytes, 'vx_admit': ref, 'vx_admit_bytes': ref_bytes})
    if not cells:
        sys.exit('No measurements yet; run measure first.')
    count = lambda key, want, truth: sum(c[key] == want and c['vllm_fits'] == truth for c in cells)
    scores = {key: {'agree': count(key, True, True) + count(key, False, False), 'false_accept': count(key, True, False),
                    'false_reject': count(key, False, True), 'cells': len(cells)} for key in ['vx_exact', 'vx_admit']}
    (out / 'fit.json').write_text(json.dumps({'vllm': VLLM, 'scores': scores, 'models': summary, 'cells': cells}, indent=1) + '\n')
    gib = lambda b: f'{b / 2**30:.1f}'
    # vLLM budgets 90% of the device; whatever is neither weights nor KV cache is its runtime overhead.
    other = lambda m: 0.9 * m['device_total_bytes'] / 2**30 - m['vllm_weights_gib'] - m['vllm_kv_gib']
    lines = ['| GPU | Model | device GiB | weights GiB | KV GiB | other GiB | vLLM KV tokens | Vx implied tokens |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {m['gpu']} | {m['model'].split('/')[1]} | {gib(m['device_total_bytes'])} | {m['vllm_weights_gib']:.1f} | "
              f"{m['vllm_kv_gib']:.1f} | {other(m):.1f} | {m['vllm_kv_tokens']:,.0f} | {m['vx_implied_tokens']:,} |" for m in summary]
    lines += ['', '| Vx program | agree | false accept | false reject | cells |', '|---|---:|---:|---:|---:|']
    lines += [f"| {k} | {v['agree']} | {v['false_accept']} | {v['false_reject']} | {v['cells']} |" for k, v in scores.items()]
    lines += ['', '| GPU | Model | context | batch | vLLM | Vx exact | Vx admit.vx |', '|---|---|---:|---:|---|---|---|']
    mark = lambda ok: 'fits' if ok else '—'
    lines += [f"| {c['gpu']} | {c['model'].split('/')[1]} | {c['context']} | {c['batch']} | {mark(c['vllm_fits'])} | "
              f"{mark(c['vx_exact'])} | {mark(c['vx_admit'])} |" for c in cells if c['vllm_fits'] != c['vx_exact'] or c['vllm_fits'] != c['vx_admit']]
    (out / 'fit.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='command', required=True)
m = sub.add_parser('measure', help='rent one GPU through Fission and run guest.py')
m.add_argument('gpu', help=f'Modal GPU type; tables use {", ".join(MACHINES)}')
m.add_argument('--models', nargs='+', default=MODELS)
m.add_argument('--duration', default='75m')
m.add_argument('--work', default='55m')
m.add_argument('--budget', default='6')
m.add_argument('--name')
m.add_argument('--approve', action='store_true')
t = sub.add_parser('table', help='run Vx admission for every cell and compare with measurements')
t.add_argument('--vx', default=str(ROOT / 'Vx'))
args = parser.parse_args()
measure(args) if args.command == 'measure' else table(args)
