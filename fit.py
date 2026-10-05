"""Compare Vx admission verdicts with what vLLM actually fits on rented GPUs."""
import argparse, json, math, re, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-72B-Instruct']
MACHINES = {'H100': 'fleet/h100-sxm.vx', 'A100-80GB': 'fleet/a100-80.vx', 'A100-40GB': 'fleet/a100-40.vx', 'H200': 'fleet/h200.vx', 'B200': 'fleet/b200.vx'}
CONTEXTS, BATCHES = [4096, 8192, 16384, 32768], [1, 4, 16, 64]
VLLM, DEFAULT_UTIL = '0.31.0', 0.92  # vLLM 0.31's CacheConfig default; runs that did not pass a utilization used it
UTILS = [0.9, 0.92, 0.95]


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
    output = f'{args.gpu}{args.tag}'
    try:
        state = fission('open', name, '--plan', plan['id'], '--approve')
        if not (state.get('guestGpu') or {}).get('verified'):
            raise RuntimeError(f"GPU check failed: {state.get('preparationError')}")
        print(json.dumps(state['guestGpu']['devices']))
        fission('wait', name, 'bootstrap', '--duration', '10m', '--max-spend', '0.01')
        fission('upload', name, str(ROOT / 'guest.py'), '/workspace/guest.py')
        fission('run', name, 'fit', '--duration', args.work, '--', 'sh', '-c',
                'command -v cc >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc g++ >/dev/null); '
                f'pip install -q vllm=={VLLM} && python3 /workspace/guest.py "$@"', 'guest', args.gpu, *args.models,
                '--utils', *map(str, args.utils), f'--tag={args.tag}', f'--attention-backend={args.attention_backend}', *(['--skip-native'] if args.skip_native else []))
        job = fission('wait', name, 'fit', '--duration', args.work, '--max-spend', '0.05', check=False)
        print(json.dumps(job if isinstance(job, str) else {k: job.get(k) for k in ['phase', 'waitingStopped']}))
        target = ROOT / 'measurements'
        target.mkdir(exist_ok=True)
        for remote, local in [(f'/workspace/out/{output}.json', f'{output}.json'),
                              ('/workspace/.fission/jobs/fit/output.log', f'{output}.log')]:
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


def probe(args):
    # One short sandbox per GPU: run probe.py once and keep its JSON line.
    name = args.name or f'vxcap-{args.gpu.lower()}-{time.strftime("%m%d%H%M")}'
    plan = fission('plan', name, '--provider', 'modal-tempo', '--gpu', args.gpu, '--duration', args.duration, '--budget', args.budget)
    print(json.dumps({k: plan.get(k) for k in ['id', 'body', 'creationQuote', 'totalCap']}))
    if not args.approve:
        return print('Preview only; repeat with --approve to pay.')
    try:
        state = fission('open', name, '--plan', plan['id'], '--approve')
        if not (state.get('guestGpu') or {}).get('verified'):
            raise RuntimeError(f"GPU check failed: {state.get('preparationError')}")
        fission('wait', name, 'bootstrap', '--duration', '5m', '--max-spend', '0.005')
        fission('upload', name, str(ROOT / 'probe.py'), '/workspace/probe.py')
        result = subprocess.run(['fission', 'advanced', 'exec', name, '--json', '--', 'python3', '/workspace/probe.py'],
                                capture_output=True, text=True)
        output = json.loads(result.stdout)
        line = next(l for l in output['stdout'].splitlines() if l.startswith('PROBE '))
        target = ROOT / 'measurements' / 'capacity'
        target.mkdir(parents=True, exist_ok=True)
        (target / f'{args.gpu}.json').write_text(json.dumps({'gpu': args.gpu, **json.loads(line[6:])}, indent=1) + '\n')
        print(line)
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


def residents(s, context, batch, reserve=0):
    # The three residents of fleet/admit.vx, sized from the real checkpoint, plus an optional engine reserve.
    tensors = [('weights', [math.ceil(s['weight_bytes'] / 2 / s['d']), s['d']]),
               ('kv_cache', [2 * s['layers'] * batch * context, s['kv_heads'] * s['head_dim']]),
               ('activations', [batch * context, s['d']])]
    if reserve:
        tensors.append(('engine_reserve', [math.ceil(reserve / 2 / 4096), 4096]))
    return tensors


def exact_program(tensors):
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


def machine_file(vx, machine, capacity, path):
    # A recorded copy of the fleet file. Optionally replace the HBM capacity; always restore the
    # HBM -> CPU_DRAM edge some fleet files omit (h200.vx), which admit.vx needs to read results back.
    source = (vx / machine).read_text()
    if capacity is not None:
        source, count = re.subn(r'(Memory HBM \{\s*capacity: )[^,]+,', rf'\g<1>{capacity} B,', source)
        assert count == 1, f'{machine}: HBM capacity not found'
    patched = 'Memory::HBM -> Memory::CPU_DRAM' not in source
    if patched:
        edge = re.search(r'\n(\s*)transfer Memory::CPU_DRAM -> Memory::HBM : ([^,]+),', source)
        source = source.replace(edge.group(0), edge.group(0) + f'\n{edge.group(1)}transfer Memory::HBM -> Memory::CPU_DRAM : {edge.group(2)},')
    path.write_text(source)
    return path, patched


def admit(vx, machine, source, path):
    path.write_text(source)
    diagnostics = path.with_suffix('.json')
    diagnostics.unlink(missing_ok=True)
    run = subprocess.run([str(vx / 'target/debug/vxc'), '--machine', str(machine), '--host', str(vx / 'fleet/host-x86-e5-2666v3.vx'),
                          str(path), '--action', 'emit-mlir', '--diagnostics-json', str(diagnostics)], capture_output=True, text=True)
    record = json.loads(diagnostics.read_text())
    if record['verdict'] not in ['admitted', 'rejected'] or (run.returncode == 0) != (record['verdict'] == 'admitted'):
        raise RuntimeError(f'{path.name}: {run.stderr[-2000:]}')
    if record['verdict'] == 'rejected' and not any(d['code'] in ['E6009', 'E6010'] for d in record['diagnostics']):
        raise RuntimeError(f'{path.name} rejected for a reason other than capacity: {record["diagnostics"]}')
    resident = [r['total_bytes'] for r in record.get('resident_sets', []) if r['space'] == 'HBM']
    return record['verdict'] == 'admitted', max(resident, default=None)


def launches(gpu):
    # Earlier baseline files and later sweep files for one GPU, merged per model.
    merged = {}
    for path in sorted((ROOT / 'measurements').glob(f'{gpu}*.json')):
        report = json.loads(path.read_text())
        if report.get('gpu') != gpu:
            continue
        for model, entry in report['models'].items():
            smi_bytes = int(report['nvidia_smi'].split(',')[1]) * 2**20
            merged.setdefault(model, []).extend({**l, 'util': l.get('util', DEFAULT_UTIL), 'file': path.name, 'smi_bytes': smi_bytes}
                                                for l in entry['launches'])
    return merged


def fits(runs, context, batch, native):
    # A configuration fits only if it fitted in every repeated vLLM start at that utilization.
    return bool(runs) and context <= native and all(l['loaded'] and batch * context <= l.get('kv_tokens', 0) for l in runs)


def table(args):
    vx = Path(args.vx).resolve()
    out = ROOT / 'results'
    for sub in ['vx', 'machines']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / 'paths.json').write_text(json.dumps({'vx': str(vx)}) + '\n')
    cells, summary, checks = [], [], {'bytes_compared': 0, 'bytes_mismatched': 0}
    variants = {'vx_admit': 'fleet/admit.vx formula, fleet capacity', 'vx_fleet': 'exact weights, fleet capacity',
                'vx_device': 'exact weights, device capacity', 'vx_reserve': "exact, device capacity, reserve = vLLM's unused share"}
    for gpu, fleet in MACHINES.items():
        runs = launches(gpu)
        # What CUDA reports as allocatable on this GPU type; nvidia-smi's larger total includes memory the driver keeps.
        totals = [l['total_bytes'] for records in runs.values() for l in records if l.get('total_bytes')]
        for model, records in runs.items():
            s = shape(model)
            short = [l for l in records if l['max_model_len'] == 4096 and not l.get('long_prompt')]
            by_util = {u: [l for l in short if l['util'] == u] for u in sorted({l['util'] for l in short})}
            if not by_util:
                continue
            device = totals[0] if totals else records[0]['smi_bytes']
            fleet_bytes = int(re.search(r'Memory HBM \{\s*capacity: ([\d.]+) GiB', (vx / fleet).read_text()).group(1)) * 2**30
            name = model.split('/')[1]
            device_file, patched = machine_file(vx, fleet, device, out / 'machines' / f'{gpu}-device.vx')
            fleet_file, _ = machine_file(vx, fleet, None, out / 'machines' / f'{gpu}-fleet.vx')
            summary.append({'gpu': gpu, 'model': name, 'fleet_bytes': fleet_bytes, 'return_edge_added': patched, 'device_bytes': device, 'weight_bytes': s['weight_bytes'],
                            'kv_bytes_per_token': 2 * s['layers'] * s['kv_heads'] * s['head_dim'] * 2,
                            'kv_tokens': {u: [l.get('kv_tokens') if l['loaded'] else None for l in runs] for u, runs in by_util.items()},
                            'other_gib': {u: [round(u * device / 2**30 - l['weights_gib'] - l['kv_gib'], 2) for l in runs if l['loaded'] and l.get('kv_gib')]
                                          for u, runs in by_util.items()},
                            'native': [{'util': l['util'], 'ran': l['loaded'], 'prompt_tokens': l.get('prompt_tokens'),
                                        'refused_at': l.get('estimated_max_len')} for l in records if l['max_model_len'] == 32768]})
            for context in CONTEXTS:
                for batch in BATCHES:
                    tag = f'{gpu}-{name}-{context}-{batch}'
                    tensors = residents(s, context, batch)
                    expected = sum(a * b * 2 for _, (a, b) in tensors)
                    cell = {'gpu': gpu, 'model': name, 'context': context, 'batch': batch, 'resident_bytes': expected}
                    for util in UTILS:
                        cell[f'vllm_{util}'] = fits(by_util[util], context, batch, s['native_context']) if util in by_util else None
                    cell['vx_fleet'], used = admit(vx, fleet_file, exact_program(tensors), out / 'vx' / f'{tag}-fleet.vx')
                    if cell['vx_fleet']:
                        checks['bytes_compared'] += 1
                        checks['bytes_mismatched'] += used != expected
                    cell['vx_device'], _ = admit(vx, device_file, exact_program(tensors), out / 'vx' / f'{tag}-device.vx')
                    for util in UTILS:
                        if cell[f'vllm_{util}'] is not None:
                            program = exact_program(residents(s, context, batch, round((1 - util) * device)))
                            cell[f'vx_reserve_{util}'], _ = admit(vx, device_file, program, out / 'vx' / f'{tag}-reserve-{util}.vx')
                    cell['vx_admit'], _ = admit(vx, fleet_file, reference_program(vx, s, context, batch), out / 'vx' / f'{tag}-admit.vx')
                    cells.append(cell)
    if not cells:
        sys.exit('No measurements yet; run measure first.')

    def score(key, util):
        truth = f'vllm_{util}'
        key = f'vx_reserve_{util}' if key == 'vx_reserve' else key
        subset = [c for c in cells if c[truth] is not None]
        return {'agree': sum(c[key] == c[truth] for c in subset), 'false_accept': sum(c[key] and not c[truth] for c in subset),
                'false_reject': sum(c[truth] and not c[key] for c in subset), 'cells': len(subset),
                'false_accepts': [f"{c['gpu']} {c['model']} {c['context']}x{c['batch']}" for c in subset if c[key] and not c[truth]],
                'false_rejects': [f"{c['gpu']} {c['model']} {c['context']}x{c['batch']}" for c in subset if c[truth] and not c[key]]}
    scores = {key: {str(util): score(key, util) for util in UTILS} for key in variants}
    (out / 'fit.json').write_text(json.dumps({'vllm': VLLM, 'checks': checks, 'scores': scores, 'models': summary, 'cells': cells}, indent=1) + '\n')
    gib = lambda b: f'{b / 2**30:.1f}'
    fmt = lambda v: f"{v['agree']}/{v['cells']} · {v['false_accept']} FA · {v['false_reject']} FR"
    lines = [f"Vx resident bytes equal the arithmetic sum in {checks['bytes_compared'] - checks['bytes_mismatched']} of {checks['bytes_compared']} admitted cells.", '',
             '| Vx program | vs vLLM util 0.90 | vs vLLM util 0.92 (default) | vs vLLM util 0.95 |', '|---|---|---|---|']
    lines += [f"| {label} | {' | '.join(fmt(scores[k][str(u)]) for u in UTILS)} |" for k, label in variants.items()]
    tokens = lambda runs: ' / '.join(f'{t:,.0f}' if t else 'fails' for t in runs)
    lines += ['', '| GPU | Model | fleet GiB | device GiB | weights GiB | KV tokens 0.90 (starts) | KV tokens 0.92 | KV tokens 0.95 | runtime overhead GiB | 32k prompt |',
              '|---|---|---:|---:|---:|---|---|---|---|---|']
    for m in summary:
        native = ', '.join(f"{n['util']}: {'ran ' + format(n['prompt_tokens'], ',') if n['ran'] else 'refused' if n['refused_at'] else 'OOM'}" for n in m['native'])
        lines.append(f"| {m['gpu']} | {m['model'].replace('-Instruct', '')} | {gib(m['fleet_bytes'])} | {gib(m['device_bytes'])} | {gib(m['weight_bytes'])} | "
                     f"{tokens(m['kv_tokens'].get(0.9, [])) or '—'} | {tokens(m['kv_tokens'].get(0.92, [])) or '—'} | {tokens(m['kv_tokens'].get(0.95, [])) or '—'} | "
                     f"{', '.join(sorted({str(round(o, 1)) for os in m['other_gib'].values() for o in os})) or '—'} | {native} |")
    capacity = []
    for gpu, fleet in MACHINES.items():
        probe_file = ROOT / 'measurements' / 'capacity' / f'{gpu}.json'
        if not probe_file.exists():
            continue
        p = json.loads(probe_file.read_text())
        fleet_file, _ = machine_file(vx, fleet, None, out / 'machines' / f'{gpu}-fleet.vx')
        # One tensor 2 MiB larger than the largest allocation the device accepted.
        elements = (p['largest_single_bytes'] + 2 * 2**20) // 2
        program = exact_program([('just_too_big', [elements // 4096, 4096])])
        admitted, _ = admit(vx, fleet_file, program, out / 'vx' / f'{gpu}-largest-plus-2mib.vx')
        fleet_bytes = int(re.search(r'Memory HBM \{\s*capacity: ([\d.]+) GiB', (vx / fleet).read_text()).group(1)) * 2**30
        capacity.append({'gpu': gpu, 'fleet_bytes': fleet_bytes, **{k: p[k] for k in ['smi_total_bytes', 'cuda_total_bytes', 'free_after_context_bytes', 'largest_single_bytes', 'fillable_bytes']},
                         'vx_admits_tensor_device_refuses': admitted})
    data = json.loads((out / 'fit.json').read_text())
    data['capacity'] = capacity
    (out / 'fit.json').write_text(json.dumps(data, indent=1) + '\n')
    lines += ['', '| GPU | fleet file GiB | nvidia-smi GiB | CUDA total GiB | largest single allocation GiB | fleet overstates by | Vx admits one tensor the device refuses |',
              '|---|---:|---:|---:|---:|---:|---|']
    lines += [f"| {c['gpu']} | {gib(c['fleet_bytes'])} | {gib(c['smi_total_bytes'])} | {gib(c['cuda_total_bytes'])} | {c['largest_single_bytes'] / 2**30:.2f} | "
              f"{(c['fleet_bytes'] - c['largest_single_bytes']) / 2**30:.2f} GiB ({(c['fleet_bytes'] / c['largest_single_bytes'] - 1) * 100:.1f}%) | "
              f"{'yes' if c['vx_admits_tensor_device_refuses'] else 'no'} |" for c in capacity]
    (out / 'fit.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='command', required=True)
m = sub.add_parser('measure', help='rent one GPU through Fission and run guest.py')
m.add_argument('gpu', help=f'Modal GPU type; tables use {", ".join(MACHINES)}')
m.add_argument('--models', nargs='+', default=MODELS[:3])
m.add_argument('--utils', nargs='+', type=float, default=[0.9])
m.add_argument('--tag', default='', help='suffix for the measurement file, such as --tag=-sweep')
m.add_argument('--attention-backend', default='', help='vLLM attention backend override, such as TRITON_ATTN')
m.add_argument('--skip-native', action='store_true', help='repeat starts only; no 32k-prompt launches')
m.add_argument('--duration', default='75m')
m.add_argument('--work', default='55m')
m.add_argument('--budget', default='6')
m.add_argument('--name')
m.add_argument('--approve', action='store_true')
p = sub.add_parser('probe', help='measure allocatable device memory on one GPU through Fission')
p.add_argument('gpu')
p.add_argument('--duration', default='8m')
p.add_argument('--budget', default='1.2')
p.add_argument('--name')
p.add_argument('--approve', action='store_true')
t = sub.add_parser('table', help='run Vx admission for every cell and compare with measurements')
t.add_argument('--vx', default=str(ROOT / 'Vx'))
args = parser.parse_args()
{'measure': measure, 'probe': probe, 'table': table}[args.command](args)
