"""Draw the README figures as plain SVG from results/ and measurements/ (run fit.py table first)."""
import json, statistics, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
INK, MUTED, GRID = '#1f2328', '#656d76', '#d0d7de'
BLUE, ORANGE, GREEN, RED, GREY = '#0969da', '#bc4c00', '#1a7f37', '#cf222e', '#8c959f'
RATE = {'H100': 3.9492, 'A100-80GB': 2.4984, 'A100-40GB': 2.0988, 'H200': 4.5396, 'B200': 6.2496}  # USDC.e per hour, Modal quotes on 2026-10-05


def text(x, y, value, size=12, anchor='start', color=INK, weight='normal'):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{value}</text>'


def svg(name, width, height, title, body):
    content = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
               f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">'
               f'<rect width="100%" height="100%" fill="white"/>{text(16, 26, title, 15, weight="600")}{"".join(body)}</svg>\n')
    (OUT / name).write_text(content)


def bars(name, title, rows, unit, width=760, label_width=300, scale=None, note=''):
    # rows: (label, value, color, annotation)
    height = 60 + 30 * len(rows) + (24 if note else 0)
    top = max(v for _, v, _, _ in rows) if scale is None else scale
    room = 7 * max(len(a or '') for _, _, _, a in rows) + 16  # leave the longest annotation fully visible
    span = width - label_width - room
    body = []
    for i, (label, value, color, annotation) in enumerate(rows):
        y = 48 + 30 * i
        length = span * value / top if top else 0
        body += [text(label_width - 10, y + 15, label, 12, 'end'),
                 f'<rect x="{label_width}" y="{y + 3}" width="{max(length, 1):.1f}" height="18" fill="{color}" rx="2"/>',
                 text(label_width + length + 6, y + 16, annotation or f'{value:g} {unit}', 12, color=MUTED)]
    if note:
        body.append(text(16, height - 12, note, 11, color=MUTED))
    svg(name, width, height, title, body)


def main():
    OUT.mkdir(exist_ok=True)
    fit = json.loads((ROOT / 'results' / 'fit.json').read_text())
    labels = {'vx_admit': 'fleet/admit.vx as shipped', 'vx_fleet': 'exact checkpoint weights',
              'vx_device': '+ capacity the device reports', 'vx_reserve': "+ vLLM's unused 8% as a reserve tensor"}
    rows = []
    for key, label in labels.items():
        s = fit['scores'][key]['0.92']
        rows.append((label, s['false_accept'], RED if s['false_accept'] else GREEN,
                     f"{s['false_accept']} false accepts · {s['false_reject']} false rejects"))
    bars('errors.svg', 'Vx admits configurations vLLM cannot serve — until the program says what vLLM reserves', rows, '',
         width=900, note=f"{s['cells']} cells ({', '.join(sorted({c['gpu'] for c in fit['cells'] if c.get('vllm_0.92') is not None}))}; Qwen2.5 7B–32B; context 4k–32k × batch 1–64) vs vLLM 0.31 defaults (utilization 0.92).")

    rows = []
    for c in fit['capacity']:
        gib = lambda b: b / 2**30
        rows += [(f"{c['gpu']} · fleet file", gib(c['fleet_bytes']), RED if c['fleet_bytes'] > c['smi_total_bytes'] * 1.02 else ORANGE, f"{gib(c['fleet_bytes']):.1f} GiB declared"),
                 (f"{c['gpu']} · nvidia-smi", gib(c['smi_total_bytes']), GREY, f"{gib(c['smi_total_bytes']):.1f} GiB"),
                 (f"{c['gpu']} · largest allocation", gib(c['largest_single_bytes']), BLUE, f"{gib(c['largest_single_bytes']):.2f} GiB usable"),
                 (f"{c['gpu']} · vLLM default budget", 0.92 * gib(c['cuda_total_bytes']), GREEN, f"{0.92 * gib(c['cuda_total_bytes']):.1f} GiB (92%)")]
    bars('capacity.svg', "What Vx's machine files declare vs what one process can allocate", rows, 'GiB', label_width=260,
         note='Largest single allocation via cuMemAlloc on a fresh context (Modal, gVisor). B200: "192 GB" was entered as 192 GiB.')

    # Time and money to learn that a configuration does not fit.
    vx = ROOT / 'results' / 'vx' / 'H200-Qwen2.5-72B-Instruct-4096-1-vx_reserve10.vx'
    machine = ROOT / 'results' / 'machines' / 'H200-device.vx'
    vxroot = Path(json.loads((ROOT / 'results' / 'paths.json').read_text())['vx'])
    seconds = []
    for _ in range(5):
        start = time.perf_counter()
        subprocess.run([str(vxroot / 'target/debug/vxc'), '--machine', str(machine), '--host', str(vxroot / 'fleet/host-x86-e5-2666v3.vx'),
                        str(vx), '--action', 'emit-mlir'], capture_output=True)
        seconds.append(time.perf_counter() - start)
    vx_seconds = statistics.median(seconds)
    cells = {(c['gpu'], c['model'], c['context'], c['batch']): c for c in fit['cells']}
    cases = []
    for path in sorted((ROOT / 'measurements').glob('*.json')):
        report = json.loads(path.read_text())
        for model, entry in report['models'].items():
            for launch in entry['launches']:
                util = launch.get('util', 0.92)
                if launch['loaded'] or util not in (0.9, 0.92, 0.95):
                    continue
                cell = cells.get((report['gpu'], model.split('/')[1], launch['max_model_len'], 1), {})
                caught = cell.get(f'vx_reserve_{util}') is False
                wall = entry['download_seconds'] + launch['seconds']
                why = 'refused' if launch.get('estimated_max_len') else 'OOM'
                cases.append((f"{report['gpu']} {model.split('-')[1]} util {util} ctx {launch['max_model_len'] // 1024}k: {why}", wall,
                              wall * RATE[report['gpu']] / 3600, caught))
    rows = [('Vx: compile one admission program', vx_seconds, GREEN, f'{vx_seconds:.2f} s · no GPU')]
    rows += [(label, wall, ORANGE if caught else RED, f"{wall:,.0f} s · {cost:.2f} USDC.e · Vx {'refused it too' if caught else 'admitted it (miss)'}")
             for label, wall, cost, caught in sorted(set(cases), key=lambda c: -c[1])]
    bars('time-to-catch.svg', 'Time to learn a configuration does not fit: Vx at compile time vs vLLM on the GPU', rows, 's', width=980, label_width=300,
         note="vLLM = checkpoint download + engine start until it fails (no provisioning/pip). Vx = the reserve program matching vLLM's utilization.")

    print('wrote', ', '.join(p.name for p in sorted(OUT.glob('*.svg'))))


main()
