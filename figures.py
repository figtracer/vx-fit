"""Draw the README figures as plain SVG from results/ and measurements/ (run fit.py table first)."""
import json, math, statistics, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
INK, MUTED, GRID = '#1f2328', '#656d76', '#d8dee4'
BLUE, ORANGE, GREEN, RED, GREY = '#0969da', '#bc4c00', '#1a7f37', '#cf222e', '#afb8c1'
RATE = {'H100': 3.9492, 'A100-80GB': 2.4984, 'A100-40GB': 2.0988, 'H200': 4.5396, 'B200': 6.2496}  # USDC.e per hour, Modal quotes on 2026-10-05
WIDTH = 920


def text(x, y, value, size=13, anchor='start', color=INK, weight='normal'):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{value}</text>'


def rect(x, y, w, h, color):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 1):.1f}" height="{h}" fill="{color}" rx="2"/>'


def line(x1, y1, x2, y2, color=GRID, width=1, dash=''):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""}/>'


def svg(name, height, title, subtitle, body, note):
    head = [text(24, 34, title, 18, weight='600'), text(24, 56, subtitle, 13, color=MUTED)]
    foot = [text(24, height - 16, note, 11.5, color=MUTED)]
    (OUT / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
                            f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">'
                            f'<rect width="100%" height="100%" fill="white"/>{"".join(head + body + foot)}</svg>\n')


def errors(fit):
    steps = [('vx_admit', 'fleet/admit.vx as shipped', 'reference program and machine files'),
             ('vx_fleet', 'exact checkpoint weights', 'replace the 12·layers·d² estimate'),
             ('vx_device', 'device capacity', 'what CUDA reports, not the spec sheet'),
             ('vx_reserve', "vLLM's 8% reserve", 'one extra tensor in the program')]
    left, span, row = 300, 420, 58
    total = fit['scores']['vx_admit']['0.92']['cells']
    scale = max(fit['scores'][k]['0.92']['false_accept'] + fit['scores'][k]['0.92']['false_reject'] for k, _, _ in steps)
    body = []
    for i, (key, label, detail) in enumerate(steps):
        s, y = fit['scores'][key]['0.92'], 86 + row * i
        fa, fr = s['false_accept'], s['false_reject']
        body += [text(left - 14, y + 16, label, 14, 'end', weight='600'), text(left - 14, y + 34, detail, 12, 'end', MUTED),
                 rect(left, y + 4, span * fa / scale, 26, RED), rect(left + span * fa / scale, y + 4, span * fr / scale, 26, GREY) if fr else '',
                 text(left + span * (fa + fr) / scale + 10, y + 22, f'{fa + fr} wrong · {s["agree"]}/{total} agree', 13, weight='600')]
    legend_y = 86 + row * len(steps) + 6
    body += [rect(left, legend_y, 12, 12, RED), text(left + 18, legend_y + 11, 'Vx admits, vLLM cannot serve', 12, color=MUTED),
             rect(left + 220, legend_y, 12, 12, GREY), text(left + 238, legend_y + 11, 'Vx refuses, vLLM serves', 12, color=MUTED)]
    gpus = ', '.join(sorted({c['gpu'] for c in fit['cells'] if c.get('vllm_0.92') is not None}))
    svg('errors.svg', legend_y + 50, "Where Vx's wrong fit verdicts came from",
        f'Each step fixes one input to the same Vx check; {total} configurations compared with vLLM 0.31 at its defaults.', body,
        f'{gpus} · Qwen2.5 7B–32B · context 4k–32k × batch 1–64 · gpu_memory_utilization 0.92.')


def capacity(fit):
    rows = fit['capacity']
    left, right, row = 150, WIDTH - 230, 52
    low, high = 0, max(c['fleet_bytes'] for c in rows) / 2**30
    x = lambda gib: left + (right - left) * (gib - low) / (high - low)
    body = []
    for tick in range(0, int(high) + 1, 40):
        body += [line(x(tick), 76, x(tick), 76 + row * len(rows), GRID), text(x(tick), 92 + row * len(rows), f'{tick} GiB', 11, 'middle', MUTED)]
    for i, c in enumerate(rows):
        y = 84 + row * i
        declared, usable = c['fleet_bytes'] / 2**30, c['largest_single_bytes'] / 2**30
        gap = declared - usable
        color = RED if gap > 5 else ORANGE
        body += [text(left - 14, y + 22, c['gpu'], 14, 'end', weight='600'),
                 rect(left, y + 8, x(usable) - left, 20, BLUE),
                 line(x(usable), y + 18, x(declared), y + 18, color, 3),
                 f'<circle cx="{x(declared):.1f}" cy="{y + 18}" r="6" fill="{color}"/>',
                 text(x(declared) + 14, y + 16, f'{gap:.1f} GiB over', 13, color=color, weight='600'),
                 text(x(declared) + 14, y + 32, f'{declared:.0f} declared · {usable:.1f} usable', 11.5, color=MUTED)]
    legend_y = 84 + row * len(rows) + 22
    body += [rect(left, legend_y, 12, 12, BLUE), text(left + 18, legend_y + 11, 'largest single allocation the GPU accepts (cuMemAlloc)', 12, color=MUTED),
             f'<circle cx="{left + 386}" cy="{legend_y + 6}" r="6" fill="{ORANGE}"/>', text(left + 398, legend_y + 11, "capacity in Vx's machine file", 12, color=MUTED)]
    svg('capacity.svg', legend_y + 48, "Vx's machine files vs what one process can allocate",
        'Every file admits a tensor the GPU refuses; on B200, "192 GB" was entered as 192 GiB.', body,
        'Fresh CUDA context on Modal (gVisor), driver 580.95.05. The gap is the CUDA context plus driver reserve, or a unit error.')


def time_to_catch(fit):
    vxroot = Path(json.loads((ROOT / 'results' / 'paths.json').read_text())['vx'])
    program = ROOT / 'results' / 'vx' / 'H200-Qwen2.5-72B-Instruct-4096-1-reserve-0.9.vx'
    machine = ROOT / 'results' / 'machines' / 'H200-device.vx'
    seconds = []
    for _ in range(5):
        start = time.perf_counter()
        subprocess.run([str(vxroot / 'target/debug/vxc'), '--machine', str(machine), '--host', str(vxroot / 'fleet/host-x86-e5-2666v3.vx'),
                        str(program), '--action', 'emit-mlir'], capture_output=True)
        seconds.append(time.perf_counter() - start)
    cells = {(c['gpu'], c['model'], c['context'], c['batch']): c for c in fit['cells']}
    cases = set()
    for path in sorted((ROOT / 'measurements').glob('*.json')):
        report = json.loads(path.read_text())
        for model, entry in report['models'].items():
            for launch in entry['launches']:
                util = launch.get('util', 0.92)
                if launch['loaded'] or util not in (0.9, 0.92, 0.95):
                    continue
                cell = cells.get((report['gpu'], model.split('/')[1], launch['max_model_len'], 1), {})
                wall = entry['download_seconds'] + launch['seconds']
                kind = 'refuses to start' if launch.get('estimated_max_len') else 'out of memory'
                cases.add((f"{report['gpu']} · {model.split('-')[1]} · {launch['max_model_len'] // 1024}k context · {util:.2f}", kind, wall,
                           wall * RATE[report['gpu']] / 3600, cell.get(f'vx_reserve_{util}') is False))
    rows = [('Vx compiler', 'any configuration, no GPU', statistics.median(seconds), 0, True)] + sorted(cases, key=lambda c: -c[2])
    left, right, row = 330, WIDTH - 250, 44
    lo, hi = math.log10(0.01), math.log10(1000)
    x = lambda s: left + (right - left) * (math.log10(s) - lo) / (hi - lo)
    body = []
    for tick, label in [(0.01, '10 ms'), (0.1, '100 ms'), (1, '1 s'), (10, '10 s'), (60, '1 min'), (600, '10 min')]:
        body += [line(x(tick), 74, x(tick), 74 + row * len(rows), GRID), text(x(tick), 90 + row * len(rows), label, 11, 'middle', MUTED)]
    for i, (label, kind, wall, cost, caught) in enumerate(rows):
        y = 80 + row * i
        vx = i == 0
        color = GREEN if vx else ORANGE if caught else RED
        verdict = '' if vx else ('Vx also refuses' if caught else 'Vx admits it (miss)')
        body += [text(left - 14, y + 14, label, 13, 'end', weight='600'), text(left - 14, y + 30, kind, 12, 'end', MUTED),
                 rect(left, y + 6, x(wall) - left, 22, color),
                 text(x(wall) + 10, y + 16, f'{wall:.2f} s' if vx else f'{wall / 60:.1f} min · {cost:.2f} USDC.e', 13, weight='600'),
                 text(x(wall) + 10, y + 31, verdict or 'compile time', 11.5, color=color if not vx else MUTED)]
    svg('time-to-catch.svg', 80 + row * len(rows) + 52, 'Time to learn a configuration does not fit',
        'Vx answers at compile time; vLLM downloads the checkpoint and starts the engine before it can say no.', body,
        "Log scale. vLLM time = download + engine start until failure (excludes provisioning). Vx = the reserve program matching vLLM's utilization.")


def main():
    OUT.mkdir(exist_ok=True)
    fit = json.loads((ROOT / 'results' / 'fit.json').read_text())
    errors(fit)
    capacity(fit)
    time_to_catch(fit)
    print('wrote', ', '.join(p.name for p in sorted(OUT.glob('*.svg'))))


main()
