"""Probe Vx's capacity checker at its edges: compare its verdict and byte count with the code it generates."""
import argparse, json, re, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TILE = 'Tensor<f32, [1024, 768]>::uninit()'  # 3 MiB; the space holds 4 MiB, so two tiles never fit.
MACHINE = '''Memory CPU_DRAM {}
Memory W { within: Memory::CPU_DRAM, capacity: 4 MiB, managed: explicit }
Topology Dev {
  arch: nvptx64, memory: Memory::W, visible: [Memory::W],
  transfer Memory::CPU_DRAM -> Memory::W : 10
  transfer Memory::W -> Memory::CPU_DRAM : 10
}
'''


def single(element, dims):
    return f'''fn main() -> i32 {{
  let a = Tensor<{element}, [{dims}]>::uninit();
  let s = transfer(a, Memory::W);
  let _b = transfer(s, Memory::CPU_DRAM);
  return 0;
}}'''


# (name, program, fits at run time?) -- the last field is the true answer the checker should give.
CASES = [
    ('two tiles in sibling if-arms', f'''fn main() -> i32 {{
  let c: i32 = 1;
  if c > 0 {{ let a = {TILE}; let s = transfer(a, Memory::W); let _b = transfer(s, Memory::CPU_DRAM); }}
  if c > 0 {{ let a = {TILE}; let s = transfer(a, Memory::W); let _b = transfer(s, Memory::CPU_DRAM); }}
  return 0;
}}''', True),
    ('tile escapes an if-arm by assignment', f'''fn main() -> i32 {{
  let a = {TILE};
  let mut s = transfer(a, Memory::W);
  let c: i32 = 1;
  if c > 0 {{ let b = {TILE}; s = transfer(b, Memory::W); }}
  let _b = transfer(s, Memory::CPU_DRAM);
  return 0;
}}''', False),
    ('tile escapes a loop body by assignment', f'''fn main() -> i32 {{
  let a = {TILE};
  let mut s = transfer(a, Memory::W);
  for i in 0..4 {{ let b = {TILE}; s = transfer(b, Memory::W); }}
  let _b = transfer(s, Memory::CPU_DRAM);
  return 0;
}}''', False),
    ('continue skips the rest of a loop body', f'''fn main() -> i32 {{
  for i in 0..4 {{
    let a = {TILE};
    let s = transfer(a, Memory::W);
    if i > 0 {{ continue; }}
    let _b = transfer(s, Memory::CPU_DRAM);
  }}
  return 0;
}}''', True),
    ('break leaves a loop, then a second tile', f'''fn main() -> i32 {{
  for i in 0..4 {{
    let a = {TILE};
    let s = transfer(a, Memory::W);
    if i > 0 {{ break; }}
    let _b = transfer(s, Memory::CPU_DRAM);
  }}
  let b = {TILE};
  let t = transfer(b, Memory::W);
  let _c = transfer(t, Memory::CPU_DRAM);
  return 0;
}}''', True),
    ('tile held across a call made in a loop', f'''fn g() -> i32 {{ let y = {TILE}; let _s = transfer(y, Memory::W); return 1; }}
fn main() -> i32 {{
  let x = {TILE};
  let sx = transfer(x, Memory::W);
  let mut t: i32 = 0;
  for i in 0..2 {{ t = t + g(); }}
  let _b = transfer(sx, Memory::CPU_DRAM);
  return t;
}}''', False),
    ('recursion holding a tile', f'''fn r(n: i32) -> i32 {{
  let a = {TILE};
  let s = transfer(a, Memory::W);
  let mut k: i32 = 0;
  if n > 0 {{ k = r(n - 1); }}
  let _b = transfer(s, Memory::CPU_DRAM);
  return k;
}}
fn main() -> i32 {{ return r(3); }}''', False),
    ('copy within the same space', f'''fn main() -> i32 {{
  let a = {TILE};
  let s = transfer(a, Memory::W);
  let s2 = transfer(s, Memory::W);
  let _b = transfer(s2, Memory::CPU_DRAM);
  return 0;
}}''', False),
    ('bool tensor, 6 MiB of elements', single('bool', '6291456'), False),
    ('i4 tensor, 6 MiB of elements', single('i4', '6291456'), False),
    ('u4 tensor, 6 MiB of elements', single('u4', '6291456'), False),
    ('i8 tensor, 3 MiB (control)', single('i8', '3145728'), True),
    ('f32 shape whose bytes overflow 64 bits', single('f32', '1073741824, 1073741824, 4'), False),
    ('const-generic shape that folds negative', '''fn f<const N : i32>() -> i32 {
  let a : Tensor<f32, [N - 2 * N]> = Tensor<f32>([N - 2 * N]);
  let s = transfer(a, Memory::W);
  let _b = transfer(s, Memory::CPU_DRAM);
  return 0;
}
fn main() -> i32 { return f<1048576>(); }''', False),
    ('const-generic N*N past i32 (control)', '''fn f<const N : i32>() -> i32 {
  let a : Tensor<f32, [N * N]> = Tensor<f32>([N * N]);
  let s = transfer(a, Memory::W);
  let _b = transfer(s, Memory::CPU_DRAM);
  return 0;
}
fn main() -> i32 { return f<65536>(); }''', False),
]


def vxc(vx, path, action, diagnostics=None):
    command = f'set -a; . ./config.local; set +a; ./target/debug/vxc {path} --action {action}'
    if diagnostics:
        command += f' --diagnostics-json {diagnostics}'
    return subprocess.run(['sh', '-c', command], cwd=vx, capture_output=True, text=True)


def allocated(llvm):
    # Bytes passed to the first vx_plugin_alloc_and_transfer, folded through the chain of i64 multiplies (wrapping, as LLVM does).
    constants = {k: int(v) for k, v in re.findall(r'(%\d+) = llvm\.mlir\.constant\((-?\d+) : (?:index|i64)\) : i64', llvm)}
    products = {m[0]: (m[1], m[2]) for m in re.findall(r'(%\d+) = llvm\.mul (%\d+), (%\d+) : i64', llvm)}

    def value(name):
        if name in constants:
            return constants[name]
        if name in products:
            left, right = (value(n) for n in products[name])
            return None if left is None or right is None else (left * right) % 2**64
        return None
    call = re.search(r'vx_plugin_alloc_and_transfer\((%\d+),', llvm)
    return value(call.group(1)) if call else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vx', default=str(ROOT / 'Vx'))
    vx = Path(parser.parse_args().vx).resolve()
    out = ROOT / 'results' / 'edges'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (name, body, fits) in enumerate(CASES):
        path, diagnostics = out / f'{index:02d}.vx', out / f'{index:02d}.json'
        path.write_text(MACHINE + body + '\n')
        diagnostics.unlink(missing_ok=True)
        start = time.perf_counter()
        vxc(vx, path, 'emit-mlir', diagnostics)
        seconds = time.perf_counter() - start
        record = json.loads(diagnostics.read_text())
        counted = max([r['total_bytes'] for r in record.get('resident_sets', [])], default=None)
        llvm = vxc(vx, path, 'emit-llvm').stdout if record['verdict'] == 'admitted' else ''
        admitted = record['verdict'] == 'admitted'
        rows.append({'case': name, 'fits_at_run_time': fits, 'verdict': record['verdict'],
                     'codes': sorted({d['code'] for d in record['diagnostics']}), 'checker_bytes': counted,
                     'allocated_bytes': allocated(llvm) if admitted else None, 'seconds': round(seconds, 3),
                     'sound': not (admitted and not fits), 'precise': not (fits and not admitted)})
    (ROOT / 'results' / 'edges.json').write_text(json.dumps(rows, indent=1) + '\n')
    lines = ['| case | fits at run time | Vx verdict | codes | checker bytes | bytes allocated | result |', '|---|---|---|---|---:|---:|---|']
    for r in rows:
        result = 'ok' if r['sound'] and r['precise'] else 'UNSOUND (admits a misfit)' if not r['sound'] else 'refuses a fit'
        lines.append(f"| {r['case']} | {'yes' if r['fits_at_run_time'] else 'no'} | {r['verdict']} | {', '.join(r['codes']) or '—'} | "
                     f"{r['checker_bytes'] if r['checker_bytes'] is not None else '—'} | {r['allocated_bytes'] if r['allocated_bytes'] is not None else '—'} | {result} |")
    (ROOT / 'results' / 'edges.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


main()
