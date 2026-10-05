# vx-fit

Is [Vx](https://github.com/vx-lang/Vx)'s compile-time memory admission right, on real GPUs and at the edges of its checker?

- **vs vLLM:** Qwen2.5 7B–72B on rented H100, A100-80GB, A100-40GB, H200 and B200. At 16 context × batch configurations per model, Vx's verdict is compared with what vLLM 0.31 allocates.
- **vs the hardware:** declared capacities compared with what one process can allocate on each GPU.
- **vs its own code generation:** edge-case programs where the bytes Vx counts are compared with the bytes it allocates.

## Run

Requires Rust, LLVM 22, Z3, CMake, Python 3.10+ and, for new measurements, [Fission](https://github.com/figtracer/fission) with a funded Tempo wallet.

```sh
git clone --recurse-submodules https://github.com/figtracer/vx-fit.git
cd vx-fit
(cd Vx && ./setup.sh && source config.local && cargo build --locked --bin vxc -p vxc)
python3 edges.py                                   # checker edge cases, no GPU
python3 fit.py table && python3 figures.py         # tables and figures from committed measurements
python3 fit.py measure H100 --utils 0.9 0.95 --tag=-new --approve   # rents one GPU
python3 fit.py probe H100 --approve                # allocatable memory on one GPU
```

Without `--approve`, `measure` and `probe` only print the quote. Paid measurements are committed in `measurements/`,
and failed lab runs in `measurements/lab-failures/`. Generated programs and tables go to ignored `results/`.

## Results

Measured October 5, 2026 against Vx `42ada79`. Filed upstream as Vx#1202–#1206, with pilot data on Vx#289. The code and fleet files behind each finding are unchanged on Vx `main` (`0dd89a4`).

### Checker edges

`edges.py` runs 15 programs. The checker handles everything its design document promises: peaks over sibling
blocks, `continue`/`break`/early `return` (each frees on every path in the emitted IR), escape by assignment,
recursion, tiles held across calls, and copies within a space. Three classes are **unsound**: the program is
admitted, but the generated code allocates more than the space holds.

| case | checker counts | code allocates | why |
|---|---:|---:|---|
| `Tensor<bool, [6291456]>` in 4 MiB | 786,432 B | 6,291,456 B | capacity assumes packed bits; codegen stores one byte per `i1` |
| `Tensor<i4/u4, [6291456]>` in 4 MiB | 3,145,728 B | 6,291,456 B | same: 4 dense bits vs one padded byte |
| `Tensor<f32, [2³⁰, 2³⁰, 4]>` | unverified (W1029) | 0 B (`malloc(0)`) | `checked_mul` overflow returns `None`, read as "dynamic shape" |
| const-generic shape folding to −2²⁰ | unverified (W1029) | 2⁶⁴ − 4 MiB | a negative dimension is also reported as "dynamic shape" |

### Machine files vs hardware

![capacity](figures/capacity.svg)

On every GPU, Vx admits a single tensor 2 MiB larger than the largest one `cuMemAlloc` accepts. The fleet files
declare the marketed total, while the CUDA context and driver keep 0.9–1.7 GiB. `fleet/b200.vx` declares 192 GiB
for a device with 179.1 GiB (183,359 MiB, i.e. 192 GB). `fleet/h200.vx` and `fleet/mi300x.vx` have no HBM → host
edge, so the unmodified `fleet/admit.vx` fails there with E6002 before any capacity verdict.

### Admission vs vLLM

![errors](figures/errors.svg)

| Vx program | vs vLLM util 0.90 | vs vLLM util 0.92 (default) | vs vLLM util 0.95 |
|---|---|---|---|
| `fleet/admit.vx` as shipped | 113/144 · 31 false accepts | 111/128 · 17 FA | 80/96 · 16 FA |
| exact checkpoint weights | 125/144 · 19 FA | 117/128 · 11 FA | 88/96 · 8 FA |
| + device capacity | 127/144 · 17 FA | 117/128 · 11 FA | 90/96 · 6 FA |
| + reserve tensor = vLLM's unused share | **138/144 · 6 FA · 0 FR** | 125/128 · 2 FA · 1 FR | **95/96 · 0 FA · 1 FR** |

- **Weight formula.** `fleet/admit.vx`'s `12·layers·d²` gives 57% of the real 7B weights and 61% of the 32B weights.
- **Policy.** vLLM uses 92% of the device by default. That is policy, which Vx's own `b200.vx` says belongs in the
  admission program, and it is acknowledged in [Vx#289](https://github.com/vx-lang/Vx/issues/289). One reserve tensor encodes it.
- **Policy, not physics.** H100 32B at 32k: vLLM refuses at 0.90–0.92, and a real 32,704-token prompt **runs at 0.95**.
  A100-40GB 14B at 32k: refused at 0.90, and the real prompt runs at 0.92. At 0.98–0.99 vLLM's own startup check
  passes on H100, then the prompt runs out of memory.
- **What is left is runtime overhead the program does not model.** That is 3.8–5.5 GiB on H100, 1.7–2.6 GiB on A100,
  5.5 GiB on H200 and 5.3 GiB on B200. Every remaining false accept is the largest model on its GPU at exactly 32,768
  tokens in flight. vLLM's numbers repeat within ±1.3% across starts.

![time to catch](figures/time-to-catch.svg)

## Limits

- vLLM's allocated KV cache and real prompts are the reference; concurrency was not driven to saturation.
- The sandbox image lacks `nvcc`: FlashInfer's sampler is disabled, `gcc` is installed for torch.compile, and B200
  uses Triton attention. The A100-80GB allocation probe died twice under gVisor; its CUDA total comes from vLLM.
- The first H100/A100 runs left vLLM's utilization at its default, which is 0.92, not 0.90. They are labelled 0.92.
- Modal sandboxes under gVisor, not bare metal; one GPU per run, no tensor parallelism.
