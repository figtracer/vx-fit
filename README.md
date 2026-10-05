# vx-fit

Do [Vx](https://github.com/vx-lang/Vx) admission verdicts match what vLLM actually fits on real GPUs?

Qwen2.5 7B–72B on rented H100, A100-80GB, H200 and B200 GPUs. For each model and GPU, Vx's compile-time
verdict is compared with vLLM 0.31.0 at 16 serving configurations: context 4k–32k × batch 1–64.

## Run

Requires Rust, LLVM 22, Z3, CMake, Python 3.10+ and [Fission](https://github.com/figtracer/fission) with a funded Tempo wallet.

```sh
git clone --recurse-submodules https://github.com/figtracer/vx-fit.git
cd vx-fit
(cd Vx && ./setup.sh && source config.local && cargo build --locked --bin vxc -p vxc)
python3 fit.py measure H100 --utils 0.9 0.95 --tag=-sweep --approve   # rents one GPU through Fission
python3 fit.py table
```

Without `--approve`, `measure` only prints the quote. Paid measurements are committed in `measurements/`
(failed lab runs are in `measurements/lab-failures/`), so `table` runs without renting anything.

## Method

- **vLLM:** each checkpoint is started with `max_model_len` 4096 at `gpu_memory_utilization` 0.90 (default)
  and 0.95, and must generate text. The KV cache vLLM allocates decides each cell: it fits if
  `batch × context` tokens fit in every repeated start. Boundary models are also run with one real 32k-token prompt.
- **Vx:** weights, KV cache and activations placed in `Memory::HBM`, as in `fleet/admit.vx`, but sized from the
  real checkpoint. Variants: the pinned fleet capacity, the capacity CUDA reports on the device, and the same plus an
  engine reserve tensor encoding vLLM's utilization policy. `fleet/admit.vx` itself runs unmodified as a baseline.

## Results

Measured October 5, 2026. Vx's resident-set arithmetic was exact in every admitted cell (98/98).

| Vx program | vs vLLM 0.90 (default) | vs vLLM 0.95 |
|---|---|---|
| `fleet/admit.vx` formula, fleet capacity | 117/144 · 27 false accepts | 80/96 · 16 false accepts |
| exact weights, fleet capacity | 127/144 · 17 false accepts | 88/96 · 8 false accepts |
| exact weights, device capacity | 129/144 · 15 false accepts | 90/96 · 6 false accepts |
| + 10% engine reserve (vLLM 0.90) | **141/144** · 2 FA · 1 FR | 93/96 · 0 FA · 3 FR |
| + 5% engine reserve (vLLM 0.95) | 139/144 · 4 FA · 1 FR | **95/96** · 0 FA · 1 FR |

| GPU | Model | fleet GiB | device GiB | weights GiB | KV tokens 0.90 (runs) | KV tokens 0.95 | 32k prompt |
|---|---|---:|---:|---:|---|---|---|
| H100 | 7B | 80.0 | 79.2 | 14.2 | 995,152 / 1,024,816 | 1,069,616 | |
| H100 | 14B | 80.0 | 79.2 | 27.5 | 216,288 / 224,928 | 237,904 | |
| H100 | 32B | 80.0 | 79.2 | 61.0 | 19,264 / 25,744 | 35,472 | refused at 0.90, **ran at 0.95**, OOM at 0.98–0.99 |
| A100-80GB | 7B | 80.0 | 79.3 | 14.2 | 1,063,680 | — | |
| A100-80GB | 14B | 80.0 | 79.3 | 27.5 | 234,320 | — | |
| A100-80GB | 32B | 80.0 | 79.3 | 61.0 | 37,824 | — | |
| H200 | 32B | 141.0 | 139.8 | 61.0 | 242,736 | 271,360 | |
| H200 | 72B | 141.0 | 139.8 | 135.4 | out of memory at 0.90–0.99 | — | |
| B200 | 32B | **192.0** | **178.4** | 61.0 | 385,728 | 422,256 | |

- Once the admission program encodes the engine's policy, Vx agrees on 95/96 cells with no false accepts.
  The leftover errors come from the program, not the checker. vLLM's runtime overhead (2–5.5 GiB on H100)
  isn't modelled, and `admit.vx`'s activation term assumes every token is live at once.
- `fleet/b200.vx` declares 192 GiB; the device has 179.1 GiB (183,359 MiB, i.e. 192 GB). `fleet/h200.vx` has
  no HBM → host edge, so the unmodified `admit.vx` fails there with E6002. A recorded copy adds the edge.
- `admit.vx`'s `12·layers·d²` weight formula gives 57% of the real 7B weights and 61% of the 32B weights.
- vLLM's own numbers move between runs (H100 32B: 19,264 vs 25,744 tokens). At 0.98–0.99 its startup check
  passed, but the real 32k prompt then ran out of memory.

## Limits

- vLLM's allocated KV cache is the reference; concurrent requests were not driven to saturation.
- One to two starts per configuration; vLLM's overhead varied by up to 1.6 GiB between H100 sessions.
- The sandbox image lacks `nvcc`, so FlashInfer's sampler is disabled (`VLLM_USE_FLASHINFER_SAMPLER=0`);
  `gcc` is installed for torch.compile. B200 uses Triton attention because FlashInfer's Blackwell backend needs `nvcc`.
- Modal sandboxes under gVisor, not bare metal; one GPU per run, no tensor parallelism.
