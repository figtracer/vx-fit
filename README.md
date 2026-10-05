# vx-fit

Do [Vx](https://github.com/vx-lang/Vx) admission verdicts match what vLLM actually fits on real GPUs?

For three Qwen2.5 checkpoints on rented H100 and A100-80GB GPUs, Vx's verdict is compared
with vLLM's own memory accounting. That covers 16 serving configurations per model and GPU:
context 4k–32k × batch 1–64.

## Run

Requires Rust, LLVM 22, Z3, CMake, Python 3.10+ and [Fission](https://github.com/figtracer/fission) with a funded Tempo wallet.

```sh
git clone --recurse-submodules https://github.com/figtracer/vx-fit.git
cd vx-fit
(cd Vx && ./setup.sh && source config.local && cargo build --locked --bin vxc -p vxc)
python3 fit.py measure H100 --approve        # rents one GPU through Fission, about 4.6 USDC.e
python3 fit.py measure A100-80GB --approve   # about 2.9 USDC.e
python3 fit.py table
```

Without `--approve`, `measure` only prints the quote. Paid measurements are committed in
`measurements/`, so `table` runs without renting anything. Generated Vx programs and tables go to ignored `results/`.

## Method

- **vLLM truth:** each checkpoint is started once with vLLM 0.31.0 defaults (`gpu_memory_utilization` 0.9,
  `max_model_len` 4096) and must generate text. Its reported KV cache capacity decides each cell:
  a configuration fits if `batch × context` tokens fit in that capacity and the context is within the
  model's 32k limit. When the capacity is below 32k, vLLM is also started at 32k to observe its refusal directly.
- **Vx exact:** weights, KV cache and activations placed in `Memory::HBM`, as in `fleet/admit.vx`, but with
  the real checkpoint size. Admitted against the pinned `fleet/h100-sxm.vx` and `fleet/a100-80.vx`.
- **Vx admit.vx:** the unmodified reference program, using its `12·layers·d²` weight formula.

## Results

Measured on October 5, 2026 (`measurements/`). Vx never rejected a configuration that vLLM fits, but it
**admitted 11 of 96 that vLLM does not fit**. The unmodified `admit.vx` admitted 15.

| Vx program | agree | false accept | false reject |
|---|---:|---:|---:|
| exact checkpoint size | 85 | 11 | 0 |
| `fleet/admit.vx` | 81 | 15 | 0 |

| GPU | Model | device GiB | weights GiB | KV GiB | other GiB | vLLM KV tokens | Vx implied tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| H100 | Qwen2.5-7B | 79.2 | 14.3 | 54.7 | 2.2 | 1,024,816 | 1,095,425 |
| H100 | Qwen2.5-14B | 79.2 | 27.6 | 41.2 | 2.5 | 224,928 | 272,467 |
| H100 | Qwen2.5-32B | 79.2 | 61.0 | 6.3 | 3.9 | 25,744 | 74,789 |
| A100-80GB | Qwen2.5-7B | 79.3 | 14.3 | 56.8 | 0.2 | 1,063,680 | 1,095,425 |
| A100-80GB | Qwen2.5-14B | 79.3 | 27.6 | 42.9 | 0.8 | 234,320 | 272,467 |
| A100-80GB | Qwen2.5-32B | 79.3 | 61.0 | 9.2 | 1.0 | 37,824 | 74,789 |

- Every disagreement is a false accept near the capacity boundary. Vx counts all 80 GiB declared
  by the machine file. vLLM sees about 79.2 GiB, uses 90% of it, and keeps 0.2–3.9 GiB for its runtime.
  For the 32B model, that leaves a third of the KV cache Vx assumes on H100 and half on A100.
- Direct case: Qwen2.5-32B with a 32k context on H100. Vx admits it (69.3 GiB). vLLM refuses to start:
  *estimated maximum model length is 25728*.
- `admit.vx`'s `12·layers·d²` formula gives 57% of the real 7B weights and 61% of the 32B weights. It omits
  embeddings and the output layer and assumes a 4·d MLP. The 14B estimate happens to land within 2%.
  The underestimate accounts for its four extra false accepts, all on 32B.

## Limits

- vLLM's accounting is the reference; concurrent requests were not driven to saturation.
- The sandbox image lacks `nvcc`, so FlashInfer's sampler is disabled (`VLLM_USE_FLASHINFER_SAMPLER=0`);
  `gcc` is installed for torch.compile. Neither changes the memory reservation.
- One run per GPU; Modal sandboxes under gVisor, not bare metal.
