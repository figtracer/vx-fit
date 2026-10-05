# Scope

- Keep this repository a compact working experiment, not a tutorial.
- One question: are Vx admission verdicts right, against vLLM, the hardware, and Vx's own generated code?
- Keep README text to setup, execution, results and material limitations. Explanations belong in a separate Gist.
- Commit paid GPU measurements in `measurements/`; keep generated Vx programs and tables in ignored `results/`.
- Rent GPUs only through Fission with an explicit `--approve`; never repeat a paid step after an unclear outcome.
