# Phase Without Position

**Can complex-valued transformer weights infer the `cos + i·sin` structure of RoPE, removing the need
for an explicit positional module?**

**Answer: no.** The mechanism is falsified at three scales, and the parameter-savings story does not
survive capacity matching. What does survive, at 5 seeds and under both capacity controls, is one
specific capability: complex weights roughly **double exact-match length generalization on order
reversal**, and an ablation attributes the effect to the complex weights themselves.

## Results (Unity 5-seed sweep + ablation, 160/160 runs converged)

Exact-match sequence accuracy past the trained length (train lengths 6-16), mean ± std over 5 seeds:
complex+NoPE (d160) vs a parameter-matched real+NoPE (d224). Full log and claim-by-claim verdicts in
[`docs/results.md`](docs/results.md); every number reproduces from the committed
`results/unity_*/result.json` via `uv run python -m experiments.stats`.

| task | eval length | complex+NoPE | real+NoPE (param-matched) | p (Welch) | verdict |
|---|---|---|---|---|---|
| **reverse** | 24 (1.5x train) | **0.86 ± 0.02** | 0.41 ± 0.08 | **0.0002** | real win; survives equal-dim and equal-param controls |
| copy | 32 | 0.37 ± 0.27 | 0.39 ± 0.36 | 0.92 | parity (extrapolation is bimodal across seeds in every NoPE condition) |
| sort | 96 | 0.27 ± 0.19 | 0.25 ± 0.28 | 0.90 | parity; the pilot's "10x sort" was n=1 seed noise |
| needle | 128 | 0.85 ± 0.28 | 0.92 ± 0.18 | 0.67 | parity; the pilot gap was a capacity artifact |

- **Mechanism: falsified.** Trained complex QK weights show no RoPE-like phase structure at three
  scales (d64, d128, and the d160 sweep checkpoints themselves); entry-phase and eigenvalue-angle
  spectra are statistically indistinguishable from the random init even when the task is fully solved.
- **The reverse win is the complex weights themselves.** Removing the learnable input phase (the only
  other complex-specific ingredient besides CReLU) leaves reverse unchanged (0.84 ± 0.06 vs 0.86 ± 0.02,
  p = 0.68) and still beats the param-matched baseline by the full margin; on needle the ablated model
  is at ceiling (1.00 ± 0.01 through length 128) with near-zero seed variance.
- **"Complex enhances RoPE" also died under param matching:** needle seq@128 is 0.45 ± 0.14 for
  complex+RoPE vs 0.57 ± 0.08 for param-matched real+RoPE, in real's favor.
- **No compression story:** complex costs ~2x parameters at equal dim and buys nothing at matched
  parameters outside reverse.
- **Training recipe:** CReLU is the piece that matters; all 160 runs at d160/10k steps converged
  in-distribution, with or without the learnable input phase. Every RoPE and learned-PE run collapses
  to 0 exact-match past the trained length, matching the literature ordering (NoPE extrapolates best).

## What this is

Capability-first study comparing **complex-NoPE** transformers against **real+RoPE**, **real+NoPE**,
and **real+learned-PE** baselines on positional benchmarks (synthetic algorithmic tasks, needle/passkey
retrieval, length generalization), with a mechanistic phase probe and a pre-registered hypothesis
(`docs/hypothesis.md`). Knowledge base: [`docs/`](docs/).

## Setup (uv)

```bash
uv sync                 # create venv + install (torch cu124, marimo, plotly, ...)
uv run pytest           # gradcheck + R^2-equivalence tests for complex layers
bash scripts/smoke_test.sh   # tiny GPU/CPU sanity run
```

## Run experiments

```bash
uv run python -m experiments.train --field complex --pe none --task reverse   # one condition
uv run python -m experiments.sweep --tasks reverse copy --seeds 0 1 2          # comparison matrix
uv run python -m experiments.stats                                              # stats behind the tables
uv run marimo edit notebooks/03_analysis.py                                     # interactive plotly dashboard
sbatch scripts/unity_sweep.sbatch                                              # full run on a Unity A100
```

## Layout

| Path | Purpose |
|---|---|
| `complexnn/` | reusable complex-valued layers, attention, models, init; real PE baselines |
| `experiments/` | configs (tyro), data generators, `train.py`, `eval.py`, `stats.py` |
| `notebooks/` | marimo + plotly analysis dashboards |
| `scripts/` | `smoke_test.sh` (local GTX 1050), Unity A100 / SLURM sbatch scripts |
| `docs/` | persistent knowledge base + results writeup |
| `results/` | sweep metric JSONs and probe metrics (committed); checkpoints/figures gitignored |
| `paper/` | NeurIPS-style writeup (`main.tex`), figures via `scripts/make_paper_figs.py` |

## Compute

Local **GTX 1050 (4 GB)** for smoke tests; **UMass Unity A100 (32 GB)** via SLURM for real runs.
One config switches between them.
