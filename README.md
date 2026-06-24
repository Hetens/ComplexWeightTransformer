# Phase Without Position

**Can complex-valued transformer weights infer the `cos + i·sin` structure of RoPE, removing the need
for an explicit positional module?**

Capability-first proof-of-concept comparing **complex-NoPE** transformers against **real+RoPE**,
**real+NoPE**, and **real+learned-PE** baselines on base positional benchmarks (synthetic algorithmic
tasks, needle/passkey retrieval, length-generalization). Secondary question: does folding position into
complex weights pay off on the quality-vs-parameters Pareto frontier?

Full research proposal: `~/.claude/plans/i-had-an-idea-frolicking-honey.md`.
Knowledge base: [`docs/`](docs/).

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
uv run marimo edit notebooks/03_analysis.py                                     # interactive plotly dashboard
sbatch scripts/unity_sweep.sbatch                                              # full run on a Unity A100
```

## Findings so far (see [`docs/results.md`](docs/results.md))
- `complex+RoPE` is stable and **extrapolates better than `real+RoPE`** (preliminary, single task).
- `complex+NoPE` (the headline idea — drop the positional module) trains **unreliably / bimodally**; at
  best it matches `real+NoPE`. The clean "complex weights subsume positional encoding" claim is **not
  validated** yet; the promising reframing is "complex weights *enhance* RoPE."

## Layout

| Path | Purpose |
|---|---|
| `complexnn/` | reusable complex-valued layers, attention, models, init; real PE baselines |
| `experiments/` | configs (tyro), data generators, `train.py`, `eval.py` |
| `notebooks/` | marimo + plotly analysis dashboards |
| `scripts/` | `smoke_test.sh` (local GTX 1050), `unity_train.sbatch` (Unity A100 / SLURM) |
| `docs/` | persistent knowledge base + results writeup |
| `results/` | metrics json, checkpoints, exported figures (gitignored) |

## Compute
Local **GTX 1050 (4 GB)** for smoke tests; **UMass Unity A100 (32 GB)** via SLURM for real runs.
One config switches between them.
