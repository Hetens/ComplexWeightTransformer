# Phase Without Position

**Can complex-valued transformer weights infer the `cos + i·sin` structure of RoPE, removing the need
for an explicit positional module?**

**Answer: no**, and the one effect that survived that answer turns out not to be about complex numbers
either. Two rounds, 270 runs. The mechanism is falsified at three scales and again in the activations.
One behavioral effect survived round one (complex+NoPE roughly doubles exact-match length
generalization on order reversal); round two then showed it is **not a property of `ℂ`** (a
block-diagonal control with no complex arithmetic reproduces it) and **not out of reach of real
width** (a tuned real model matches it at ~4x the parameters). What remains is a
**parameter-efficiency** result on one task out of six.

## Results (270 runs: a 140-run sweep + 20-run ablation in round one, plus 110 in round two)

Exact-match sequence accuracy past the trained length (train lengths 6-16), mean ± std over 5 seeds.
Full log and claim-by-claim verdicts in [`docs/results.md`](docs/results.md); every number reproduces
from the committed `results/unity_*/result.json` via `uv run python -m experiments.stats`.

**Round one, complex+NoPE (d160) vs a parameter-matched real+NoPE (d224):**

| task | eval length | complex+NoPE | real+NoPE (param-matched) | p (Welch) | verdict |
|---|---|---|---|---|---|
| **reverse** | 24 (1.5x train) | **0.86 ± 0.02** | 0.41 ± 0.08 | **0.0002** | the one surviving effect; see round two for what it is |
| copy | 32 | 0.37 ± 0.27 | 0.39 ± 0.36 | 0.92 | parity (extrapolation is bimodal across seeds in every NoPE condition) |
| sort | 96 | 0.27 ± 0.19 | 0.25 ± 0.28 | 0.90 | parity; the pilot's "10x sort" was n=1 seed noise |
| needle | 128 | 0.85 ± 0.28 | 0.92 ± 0.18 | 0.67 | parity; the pilot gap was a capacity artifact |

**Round two, what the reverse effect actually is.** All four tyings below have *identical* parameter
count (3,085,920) and *identical* activation width (320); only the weight-sharing pattern differs:

| tying on reverse @24 | algebra | seq@24 | vs complex |
|---|---|---|---|
| `[[A,-B],[B,A]]` | complex, `i²=-1` | 0.856 ± 0.018 | reference |
| `[[A,B],[B,A]]` | split-complex, `j²=+1` | 0.846 ± 0.05 | p = 0.70 |
| `[[A,0],[0,B]]` | **none (no mixing at all)** | 0.830 ± 0.05 | p = 0.31 |
| `[[A,0],[B,A]]` | dual, `ε²=0` | 0.754 ± 0.08 | p = 0.044 |

and a real+NoPE width scan, both arms tuned over the same learning-rate grid, converged runs only:

| real width | params | x complex | seq@24 | vs complex |
|---|---|---|---|---|
| d224 (matched params) | 3.02M | 1.0x | 0.407 ± 0.085 | +0.450, p = 0.0002 |
| d320 (**matched width**) | 6.16M | 2.0x | 0.654 ± 0.234 | +0.202, p = 0.13 |
| d448 | 12.06M | 3.9x | 0.811 ± 0.090 | +0.045, p = 0.33 |
| d640 | 24.61M | 8.0x | **0.868 ± 0.040** | -0.012, p = 0.57 |

(complex+NoPE d160 is 0.856 ± 0.018 at 3.09M. The d320 row is the control that matters most: it is an
ordinary real transformer at *exactly* the same activation width 320, with the tying removed, so it
costs twice as much and still scores lower. Width alone does not buy the effect; the tying is what
makes that width cheap, and reliable.)

- **Mechanism: falsified, twice.** Trained complex QK weights show no RoPE-like phase structure at
  three scales. Probing the *activations* instead finds every NoPE model builds a positional rotation,
  but it encodes **fractional** position (total swept angle invariant across a 4x length range), so
  there is no fixed rotation for weights to store. Two further pre-registered mechanistic predictions
  failed, one of them backwards: the parameter-matched real models own *cleaner* attention staircases
  than the complex models that outscore them.
- **The reverse win is not the complex weights.** A block-diagonal control with no complex arithmetic
  and no cross-stream mixing reproduces it (p = 0.31). The `learn_imag` ablation had already ruled out
  the input phase; the algebra ablation rules out `ℂ` itself.
- **It is also not unreachable by real width.** The real curve is flat only over d160-d224, the two
  widths round one happened to sample, then rises to 0.87. Round one's 0.41 is what a real transformer
  does *at matched parameters*, not what it can do.
- **A compression story, partially reinstated.** On reverse only, a two-stream tying reaches at 3.09M
  what an unconstrained real model needs ~12M to match. Round one recorded H1_comp as not supported
  from the d224 point alone; that was correct on its evidence and under-powered on the parameter axis.
- **The tying buys consistency as much as accuracy:** complex ± 0.018 against ± 0.090-0.234 for the
  wide real baselines, which is why the significance boundary (2x) and the point-estimate crossover
  (4x) disagree.
- **Two more pre-registered predictions died in round two.** The retrieval-offset law (that the
  advantage should follow tasks whose fetch offset does not depend on input length) failed on two
  purpose-built tasks, with the larger point estimate on the wrong one. And re-scoring needle out to a
  1024-token haystack falsified the prediction that complex would hold ceiling: it falls fastest, and
  the parameter-matched real baseline is ahead by 1024.
- **Training recipe:** CReLU is the piece that matters; the d160-tuned lr does *not* transfer up the
  width axis (real d640 fails to train at all at 2e-3 and reaches 0.87 at 5e-4).

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

Round two (all default off, so the commands above reproduce round one unchanged):

```bash
# weight-tying ablation: same width, same params, only the sharing pattern changes
uv run python -m experiments.sweep --tasks reverse --core none --algebras split dual diag
# real width scan, and the lower learning rates the widest baselines need to train at all
uv run python -m experiments.sweep --tasks reverse --core none --real-width-scan 320 448 640
uv run python -m experiments.sweep --tasks reverse --core none --real-width-scan 448 640 \
    --lr-real 5e-4 --tag-note lr5e4
# offset-law tasks; re-score committed checkpoints at longer haystacks; activation-level probe
uv run python -m experiments.sweep --tasks revskip skipcopy --core pair
uv run python -m experiments.reeval --match "unity_needle_*_none_*" --lengths 256 512 1024
uv run python -m experiments.activation_probe --lengths 16 24 32 48 64
```

## Layout

| Path | Purpose |
|---|---|
| `complexnn/` | reusable complex-valued layers, attention, models, init; real PE baselines |
| `experiments/` | configs (tyro), data generators, `train.py`, `sweep.py`, `stats.py`, `reeval.py` (re-score committed checkpoints at new lengths), `phase_probe.py` (weight-level), `activation_probe.py` (residual-stream level) |
| `notebooks/` | marimo + plotly analysis dashboards |
| `scripts/` | `smoke_test.sh` (local GTX 1050), Unity SLURM sbatch scripts: `unity_sweep` / `unity_ablation` (round one), `unity_followup` / `unity_width_lr` / `unity_complex_lr` (round two) |
| `docs/` | persistent knowledge base + results writeup |
| `results/` | sweep metric JSONs and probe metrics (committed); checkpoints/figures gitignored |
| `paper/` | NeurIPS-style writeup (`main.tex`), figures via `scripts/make_paper_figs.py` |

## Compute

Local **GTX 1050 (4 GB)** for smoke tests; **UMass Unity A100 (32 GB)** via SLURM for real runs.
One config switches between them.
