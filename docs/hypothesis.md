# Pre-registered hypothesis

Stated before running experiments so results cannot be retrofitted. Background:
[`background.md`](background.md). Feasibility: [`feasibility.md`](feasibility.md).

---

## Construct under test

A **complex-weight transformer (complex-NoPE)**: all linear maps, norms, attention, and MLP operate over
`ℂ` (`W = A + iB`), with **no explicit positional encoding**. Real token IDs → real embeddings, lifted to
`ℂ` (imaginary part zero-init or learned); output projected `ℂ→ℝ` (`[Re ‖ Im]` concat) before the head.

**Claim being tested:** because a complex linear layer is intrinsically a phase-equivariant map (see
`background.md` §5.1), a complex-NoPE model can *infer* RoPE-like relative-phase structure from the causal
mask + depth alone, recovering the capability that explicit RoPE supplies — and eventually doing so with
fewer parameters.

---

## Comparison conditions (all share the same backbone shape & training recipe)

| ID | Weight field | Positional encoding |
|----|--------------|---------------------|
| `R-RoPE`   | real    | RoPE (strong baseline) |
| `R-NoPE`   | real    | none (isolates "no explicit position") |
| `R-Learned`| real    | learned absolute PE (classic baseline) |
| **`C-NoPE`** | **complex** | **none (the candidate)** |
| `C-RoPE`   | complex | RoPE (diagnostic: does complex help *on top of* RoPE?) |

### Capacity-matching (fairness — runs done both ways)
A complex `d×d` layer = `2d²` real DOF. We compare at:
- **(a) matched real-DOF** — equal expressivity budget → the **capability** question.
- **(b) matched complex-entries / fewer real-DOF** — fewer numbers → the **compression** question.

---

## Hypotheses

### Primary — Capability
- **H1_cap:** at **matched real-DOF**, `C-NoPE` achieves **higher** length-extrapolation accuracy than
  `R-NoPE`, **and** is **statistically competitive with** `R-RoPE` (within a pre-set margin).
- **H0_cap:** `C-NoPE ≤ R-NoPE` on length extrapolation, within seed noise.

### Secondary — Compression
- **H1_comp:** on the **quality-vs-total-real-parameters** Pareto frontier, `C-NoPE` lies **on or above**
  the `R-RoPE` frontier (matches quality with no positional module and/or fewer params).
- **H0_comp:** no Pareto improvement at equal quality.

### Mechanistic (supporting, not sufficient alone)
- **H1_mech:** trained `C-NoPE` weights exhibit **emergent structured phase rotations** (RoPE-like banding
  in the QK weight phase spectra). Absence of structure with good accuracy ⇒ benefit is a confound.

---

## Variables

- **Independent:** weight field {real, complex}; PE {RoPE, NoPE, learned}; capacity-match {real-DOF, entries};
  train length `L`; test length {`L, 2L, 4L`}; recursion depth (where applicable); seed.
- **Dependent (metrics):**
  - per-length **token accuracy** and **exact-sequence accuracy** (synthetic tasks)
  - **retrieval accuracy vs key depth** (needle/passkey)
  - **validation loss / perplexity**
  - **total real parameters**, **FLOPs/token**, **peak memory**
  - **learned-phase structure** (phase histograms / spectra of QK & MLP weights)
- **Controlled:** optimizer, LR schedule, batch tokens, total training steps/tokens, vocab, depth/width
  (up to the capacity-match definition), data, eval protocol.

---

## Success thresholds (pre-set)

Let `Δ = mean(C-NoPE) − mean(R-NoPE)` on extrapolation accuracy at `2L` (primary length point),
averaged over tasks, with per-condition **≥ 3 seeds**.

- **H1_cap supported** iff `Δ > 0` by **more than the pooled seed std** (a simple, pre-committed effect-size
  bar; we also report a two-sample t-test p-value as secondary evidence) **and** `C-NoPE` is within
  **3 accuracy points** of `R-RoPE` at `2L`.
- **H1_comp supported** iff `C-NoPE` is not Pareto-dominated by `R-RoPE` across the swept sizes
  (≥ one size where it matches/beats `R-RoPE` quality at ≤ its parameter count).
- **H1_mech supported** iff QK phase spectra show non-uniform, frequency-structured concentration absent in
  `R-NoPE` (qualitative + a uniformity test on phase histograms).

Baselines must first **reproduce the known ordering** from the literature (e.g., NoPE ≥ RoPE on several
length-gen tasks; learned-APE collapses past `L`) before complex results are trusted.

---

## Falsification table

| # | Observation that **rejects** the hypothesis | Which claim dies |
|---|---|---|
| F1 | `C-NoPE ≤ R-NoPE` on extrapolation within seed std | H1_cap (no benefit from complex weights) |
| F2 | `C-NoPE` stays well below `R-RoPE` (>3 pts) at `2L`/`4L` | H1_cap (capacity exists, position not realized) |
| F3 | No emergent phase structure despite decent accuracy | H1_mech (benefit is a confound, not the proposed mechanism) |
| F4 | `C-NoPE` never reaches the `R-RoPE` Pareto frontier | H1_comp (compression claim) |
| F5 | Complex models can't train stably to a fair comparison despite correct grads/init | feasibility for this regime |

Any of F1–F2 rejects the **primary** hypothesis; F4 rejects the **secondary**; F3/F5 weaken
interpretation/feasibility. A clean negative is a valid, reportable outcome.

---

## Statistical & reproducibility protocol
- ≥ 3 seeds per condition; report **mean ± std**; pre-committed effect-size bar above + t-test as support.
- Fixed seeds; full config logged to `results/<run>/config.json`; `uv.lock` pins the environment.
- Identical data, steps, and eval harness across conditions; only the IV changes.
- All figures regenerated from `results/` by the marimo notebooks (`notebooks/03_analysis.py`).
