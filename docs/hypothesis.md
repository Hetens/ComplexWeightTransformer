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
mask + depth alone, recovering the capability that explicit RoPE supplies, and eventually doing so with
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

### Capacity-matching (fairness: runs done both ways)
A complex `d×d` layer = `2d²` real DOF. We compare at:
- **(a) matched real-DOF**: equal expressivity budget → the **capability** question.
- **(b) matched complex-entries / fewer real-DOF**: fewer numbers → the **compression** question.

---

## Hypotheses

### Primary: Capability
- **H1_cap:** at **matched real-DOF**, `C-NoPE` achieves **higher** length-extrapolation accuracy than
  `R-NoPE`, **and** is **statistically competitive with** `R-RoPE` (within a pre-set margin).
- **H0_cap:** `C-NoPE ≤ R-NoPE` on length extrapolation, within seed noise.

### Secondary: Compression
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

Any of F1-F2 rejects the **primary** hypothesis; F4 rejects the **secondary**; F3/F5 weaken
interpretation/feasibility. A clean negative is a valid, reportable outcome.

---

## Statistical & reproducibility protocol
- ≥ 3 seeds per condition; report **mean ± std**; pre-committed effect-size bar above + t-test as support.
- Fixed seeds; full config logged to `results/<run>/config.json`; `uv.lock` pins the environment.
- Identical data, steps, and eval harness across conditions; only the IV changes.
- All figures regenerated from `results/` by the marimo notebooks (`notebooks/03_analysis.py`).

---

# Follow-up pre-registration (2026-07-26)

Stated before the follow-up sweep runs, same rule as above: predictions first, so nothing can be
retrofitted. The first round left exactly one surviving effect (complex+NoPE roughly doubles
exact-match length generalization on `reverse`, robust to both capacity controls and to the
`learn_imag` ablation) and no account of *why*. Two structural observations motivate what follows.

**(a) The construct is a weight-tying constraint, not complex arithmetic.** Unrolled to reals,
`ComplexLinear` is the map `[[A,-B],[B,A]]` on the stacked vector `[Re; Im]`, `crelu` is ReLU on both
halves, `ComplexRMSNorm` is a 2d-wide RMS norm with gains tied in pairs, and the attention logit
`Re(q k^H) = q.re·k.re + q.im·k.im` is exactly the real dot product in the doubled space. So
**complex+NoPE at d=160 is a real transformer of activation width 320 whose internal matrices are
constrained to half the free parameters an unconstrained 320x320 matrix would need.** The committed
control (real d224, 3.02M params) matches parameters but *not* width (224 vs 320), so the first round
cannot separate "complex helps" from "width 320 helps, and tying is how you afford it".

**(b) `reverse` is the only task whose retrieval offset is independent of input length.** With the
layout `[x_0..x_{L-1}] SEP [ans_0..ans_{A-1}]` and the LM shift, answer token `j` is predicted at
input index `L+j`, so the attention offset required to fetch it from source index `src(j)` is
`src(j) - (L+j)`: `-L` for copy, `-(2j+1)` for reverse. End-anchored tasks give an offset affine in
`j` with no `L` in it, solvable by a counter started at SEP; start-anchored tasks require measuring
`L`. This predicts the observed pattern (reverse wins, copy/sort/needle do not) and is testable by
constructing new tasks rather than by reinterpreting the old ones.

## New comparison conditions

| ID | Change vs `C-NoPE` | Purpose |
|----|--------------------|---------|
| `C-split` | tying `[[A,B],[B,A]]` (split-complex, `j²=+1`) | reflection algebra, same params/width |
| `C-dual`  | tying `[[A,0],[B,A]]` (dual, `ε²=0`)          | nilpotent algebra, same params/width |
| `C-diag`  | tying `[[A,0],[0,B]]` (two unmixed streams)   | **no** algebra: pure width-at-equal-params control |
| `R-wide`  | real+NoPE at d320 / d448 / d640               | traces the real quality-vs-width curve past the matched point |

`complex`, `split` and `dual` exhaust the 2D real unital associative algebras up to isomorphism, so
the set is a complete classification plus one unstructured control. All four tyings hold exactly two
real matrices, hence **identical parameter count and identical activation width**; only the sharing
pattern differs. Algebra arms use `learn_imag=True` by necessity, not taste: under `diag` the streams
never mix, so a zero-initialised imaginary stream would stay identically zero and the control would
silently degenerate to a real d160 model.

## Hypotheses

### H2_alg: is the effect ℂ, or structured weight sharing?
Two mutually exclusive readings, both pre-committed:
- **H2_alg-algebra:** the reverse effect is specific to the rotation structure. `C-NoPE` exceeds each
  of `C-split`, `C-dual`, `C-diag` on reverse seq@24 by more than the pooled seed std.
- **H2_alg-tying:** the effect is parameter-efficient width. All four tyings land within pooled seed
  std of each other and **all** exceed the parameter-matched `R-NoPE d224` (0.41 ± 0.08).
- *Diagnostic middle case:* `C-diag` alone falls back to baseline while `split`/`dual` do not ⇒ what
  matters is cross-stream mixing, not which algebra supplies it.

### H2_width: is the gap reachable by real width?
- **H2_width:** real+NoPE reverse seq@24 stays within seed noise of its d160/d224 value (~0.41)
  through d640, i.e. the curve is flat and the gap is **not** a capacity gap.
- **H0_width:** real+NoPE reaches complex's 0.86 ± 0.02 at some swept width ⇒ the surviving positive
  is a capacity artifact, exactly as `needle` turned out to be, and the headline must be withdrawn.

### H3_offset: the retrieval-offset law
- **H3_offset:** `C-NoPE` beats parameter-matched `R-NoPE` on **`revskip`** (end-anchored, offset
  `-(3j+1)`, length-independent) by a margin whose sign matches `reverse`, and does **not** beat it on
  **`skipcopy`** (start-anchored, offset `j-L`, length-dependent). The pair is matched on stride and
  on answer length, so anchor is the only difference.
- **H0_offset:** complex wins both or neither ⇒ length-independence is not the operative property and
  the law is wrong.

### H4_needle: under-powered, or genuinely null?
- **H4_needle:** re-evaluated at 256/512/1024, `C-NoPE (no input phase)` stays at ceiling
  (seq ≥ 0.99) while the parameter-matched `R-NoPE d224` continues its observed decay
  (1.00 → 0.92 over 16 → 128), separating the arms at p < 0.05 with the existing 5 seeds.
- **H0_needle:** both arms decay together, or complex decays first ⇒ the first-round "capacity
  artifact" verdict stands unchanged and no further needle claim is made.

### H1_mech_act: the mechanism, measured in the right tensor
`H1_mech` was registered on the *weights* and is falsified at three scales; the weights of a NoPE model
are position-independent, so that was the wrong place to look. Re-registered at the activation level:
- **H1_mech_act:** on reverse checkpoints, the position-conditional residual stream of `C-NoPE`
  contains a rotating subspace (top-2 principal plane, unwrapped in-plane angle linear in position)
  with fit quality above the shuffled-position null (empirical p < 0.05), and `R-NoPE` does not; and
  at least one attention head walks the required `-(2j+1)` staircase with hit rate > 0.9 in `C-NoPE`
  more cleanly than in the parameter-matched `R-NoPE`.
- **H0_mech_act:** neither field shows a rotating subspace above null, or both do equally ⇒ the
  mechanism is not a positional clock and the reverse win still has no mechanistic account.

## Follow-up falsification table

| # | Observation | Which claim dies |
|---|-------------|------------------|
| F6 | `C-diag` matches `C-NoPE` on reverse | "complex numbers matter"; the effect is width-at-equal-params |
| F7 | `R-wide` reaches 0.86 at any swept width | the surviving positive is a capacity artifact (H2_width) |
| F8 | complex beats real on `skipcopy` as much as on `revskip` | the offset law (H3_offset) |
| F9 | complex loses to real on `revskip` | the offset law, in the other direction |
| F10 | no rotating subspace above null in `C-NoPE` | H1_mech_act; mechanism stays open |

F6 and F7 are the consequential ones: either would substantially rewrite the paper's framing, F6 by
reattributing the effect away from ℂ, F7 by removing the effect.
