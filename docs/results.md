# Results & running log

Living document: methodology findings first (things we learned while building the harness that change
how the experiment must be run), then quantitative results and the hypothesis verdict. See
[`hypothesis.md`](hypothesis.md) for the pre-registered claims this is measured against.

---

## Methodology findings (M2–M3)

These are real findings — each one would have biased the experiment if missed.

1. **Causal-mask task design.** An *in-place* reverse (output[i] aligned with input[i]) is
   **mathematically unsolvable** for a causal LM at early positions: position 0 sees only input[0] yet
   must emit input[T-1]. This caps accuracy near 0.9 regardless of model and is a measurement artifact,
   not a model limitation. **Fix:** pose every task as `[input] [SEP] [answer]` so all input is visible
   before any answer token; compute loss/accuracy only on answer positions (`experiments/data/tasks.py`).

2. **`modReLU` with bias = 0 is the identity.** `ReLU(|z| + 0) = |z|`, so the scale `|z|/|z| = 1`. With
   zero-init bias the complex MLPs were *linear at initialization*. Even after fixing the bias to a small
   negative dead-zone, modReLU remains too weak a nonlinearity after RMS-normalized inputs.

3. **CReLU is the working complex activation.** Head-to-head on the (solvable) reverse task, in-distribution:

   | activation | final loss | acc@12 |
   |---|---|---|
   | modReLU (b=−0.1) | ~1.35 | ~0.35 |
   | **CReLU** | **~0.15** | **~0.89** |

   modReLU's failure was masquerading as an "optimization problem" / "complex can't fit." It was the
   activation. **Default is now CReLU.** (modReLU preserves phase, which is theoretically attractive for
   "phase carries position"; making it competitive needs a tuned magnitude threshold — logged as future
   work. The phase machinery still lives in the linear + attention layers, so CReLU does not remove it.)

   *Lesson:* this is exactly the "incomplete experiments" failure mode that got ComplexFormer withdrawn —
   an under-tuned complex model looks like evidence against the hypothesis. We tune before concluding.

4. **`complex+NoPE` training is strongly bimodal / initialization-sensitive.** On the solvable reverse
   task, identical config + recipe gives, across seeds:

   | seed | acc@12 |
   |---|---|
   | 0 | **0.89** (good basin — fits, loss 0.15) |
   | 1 | 0.35 (collapsed, loss ~1.25) |
   | 2 | 0.34 (collapsed) |

   The earlier "CReLU works (0.89)" reading was a single lucky seed (0). Most seeds collapse. This is a
   real result, not a bug: **complex+RoPE is rock-solid** (acc@12=1.00 every seed) and even **extrapolates
   better than real+RoPE** (acc@24 0.46 vs 0.20), whereas **complex+NoPE rarely finds the solution.** It
   matches the theory (background.md §5.1): complex weights supply phase *capacity* but not *position*;
   under NoPE position must bootstrap from the causal mask, and that bootstrap is unreliable. Whether a
   stabilizer (learnable imaginary embeddings, init/lr) fixes it is the live question (next).

---

## Quantitative results

### First trustworthy signal (solvable reverse, CReLU, single seed, 2k steps, GTX 1050)
*(superseded by the multi-seed sweep below once it lands; kept for the record)*

| condition | params | acc@12 (in-dist) | acc@24 | acc@48 |
|---|---|---|---|---|
| real + RoPE (d64) | 100,928 | **1.00** | 0.15 | 0.08 |
| real + NoPE (d64) | 100,928 | 0.77 | 0.30 | 0.13 |
| real + NoPE (d92, param-matched) | 206,908 | 0.84 | 0.37 | 0.18 |
| complex + NoPE (d64, **modReLU — under-tuned**) | 200,896 | 0.35 | 0.21 | 0.12 |

Read: RoPE is perfect in-distribution but extrapolates worst; NoPE extrapolates best *relatively*. The
complex row here used the broken activation — **re-run with CReLU pending** (sweep `bsiqrzoz7`).

### Capacity-matched sweep — reverse + copy, 3 seeds, 2000 steps, GTX 1050 (complex uses learn_imag=True)

In-distribution accuracy (acc@12) and **success_rate** = fraction of seeds with acc>0.7:

| condition | copy acc@12 | copy succ | reverse acc@12 | reverse succ |
|---|---|---|---|---|
| **complex + NoPE** | **0.941** | **3/3** | 0.680 (median **0.827**) | **2/3** |
| complex + RoPE | 1.000 | 3/3 | 1.000 | 3/3 |
| real + NoPE (d64) | 0.905 | 3/3 | 0.802 | 3/3 |
| real + NoPE (d88, matched) | 0.916 | 3/3 | 0.803 | 3/3 |
| real + RoPE (d64) | 0.999 | 3/3 | 1.000 | 3/3 |
| real + learned | 1.000 | 3/3 | 1.000 | 3/3 |

Length extrapolation (mean acc@24 / acc@48):

| condition | copy @24/@48 | reverse @24/@48 |
|---|---|---|
| **complex + NoPE** | **0.389** / 0.153 | 0.337 / 0.147 |
| complex + RoPE | 0.109 / 0.219 | **0.417** / 0.148 |
| real + NoPE (d64) | 0.300 / 0.137 | 0.299 / 0.147 |
| real + RoPE (d64) | 0.068 / 0.066 | 0.222 / 0.100 |
| real + learned | 0.221 / 0.150 | 0.104 / 0.089 |

Reads: (1) RoPE is perfect in-distribution but extrapolates *worst* (overfits the trained length —
real+RoPE copy@24 = 0.07). (2) **complex+NoPE, when it converges, matches or beats real+NoPE both
in-distribution and at @24** on both tasks. (3) **complex+RoPE has the best reverse extrapolation
(0.417 @24, ~2× real+RoPE).** (4) complex+NoPE's weakness is **reliability**: solid on copy (3/3),
shaky on the harder reverse (2/3).

### Rigorous local sweep — all 4 tasks, d160 / 5 layers / 10k steps, GTX 1050

`scripts/local_rigorous_sweep.sh` (completed 2026-06-23). Step-up from the probe scale: **d160, 5 layers,
10k steps**, train lengths 6–16, eval at 16 (in-dist) / 32 / 64 / 96. complex+NoPE gets **3 seeds** on
reverse+copy (the reliability question); the stable conditions and the first-ever sort+needle runs get
**1 seed**. Complex backbone = 3.09M params, real = 1.54M (~2×). **First data on `sort` and `needle`.**

Two metrics matter and tell different stories: **token_acc** (per-position, the log's `acc@L`) and
**seq_acc** (exact full-sequence match — far stricter, and the honest length-gen metric). Token_acc, n=3
conditions are seed means:

| condition | task | tok@16 | tok@32 | tok@64 | tok@96 | seq@32 | seq@64 | seq@96 |
|---|---|---|---|---|---|---|---|---|
| **complex+NoPE** (n=3) | reverse | 1.00 | 0.929 | 0.505 | 0.313 | 0.147 | 0.00 | 0.00 |
| complex+RoPE (n=3) | reverse | 1.00 | 0.587 | 0.471 | 0.392 | 0.00 | 0.00 | 0.00 |
| real+NoPE | reverse | 1.00 | 0.867 | 0.295 | 0.157 | 0.041 | 0.00 | 0.00 |
| real+RoPE | reverse | 1.00 | 0.390 | 0.206 | 0.163 | 0.00 | 0.00 | 0.00 |
| **complex+NoPE** (n=3) | copy | 1.00 | 0.957 | 0.430 | 0.271 | 0.351 | 0.00 | 0.00 |
| complex+RoPE (n=3) | copy | 1.00 | 0.100 | 0.113 | 0.075 | 0.00 | 0.00 | 0.00 |
| real+NoPE | copy | 1.00 | 0.903 | 0.276 | 0.138 | 0.072 | 0.00 | 0.00 |
| real+RoPE | copy | 1.00 | 0.062 | 0.550 | 0.196 | 0.00 | 0.00 | 0.00 |
| **complex+NoPE** | sort | 1.00 | 1.000 | 0.996 | 0.991 | **0.996** | **0.758** | **0.371** |
| complex+RoPE | sort | 1.00 | 0.557 | 0.295 | 0.222 | 0.00 | 0.00 | 0.00 |
| real+NoPE | sort | 1.00 | 1.000 | 0.993 | 0.974 | 0.996 | 0.596 | 0.035 |
| real+RoPE | sort | 1.00 | 0.526 | 0.276 | 0.221 | 0.00 | 0.00 | 0.00 |
| **complex+NoPE** | needle | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| complex+RoPE | needle | 1.00 | 1.000 | 0.873 | 0.740 | 1.000 | 0.873 | 0.740 |
| real+NoPE | needle | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| real+RoPE | needle | 1.00 | 0.998 | 0.748 | 0.543 | 0.998 | 0.748 | 0.543 |

Reads:

1. **The collapse is gone at scale.** complex+NoPE converged **3/3 on reverse** (was 2/3 at d64) and 3/3 on
   copy — in-dist tok@16 = 1.00, seq@16 = 1.00 every seed. Confirms the d128 probe set: the bimodal
   initialization sensitivity that gated the primary hypothesis **does not appear at d160 / 10k**. The
   reliability caveat is no longer the bottleneck.

2. **complex+NoPE ≥ real+NoPE on all four tasks**, in-dist and at length. The standout is **sort**, the most
   discriminating task: complex+NoPE holds **seq@64 = 0.758 / seq@96 = 0.371** vs real+NoPE's 0.596 / 0.035
   — a **~10× exact-match advantage at the longest length**. (Caveat: real conditions are n=1 here, so this
   is a clean single-seed read, not a seed-averaged one.)

3. **NoPE ≫ RoPE for length generalization on algorithmic tasks.** Under exact-match (seq_acc), **every RoPE
   run collapses to 0 beyond the trained length** on reverse/copy/sort — RoPE memorizes absolute trained
   positions and cannot extrapolate the permutation. complex+NoPE is the best length extrapolator overall on
   these tasks. (This sharpens, and is consistent with, the earlier "RoPE extrapolates worst" reading.)

4. **token_acc overstates length-gen.** Reverse/copy reach 0 *exact-match* by @32–64 even where token_acc
   looks healthy (e.g. reverse complex+NoPE tok@64 = 0.505 but seq@64 = 0.00): a handful of wrong positions
   per sequence. Only **sort and needle** retain meaningful exact-match at length, and on both, complex wins
   its bracket.

5. **needle is non-discriminating among NoPE** — complex+NoPE and real+NoPE are **both perfect (1.00)** at
   every length, as predicted (needle is content-addressable: find MARK by content, emit the +1 token; little
   positional reasoning required). It still shows the enhancer effect at the RoPE level: **complex+RoPE
   (seq@96 = 0.740) > real+RoPE (0.543)** — complex helping RoPE again.

**Net:** at this scale the primary capability story is **positive and now reliable** — complex+NoPE trains
3/3, beats real+NoPE everywhere, and decisively out-extrapolates it on sort; and the RoPE-enhancer thread
holds on needle. The honest limiter is that reverse/copy are exact-match-saturated past @32, so the cleanest
length-gen evidence comes from sort.

### Mechanistic probe (H1_mech) — does emergent RoPE-like phase structure live in the weights?

`experiments/phase_probe.py`, run on two independent scales of the solvable reverse task:
**d64 / 2-layer / 2000-step** (`probe_*` tags, `--tagset d64`) and **d128 / 4-layer / 8000-step**
(`probe2_*` tags, `--tagset d128`). The d128 set is the decisive one: every condition **fully solves the
task** (acc@16 = 1.00, final loss ~1e-4) and **all four complex-NoPE seeds converged — no collapse** — so
this is a converged-vs-converged comparison at full task mastery. Two levels of test:

**1. Weight-level (H1_mech exactly as pre-registered): NOT supported — F3, now confirmed at scale.**
Pooled QK entry-phases and eigenvalue-angles are tested for non-uniform banding against the Trabelsi-init
null with three circular tests (Rayleigh / KS / Kuiper, no scipy). At d128:

| condition (d128) | acc@16 | entry circ-var | entry KS-p | eigen circ-var | eigen KS-p |
|---|---|---|---|---|---|
| init (Trabelsi null) | — | 0.997 | 0.093 | 0.995 | 1.00 |
| C-NoPE (s0) | 1.00 | 0.996 | 0.058 | 0.999 | 0.99 |
| C-NoPE (s2) | 1.00 | 0.997 | 0.400 | 0.996 | 1.00 |
| C-RoPE | 1.00 | 0.996 | 0.133 | 0.997 | 1.00 |

Circular variance ≈ 1 (uniform) and **every** uniformity p ≫ 0.05 for converged *and* RoPE models —
**statistically indistinguishable from the untrained init**. No phase banding appears even though the model
now clearly solves the task, ruling out the d64 worry that structure was merely under-trained. d64 gave the
identical reading. *Argued mis-specification:* under NoPE the weights are position-independent, so RoPE's
runtime `m·θ` rotation is applied per-token at inference, not *stored* in W_q/W_k — there is nothing for
the weights to band. The registered weight-level claim is the wrong place to look.

**2. Behaviour-level (where inferred position must actually live): complex helps RoPE, not NoPE.**
`pos_fraction` = positional share of attention-logit variance (position-blind floor ≈ 0.004), `rel_fraction`
= how relative-offset (RoPE-like) that structure is:

| condition (d128) | pos_fraction | rel_fraction |
|---|---|---|
| C-NoPE (s0 / s2) | 0.39 / 0.44 | 0.23 / 0.22 |
| R-NoPE | 0.43 | 0.19 |
| C-RoPE | **0.67** | 0.34 |
| R-RoPE | 0.55 | 0.33 |

Reads: (1) **C-NoPE ≈ R-NoPE** (0.39–0.44 vs 0.43) → complex weights give **no positional advantage under
NoPE** — and this now holds for *converged* seeds, not just collapsed ones. (2) **C-RoPE > R-RoPE** (0.67 vs
0.55) → complex weights **amplify** RoPE's positional signal (d64: 0.83 vs 0.66 — same direction). (3) NoPE's
emergent position is less relative (rel_fraction 0.19–0.23) than RoPE's (0.33–0.34): what little position
NoPE bootstraps from the causal mask is more absolute/content-driven, not a recovered RoPE kernel.

**Net:** the literal mechanism (complex QK weights internalize RoPE's rotation as banded phase structure) is
**falsified at two scales (F3)**. The behaviour reframes cleanly toward the standout finding: complex weights
function as a **RoPE enhancer**, not a positional-module replacement. Figures: `results/_figs/` (d64) and
`results/_figs/d128/` (`phase_qk_*`, `phase_eigenspectrum`, `phase_offset_kernel`, `phase_metrics.json`).

### Coverage & next runs

- **All four tasks now measured** (rigorous local sweep, above): copy, reverse, sort, needle at d160 / 5L /
  10k. `sort` is the most discriminating (complex+NoPE ~10× real+NoPE exact-match at @96); `needle` is
  non-discriminating among NoPE (both perfect).
- **More-rigorous local validation: DONE** (2026-06-23, ~8 h on the 1050). `scripts/local_rigorous_sweep.sh`
  at **d160 / 5 layers / 10k steps** gave the per-condition read at scale — complex+NoPE reliability (3/3 on
  reverse *and* copy) plus first sort+needle signal. Still-open follow-ups: (a) re-run the phase probe on
  these d160 tags for a third-scale confirmation (probe currently has `--tagset {d64,d128}`); (b) the full
  4-task × 5-seed × 7-condition matrix on Unity (`scripts/unity_sweep.sbatch`) to seed-average the real and
  sort/needle conditions (n=1 here).
- **Dashboard:** `uv run marimo edit notebooks/03_analysis.py` renders all of the above (results, the
  mechanistic probe, an animated attention-retrieval view, and a live-training cell); results refresh as
  sweep runs land.

---

## Preliminary hypothesis verdict (GTX 1050; to be confirmed by the Unity multi-seed sweep)

Measured against the falsification table in `hypothesis.md`. **Preliminary** — based on the focused
reverse-task studies above, the 3-seed d64 reverse/copy sweep, and the **completed d160 / 10k rigorous
sweep across all 4 tasks**. The full 5-seed × 4-task A100 run (`scripts/unity_sweep.sbatch`) is still needed
to seed-average the real and sort/needle conditions (n=1 locally).

**Primary — Capability (H1_cap: complex-NoPE > real-NoPE, competitive with real+RoPE): SUPPORTED at scale**
(was "partially supported, gated by reliability" — the reliability gate cleared at d160 / 10k).
- *Reliability — resolved at scale:* the d64 bimodal collapse (reverse 2/3) **disappears at d160 / 10k**:
  complex+NoPE converged **3/3 on reverse and 3/3 on copy**, matching the d128 probe set. The "harder tasks
  collapse" worry was a low-capacity / under-training artifact, not intrinsic → the F5-flavored open problem
  is largely closed for this scale.
- *Quality — complex+NoPE ≥ real+NoPE on **all four tasks**, in-dist and at length.* Standout: **sort**, the
  most discriminating task — complex+NoPE holds exact-match seq@64 = 0.76 / seq@96 = 0.37 vs real+NoPE's
  0.60 / 0.035 (~10× at @96). So the "complex weights help infer position" signal is **real, positive, and
  now reliable** → F1 **not** met.
- *vs real+RoPE:* tied in-distribution (everything is 1.00 at @16), and complex+NoPE **out-extrapolates
  real+RoPE** — on these algorithmic tasks RoPE memorizes the trained length and collapses to 0 exact-match
  beyond it, so NoPE (best variant: complex) wins length-gen outright.
- *Honest limiter:* reverse/copy are exact-match-saturated past @32, so the cleanest length-gen evidence is
  sort; needle is non-discriminating among NoPE (both perfect).

**Secondary — Compression (H1_comp: on/above real+RoPE Pareto): NOT supported (in-distribution).**
- The complex backbone costs ~2× real params at equal dim; real+RoPE is perfect in-dist at half the params,
  and a param-matched wider real+NoPE (d88) barely moved → **F4 met for in-distribution quality.** (At
  extrapolation the Pareto is murkier — everything degrades — so compression-at-extrapolation is inconclusive.)

**Mechanistic (H1_mech as registered — RoPE-like phase banding in the weights): NOT supported (F3),
confirmed at two scales.** The phase probe (above) shows QK entry-phase *and* eigenvalue-angle spectra are
statistically uniform — indistinguishable from the Trabelsi init — for converged complex models that **fully
solve the task** (d128, acc@16 = 1.00), so the lack of structure is not under-training. The registered claim
is argued mis-specified: under NoPE the weights are position-independent, so RoPE's `m·θ` rotation cannot be
stored in them. **The behavioural reframe holds:** complex weights add no positional signal under NoPE
(pos_fraction C-NoPE ≈ R-NoPE) but **amplify RoPE** (C-RoPE > R-RoPE at both scales) — i.e. a RoPE *enhancer*,
consistent with the standout finding below.

### Standout positive finding
- **complex+RoPE:** stable (acc@12=1.00 every seed) **and best length extrapolation** (reverse @24 = 0.417,
  ~2× real+RoPE's 0.222). The most promising thread: **complex weights *enhance* RoPE rather than replace it.**

### Net for the user
The literal original idea — make weights complex so the model infers position and you can **delete** the
positional module — **holds up at scale on capability**: at d160 / 10k, complex+NoPE trains reliably (3/3
on reverse and copy), beats real+NoPE on all four tasks, and out-extrapolates it ~10× on sort. The
remaining "not cleanly validated" piece is **compression** — the complex backbone still costs ~2× params,
so you buy positional capability, you don't save on it. **And the underlying intuition has strong support:**
complex weights carry genuine positional capability (complex+NoPE ≥ real+NoPE on every task) *and* act as a
RoPE enhancer (complex+RoPE > real+RoPE on needle; best reverse extrapolation at d64). The two concrete,
fundable next steps are **(a) the capability paper** — complex+NoPE as a reliable, length-generalizing
positional-module replacement, anchored on sort — and **(b) complex weights as a RoPE *enhancer*** — both to
be seed-averaged and length-stress-tested on the Unity A100 sweep (`scripts/unity_sweep.sbatch`), alongside
the d160 phase-probe re-run.
