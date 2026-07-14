# Results & running log

**TL;DR (final verdict, 2026-07-10, Unity 5-seed sweep + ablation, 160/160 runs converged).**
H1_mech is falsified at three scales: trained complex QK weights show no RoPE-like phase structure,
statistically indistinguishable from the random init even at full task mastery. H1_comp is not
supported: complex costs ~2x parameters at equal dim and buys nothing at matched parameters outside
reverse. One capability survives 5 seeds and both capacity controls: complex+NoPE roughly doubles
exact-match length generalization on reverse (seq@24 = 0.86 ± 0.02 vs 0.41 ± 0.08 param-matched,
p = 0.0002), and the learn_imag ablation attributes it to the complex weights themselves (ablated
0.84 ± 0.06, p = 0.68 vs full). The pilot's sort, needle, and RoPE-enhancer readings did not survive.
Full verdicts: [Hypothesis verdict, FINAL](#hypothesis-verdict-final-unity-5-seed-sweep-2026-07-09--ablationprobe-2026-07-10).

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

### Unity 5-seed confirmatory sweep (job 61568498, completed 2026-07-09): the decisive run

`scripts/unity_sweep.sbatch` on a Unity A100: 4 tasks x 5 seeds x 7 conditions (complex and real at
d160, plus param-matched real+NoPE and real+RoPE at d224), d160 / 5 layers / 10k steps, train lengths
6-16, eval to 128, 512 sequences per eval cell. All 140 runs converged in-distribution (minimum token
accuracy at length 16 is 1.00 for every run). Metric JSONs are committed (`results/unity_*/result.json`,
commit b684ea9); checkpoints remain on Unity. Numbers below are exact-match (sequence) accuracy,
mean ± std over 5 seeds, from `experiments/stats.py` (exact Welch t-tests, no scipy).

**Claim-by-claim verdicts against the local pilot:**

1. **REPLICATED, the surviving positive: `reverse`.** complex+NoPE holds seq@24 = **0.86 ± 0.02** vs
   real+NoPE **0.42 ± 0.14** (d160, p = 0.002) and **0.41 ± 0.08** (d224 param-matched, p = 0.0002);
   at @32 it is 0.15 ± 0.02 vs 0.03 ± 0.01 (p < 1e-4). Tight across seeds, survives both the equal-dim
   and the equal-parameter control. This is the headline now.

2. **DID NOT REPLICATE: the "10x sort" headline.** At 5 seeds, sort seq@96 is 0.27 ± 0.19 (complex+NoPE)
   vs 0.25 ± 0.24 (real+NoPE d160, p = 0.88) and 0.25 ± 0.28 (d224, p = 0.90). The n=1 local read
   (0.37 vs 0.035) was seed noise. Sort is parity, full stop.

3. **EXPOSED AS A CAPACITY ARTIFACT: needle.** complex+NoPE d160 (0.85 ± 0.28 @128) beats real+NoPE
   d160 (0.61 ± 0.25, p = 0.19) but **loses to the param-matched real+NoPE d224 (0.92 ± 0.18)**. The
   "both perfect, non-discriminating" local read also fails at 96-128: needle does discriminate, by
   parameter count.

4. **DID NOT SURVIVE PARAM MATCHING: the RoPE-enhancer thread.** Needle seq@128: complex+RoPE
   0.45 ± 0.14 vs real+RoPE d160 0.40 ± 0.09 (p = 0.53, n.s.) and vs real+RoPE d224 **0.57 ± 0.08**
   (real wins, p = 0.13). The enhancer framing is dead.

5. **Literature ordering replicates** (the protocol's trust gate): every RoPE run collapses to 0
   exact-match beyond the trained length on copy/reverse/sort; learned-APE collapses immediately;
   NoPE variants extrapolate best.

6. **Honest wart: copy extrapolation is bimodal in every NoPE condition** (seq@32 std ±0.27-0.36; seeds
   split between ~0.1 and ~0.7). Report per-seed dots, not just bars (paper Fig. 2).

**Pre-registered H1_cap at 2L = 32** (success bar: task-averaged delta > pooled seed std):
- token accuracy: **SUPPORTED** vs both baselines (delta +0.045 equal-dim / +0.056 matched), but the
  effect is carried entirely by reverse (+0.16, p ≤ 0.0008); copy/sort/needle deltas are ~0.
- sequence accuracy: **NOT SUPPORTED** as task-averaged (delta +0.02-0.03 vs pooled std ~0.15, swamped
  by copy's variance), while reverse alone is strongly significant (p < 1e-4).
- The "within 3 points of R-RoPE" clause passes trivially (RoPE is at 0 beyond the trained length).

### Unity ablation + d160 probe (job 61597842, completed 2026-07-10): attribution closed

`scripts/unity_ablation.sbatch`: (a) the **learn_imag ablation**, complex+NoPE with learn_imag=False
("no input phase"), 4 tasks x 5 seeds, tag suffix `_noimag`; (b) the **d160 phase-probe re-run** on the
Unity reverse checkpoints (`experiments/phase_probe.py --tagset d160`, now including the param-matched
real+RoPE d224 control and one ablated model). All 20 ablation runs converged in-distribution
(token@16 = 1.00), so the full sweep count is now 160/160. Metric JSONs committed
(`results/unity_*_noimag_*/result.json`); probe metrics committed (`results/_figs/d160/phase_metrics.json`).

1. **The reverse win is the complex weights, not the input phase.** Ablated reverse seq@24 =
   **0.84 ± 0.06** vs full complex 0.86 ± 0.02 (p = 0.68, no change) and vs param-matched real+NoPE
   d224 0.41 ± 0.08 (**+0.44, p < 1e-4**, the full margin). Attribution closed.
2. **The input phase's reliability motivation is retired at d160.** It was added because d64 pilots
   collapsed without it; at d160 all 20 ablated runs converge, and on needle the ablated model is the
   most consistent condition trained: **1.00 ± 0.01 at every length through 128** (vs 0.85 ± 0.28 with
   the phase, 0.92 ± 0.18 for real d224; pairwise n.s., p >= 0.31, but the seed-variance collapse is
   notable). Sort/copy: parity with full complex. The recipe reduces to CReLU.
3. **d160 probe: H1_mech stays falsified at a third scale, on the decisive checkpoints.** Entry and
   eigen circular variance 0.995-0.999, all Rayleigh/KS/Kuiper p >> 0.05, for 3 converged C-NoPE seeds,
   the ablated model, and C-RoPE: all statistically at the Trabelsi init null while solving the task.
4. **pos_fraction (behavior-level position):** C-NoPE 0.39-0.48 across seeds (ablated 0.40) vs R-NoPE
   0.54 (d160) / 0.42 (d224 matched): no additional position in attention on the very task complex
   wins. The pilot's C-RoPE > R-RoPE asymmetry **persists against the param-matched control**
   (0.61 vs 0.47 d160 / 0.37 d224) but stays uninterpreted: the behavioral enhancer claim it was
   read through is dead.

### Coverage & next runs

- **Unity 5-seed sweep: DONE. Ablation + d160 probe: DONE** (above). Results committed; figures in
  `paper/figs/` via `scripts/make_paper_figs.py`; numbers via `experiments/stats.py`. The paper's two
  TODO markers are filled; no runs are staged.
- The d64/d128 probe artifacts remain lost (numbers survive only in the tables above); the d160
  probe artifacts are the committed evidence base for the probe claims.
- **Dashboard:** `uv run marimo edit notebooks/03_analysis.py` renders all of the above (results, the
  mechanistic probe, an animated attention-retrieval view, and a live-training cell); results refresh as
  sweep runs land.

---

## Hypothesis verdict, FINAL (Unity 5-seed sweep 2026-07-09 + ablation/probe 2026-07-10)

Measured against the falsification table in `hypothesis.md`. This supersedes the preliminary GTX 1050
verdict (see git history for the pre-Unity reading, which rested on n=1 sort and enhancer effects that
did not replicate).

**Primary, Capability (H1_cap: complex-NoPE > real-NoPE, competitive with real+RoPE): MIXED, and
narrower than the pilot suggested.**
- On the pre-registered task-averaged test at 2L, H1_cap is **supported on token accuracy** (delta
  +0.045 equal-dim / +0.056 param-matched, both above the pooled seed std) but **not supported on
  sequence accuracy** (delta ~+0.03 vs pooled std ~0.15). Both readings are carried entirely by one
  task: reverse.
- Per task: **reverse is a real, large, param-matched-robust win** (seq@24 0.86 vs 0.41, p = 0.0002);
  copy and sort are parity; needle's gap is a capacity artifact (the d224 real control wins).
  F1 is met on three of four tasks and decisively not met on reverse.
- vs real+RoPE: complex+NoPE trivially out-extrapolates it (RoPE hits 0 exact-match past the trained
  length), so the competitiveness clause holds.
- *Reliability is fully resolved:* 160/160 runs converged in-distribution (140 sweep + 20 ablation).
  The d64 bimodality was a low-capacity artifact (F5 closed at this scale). The remaining wart is
  copy's bimodal extrapolation.
- *Attribution is closed:* the learn_imag ablation leaves reverse unchanged (0.84 ± 0.06 vs
  0.86 ± 0.02, p = 0.68) and still beats the d224 control by the full margin; the effect belongs to
  the complex weights, not the learnable input phase.

**Secondary, Compression (H1_comp): NOT supported, unchanged (F4).** Complex costs ~2x params at equal
dim and buys nothing at matched params outside reverse.

**Mechanistic (H1_mech): NOT supported (F3), confirmed at three scales** (d64, d128, and d160 on the
decisive sweep checkpoints). QK entry-phase and eigenvalue-angle spectra are statistically uniform,
indistinguishable from the Trabelsi init, for converged models that fully solve the task. The registered
claim is argued mis-specified: under NoPE the weights are position-independent, so RoPE's per-position
rotation cannot be stored in them. Note the previous behavioural reframe ("complex amplifies RoPE") also
died under param matching; the pos_fraction asymmetry (C-RoPE > R-RoPE) was re-measured at d160 and
persists against the d224 control (0.61 vs 0.37), but with no behavioral effect to explain it remains
an uninterpreted observation.

### Net for the user
The literal idea (make weights complex, delete the positional module, save parameters) is falsified in
mechanism and mostly falsified in behavior; what remains is one specific, robust capability: complex
weights roughly **double exact-match length generalization on order reversal**, the task where retrieval
must run against the causal order, and the effect survives both capacity controls (p <= 0.002) **and the
learn_imag ablation**: it belongs to the complex weights themselves. The honest paper is: two negatives
(no emergent phase structure at three scales; no general or enhancer advantage once params are matched)
plus that one positive, plus the training recipe (CReLU; the learnable input phase turned out to be
unnecessary at d160) that makes complex transformers converge reliably.
