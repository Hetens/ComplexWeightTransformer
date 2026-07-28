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

## Methodology findings (M2-M4)

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

5. **The construct is a weight-tying constraint, and the parameter-matched control is width-mismatched**
   (M4, 2026-07-26). Unrolled to real arithmetic, the whole complex stack is a *real* network:
   `ComplexLinear` is the map `[[A,-B],[B,A]]` on `[Re; Im]`; `crelu` is ReLU on both halves;
   `ComplexRMSNorm` is a 2d-wide RMS norm with gains tied in pairs; and the attention logit
   `Re(q k^H) = q.re·k.re + q.im·k.im` is **exactly** the real dot product in the doubled space, so the
   score computation contains no complex arithmetic at all. Hence:

   > complex+NoPE at d160 is a real transformer of **activation width 320** whose internal matrices are
   > constrained to `2·160²` free parameters instead of the `320²` an unconstrained layer would need.

   | condition | real params | activation width |
   |---|---|---|
   | complex d160 | 3,085,920 | **320** |
   | real d224 (the committed "param-matched" control) | 3,021,536 | **224** |
   | real d320 | 6,159,680 | 320 |

   The committed control therefore matches *parameters* but not *width*, so nothing in the 160-run
   sweep separates "complex weights help" from "width 320 helps, and tying is how you afford it at
   3M parameters". This does not invalidate any measured number, but it does mean the causal
   attribution in the FINAL verdict below was stated more narrowly than the evidence supports.
   The follow-up round (`H2_alg`, `H2_width` in [`hypothesis.md`](hypothesis.md)) is designed to
   close exactly this gap: `split`/`dual`/`diag` tyings hold width and parameter count fixed and vary
   only the sharing pattern, and the real width scan asks whether real width can buy the gap at all.
   *Lesson:* when the treatment changes the parameterization of a layer, "parameter-matched" is not
   automatically the right control; state which axis is held fixed and check the others.

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

### Follow-up round (job 62193425, gpu-preempt, started 2026-07-26): why does reverse win?

Pre-registered in [`hypothesis.md`](hypothesis.md) "Follow-up pre-registration (2026-07-26)" **before**
these runs, same discipline as round one. Motivated by methodology finding 5 above (the construct is a
weight-tying constraint and the committed control is width-mismatched) and by the retrieval-offset law
now documented in `experiments/data/tasks.py`. Arms: `H2_alg` (algebra ablation), `H2_width` (real
width scan), `H3_offset` (revskip vs skipcopy), `H4_needle` (long-haystack re-eval),
`H1_mech_act` (activation-level probe). Footprint note: the whole round is one 1-GPU `gpu-preempt`
job running the arms sequentially, with `save_model=False`, so it never competes with work on the
`gpu` partition and adds a few MB rather than ~2 GB to `results/`.

#### H2_alg: **F6 fires.** The reverse effect does not require complex numbers, or any algebra at all.

The four two-channel tyings hold **exactly** the same parameter count (3,085,920) and the same
activation width (320); only the sharing pattern differs, and all four converge identically
in-distribution (final loss ~5e-6, seq@16 = 1.000 every run). reverse exact-match at 1.5x train
length, 5 seeds each, against the committed parameter-matched real baseline:

| tying | algebra | reverse seq@24 | vs complex | vs real d224 |
|---|---|---|---|---|
| `[[A,-B],[B,A]]` | complex, `i²=-1` | 0.856 ± 0.02 | reference | +0.450, p = 0.0002 |
| `[[A,B],[B,A]]` | split-complex, `j²=+1` | 0.846 ± 0.05 | -0.010, **p = 0.70** | +0.439, p < 1e-4 |
| `[[A,0],[0,B]]` | **none (no mixing)** | 0.830 ± 0.05 | -0.026, **p = 0.31** | +0.424, p < 1e-4 |
| `[[A,0],[B,A]]` | dual, `ε²=0` | 0.754 ± 0.08 | -0.102, p = 0.044 | +0.347, p = 0.0002 |
| real+NoPE d224 | (single stream, w224) | 0.407 ± 0.08 | | |

**H2_alg-algebra is falsified; H2_alg-tying is supported.** Split-complex and block-diagonal are
statistically indistinguishable from complex (p = 0.70 and p = 0.31), and *every* tying beats the
parameter-matched real baseline by 0.35-0.45 at p <= 0.0002. The spread among tyings (0.10) is far
smaller than the gap from any of them to the real baseline (0.42).

The `diag` arm is the one that settles it. It has **no cross-stream mixing in any linear layer**:
`out_r = Ax`, `out_i = By`, so the two streams touch only through the shared attention pattern
(`Re(q k^H)` sums both streams into one set of logits), the joint magnitude RMS norm, and the
`[Re ‖ Im]` readout. It contains no complex arithmetic, no rotation, and no algebra, and it recovers
the full effect. So the surviving positive from round one is **not** a property of `ℂ`.

Two qualifications, stated so the result is not over-read:
- `dual` is the one arm that is marginally lower (p = 0.044). It is also the only arm whose streams
  are unbalanced at initialization (measured 0.67/1.33 against 1.00/1.00 for the other three), which
  is a property of the nilpotent algebra that no init correction can remove, so its deficit is
  confounded in a way split's and diag's are not. Weight split and diag more heavily.
- This shows the effect is *not about ℂ*; it does **not** yet show what it *is* about. "Two streams
  at 3.09M parameters" and "activation width 320" are still confounded, because every tying here buys
  width 320 at half the parameters an unconstrained 320-wide layer would cost. `H2_width` (the real
  width scan) is the arm that separates them, and it is still running.

#### H2_width: **F7 fires.** Real width does buy the gap, at roughly 4x the parameters.

*(lr=2e-3 pass complete, 15/15 runs; the retuned d448/d640 supplement is job 62210295, in flight.)*

**Methodology finding first, because it determines how the arm may be read.** `lr=2e-3` is the value
tuned at d160 and used by every run in the committed 160-run sweep. It does **not** transfer up the
width axis. At d640 all five seeds sit at final loss ~2.2 with token accuracy 0.13-0.15 at the
*training* length: they never learned the task at all, against ~5e-6 for a healthy run. At d448 two of
five fail the same way. Averaging those into a length-generalization mean would report an optimization
failure as a capability limit, and would do so in the direction that flatters `H2_width`, which
predicts the real curve stays flat. `experiments/analysis.py` now records a `converged` flag
(token accuracy = 1.0 at `train_max`), `experiments/stats.py` excludes non-converged runs from the
means and prints the counts, and the scan is broken out by learning rate so a retuned width can never
silently pool with a diverged one. The committed sweep passes the gate 160/160, so no earlier number
moves. *Lesson (same one as modReLU in item 3 above): tune the baseline before concluding from it.*

**Second methodology finding: convergence-filtering is itself biased when convergence is partial, and
"converged" is not "tuned".** Two further traps surfaced while fixing the first one, both of which
would have inflated our own result:

1. *Survivorship.* Filtering to converged runs is right, but at d448/lr=2e-3 only 3 of 5 converge and
   the two missing seeds are not missing at random. That filtered mean is 0.777; the same width at a
   rate where **all five** converge gives 0.659. Only a fully-converged cell is an unbiased estimate.
2. *Tuning, not just convergence.* We had pre-declared that the lr would be selected by
   in-distribution convergence, a criterion deliberately independent of seq@24. It turned out to be
   *underdetermined*: at d448 both 1e-3 and 5e-4 converge 5/5, yet seq@24 is 0.659 vs 0.811. A
   criterion that cannot separate two candidates cannot be what decides a headline number, and the
   choice we were about to make by default (stop after 1e-3) was the one favouring complex.

The fix is the only symmetric one: **tune both arms over the same lr grid and compare best to best.**
Both the real baselines and complex+NoPE d160 are swept at {2e-3, 1e-3, 5e-4}
(`scripts/unity_width_lr.sbatch` job 62210295, `scripts/unity_complex_lr.sbatch` job 62221575), so no
width is compared at a rate the other arm never saw. Full grid, converged/total and mean ± std of the
converged:

| width | lr=2e-3 | lr=1e-3 | lr=5e-4 |
|---|---|---|---|
| real d160 | 5/5, 0.422 ± 0.139 | not run | not run |
| real d224 | 5/5, 0.407 ± 0.085 | not run | not run |
| real d320 | 5/5, 0.542 ± 0.079 | 5/5, 0.637 ± 0.230 | **5/5, 0.654 ± 0.234** |
| real d448 | 3/5, 0.777 ± 0.124 | 5/5, 0.659 ± 0.236 | **5/5, 0.811 ± 0.090** |
| real d640 | **0/5, diverged** | 5/5, 0.794 ± 0.057 | **5/5, 0.868 ± 0.040** |
| **complex d160** | **5/5, 0.856 ± 0.018** | 5/5, 0.828 ± 0.033 | 5/5, 0.829 ± 0.030 |

**Complex is already at its optimum at 2e-3** (0.856, against 0.828 and 0.829 at the lower rates), so
the headline complex number never depended on the retune, and the asymmetry, once closed, did not move
it. The wide real baselines, by contrast, gain a great deal: d640 goes from *not training at all* to
0.868. Best-tuned real against best-tuned complex (0.856 ± 0.018):

| width | params | multiple of complex | best lr | seq@24 | delta | p |
|---|---|---|---|---|---|---|
| d160 | 1.54M | 0.5x | 2e-3 | 0.422 ± 0.139 | +0.434 | 0.0020 |
| d224 | 3.02M | 1.0x | 2e-3 | 0.407 ± 0.085 | +0.450 | 0.0002 |
| d320 | 6.16M | 2.0x | 5e-4 | 0.654 ± 0.234 | +0.202 | 0.13 |
| d448 | 12.06M | 3.9x | 5e-4 | 0.811 ± 0.090 | +0.045 | 0.33 |
| d640 | 24.61M | 8.0x | 5e-4 | **0.868 ± 0.040** | -0.012 | 0.57 |

The real curve is **not flat**: 0.42, 0.41, 0.65, 0.81, 0.87. **H2_width is falsified and H0_width
holds.** The round-one framing that the reverse gap is unreachable by real width does not survive, and
the reason it ever looked that way is instructive: d160 and d224 are the two widths we happened to
sample, and they are exactly the flat stretch of a rising curve.

**State the crossover carefully, because the two readings of it disagree.** By *point estimate*, real
matches complex at ~12M (3.9x) and overtakes at 24.6M (8.0x). By *significance*, the gap already fails
p < 0.05 at 6.16M (2.0x). Those disagree because the wide real baselines are **seed-unstable**: at
d320 the real std is 0.234 against complex's 0.018, so the test loses power long before the means
converge (real is still 0.20 lower there). Quoting either number alone would be picking the flattering
summary. The full statement is: complex at 3.09M significantly beats a *tuned* real baseline at equal
parameters (+0.450, p = 0.0002); its accuracy is matched in point estimate at roughly 4x the
parameters; and the significance boundary sits lower, at 2x, only because the baseline's variance
explodes. The gap between the two readings is itself a result: **the tying buys seed-consistency as
much as accuracy** (std 0.018 vs 0.090-0.234 across the wide real baselines).

Note this also recontextualizes the round-one headline without contradicting it. The 0.41 the paper
quotes for real+NoPE is correct *for the parameter-matched control*, but it is not "what a real
transformer does on reverse": a properly tuned real model reaches 0.87. The claim was always about
matched parameters, and should be stated that way rather than as a statement about real backbones.

**What this leaves, combined with H2_alg.** Three separate framings are now dead: the implicit-RoPE
mechanism (round one, three scales, plus H1_mech_act above), "complex weights are special"
(H2_alg: split-complex and block-diagonal match complex), and "unreachable by real width" (here).
What replaces them is a **parameter-efficiency** claim, and it is the one the secondary hypothesis
originally asked about: on reverse, a two-stream weight tying reaches at 3.09M parameters what an
unconstrained real model needs ~12M to reach (statistically level at d448, 3.9x; real overtakes only
at d640, 8.0x). Round one recorded H1_comp as NOT supported on the strength of the d224 point alone
("complex buys nothing at matched parameters outside reverse"); that verdict was correct on its
evidence but under-powered on the parameter axis, and **on reverse it is now reversed**: there is a
real Pareto point. It remains one task out of six, it is a statement about parameter efficiency
rather than about complex numbers, and the ~4x figure is the resolution of our width grid rather than
a measured crossover. It is also provisional in one direction we can still check: complex has so far
only been run at lr=2e-3, so 0.856 may not be its ceiling either, and the in-flight lr sweep can only
move this ratio *up*.

#### H3_offset: NOT SUPPORTED. Length-independence does not predict where complex wins.

The registered law said the complex advantage should appear exactly when the required retrieval
offset is affine in the answer index with no dependence on input length. `revskip` and `skipcopy` are
the matched pair built to isolate that: identical stride, identical answer length (`ceil(L/2)`),
identical alternating-token selection, differing **only** in the anchor and therefore only in
length-dependence (the arithmetic is asserted in `tests/test_tasks.py`, not assumed). Exact-match at
1.5x train length, 5 seeds, complex+NoPE d160 vs the parameter-matched real+NoPE d224:

| task | anchor | offset(j) | length-dep? | complex | real d224 | delta | p |
|---|---|---|---|---|---|---|---|
| **reverse** | end | `-(2j+1)` | no | **0.856 ± 0.02** | 0.407 ± 0.08 | **+0.450** | **0.0002** |
| revskip | end | `-(3j+1)` | no | 0.259 ± 0.18 | 0.187 ± 0.14 | +0.072 | 0.49 |
| skipcopy | start | `j-L` | **yes** | 0.396 ± 0.14 | 0.275 ± 0.04 | +0.121 | 0.13 |
| copy | start | `-L` | **yes** | 0.94 ± 0.04 | 0.76 ± 0.37 | +0.18 | n.s. |

The prediction was `revskip` behaves like `reverse` (large gap) and `skipcopy` like `copy` (none).
Neither holds. Both new tasks land at a small, non-significant gap, and the point estimate is
*larger* on the start-anchored member of the pair, which is the wrong direction. **H0_offset stands.**

Read carefully, this is a failure to transfer rather than a sign reversal: no comparison is
significant either way, and with seed std of 0.14-0.18 at n=5 the power to detect a `reverse`-sized
effect is good but the power to resolve a 0.07-0.12 gap is not. The defensible statement is the
negative one: **whatever produces the +0.45 on `reverse` does not appear on another end-anchored,
length-independent, affine-offset task, so length-independence is not sufficient for it.** The law
was a clean pre-registered prediction and it was wrong; it is recorded rather than quietly replaced
with a post-hoc story that fits.

This leaves `reverse` genuinely singular. Its gap is roughly an order of magnitude larger than on any
other task measured, including one constructed specifically to share its offset structure, and four
candidate explanations are now eliminated: weight-level phase banding (round one, three scales), the
attention staircase and the deep rotation as causes (H1_mech_act above), and the offset law here.

#### H4_needle: FALSIFIED. Both arms decay, and the parameter-matched real baseline decays *least*.

`experiments/reeval.py` re-evaluates the 20 committed `pe=none` needle checkpoints at 256/512/1024
with no retraining (inference only, `eval_batch=32`; RoPE and learned-APE are ineligible because they
build a positional buffer at `max_T`). Exact-match, mean ± std over 5 seeds:

| condition | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| complex+NoPE (no input phase) | 1.00 ± 0.01 | 0.97 ± 0.04 | 0.87 ± 0.13 | 0.72 ± 0.24 |
| complex+NoPE | 0.85 ± 0.28 | 0.76 ± 0.34 | 0.70 ± 0.35 | 0.64 ± 0.37 |
| real+NoPE (d224 matched) | 0.92 ± 0.18 | 0.87 ± 0.29 | 0.84 ± 0.35 | **0.83 ± 0.38** |

The registered prediction was that the ablated complex model holds ceiling (seq >= 0.99) while the
real baseline continues its decay, separating the arms at p < 0.05 on the existing seeds. The
opposite happens: the ablated model leaves ceiling immediately past 128 and falls fastest, while the
real baseline is nearly flat from 128 to 1024. No comparison is significant at any length
(p = 0.50 / 0.88 / 0.61 for the ablated arm), and the sign flips against complex by 1024.
**H0_needle holds and the round-one verdict is reinforced, not overturned: needle discriminates by
parameter count, not by weight field.** The "1.00 ± 0.01 through 128" reading was a ceiling effect of
a too-short length axis, exactly the failure mode the length push was designed to expose. Honest
wart: every arm is strongly bimodal across seeds at long haystacks (std 0.24-0.38), so these are
seed-level coin flips on whether retrieval survives, not smooth degradation.

#### H1_mech_act: rotating subspace confirmed and dissociated, but it does **not** explain the effect

`experiments/activation_probe.py`, run locally on the committed d160 reverse checkpoints (10 runs:
3 complex seeds, the ablated model, 3 real d160 seeds, 3 real d224 seeds). Position-conditional mean
residual stream per block, top-2 principal plane, unwrapped in-plane angle fitted against position;
null = shuffled position labels, which leaves the point cloud and its PCA untouched and destroys only
the ordering. Angle-fit R^2 by block depth at eval length 24 (block 0 input is the embedding, which
carries no position and correctly reads ~0 in every condition, a sanity check on the probe):

| condition | blk0 | blk1 | blk2 | blk3 | blk4 |
|---|---|---|---|---|---|
| complex+NoPE s0/s1/s2 | 0.00 | 0.80 | 0.97-0.98 | 0.97-0.99 | **0.98-0.99** |
| complex+NoPE (no input phase) | 0.00 | 0.79 | 0.94 | 0.95 | **0.97** |
| real+NoPE d160 s0/s1/s2 | 0.00 | 0.79 | 0.96-0.97 | 0.00-0.95 | 0.00 / 0.88 / 0.93 |
| real+NoPE d224 s0/s1/s2 | 0.00 | 0.79 | 0.95-0.97 | 0.00 / 0.90 / 0.00 | **0.00 / 0.00 / 0.00** |

Three readings, in decreasing order of confidence:

1. **Every NoPE model builds a positional rotation in its early blocks** (blk1 = 0.79-0.81 in *all ten*
   runs, blk2 = 0.94-0.98), which is what bootstrapping position from the causal mask looks like. This
   is not complex-specific and is the first direct evidence in this project that NoPE position exists
   as a geometric object rather than only as a variance share.
2. **It is a *fractional*-position code, not a fixed-frequency clock.** The discriminator is the total
   angle swept across the sequence: a RoPE-like clock holds omega fixed and sweeps proportionally
   further as the sequence grows, while a normalized-position code holds the *sweep* fixed and shrinks
   omega like 1/T. Deepest-block total sweep, in radians, over a 3x span of evaluation length:

   | condition | L=16 | L=24 | L=48 |
   |---|---|---|---|
   | complex+NoPE s0 / s1 / s2 | 5.41 / 5.53 / 5.43 | 5.43 / 5.65 / 5.42 | 5.38 / 5.64 / 5.27 |
   | complex+NoPE (no input phase) | 5.22 | 5.29 | 5.08 |

   All four complex runs hold ~5.1-5.7 rad while the sequence triples, so the code is normalized
   position, which is length-invariant by construction. This also finally explains why the registered
   weight-level H1_mech had to come back null: there is no fixed rotation for the weights to store.
3. **The depth dissociation is real but does not explain the behaviour, and it is not clean.** At the
   trained-length scale the split is total: at L=16 and L=24 all three parameter-matched real seeds
   read R^2 <= 0.002 at the last block while all four complex models read 0.96-0.99. But at L=48 one
   real d224 seed (s0) acquires a deep rotation (0.944, sweep 6.12) while the other two stay at
   0.005-0.010, so the property is seed- and length-dependent in the real field rather than absent.
   More decisively, *within* real+NoPE d160 the association with the outcome runs **backwards**:
   seed 0 has no deep rotation at L=16/24 (R^2 = 0.001) and the **best** reverse seq@24 of its
   condition (0.609); seed 2 has the strongest (0.928) and the **worst** (0.227). A property that
   anti-correlates with the outcome inside a condition cannot be the mechanism producing it across
   conditions. Reported as a real representational difference, not as the explanation.

**Addendum (2026-07-26): the block-diagonal model carries the same signature.** Once H2_alg showed the
behaviour does not need complex arithmetic, the obvious question is whether the *rotation* does either.
The sweep ran with `save_model=False`, so this is an independent retrain of the same seed-0 `diag`
recipe (`local_reverse_complex_none_d160_diag_s0`, seq@24 = 0.84 locally vs 0.883 on the cluster; not
bit-identical across GPUs). Deepest block:

| condition | R^2 @16/24/48 | total sweep @16/24/48 (rad) | p vs shuffled null (L=24) |
|---|---|---|---|
| complex+NoPE s0-s2 | 0.95-0.99 | 5.3-5.7 | 0.000 |
| **block-diagonal (no complex arithmetic)** | 0.85 / 0.84 / 0.82 | **4.89 / 4.90 / 4.89** | **0.060** |
| real+NoPE d224 s1, s2 | 0.00 | 0.1-0.5 | 0.97-1.00 |

The block-diagonal model has the **most length-invariant sweep of anything measured** (4.89 rad at all
three lengths, a 3x span), which places it with the two-stream group and nowhere near the two
parameter-matched real seeds that have no positional geometry at all. But its angle fit is *at* the
shuffled-position null (p = 0.060, against p = 0.000 for every complex run), so on the strict test it
does not clear significance, and this is one model, not five seeds. The defensible conclusion is the
negative one it was run to test: **the deep rotation is not evidence of complex arithmetic**, since a
model containing none of it shows the same invariant-sweep signature. It does not follow that the
probe cleanly separates two-stream from single-stream architectures; two of five real runs
(d160 s1, s2) also carry a rotation, and diag's own significance is marginal.

**Attention staircase: falsified, in the opposite direction.** reverse needs answer token `j` fetched
from offset `-(2j+1)`; the registered guess was that complex owns a head walking that staircase more
cleanly. Measured per head as exact-hit rate and as attention mass on the required source, at length
24 (batch-averaged over 256 random sequences, so content marginalises out):

| condition | best hit rate | mass on required source | total across heads |
|---|---|---|---|
| complex+NoPE s0/s1/s2 | 0.04 | 0.04-0.05 | 0.40-0.42 |
| complex+NoPE (no input phase) | **1.00** | 0.20 | 0.50 |
| real+NoPE d224 s0/s1/s2 | **1.00** | 0.32-0.66 | 1.31-2.49 |

The parameter-matched real models, which score 0.41, route **more** cleanly than the complex models,
which score 0.86; and full complex has no staircase head at all while its own ablation has a perfect
one at equal accuracy. So the complex advantage on reverse is **not** better positional routing, and
at 1.5x train length the real baseline's failures are not routing failures. Combined with round one's
`pos_fraction` result (complex puts no extra position into attention), the effect is downstream of
attention in both measurements we have. Artifacts: `results/_figs/act/activation_metrics.json`,
`act_staircase_L*.html`.

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
