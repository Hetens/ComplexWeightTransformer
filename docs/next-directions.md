# Next directions

Where this project can go after two rounds. Written 2026-07-28, after the round-two commit (`c4ed0a7`).

The framing that matters: **round two killed the question we started with and handed us two better ones.**
Complex weights do not infer RoPE, and the one surviving behavioural effect is not about `ℂ`. But the
two controls that killed it are each interesting in their own right:

* the **block-diagonal tying** (`[[A,0],[0,B]]`, no complex arithmetic, no cross-stream mixing at all)
  reaches 0.830 against a dense real model's 0.407 **at identical parameters**, and
* the **activation probe** found a positional code that is length-normalised rather than absolute.

Both of those connect to live literatures that the project was not aimed at. This document maps the
connections, and for each direction states the decisive experiment and what would falsify it.

Everything below is ordered by what I would actually run first.

---

## The fractional-position measurement, and its positive control

Direction 2 rests on the claim that NoPE models encode *fractional* position. `docs/results.md`
supports it from a null result: the total swept angle does not grow as the sequence gets longer. A null
is weak on its own, because a statistic that simply cannot see a fixed-frequency code would produce the
same null for free. So the **positive control** is the same probe run on the committed RoPE
checkpoints, which are handed an external fixed-frequency rotation by construction. If the statistic
can see a fixed-frequency code anywhere, it must see one there.

This is now part of the committed probe rather than a side script. Reproduce with:

```bash
uv run python -m experiments.activation_probe --lengths 16 24 32 48 64
```

which writes `results/_figs/act/activation_metrics.json` (committed). Deepest block, reverse,
`omega` = fitted rad/step of the residual-stream rotation, sweep = total angle swept end to end:

| run | seq@24 | omega @16 | @24 | @32 | @48 | @64 | sweep @16 -> @64 | reading |
|---|---|---|---|---|---|---|---|---|
| real+**RoPE** d160 | 0.00 | 0.984 | 0.991 | 0.865 | 0.875 | 0.848 | **30.5 -> 107.7** | **fixed frequency** |
| complex+NoPE d160 | 0.84 | 0.174 | 0.116 | 0.086 | 0.057 | 0.042 | 5.41 -> 5.34 | **fractional** |
| complex+diag NoPE | 0.84 | 0.158 | 0.104 | 0.078 | 0.051 | 0.038 | 4.89 -> 4.88 | **fractional** |
| complex+RoPE d160 | 0.00 | 0.148 | 0.096 | 0.071 | 0.049 | 0.036 | 4.58 -> 4.57 | fractional |
| real+NoPE d160 | 0.61 | 0.007 | 0.002 | 0.106 | 0.064 | 0.046 | 0.23 -> 5.87 | none in-distribution |

The control fires. The one model whose `omega` stays flat as length grows is the one with a
fixed-frequency code injected into it, and its total sweep grows 3.5x over a 4x length span. Every
NoPE model that has a deep rotation at all has `omega` falling like `1/T` instead, holding the total
sweep constant to within about 2% over that same 4x span. That is a length-normalised position code,
measured, with a working positive control.

Four honest caveats.

1. RoPE conditions are n=1 seed, so they fix the sign of the contrast, not its size.
2. The control is clean in **one** of the two RoPE models probed. The parameter-matched real+RoPE d224
   model has no coherent deepest-block rotation past length 32 at all (R^2 drops to 0.01-0.08), so
   there is nothing there for the statistic to characterise. The contrast is carried by the d160 run.
3. This reads only the deepest block, so it says nothing about the early-layer rotation that
   `results.md` reports in every model.
4. The shuffled-position null loses power on fast rotations (real+RoPE sits at R^2 ~ 0.80 yet does not
   clear the null), so `omega`-invariance, not the p-value, is what carries the RoPE row.

**The result also complicates the obvious story, in a useful way.** The naive reading of "fractional
position is the length-generalisation bottleneck" predicts that the fixed-frequency model should
extrapolate best. It is the worst: real+RoPE has a clean absolute clock and scores **0.00** at 1.5x
training length, while complex+NoPE has a fractional code and scores 0.84. See direction 2.

---

## Direction 1: the finding is multi-stream tying, and that is a hot architecture question

**Claim to test.** Our round-two result is not "complex numbers do nothing". It is: *two parallel
residual streams that share one attention distribution beat one dense stream at equal parameters,
and the mixing between the streams does not matter.*

That is exactly the ablation the multi-stream residual literature has not run at fixed parameters.
[Hyper-Connections](https://arxiv.org/abs/2409.19606) (ByteDance, ICLR 2025) widens the residual
stream into n parallel streams with *learnable mixing matrices*, and DeepSeek's
[mHC](https://arxiv.org/abs/2603.14833) constrains those matrices to a manifold. The premise of both
is that the learned mixing is the point. Our four tyings are a direct test of that premise, because
they are four different mixing patterns at identical width and identical parameter count:

| mixing pattern | algebra | seq@24 |
|---|---|---|
| full, `[[A,-B],[B,A]]` | complex | 0.856 ± 0.018 |
| full, `[[A,B],[B,A]]` | split-complex | 0.846 ± 0.05 |
| **none**, `[[A,0],[0,B]]` | (not an algebra) | **0.830 ± 0.05** |
| triangular, `[[A,0],[B,A]]` | dual | 0.754 ± 0.08 |
| single dense stream, same params | real d224 | 0.407 ± 0.085 |

The gap between *any* two-stream tying and the single-stream baseline is 0.35 to 0.45. The gap
between full mixing and **no mixing at all** is 0.026, p = 0.31. On this task, at this scale, the
streams are what matters and the mixing is free.

**What the streams still share under `diag`.** The weights stop mixing, but three couplings survive,
and naming them precisely is what makes the follow-up designable:

| # | where | code | effect |
|---|---|---|---|
| C1 | attention logits | `attention.py:132`, `q.real @ k.real^T + q.imag @ k.imag^T`, one softmax applied to both streams' values (`:138`) | the streams must **agree on where to attend** |
| C2 | norm | `layers.py:148`, `ms = (z.real**2 + z.imag**2).mean(-1)`, one scalar rescaling both | the streams **share a gain**, so one cannot blow up alone |
| C3 | readout | `models.py:111`, `nn.Linear(2*dim, vocab)` over `[Re ‖ Im]` | the logits are a **learned combination** of both streams |

So the winning model is two width-160 transformers that share an attention pattern, a normalisation
gain, and a learned output head. C3 alone already makes it an ensemble with a learned combiner, which
is why "is this just ensembling?" is the first question to settle and not the last. The variance result
(complex ± 0.018 versus real d640's ± 0.234) is consistent with any of the three.

**Decisive experiment: knock out C1, C2, C3 one at a time.** All at fixed parameters, on reverse,
5 seeds, against the `diag` reference of 0.830.

| arm | change | tests |
|---|---|---|
| E0 **true ensemble** | two independently trained real d160 models, average logits | is any of this more than ensembling? |
| E1 **split attention** | each stream gets its own logits and softmax (kills C1, keeps C2, C3) | is shared attention load-bearing? |
| E2 **split norm** | per-stream RMS norm (kills C2, keeps C1, C3) | is the shared gain load-bearing? |
| E3 **split readout** | logits = sum of two per-stream heads, no learned combination (weakens C3) | is the learned combiner load-bearing? |

E0 is the one to run first, and it is the cheapest: it needs no architecture change at all, just two
d160 runs and a logit average at eval. If E0 reaches 0.83, the whole story is "two narrow models beat
one wide one at fixed parameters", which is ensembling, is already well understood, and is not worth a
paper. Everything downstream is conditional on E0 coming in low.

If E0 is low and E1 collapses, the claim becomes concrete and novel: *the benefit comes from forcing
two streams to agree on one attention distribution*, which is a constraint no ensemble has and which
the hyper-connections line does not isolate.

**Then the k-scan.** At fixed budget P, k streams of width w each gives total width `sqrt(P·k)`:
k=1 → d224 (0.407 measured), k=2 → d160×2 (0.830 measured), k=4 → d112×4, k=8 → d79×8. Predict an
inverted U in k and monotonically falling seed variance. Add a learned k×k mixing matrix at each k
(cheap, k² parameters per layer, this is exactly the hyper-connections width-connection): our p = 0.31
says mixing is free at k=2, but narrow streams at k=8 may need it.

Cost: E0 is 10 runs and no new code. E1 to E3 are ~15 runs plus small edits to `attention.py`,
`layers.py`, `models.py`. The k-scan is ~20 runs. All fit `gpu-preempt`.

**Falsifier.** E0 reaching 0.83 kills the direction. Run it first.

**Related work to position against.** [Compute Better Spent](https://arxiv.org/abs/2406.06248) (Qiu
et al., ICML 2024) already shows block-diagonal and Monarch structures beating dense at fixed compute,
and warns that structured layers need their own initialisation and learning-rate scaling. We hit that
warning independently: the d160-tuned learning rate does not transfer up the width axis. Their result
is about compute efficiency on in-distribution loss; ours would be about *length generalisation* and
*seed variance*, which they do not measure.

---

## Direction 2: NoPE models learn Position Interpolation by themselves

**Claim to test.** The length-normalised code we just measured is the same object that
[Position Interpolation](https://arxiv.org/abs/2306.15595) (Chen et al., 2023) manufactures by hand.
PI rescales position indices by `T_train/T_test` to keep RoPE's angles inside the pretrained range.
A NoPE transformer, given no positional module at all, converges on a code that is already normalised
that way. Nobody has to rescale it, because `omega ∝ 1/T` falls out of training.

This is not a folk claim, it has theoretical backing from the other side.
[The Counting Power of Transformers with No Positional Encodings](https://arxiv.org/abs/2505.11199)
shows NoPE's expressive constructions carry components of the form `x_i/(n+1)`, normalised by sequence
length, and explicitly notes that this normalisation is why the constructions do not transfer across
lengths. Softmax attention returns convex combinations, so any positional quantity a NoPE model can
read off cheaply is a *ratio*. Fractional position is the architecture's attractor, not a training
accident.

**Why this reframes length generalisation.** It explains both halves of the RoPE-versus-NoPE picture
that [Kazemnejad et al.](https://arxiv.org/abs/2305.19466) reported empirically:

* RoPE fails OOD with a *cliff* (our real+RoPE: 1.00 at 16, then 0.00) because a fixed-frequency code
  evaluated past training length produces angles never seen in training.
* NoPE degrades *gracefully* (our complex+NoPE: 1.00, 0.84, 0.12, 0.00) because its code is never OOD
  by construction. What it loses instead is **resolution**: with total sweep `C ≈ 5.4` rad held fixed,
  adjacent positions are separated by `C/T` radians, which shrinks as the sequence grows.

That yields a quantitative, falsifiable law. Read the collapse off our own numbers: the model works at
`omega = 0.116` (L=24, seq 0.84), breaks at `omega = 0.086` (L=32, seq 0.12), and is gone by
`omega = 0.057` (L=48, seq 0.00). So there is an angular discrimination threshold `δ ≈ 0.1` rad, and
the prediction is:

> length generalisation fails at `T* ≈ C/δ`, so **anything that increases the total sweep `C` should
> push `T*` out proportionally.**

**Decisive experiment.** Manipulate `C` and check whether `T*` moves with it.
1. Train on a wider band of lengths (6 to 32 instead of 6 to 16) and re-probe. Does `C` grow, and does
   `T*` move in step?
2. Add a second rotating subspace at a different scale, the NoPE analogue of what NTK-aware scaling and
   [YaRN](https://arxiv.org/abs/2309.00071) do for RoPE (preserve high-frequency components while
   stretching low ones). Predict: multi-scale sweep beats single-scale at long lengths.
3. Causal check: patch the fitted rotation at eval time to hold `omega` fixed rather than letting it
   fall as `1/T`. If `T*` does not move, the rotation is correlational and direction 2 is dead.

Test 3 is the cheap one and should go first: it is inference-only on checkpoints we already hold
locally, no cluster time at all.

**Falsifier.** The dissociation from round two still stands and is the main threat: real+NoPE d160 has
**no** deep rotation at L=16 or L=24 (R² = 0.002, p = 0.97) and still scores 0.61. So the rotation is
plainly not necessary for the behaviour. If test 3 comes back null, the honest conclusion is that the
fractional code is a real and cleanly-measured property of these models that does not drive the
capability, which is worth writing up as a negative result but is not a research programme.

---

## Direction 3: phase pays under composition, not under scoring

**Claim to test.** Our null and Mamba-3's positive result are the same fact seen from two sides, and
together they give a rule for when complex parameterisation is worth it.

Why our attention result had to be null, structurally: the complex attention score is
`Re(q k^H) = q_re·k_re + q_im·k_im`, which is *exactly* the real dot product in the doubled space.
There is no complex arithmetic in the score. Phase enters once, in a single bilinear form, and a
single bilinear form cannot exploit it.

[Mamba-3](https://arxiv.org/abs/2603.15569) (ICLR 2026) is the contrast case. It reintroduces
complex-valued state updates precisely because real eigenvalues cannot express rotational dynamics:
parity needs `h_t = R(πx_t)h_{t-1}`, and Mamba-2 scores 0.90% on it against Mamba-3's 100%. The
difference is that in a recurrence the rotation is **accumulated**, `∏ R_i`, so phase becomes a group
action and the model gets state tracking. Tellingly, Mamba-3 implements this as RoPE-style rotations
applied to B and C rather than with actual complex arithmetic, which is the same conclusion we reached:
the useful object is the rotation structure, not the number field.

So the proposed rule: **phase buys you something exactly when it is composed, and nothing when it is
scored once.**

**Decisive experiment.** This is directly runnable in our harness and would convert a negative result
into a predictive one. Add parity and modular arithmetic to `experiments/data/tasks.py` (both are
small; parity is the canonical state-tracking probe), then cross two factors:

| | attention (score once) | linear recurrence (compose) |
|---|---|---|
| complex | predict ≈ diag | predict ≫ diag, real |
| diag (block-diagonal) | 0.830 measured | predict ≈ real |
| real dense, matched params | 0.407 measured | predict fails parity |

The prediction that makes it worth running: **the same `diag` control that erased the complex
advantage in attention should fail to erase it in the recurrent cell.** One task family, one
architecture axis, and the algebra ablation flips sign. That is a clean result either way.

Cost: the recurrent cell is new code (a diagonal linear RNN over the four algebras), maybe 150 lines,
plus 3 algebras × 2 architectures × 2 tasks × 5 seeds = 60 short runs on small models.

---

## Direction 4: the offset law as an instrument, and transfer

Round two's H3_offset failed: `revskip` (+0.072, p = 0.49) and `skipcopy` (+0.121, p = 0.13) did not
reproduce reverse's +0.450, so end-anchored length-independence does not predict where the tying wins.
The offset law itself is still a good instrument, and two recent results say how to aim it better.

[Golowich et al.](https://arxiv.org/abs/2502.16792) prove length generalisation happens when each
predicted token depends on a small fixed number of previous tokens (k-sparse dependency), which
covers our whole task family at k=1 and therefore does *not* separate copy from reverse. But the tasks
differ in the *form* of the offset: copy is constant in j (`-L`), reverse is affine in j (`-(2j+1)`),
revskip is affine with a steeper slope (`-(3j+1)`). Our results so far, ordered by advantage, are
reverse (+0.450) > skipcopy (+0.121) > revskip (+0.072), which does not sort by slope. Worth mapping
the `o(j, L)` plane properly rather than sampling four points from it.

[Positional Attention](https://arxiv.org/abs/2410.01686) (ICML 2025) is the sharper tool: attention
weights that depend *only* on positional encodings, with better OOD generalisation exactly on tasks
whose algorithm is purely positional. Every task in our suite is purely positional. A positional-attention
arm belongs in the comparison as an upper bound on what a "solve it with position alone" model achieves.

[Extrapolation by Association](https://arxiv.org/abs/2506.09251) reports that length-generalisation
transfer between tasks correlates with re-use of the same attention heads. Direct experiment: co-train
reverse and revskip, then check whether reverse's advantage transfers to revskip and whether the
staircase head is shared. Our H3 result says the advantage does not appear when revskip is trained
alone, which makes it a good transfer target.

---

## Direction 5: report distributions, not means (cheap, and a real contribution)

Round one already found extrapolation is bimodal across seeds in every NoPE condition on copy, and
round two found the wide real baselines swing by ± 0.234 while the tied model sits at ± 0.018. That
variance gap is arguably the more useful finding than the mean gap: the tying buys *reliability* of
length generalisation.

This is a known-but-underreported problem.
[Transformers Can Achieve Length Generalization But Not Robustly](https://arxiv.org/abs/2402.09371)
found the same bimodality on addition and reports high seed variance. Most length-generalisation papers
still publish a mean over few seeds.

Cheap contribution, no new runs: re-analyse our 270 committed `result.json` files as **distributions**,
report the fraction of seeds that generalise rather than the mean, and show how often a 3-seed mean
would have reversed a conclusion. We have the data and the harness. This is a short methods note and it
strengthens every other direction by fixing how they get reported.

---

## What I would run first, given the cluster constraint

TRM has priority on `gpu`, so everything below is sized for `gpu-preempt` with `--requeue`.

| order | experiment | cost | why first |
|---|---|---|---|
| 1 | Direction 2, test 3 (patch `omega`, inference only) | **zero cluster time**, local checkpoints | kills or confirms the mechanism story in an afternoon |
| 2 | Direction 1, **E0** (two d160 models, average logits) | ~10 runs, **no new code** | one number decides whether direction 1 exists |
| 3 | Direction 5 (re-analysis) | zero | no new compute, improves reporting of everything else |
| 4 | Direction 1, E1 to E3 then the k-scan | ~35 runs | the positive result, if E0 comes in low |
| 5 | Direction 3 (parity, recurrent cell) | ~60 short runs + new code | highest ceiling, most new code |

Directions 2 and 1 are the two that could carry a paper on their own. Direction 3 is the one that would
turn the project's central negative result into a general rule.

## Scale, which is the standing limitation

Every number here comes from 5-layer models on synthetic tasks at 3M to 25M parameters, and the paper
lists this first among limitations. Any of the above that survives should then be checked at small-LM
pretraining scale (100M parameters, a few B tokens) against length extrapolation on real text. That is
the one item on this list that does not fit on a preemptible single-GPU budget, and it should be
attempted only after direction 1 or 2 has produced something worth the compute.
