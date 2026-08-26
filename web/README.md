# Interactive companion to *Phase Without Position*

**Live: <https://web-one-theta-88.vercel.app/>**

A static site that runs the paper's trained transformers in the reader's browser. There is no server
component and no API route.

Deployed from this directory, with the Vercel project's **Root Directory set to `web`**. That setting
is required, not cosmetic: `package.json` is here rather than at the repository root, so a build
started from the root finds nothing to build. Pushes to `main` deploy production; other branches get
preview URLs, which are behind Vercel Authentication while production is public.

## What is live and what is committed

| On the page | Where it comes from |
|---|---|
| the reverse demo, the rotating-plane probe, the shuffle null, the attention staircase | computed in a Web Worker by the checkpoints in `public/models/`, on the reader's machine |
| length generalization, the tying ablation, the width scan, the weight-level circular tests | `public/data/*.json`, projections of the committed metrics in `../results/` |

Nothing on the page is a pre-rendered figure.

## Regenerating the assets

Both scripts live in the repository root and read `../results/`:

```bash
uv run python scripts/export_web_models.py   # checkpoints -> web/public/models/  (~18 MB, fp16)
uv run python scripts/export_web_data.py     # committed metrics -> web/public/data/
```

`export_web_data.py` reuses `experiments.stats` for learning-rate selection and seed filtering, so the
site's tables cannot drift from the paper's. Two filters there are load-bearing and were both wrong on
the first pass: every claim except the width scan reads the canonical learning rate only (pooling the
grid moves complex+NoPE from 0.856 to 0.838), and the algebra ablation is NoPE-only (complex+RoPE also
reports `algebra == "complex"`, and pooling its five zero-scoring runs destroys the reference arm).

## The browser engine

`lib/engine/` is a from-scratch port of `complexnn/` and `experiments/activation_probe.py` to
TypeScript over typed arrays: no PyTorch, no ONNX, no inference runtime. It covers all four 2D weight
algebras, real and complex RoPE, and learned positional embeddings.

Two things that cost real debugging time, recorded so they are not rediscovered:

- **`F.gelu` defaults to the exact erf form,** not the tanh approximation. Using tanh leaves a 9e-3
  relative logit error that fp32 does not fix.
- **The paper reports one entry per block *input*.** A capture that also keeps the stream past the last
  block is one step deeper than `L4`, which looks like a small systematic disagreement rather than an
  off-by-one.

### Verification

`npm run verify` reproduces, from the shipped fp16 blobs, the paper's teacher-forced accuracies, its
per-block rotating-plane statistics, and Section 9's claim as a ratio across a 4x span of evaluation
length. `npm run build` runs the `--quick` subset first, so the site cannot build against a broken
engine. Current agreement, complex+NoPE d160 seed 0 at the deepest block:

| eval length | ω (browser) | ω (paper) | sweep (browser) | sweep (paper) |
|---|---|---|---|---|
| 16 | 0.1744 | 0.1744 | 5.41 | 5.41 |
| 24 | 0.1155 | 0.1155 | 5.43 | 5.43 |
| 64 | 0.0418 | 0.0420 | 5.31 | 5.34 |

### Known deviation: the RoPE control at long lengths

The RoPE positive control needs more sequences than the NoPE models to settle. At about 1 rad per
position its plane trajectory wraps many times, so residual content noise moves the angle much
further: at n=8 it reads ω = 0.91 against the committed 0.991, at n=64 it reads 0.990. That is why
the panel's default sample size is 8 for the NoPE models and 64 when a RoPE control is selected.

Even at n=64 the browser probe reads low for that one arm at the two longest evaluation lengths:

| eval length | ω (browser) | ω (committed) |
|---|---|---|
| 16, 24, 32 | 0.983, 0.990, 0.865 | 0.984, 0.991, 0.865 |
| 48 | 0.791 | 0.875 |
| 64 | 0.780 | 0.848 |

This is not a TypeScript/Python difference: the numpy prototype of the same pipeline reproduces the
browser's values at 512 sequences, so it is a property of this probe implementation rather than of
sampling. The likely cause is the choice of principal plane where the second and third eigenvalues are
close, which the RoPE arm's loose fit (R² ≈ 0.80 at these lengths) makes easy to shift. The
qualitative contrast the panel exists to show is unaffected in either direction: across 16 to 64 the
NoPE models' ω falls by about 4x while their sweep does not move, and the RoPE control's ω stays
within about 20% of itself while its sweep triples. `verify.ts` therefore asserts the RoPE control's
ω only at the lengths where the two agree, and asserts the contrast itself as a ratio.

## Performance notes

A d160 complex forward pass at T=48 is about 0.6 GFLOP. The 4x4 register-blocked GEMM in
`lib/engine/kernels.ts` sustains ~2.5 GFLOP/s in V8, against ~0.7 for the naive triple loop.

The probe does not need the paper's 512 sequences. The position-conditional mean converges almost
immediately: at n=8 the deepest block reads ω = 0.1151 against 0.1155 at n=512. That is what makes the
panel affordable live, and it is why the default is 8.

Work is spread over three workers with model affinity as a *preference* rather than a rule, and each
panel waits until it scrolls into view. Both were needed: three panels default to the same model, so
strict affinity piles every request onto one lane while the others idle.
