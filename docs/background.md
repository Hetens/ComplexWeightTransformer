# Background knowledge

The background needed to understand and execute this project, written for a reader comfortable with
deep learning but not necessarily with positional-encoding theory or complex-valued networks. Companion
files: [`related-work.md`](related-work.md) (citations), [`feasibility.md`](feasibility.md),
[`hypothesis.md`](hypothesis.md), [`embedding-compression-frontier.md`](embedding-compression-frontier.md).

---

## 1. Why transformers need positional information

Self-attention is **permutation-equivariant**: if you shuffle the input tokens, the output is shuffled
the same way but otherwise unchanged. Concretely, attention computes, for query `i` and key `j`,
a score `qᵢ · kⱼ` that depends only on the *content* of tokens `i` and `j`, not on *where* they are.
So a plain attention block sees a sequence as a **set**, not a **sequence**. "The dog bit the man" and
"The man bit the dog" would be indistinguishable to the content path.

Language, code, and grids are not sets; order and 2-D layout carry meaning. The fix is to inject
**positional information** so attention can become position-aware. How you inject it is the
*positional encoding* (PE) design space.

A subtlety this project hinges on: **decoder-only (causal) transformers are not fully permutation
equivariant** even without a PE, because the causal mask makes token `i` attend only to `j ≤ i`. That
asymmetry leaks weak positional signal, which is why "no positional encoding" (NoPE) can work at all
(see §6).

---

## 2. Taxonomy of positional encodings

| Family | Idea | Adds params? | Relative? | Length extrapolation |
|---|---|---|---|---|
| **Sinusoidal APE** (Vaswani 2017) | add fixed `sin/cos(pos/10000^{...})` to embeddings | no | no | poor |
| **Learned APE** (GPT-2/BERT) | a trainable vector per absolute position | yes (`L×d`) | no | none past trained `L` |
| **Relative bias** (Shaw 2018; T5) | add a learned scalar bias to attention logits per relative offset | small | yes | moderate |
| **ALiBi** (Press 2022) | subtract a slope×distance penalty from logits | no | yes | good (decay-based) |
| **RoPE** (Su 2021) | *rotate* Q/K by a position-dependent angle | **no** | yes | good, the de-facto standard |
| **NoPE** | nothing explicit; rely on causal mask + depth | no | implicit | often best on length-gen |

- **Absolute (APE)**: tag each position with an embedding. Simple, but ties the model to positions seen
  in training, so it extrapolates badly.
- **Relative**: encode *distance* `i−j` rather than absolute index. Matches the translation-invariance of
  language and generalizes better.
- **RoPE** is the relative method that won: zero added parameters, encodes relative distance, plays
  nicely with efficient attention. It is the method this project tries to *internalize into the weights*.

---

## 3. RoPE in depth (the thing we want complex weights to infer)

### 3.1 The construction
Split a `d`-dimensional query/key into `d/2` coordinate **pairs**. Treat each pair `(x₁, x₂)` as a 2-D
vector, or equivalently as the complex number `z = x₁ + i·x₂`. For a token at position `m`, RoPE rotates
each pair by an angle `m·θⱼ`:

```
z'  =  e^{i m θ_j} · z  =  (cos m θ_j + i sin m θ_j)(x₁ + i x₂)
```

In real 2-vector form this is a rotation matrix:

```
[x₁']   [cos mθ_j   −sin mθ_j] [x₁]
[x₂'] = [sin mθ_j    cos mθ_j] [x₂]
```

The per-pair frequencies are a geometric series `θ_j = base^{−2j/d}` (base usually 10000), so low-index
pairs rotate fast (capture short-range order) and high-index pairs rotate slowly (capture long-range
order), a multi-scale "clock."

### 3.2 Why it gives *relative* position for free
After rotating query at `m` and key at `n`, their interaction depends only on `m − n`:

```
Re( ⟨ e^{i m θ} q ,  e^{i n θ} k ⟩ )  =  Re( q̄ · k · e^{i (n − m) θ} )
```

The dot product picks up a phase `e^{i(n−m)θ}` that is a function of the **relative** offset only. That
is the whole trick: absolute rotations, applied to both sides, leave a relative phase in the score.
Magnitudes are untouched because `|e^{iα}| = 1`, so RoPE never rescales activations.

### 3.3 2-D / axial RoPE
For grids (vision, ARC-style puzzles), **axial RoPE** applies 1-D RoPE twice: the first `d/2` channels
rotate by the `x`-coordinate, the rest by the `y`-coordinate. "Mixed"/learned-frequency variants add
diagonal sensitivity. Relevant here because "spatial reasoning" claims rest on 2-D position.

---

## 4. Complex-number refresher (only what we use)

- **Euler:** `e^{iθ} = cos θ + i sin θ`. Multiplying by `e^{iθ}` **rotates** by `θ`; multiplying by a
  general complex `w = r e^{iφ}` rotates by `φ` **and** scales by `r`.
- **Modulus / phase:** `z = a + ib = |z| e^{i·arg z}`, `|z| = √(a²+b²)`.
- **Conjugate:** `z̄ = a − ib`. `z·z̄ = |z|²` (real, ≥ 0).
- **Hermitian inner product** on `ℂⁿ`: `⟨u, v⟩ = Σ uᵢ v̄ᵢ`. Its real part `Re⟨u,v⟩` equals the ordinary
  real dot product of the stacked `[Re; Im]` vectors; this is how we get real attention logits from
  complex Q/K (§5.4).

---

## 5. Complex-valued neural networks (CVNNs)

The project's core move is replacing real weights with complex ones. Here is the machinery.

### 5.1 Complex linear layer = a *constrained* real layer
A complex weight `W = A + iB` acting on `z = x + iy` gives
`Wz = (Ax − By) + i(Bx + Ay)`. Stacking real/imag parts, this is the real linear map

```
[Re]   [A  −B] [x]
[Im] = [B   A] [y]
```

So a complex `d_out×d_in` layer is **exactly** a real `2d_out×2d_in` layer whose block structure is
forced to `[[A, −B],[B, A]]`. Two consequences:
1. **Parameter count:** `2·d_out·d_in` real numbers, vs `4·d_out·d_in` for an *unconstrained* real
   `2d×2d` layer → the ~2× / "50%" efficiency that CRoPE markets.
2. **Inductive bias:** these are precisely the real maps that **commute with multiplication by `i`**
   (the matrix `J=[[0,−I],[I,0]]`). They are *phase-equivariant*: rotate the input's phase and the
   output's phase rotates the same way. RoPE's pure rotation `[[cos,−sin],[sin,cos]]` is the special
   case `A=cosα·I, B=sinα·I`. **This is exactly why complex weights are a natural home for RoPE-like
   behavior.**

> **The honest caveat (drives the experiment):** complex weights give the *capacity* to apply and
> propagate phase rotations cheaply; they do **not** by themselves create a notion of position. Under
> NoPE, the positional signal must still originate from the causal mask + depth/iteration. The bet is
> that complex weights make it *easier* for the network to turn that implicit signal into RoPE-like
> relative-phase relationships. If not, the hypothesis fails cleanly (see `hypothesis.md`).

### 5.2 Complex activations (the non-trivial part)
A function `ℂ→ℂ` that is interesting and bounded **cannot be complex-differentiable (holomorphic)**
everywhere (Liouville). So practical CVNNs use *non-holomorphic* activations and rely on `ℝ²` autograd
(Wirtinger calculus, §5.5). Common choices:
- **CReLU:** `ReLU(Re z) + i·ReLU(Im z)`, simplest, splits real/imag. Strong default.
- **modReLU:** `ReLU(|z| + b) · z/|z|`, thresholds *magnitude*, **preserves phase** (good when phase is
  the carrier of position; `b` learnable).
- **zReLU:** pass `z` only if both parts are positive (phase in the first quadrant), else 0.

### 5.3 Complex normalization
- **Proper complex LayerNorm/BN** (Trabelsi 2018): treat each unit as a 2-D vector and whiten with its
  `2×2` covariance, correct but heavier.
- **ComplexRMSNorm (our default):** `z ← z / √(mean(|z|²)+ε) · γ`. Cheap, magnitude-only, phase-preserving.
  Well suited to a project where phase carries the signal we care about.

### 5.4 Complex attention
With complex `q, k, v`: logits = `Re(Σ qᵢ k̄ᵢ)` (real part of the Hermitian inner product), softmax over
real logits, output `Σ softmax · v` (complex). If `q, k` happen to carry position-dependent phase, the
real part naturally produces a `cos((m−n)θ)`-modulated score, i.e. attention *can* recover the RoPE
relative-position effect from learned phase. This is the mechanistic crux.

### 5.5 Initialization & optimization
- **Init (Trabelsi/Rayleigh):** sample magnitude `|w| ~ Rayleigh(σ)` with `σ` from a He/Glorot variance
  budget, phase `~ Uniform(−π, π)`. Gives `Var(w) = 2σ²`; pick `σ` so total variance matches the
  real-net target for the chosen fan-in/out.
- **Optimization / Wirtinger:** for a **real-valued loss** `L` of a complex parameter `z`, treat `z` and
  `z̄` as independent. PyTorch computes and stores in `.grad` the **conjugate Wirtinger derivative**
  `∂L/∂z̄`, defined so that `z ← z − lr·z.grad` is correct steepest descent. **Practical upshot:**
  standard optimizers (Adam, SGD) work on complex `Parameter`s unchanged; we just verify correctness
  with `torch.autograd.gradcheck` against an explicit `ℝ²` reference (see `feasibility.md`).
- **Readout `ℂ→ℝ`:** the task head needs reals. Options: take `Re(z)`, take `|z|`, or concatenate
  `[Re z ‖ Im z]` (no information lost). We default to concat for the LM/classifier head.

---

## 6. NoPE: why "infer position" is even plausible

"Transformers without Positional Encodings" (Haviv 2022) and "The Impact of PE on Length Generalization"
(Kazemnejad, NeurIPS 2023) show that **causal** decoder-only models with **no** explicit PE:
- recover a usable notion of absolute position internally (decodable by mid-layers), and
- frequently **match or beat** explicit PE (including RoPE) on length-generalization tasks.

The causal mask breaks permutation symmetry enough for position to emerge from depth. This is the
theoretical license for the project: if a *real* NoPE model can infer position, a *complex* NoPE model,
which has phase machinery built into every weight, might infer it *better* or *cheaper*. That delta is
exactly H1 (`hypothesis.md`).

---

## 7. Tiny recursive reasoners (context for the original idea)

The idea was sparked by **HRM** (Hierarchical Reasoning Model) and **TRM** (Tiny Recursive Model, Samsung
SAIL, 2025): a ~7M-parameter, 2-layer network that *recurses* on a latent scratchpad `z` and a current
answer `y`, reaching SOTA-for-its-size on ARC-AGI. In such tiny models, the embedding/positional machinery
is a **large fraction** of parameters, and the *same weights* are reused at every recursion step, so a
positional scheme baked into the weights (rather than added per step) is especially attractive. We do
**not** benchmark on grid puzzles in this PoC (dropped for clean, comparable signal), but this regime is
the motivation and the eventual target for scale-up.

---

## 8. Glossary
- **APE / RPE:** absolute / relative positional encoding.
- **RoPE:** rotary positional embedding, position as a rotation of Q/K pairs.
- **NoPE:** no explicit positional encoding (rely on causal mask + depth).
- **CVNN:** complex-valued neural network.
- **Holomorphic:** complex-differentiable; most useful activations are not.
- **Wirtinger calculus:** treat `z`, `z̄` as independent variables to differentiate non-holomorphic
  functions; how complex autograd works.
- **Phase-equivariant map:** a linear map commuting with multiplication by `i`; equivalently a complex
  linear layer.
- **Length extrapolation / "train-short-test-long":** train at sequence length `L`, evaluate at `>L`.
