# Is embedding / positional compression valuable at the frontier? (the "branch out")

The user asked to step back from complex numbers specifically and ask whether **compressing the
embedding / positional machinery** has real value in current DL/AI research. Short answer: **yes, but
unevenly**: it is a major lever for small/edge/recursive models and a live (not settled) research area,
while it matters less for the largest models where embeddings are a small parameter fraction.

---

## 1. Why embeddings are a compression target
The token-embedding + unembedding matrices are `vocab × d` each, often the **single largest parameter
block in small models**:
- ~**73%** of parameters in Pythia-70M are embeddings; the share shrinks to ~**9%** by 2.8B.
- So for **tiny models** (TRM ≈ 7M params, on-device models, recursive reasoners), shrinking the
  embedding/positional machinery is a first-order win; for 70B models it is a rounding error.

This is exactly why the complex-weight idea has the **most leverage in the tiny/recursive regime** that
motivated the project (`background.md` §7).

## 2. The standard compression toolbox (real-valued, today's frontier)
- **Weight tying**: share input & output embeddings (GPT-2/BERT/Gemma). Big saving for small models;
  recent large models (Llama-3, OLMo-2, Qwen3) increasingly **untie** because the bias it imposes hurts at
  scale ("Weight Tying Biases Token Embeddings Toward the Output Space", 2026).
- **Factorized embeddings**: ALBERT-style `vocab×r · r×d` low-rank factorization; decouples vocab size
  from hidden size.
- **Low-rank / tensor-train / Kronecker**: structured factorizations of the embedding matrix
  (TensorSLM, 2025).
- **Vector quantization / residual VQ**: store embeddings as codebook indices (CARVQ, 2025).
- **Per-layer / n-gram embeddings, vocabulary customization**: densify information per token.

Counter-current: **"Scaling Embeddings Outperforms Scaling Experts"** (2026) and **"Scaling Embedding
Layers"** (2025) argue bigger embeddings can be *better* spent compute than more experts, i.e. the
community is actively re-pricing embeddings, not treating them as fixed overhead. Compression and scaling
are two sides of the same live question: **how many parameters does token/position representation deserve?**

## 3. The *positional* machinery specifically
- **RoPE adds zero parameters** but costs compute and **bounds context** (the base frequency caps usable
  length; see "Base of RoPE Bounds Context Length", NeurIPS 2024). So the cost is *compute + capability*, not
  parameters.
- **NoPE removes explicit PE entirely** and often matches/beats it on length generalization, which is the existence
  proof that positional *parameters/modules* are not strictly necessary.
- **CRoPE** (preprint) reparametrizes attention in `ℂ` and claims **~50% attention-block parameter
  reduction**, framing RoPE as "reserving half the embedding space for position." If true, that is direct
  evidence that positional capability can be **folded into weights** at a real parameter saving, the exact
  bet of this project. We can validate it.

## 4. Where complex weights fit on this map
The complex-weight + NoPE idea is a **structural** compression of the positional machinery:
- it removes the separate PE module (like NoPE),
- it uses a phase-equivariant weight parametrization that *natively* expresses rotation (unlike real NoPE),
- and, at matched complex-entries, it spends **fewer real parameters** than an unconstrained real layer
  (the CRoPE 2× argument; see `background.md` §5.1).

Implications beyond parameter count:
- **Recursion:** TRM/HRM reuse the *same* weights every step; a positional scheme baked into weights is
  applied consistently across depth "for free," whereas an additive PE must be re-injected each step.
- **KV cache / inference:** removing a PE module simplifies the inference path; complex KV doubles cache
  size, a trade-off to measure (we log memory).

## 5. Honest frontier verdict
- **High value:** edge / on-device / tiny / recursive models, and any setting where the embedding-or-PE
  block dominates parameters or where length/spatial generalization is the bottleneck.
- **Lower value:** frontier-scale dense LLMs, where embeddings are a small fraction and the community is
  arguably moving toward *larger* embeddings, not smaller.
- **Research value regardless:** "how should position be represented, and how cheaply?" is unsettled and
  active (RoPE-base limits, NoPE, complex PE, CRoPE, embedding-scaling laws). A clean capability-vs-cost
  result for complex-weight NoPE contributes to that conversation even if the compression win is regime-
  specific.

This is why the project keeps **capability** primary and **compression** secondary: the compression payoff
is real but regime-dependent, whereas a capability result (can weights infer position?) is informative
everywhere. See [`hypothesis.md`](hypothesis.md).
