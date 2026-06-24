# Related work (annotated)

Short, working bibliography (the "later: long" version will expand annotations and verify every link).
Status flags matter here: several of the closest results are **withdrawn / preprint / workshop**, which is
what creates room for this project. A few arXiv ids are marked *(confirm)* — verify before citing formally.

---

## A. RoPE and its variants
- **RoFormer / RoPE** — Su et al., 2021. Position as rotation `e^{imθ}` of Q/K pairs; relative, zero added
  params. The method we try to internalize. <https://arxiv.org/abs/2104.09864>
- **2-D / axial RoPE for ViT** — Heo et al., 2024. RoPE for grids; beats absolute PE on resolution
  extrapolation. Relevant to "spatial" claims. <https://arxiv.org/abs/2403.13298>
- **Position Interpolation** — Chen et al., 2023. Extend context by rescaling positions.
  <https://arxiv.org/abs/2306.15595>
- **Base of RoPE Bounds Context Length** — NeurIPS 2024. The RoPE base frequency caps usable context.
  <https://arxiv.org/abs/2405.14591>
- **Fourier Position Embedding** — 2024. Periodic-extension view of attention for length generalization.
  <https://arxiv.org/abs/2412.17739>

## B. Benchmarks that test positional / length capability (our "base benchmarks")
- **RULER** — Hsieh et al., COLM 2024. Synthetic long-context suite; models that ace needle-in-haystack
  still fail harder positional tasks as length grows. <https://arxiv.org/abs/2404.06654>
- **The Impact of Positional Encoding on Length Generalization** — Kazemnejad et al., NeurIPS 2023.
  Systematic APE/T5-rel/ALiBi/RoPE/**NoPE** comparison on reasoning+math; NoPE often best.
  <https://arxiv.org/abs/2305.19466> · code: <https://github.com/McGill-NLP/length-generalization>
- **Length Generalization of Causal Transformers without PE** — 2024. NoPE's limits + temperature tuning.
  <https://arxiv.org/abs/2404.12224>

## C. NoPE — implicit position (theoretical backbone)
- **Transformers without Positional Encodings** — Haviv et al., EMNLP 2022 findings. Causal LMs learn
  implicit absolute position; can beat sinusoidal on PPL. <https://arxiv.org/abs/2203.16634>
- (Kazemnejad 2023, §B, also a primary NoPE reference.)

## D. Complex positional encodings & complex transformers (closest prior art)
- **Encoding Word Order in Complex Embeddings** — Wang et al., **ICLR 2020**. Peer-reviewed; complex-valued
  *positional embeddings* (continuous functions of position), **not** complex weights. *(arXiv id confirm)*
- **ComplexFormer** — 2025. Head-specific complex-vector attention combining "semantic + positional angles";
  reported 34.2 gen-PPL on WikiText-103. **STATUS: WITHDRAWN** (authors: experiment incomplete) → a
  validation opportunity. <https://arxiv.org/abs/2505.10222>
- **CoPE: A Lightweight Complex Positional Encoding** — Amballa, 2025. real=content / imag=position;
  phase-aware first layer; beats RoPE/sinusoidal/learned on GLUE. **STATUS: workshop / OpenReview** (UMass;
  advised by McCallum). <https://arxiv.org/abs/2508.18308>
- **CRoPE: Efficient Parametrization of RoPE** — 2026. Complex QK projection; claims **~50% attention-block
  param saving**; frames RoPE as "reserving half the embedding space for position." **STATUS: unpublished
  preprint** → validate/refute its central claim. <https://arxiv.org/abs/2601.02728>
- **RoPE as Phase Modulation** — 2026. Theoretical bounds on the RoPE base for long context.
  <https://arxiv.org/abs/2602.10959>
- **Complex-Valued Relative Positional Encodings for Transformer** — IEEE (peer-reviewed, secondary venue).
  <https://ieeexplore.ieee.org/abstract/document/10105716>

## E. Complex-valued network theory & training
- **Deep Complex Networks** — Trabelsi et al., ICLR 2018. Foundational complex conv/BN/init/activations;
  we reuse the init and activation patterns. *(arXiv id confirm)*
- **Complex-Valued Neural Networks — Theory and Analysis** — 2023. Wirtinger calculus, activation/loss
  non-holomorphy, optimization. <https://arxiv.org/abs/2312.06087>
- **CauchyNet** — 2025. Holomorphic-activation, data-efficient CVNN. <https://arxiv.org/abs/2510.10195>

## F. Embedding / PE compression frontier (the "branch out" — full survey in `embedding-compression-frontier.md`)
- **Efficient Compression of the Embedding Layer in LLMs** — 2023. <https://arxiv.org/abs/2307.00526>
- **Weight Tying Biases Token Embeddings Toward the Output Space** — 2026. <https://arxiv.org/abs/2603.26663>
- **Scaling Embedding Layers in Language Models** — 2025. <https://arxiv.org/abs/2502.01637>
- **TensorSLM** (tensor-train embedding compression) — 2025. <https://arxiv.org/abs/2506.13514>
- **CARVQ** (residual VQ embedding compression) — 2025. <https://arxiv.org/abs/2510.12721>
- **Scaling Embeddings Outperforms Scaling Experts** — 2026. <https://arxiv.org/abs/2601.21204>

---

### How this project is positioned
Group D shows the *mechanism* (complex/phase ≈ position) but the strongest empirical claims are
withdrawn (ComplexFormer), preprint (CRoPE), or workshop (CoPE); peer-reviewed work (Wang ICLR 2020, IEEE)
makes the *positional encoding* complex, not the *weights*. We test **complex weights + NoPE**,
**capability-first**, on Group B benchmarks, leaning on Group C as the theoretical license — and we can
independently **validate CRoPE's 50% claim** and revisit ComplexFormer's withdrawn one.
