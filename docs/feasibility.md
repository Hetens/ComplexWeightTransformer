# Feasibility analysis

Can this actually be built and trained on the available hardware, and is it worth doing? Background:
[`background.md`](background.md). Verdict up front:

| Axis | Verdict | One-line reason |
|---|---|---|
| Mathematical | ✅ strong | complex linear layer = phase-equivariant real map; superset of RoPE rotation |
| Training | ✅ tractable | PyTorch complex autograd works (verified); guard with `gradcheck` vs `ℝ²` reference |
| Compute | ✅ fits | tiny models on GTX 1050 for smoke; A100 for real runs; complex ≈ 2× cost |
| Novelty | ⚠️ moderate | mechanism shown but in withdrawn/preprint/workshop work; weights+NoPE angle is thin |
| Null-result risk | ⚠️ acceptable | gains may not transfer to NoPE; design isolates this cleanly |

---

## 1. Mathematical feasibility — strong
A complex `d×d` layer is exactly the real `2d×2d` maps of the form `[[A,−B],[B,A]]` (commute with mult-by-`i`).
This set **contains** RoPE's block rotations as a special case, so the model has the *capacity* to express
position-dependent phase. Capacity ≠ realization (the NoPE caveat in `background.md` §5.1) — which is the
empirical question, not a feasibility blocker.

## 2. Training feasibility — tractable, with named pitfalls

**Verified locally (`scripts/check_env.py`):** `torch==2.6.0+cu124`, complex autograd yields finite
`complex64` grads; CUDA runs on the GTX 1050 (sm_61) via sm_60 cubin + PTX-JIT.

PyTorch complex specifics to respect:
- **Autograd convention:** for a real-valued loss, `.grad` holds the conjugate Wirtinger derivative
  `∂L/∂z̄`, defined so `z −= lr·z.grad` is correct descent ⇒ **Adam/SGD work unchanged** on complex params.
- **Op coverage gaps:** not every op supports complex (e.g., some norms, `F.scaled_dot_product_attention`
  fused kernels, certain reductions). **Mitigation:** implement layers on **explicit `(real, imag)` real
  tensors** where coverage is uncertain (also avoids surprises on Pascal), keeping native `cfloat` only
  where it's clean.
- **Non-holomorphic activations** (CReLU/modReLU): fine under `ℝ²` autograd; just don't expect a single
  complex derivative.
- **Numerical:** `modReLU`/magnitude norms divide by `|z|` → add `ε`; watch `complex64` precision.

**Correctness gate (M2):** every complex layer must (a) pass `torch.autograd.gradcheck` (double precision),
and (b) match an independent **`ℝ²` reference** (the same layer written with two real matrices `A,B`) to
tolerance, forward and backward. No layer is trusted until both pass.

**Empirical (M3):** layers train fine, but **complex+NoPE optimization is bimodal/initialization-sensitive**
— most seeds collapse, some converge to competitive accuracy (`docs/results.md`). `complex+RoPE` is stable.
modReLU is near-linear after RMSNorm (use CReLU); learnable imaginary embeddings raise complex+NoPE's
convergence rate (~25%→75%) but don't guarantee it. Conclusion: feasible to *build* and *train*, but the
NoPE candidate needs many seeds + stabilization to evaluate fairly.

## 3. Compute feasibility — fits the plan
- **Complex ≈ 2× memory and ≈ up to 4× MACs** of a same-*shaped* real layer (complex multiply = 4 real
  mults / 3 with Karatsuba). Budget accordingly; this is why capacity-matching is defined in real-DOF.
- **GTX 1050 (4 GB):** smoke tests + smallest synthetic runs only (≤ ~1–2 M params, short seq, small batch).
  Expect a one-time PTX-JIT warmup on first CUDA kernel.
- **Unity A100 (32 GB, 100 GB disk):** all real runs via `scripts/unity_train.sbatch` (SLURM). A100 is
  sm_80 → native in the wheel's arch list; one config switches local↔A100.

## 4. Novelty & prior-art risk — moderate (positioned honestly)
The mechanism (complex/phase attention ≈ position) is demonstrated, but the strongest claims are **not**
solidly published:
- **ComplexFormer** (arXiv:2505.10222) — **withdrawn** ("experiment incomplete"). → validation opportunity.
- **CRoPE** (arXiv:2601.02728) — **unpublished preprint**; ~50% attention-param-saving claim → validate/refute.
- **CoPE** (arXiv:2508.18308) — **workshop** paper (UMass; Amballa/McCallum); complex *PE*, not complex weights.
- Peer-reviewed: Wang et al. **ICLR 2020** (complex *embeddings*), an IEEE complex relative-PE paper.

**Our wedge:** complex **weights** + **NoPE**, evaluated **capability-first** on base positional benchmarks,
with a measured compression Pareto — thin in peer-reviewed venues and partly resting on testable
withdrawn/preprint claims. See [`related-work.md`](related-work.md).

## 5. Risks & mitigations
| Risk | Mitigation |
|---|---|
| PyTorch complex op gaps / wrong grads | explicit `(real,imag)` impl + `gradcheck` + `ℝ²` reference |
| Complex training instability | Trabelsi init, modReLU/CReLU, ComplexRMSNorm, lower LR, grad clipping |
| Null result (no capability gain) | pre-registered + informative; phase plots explain *why* (`hypothesis.md`) |
| 4 GB local ceiling | local = smoke only; real runs on A100 |
| Unconfirmed Unity SLURM/module names | `unity_train.sbatch` flagged "confirm with sinfo / module avail" |

**Overall: feasible.** No blocker is fundamental; the open question is scientific (does capability emerge),
which is exactly what the experiments are built to answer.
