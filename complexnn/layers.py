"""Complex-valued building blocks.

Activations flow through the network as torch complex tensors (cfloat/cdouble), but every module
computes on explicit real/imag tensors internally. That keeps us off op-coverage gaps (fused kernels,
some norms) and off Pascal-specific complex-kernel edges, while staying autograd-correct. Each layer
here is gated by `tests/test_layers.py` (gradcheck + an independent R^2/complex-matmul reference).

Math reference: docs/background.md §5.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "ALGEBRAS",
    "complex_kaiming_",
    "ComplexLinear",
    "ComplexRMSNorm",
    "crelu",
    "ModReLU",
    "lift_to_complex",
    "to_real_concat",
    "count_real_params",
]

# The two-channel tyings we compare. Every one of them stores exactly two real matrices (A, B), so
# they all have identical parameter count and identical activation width; only the *pattern* by which
# those two matrices are shared across the 2d x 2d real map changes. This is the instrument for
# "is the effect C, or is it structured weight sharing?" (docs/hypothesis.md, H2_alg).
#
#   complex  i^2 = -1   [[A, -B], [B, A]]   rotation-like (the original construct)
#   split    j^2 = +1   [[A,  B], [B, A]]   reflection/boost-like (split-complex a.k.a. hyperbolic)
#   dual     e^2 =  0   [[A,  0], [B, A]]   nilpotent (the third and last 2D real unital algebra)
#   diag     ---        [[A,  0], [0, B]]   NOT an algebra: two independent real streams, no mixing
#
# complex/split/dual exhaust the 2D real unital associative algebras up to isomorphism, so the set is
# a complete classification plus one unstructured control.
ALGEBRAS = ("complex", "split", "dual", "diag")

# Init variance correction. With Var(A_ij) = Var(B_ij) = sigma^2, fan-in n, and inputs of per-channel
# variance v on both streams, the total output variance Var(out_r) + Var(out_i) is 4 n sigma^2 v for
# complex/split, 3 n sigma^2 v for dual (out_r = Ax only), and 2 n sigma^2 v for diag. We scale sigma
# so every algebra leaves the block with the same activation scale, otherwise the ablation would
# confound the tying pattern with a sqrt(2) init-scale difference.
#
# CAVEAT for interpreting `dual`. This equalises the *total* only. Measured per stream at init
# (input variance 1 on both), complex/split/diag come out balanced at 1.00/1.00 while dual is
# 0.67/1.33, because the nilpotent algebra has no term feeding y into out_r. That imbalance is a
# property of the algebra, not something an init can remove, so a dual result that differs from
# complex is confounded with stream imbalance in a way the split and diag results are not. Weight
# the split/diag arms accordingly when reading the ablation.
_ALGEBRA_VAR_SCALE = {"complex": 1.0, "split": 1.0, "dual": math.sqrt(4.0 / 3.0), "diag": math.sqrt(2.0)}


def complex_kaiming_(A: Tensor, B: Tensor, fan_in: int, gain: float = 1.0, algebra: str = "complex") -> None:
    """Trabelsi-style init: magnitude ~ Rayleigh(s), phase ~ U(-pi, pi).

    For Rayleigh magnitude with scale s, E[|w|^2] = 2 s^2. We set 2 s^2 = gain^2 / fan_in so the
    per-weight variance matches a real Kaiming target of gain^2 / fan_in (docs/background.md §5.5),
    then apply the per-algebra correction above so all tyings match complex's output scale.
    """
    with torch.no_grad():
        s = _ALGEBRA_VAR_SCALE[algebra] * gain / math.sqrt(2.0 * fan_in)
        u = torch.rand_like(A).clamp_min(1e-12)
        mag = s * torch.sqrt(-2.0 * torch.log(u))  # Rayleigh via inverse-CDF
        phase = (torch.rand_like(A) * 2.0 - 1.0) * math.pi
        A.copy_(mag * torch.cos(phase))
        B.copy_(mag * torch.sin(phase))


class ComplexLinear(nn.Module):
    """y = W z (+ b), with W built from two real matrices A, B under one of ALGEBRAS.

    Stored as two real matrices A, B (so a d_out x d_in layer has 2*d_out*d_in real DOF and standard
    real optimizers/gradcheck apply). For the default `complex` algebra, W = A + iB and the map is
    equivalent to the constrained real map [[A,-B],[B,A]]. The other algebras keep the carrier
    (two real streams, same width, same parameter count) and change only the mixing pattern.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, gain: float = 1.0,
                 algebra: str = "complex"):
        super().__init__()
        assert algebra in ALGEBRAS, f"algebra must be one of {ALGEBRAS}, got {algebra!r}"
        self.in_features = in_features
        self.out_features = out_features
        self.algebra = algebra
        self.A = nn.Parameter(torch.empty(out_features, in_features))
        self.B = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)
        complex_kaiming_(self.A, self.B, in_features, gain, algebra)

    def forward(self, z: Tensor) -> Tensor:
        x, y = z.real, z.imag
        if self.algebra == "complex":       # (A + iB)(x + iy)
            out_r = F.linear(x, self.A) - F.linear(y, self.B)
            out_i = F.linear(x, self.B) + F.linear(y, self.A)
        elif self.algebra == "split":       # (A + jB)(x + jy), j^2 = +1
            out_r = F.linear(x, self.A) + F.linear(y, self.B)
            out_i = F.linear(x, self.B) + F.linear(y, self.A)
        elif self.algebra == "dual":        # (A + eB)(x + ey), e^2 = 0
            out_r = F.linear(x, self.A)
            out_i = F.linear(x, self.B) + F.linear(y, self.A)
        else:                               # diag: two independent streams, no cross-mixing
            out_r = F.linear(x, self.A)
            out_i = F.linear(y, self.B)
        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i
        return torch.complex(out_r, out_i)

    @property
    def weight(self) -> Tensor:
        """The weight as A + iB (for inspection / phase-spectrum analysis).

        Meaningful as a complex number only for algebra='complex'; for the others it is just a
        convenient container for the same two real matrices.
        """
        return torch.complex(self.A, self.B)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias_r is not None}, algebra={self.algebra}")


class ComplexRMSNorm(nn.Module):
    """Magnitude RMS norm, phase-preserving: z <- z / sqrt(mean(|z|^2) + eps) * gamma.

    gamma is real (scales magnitude only), so the phase that carries positional signal is untouched.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, z: Tensor) -> Tensor:
        ms = (z.real**2 + z.imag**2).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(ms + self.eps) * self.weight
        return torch.complex(z.real * scale, z.imag * scale)


def crelu(z: Tensor) -> Tensor:
    """CReLU: ReLU on real and imaginary parts independently."""
    return torch.complex(F.relu(z.real), F.relu(z.imag))


class ModReLU(nn.Module):
    """modReLU: ReLU(|z| + b) * z/|z| -- thresholds magnitude, preserves phase. b is learnable per-dim.

    NOTE: b must be < 0 to be a nonlinearity. With b == 0, ReLU(|z|) == |z| and the op is the identity
    (it would silently make MLPs linear at init), so we init b to a small negative dead-zone.
    """

    def __init__(self, dim: int, eps: float = 1e-6, bias_init: float = -0.1):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), bias_init))
        self.eps = eps

    def forward(self, z: Tensor) -> Tensor:
        mag = torch.sqrt(z.real**2 + z.imag**2 + self.eps)
        scale = F.relu(mag + self.bias) / mag
        return torch.complex(z.real * scale, z.imag * scale)


def lift_to_complex(x: Tensor) -> Tensor:
    """Real -> complex with zero imaginary part."""
    return torch.complex(x, torch.zeros_like(x))


def to_real_concat(z: Tensor) -> Tensor:
    """Complex -> real readout by concatenating [Re ‖ Im] on the last dim (lossless)."""
    return torch.cat([z.real, z.imag], dim=-1)


def count_real_params(module: nn.Module) -> int:
    """Real degrees of freedom (a complex parameter counts as 2). Used for capacity-matching."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in module.parameters())
