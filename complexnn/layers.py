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
    "complex_kaiming_",
    "ComplexLinear",
    "ComplexRMSNorm",
    "crelu",
    "ModReLU",
    "lift_to_complex",
    "to_real_concat",
    "count_real_params",
]


def complex_kaiming_(A: Tensor, B: Tensor, fan_in: int, gain: float = 1.0) -> None:
    """Trabelsi-style init: magnitude ~ Rayleigh(s), phase ~ U(-pi, pi).

    For Rayleigh magnitude with scale s, E[|w|^2] = 2 s^2. We set 2 s^2 = gain^2 / fan_in so the
    per-weight variance matches a real Kaiming target of gain^2 / fan_in (docs/background.md §5.5).
    """
    with torch.no_grad():
        s = gain / math.sqrt(2.0 * fan_in)
        u = torch.rand_like(A).clamp_min(1e-12)
        mag = s * torch.sqrt(-2.0 * torch.log(u))  # Rayleigh via inverse-CDF
        phase = (torch.rand_like(A) * 2.0 - 1.0) * math.pi
        A.copy_(mag * torch.cos(phase))
        B.copy_(mag * torch.sin(phase))


class ComplexLinear(nn.Module):
    """y = W z (+ b), with W = A + iB.

    Stored as two real matrices A, B (so a complex d_out x d_in layer has 2*d_out*d_in real DOF and
    standard real optimizers/gradcheck apply). Equivalent to the constrained real map [[A,-B],[B,A]].
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, gain: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = nn.Parameter(torch.empty(out_features, in_features))
        self.B = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_i", None)
        complex_kaiming_(self.A, self.B, in_features, gain)

    def forward(self, z: Tensor) -> Tensor:
        x, y = z.real, z.imag
        out_r = F.linear(x, self.A) - F.linear(y, self.B)
        out_i = F.linear(x, self.B) + F.linear(y, self.A)
        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i
        return torch.complex(out_r, out_i)

    @property
    def weight(self) -> Tensor:
        """The complex weight A + iB (for inspection / phase-spectrum analysis)."""
        return torch.complex(self.A, self.B)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias_r is not None}"


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
