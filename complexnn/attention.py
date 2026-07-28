"""Attention + RoPE for real and complex fields.

Real attention is a standard causal MHA (optionally RoPE). Complex attention computes logits as the
real part of the Hermitian inner product Re(q k^H) -- which, if q/k carry position-dependent phase,
naturally yields cos((m-n)theta)-modulated scores (docs/background.md §5.4). We avoid fused SDPA so
complex tensors stay on supported ops.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .layers import ComplexLinear

__all__ = [
    "build_rope_real",
    "apply_rope_real",
    "build_rope_complex",
    "RealSelfAttention",
    "ComplexSelfAttention",
]


# ---------------------------------------------------------------- RoPE
def build_rope_real(T: int, head_dim: int, base: float = 10000.0, device=None, dtype=torch.float32):
    """Returns (cos, sin) of shape (T, head_dim) for the half-split rotate_half convention."""
    assert head_dim % 2 == 0, "RoPE needs an even head_dim"
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim
    inv_freq = 1.0 / (base**idx)  # (head_dim/2,)
    t = torch.arange(T, device=device, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_real(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, H, T, head_dim); cos/sin: (T, head_dim)."""
    return x * cos + _rotate_half(x) * sin


def build_rope_complex(T: int, head_dim_c: int, base: float = 10000.0, device=None, dtype=torch.complex64):
    """Returns complex rotation factors e^{i m theta_j}, shape (T, head_dim_c) -- one angle per
    complex channel. Multiplying a complex channel by this rotates it (the cleanest RoPE form)."""
    real_dtype = torch.float64
    idx = torch.arange(0, head_dim_c, device=device, dtype=real_dtype) / head_dim_c
    inv_freq = 1.0 / (base**idx)  # (head_dim_c,)
    t = torch.arange(T, device=device, dtype=real_dtype)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim_c)
    rot = torch.polar(torch.ones_like(freqs), freqs)  # complex128
    return rot.to(dtype)


def _causal_fill(logits: Tensor, T: int) -> Tensor:
    mask = torch.triu(torch.ones(T, T, device=logits.device, dtype=torch.bool), diagonal=1)
    return logits.masked_fill(mask, float("-inf"))


# ---------------------------------------------------------------- attention
class RealSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, causal=True, pe="rope", max_T=2048, rope_base=10000.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim, self.causal, self.pe = n_heads, dim // n_heads, causal, pe
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        if pe == "rope":
            cos, sin = build_rope_real(max_T, self.head_dim, rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, hd)
        if self.pe == "rope":
            q = apply_rope_real(q, self.rope_cos[:T], self.rope_sin[:T])
            k = apply_rope_real(k, self.rope_cos[:T], self.rope_sin[:T])
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.causal:
            logits = _causal_fill(logits, T)
        out = logits.softmax(dim=-1) @ v
        return self.proj(out.transpose(1, 2).reshape(B, T, D))

    @torch.no_grad()
    def attn_logits(self, x: Tensor) -> Tensor:
        """Pre-mask, pre-softmax logits (B, H, T_q, T_k) for the offset-kernel probe. Mirrors forward."""
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        if self.pe == "rope":
            q = apply_rope_real(q, self.rope_cos[:T], self.rope_sin[:T])
            k = apply_rope_real(k, self.rope_cos[:T], self.rope_sin[:T])
        return (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)


class ComplexSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, causal=True, pe="none", max_T=2048, rope_base=10000.0,
                 algebra="complex"):
        super().__init__()
        assert dim % n_heads == 0
        assert not (pe == "rope" and algebra != "complex"), \
            "the RoPE phasor multiply is only defined for algebra='complex'"
        self.n_heads, self.head_dim, self.causal, self.pe = n_heads, dim // n_heads, causal, pe
        self.q = ComplexLinear(dim, dim, bias=False, algebra=algebra)
        self.k = ComplexLinear(dim, dim, bias=False, algebra=algebra)
        self.v = ComplexLinear(dim, dim, bias=False, algebra=algebra)
        self.proj = ComplexLinear(dim, dim, bias=False, algebra=algebra)
        if pe == "rope":
            rot = build_rope_complex(max_T, self.head_dim, rope_base)
            self.register_buffer("rope_rot", rot, persistent=False)

    def _heads(self, c: Tensor) -> Tensor:
        B, T, _ = c.shape
        return c.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)

    def forward(self, z: Tensor) -> Tensor:
        B, T, D = z.shape
        q, k, v = self._heads(self.q(z)), self._heads(self.k(z)), self._heads(self.v(z))
        if self.pe == "rope":
            rot = self.rope_rot[:T].to(q.dtype)
            q, k = q * rot, k * rot
        # Re(q k^H) = sum_d (q.re k.re + q.im k.im)  -> real logits
        logits = (q.real @ k.real.transpose(-1, -2) + q.imag @ k.imag.transpose(-1, -2)) / math.sqrt(
            2 * self.head_dim
        )
        if self.causal:
            logits = _causal_fill(logits, T)
        attn = logits.softmax(dim=-1)  # real attention weights
        out = torch.complex(attn @ v.real, attn @ v.imag)  # (B, H, T, hd) complex
        return self.proj(out.transpose(1, 2).reshape(B, T, D))

    @torch.no_grad()
    def attn_logits(self, z: Tensor) -> Tensor:
        """Pre-mask, pre-softmax real logits (B, H, T_q, T_k) for the offset-kernel probe. Mirrors forward."""
        T = z.shape[1]
        q, k = self._heads(self.q(z)), self._heads(self.k(z))
        if self.pe == "rope":
            rot = self.rope_rot[:T].to(q.dtype)
            q, k = q * rot, k * rot
        return (q.real @ k.real.transpose(-1, -2) + q.imag @ k.imag.transpose(-1, -2)) / math.sqrt(
            2 * self.head_dim
        )
