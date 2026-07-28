"""Transformer-LM backbone, switchable between real and complex weight fields and PE schemes.

One class covers all comparison conditions in docs/hypothesis.md:
    field in {real, complex}  x  pe in {rope, none, learned}
so the only thing that changes across an experiment is the config, not the code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import ComplexSelfAttention, RealSelfAttention
from .layers import ALGEBRAS, ComplexLinear, ComplexRMSNorm, ModReLU, crelu, lift_to_complex, to_real_concat

__all__ = ["ModelConfig", "TransformerLM"]


@dataclass
class ModelConfig:
    vocab_size: int
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: float = 4.0
    field: str = "real"        # real | complex
    pe: str = "rope"           # rope | none | learned
    causal: bool = True
    max_T: int = 512
    rope_base: float = 10000.0
    learn_imag: bool = False   # complex only: learn the imaginary part of token embeddings
    complex_act: str = "crelu"  # crelu | modrelu  (CReLU trains far better; modReLU is near-linear post-RMSNorm)
    algebra: str = "complex"   # complex only: complex | split | dual | diag (see complexnn.layers.ALGEBRAS)


# ----------------------------------------------------------------- real stack
class _RealMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _RealBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.dim)
        self.attn = RealSelfAttention(cfg.dim, cfg.n_heads, cfg.causal, cfg.pe, cfg.max_T, cfg.rope_base)
        self.n2 = nn.RMSNorm(cfg.dim)
        self.mlp = _RealMLP(cfg.dim, int(cfg.dim * cfg.mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


# ----------------------------------------------------------------- complex stack
class _ComplexMLP(nn.Module):
    def __init__(self, dim, hidden, act, algebra="complex"):
        super().__init__()
        self.fc1 = ComplexLinear(dim, hidden, bias=False, algebra=algebra)
        self.act = ModReLU(hidden) if act == "modrelu" else None
        self.fc2 = ComplexLinear(hidden, dim, bias=False, algebra=algebra)

    def forward(self, z):
        h = self.fc1(z)
        h = self.act(h) if self.act is not None else crelu(h)
        return self.fc2(h)


class _ComplexBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = ComplexRMSNorm(cfg.dim)
        self.attn = ComplexSelfAttention(cfg.dim, cfg.n_heads, cfg.causal, cfg.pe, cfg.max_T,
                                         cfg.rope_base, cfg.algebra)
        self.n2 = ComplexRMSNorm(cfg.dim)
        self.mlp = _ComplexMLP(cfg.dim, int(cfg.dim * cfg.mlp_ratio), cfg.complex_act, cfg.algebra)

    def forward(self, z):
        z = z + self.attn(self.n1(z))
        return z + self.mlp(self.n2(z))


# ----------------------------------------------------------------- model
class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.field in {"real", "complex"}
        assert cfg.pe in {"rope", "none", "learned"}
        assert cfg.algebra in ALGEBRAS
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Embedding(cfg.max_T, cfg.dim) if cfg.pe == "learned" else None
        self.tok_imag = nn.Embedding(cfg.vocab_size, cfg.dim) if (cfg.field == "complex" and cfg.learn_imag) else None

        if cfg.field == "real":
            self.blocks = nn.ModuleList(_RealBlock(cfg) for _ in range(cfg.n_layers))
            self.norm_f = nn.RMSNorm(cfg.dim)
            self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        else:
            self.blocks = nn.ModuleList(_ComplexBlock(cfg) for _ in range(cfg.n_layers))
            self.norm_f = ComplexRMSNorm(cfg.dim)
            self.head = nn.Linear(2 * cfg.dim, cfg.vocab_size, bias=False)  # reads [Re ‖ Im]

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        h = self.tok(idx)
        if self.pos is not None:
            h = h + self.pos(torch.arange(T, device=idx.device))[None]

        if self.cfg.field == "complex":
            imag = self.tok_imag(idx) if self.tok_imag is not None else torch.zeros_like(h)
            h = torch.complex(h, imag)
            for blk in self.blocks:
                h = blk(h)
            h = to_real_concat(self.norm_f(h))
        else:
            for blk in self.blocks:
                h = blk(h)
            h = self.norm_f(h)
        return self.head(h)

    def num_real_params(self) -> int:
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters())
