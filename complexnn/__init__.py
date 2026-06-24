"""complexnn — complex-valued transformer building blocks and real-valued PE baselines.

See ~/.claude/plans for the research plan and docs/ for the knowledge base.
"""

from .attention import ComplexSelfAttention, RealSelfAttention, apply_rope_real, build_rope_complex, build_rope_real
from .layers import (
    ComplexLinear,
    ComplexRMSNorm,
    ModReLU,
    complex_kaiming_,
    count_real_params,
    crelu,
    lift_to_complex,
    to_real_concat,
)
from .models import ModelConfig, TransformerLM

__version__ = "0.1.0"

__all__ = [
    "ComplexLinear",
    "ComplexRMSNorm",
    "ModReLU",
    "crelu",
    "complex_kaiming_",
    "lift_to_complex",
    "to_real_concat",
    "count_real_params",
    "RealSelfAttention",
    "ComplexSelfAttention",
    "build_rope_real",
    "apply_rope_real",
    "build_rope_complex",
    "ModelConfig",
    "TransformerLM",
]
