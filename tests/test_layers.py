"""Correctness gate for complexnn.layers: gradcheck + independent reference equivalence.

No layer is trusted until it (a) passes torch.autograd.gradcheck in double precision and (b) matches a
hand-written reference (complex-matmul or R^2) forward and backward. See docs/feasibility.md §2.
"""

import torch
from torch.autograd import gradcheck

from complexnn.layers import (
    ComplexLinear,
    ComplexRMSNorm,
    ModReLU,
    count_real_params,
    crelu,
    lift_to_complex,
    to_real_concat,
)

torch.manual_seed(0)


def _cinput(*shape):
    return torch.randn(*shape, dtype=torch.complex128, requires_grad=True)


# ---------- ComplexLinear ----------

def test_complex_linear_matches_complex_matmul():
    layer = ComplexLinear(5, 7, bias=True).double()
    z = torch.randn(3, 5, dtype=torch.complex128)
    out = layer(z)
    W = torch.complex(layer.A, layer.B)
    b = torch.complex(layer.bias_r, layer.bias_i)
    ref = z @ W.T + b
    assert torch.allclose(out, ref, atol=1e-10), (out - ref).abs().max()


def test_complex_linear_gradcheck():
    layer = ComplexLinear(4, 6, bias=True).double()
    z = _cinput(2, 4)
    assert gradcheck(layer, (z,), eps=1e-6, atol=1e-5)


def test_complex_linear_param_count():
    layer = ComplexLinear(8, 8, bias=False)
    # A and B are real (8x8 each) -> 128 real DOF == 2 * 8 * 8.
    assert count_real_params(layer) == 2 * 8 * 8


# ---------- ComplexRMSNorm ----------

def test_complex_rmsnorm_normalizes_magnitude():
    norm = ComplexRMSNorm(16).double()
    z = torch.randn(4, 16, dtype=torch.complex128) * 3.7  # arbitrary scale
    out = norm(z)
    ms = (out.real**2 + out.imag**2).mean(dim=-1)  # weight==1 -> mean |z|^2 should be ~1
    assert torch.allclose(ms, torch.ones_like(ms), atol=1e-6), ms


def test_complex_rmsnorm_preserves_phase():
    norm = ComplexRMSNorm(16).double()
    z = torch.randn(4, 16, dtype=torch.complex128)
    out = norm(z)
    # scaling by a positive real does not change phase
    assert torch.allclose(out.angle(), z.angle(), atol=1e-6)


def test_complex_rmsnorm_gradcheck():
    norm = ComplexRMSNorm(6).double()
    z = _cinput(3, 6)
    assert gradcheck(norm, (z,), eps=1e-6, atol=1e-5)


# ---------- activations ----------

def test_crelu_gradcheck():
    z = _cinput(3, 5)
    assert gradcheck(crelu, (z,), eps=1e-6, atol=1e-5)


def test_modrelu_gradcheck():
    act = ModReLU(5).double()
    # bias slightly negative so the ReLU threshold is active (tests the kink region away from 0)
    with torch.no_grad():
        act.bias.uniform_(-0.5, 0.5)
    z = _cinput(3, 5)
    assert gradcheck(act, (z,), eps=1e-6, atol=1e-5)


# ---------- readout helpers ----------

def test_lift_and_readout_roundtrip():
    x = torch.randn(2, 4)
    z = lift_to_complex(x)
    assert torch.allclose(z.real, x) and torch.allclose(z.imag, torch.zeros_like(x))
    r = to_real_concat(torch.complex(x, 2 * x))
    assert r.shape == (2, 8)
    assert torch.allclose(r[:, :4], x) and torch.allclose(r[:, 4:], 2 * x)
