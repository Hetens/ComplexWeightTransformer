"""Correctness gate for complexnn.layers: gradcheck + independent reference equivalence.

No layer is trusted until it (a) passes torch.autograd.gradcheck in double precision and (b) matches a
hand-written reference (complex-matmul or R^2) forward and backward. See docs/feasibility.md §2.
"""

import pytest
import torch
from torch.autograd import gradcheck

from complexnn.layers import (
    ALGEBRAS,
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


# ---------- weight-tying algebras (H2_alg) ----------

# The 2d x 2d real block matrix each algebra is supposed to realise, as [[top-left, top-right],
# [bottom-left, bottom-right]] acting on the stacked vector [x; y].
_BLOCKS = {
    "complex": lambda A, B: ((A, -B), (B, A)),
    "split":   lambda A, B: ((A, B), (B, A)),
    "dual":    lambda A, B: ((A, None), (B, A)),
    "diag":    lambda A, B: ((A, None), (None, B)),
}


@pytest.mark.parametrize("algebra", ALGEBRAS)
def test_algebra_matches_explicit_real_block_matrix(algebra):
    """Each tying must equal the stacked real map it claims to be. This is the whole ablation."""
    layer = ComplexLinear(5, 7, bias=False, algebra=algebra).double()
    z = torch.randn(3, 5, dtype=torch.complex128)
    out = layer(z)
    (tl, tr), (bl, br) = _BLOCKS[algebra](layer.A, layer.B)
    zero = torch.zeros(7, 5, dtype=torch.float64)
    top = torch.cat([tl if tl is not None else zero, tr if tr is not None else zero], dim=1)
    bot = torch.cat([bl if bl is not None else zero, br if br is not None else zero], dim=1)
    M = torch.cat([top, bot], dim=0)                      # (2*out, 2*in)
    stacked = torch.cat([z.real, z.imag], dim=-1) @ M.T   # (3, 2*out)
    ref = torch.complex(stacked[:, :7], stacked[:, 7:])
    assert torch.allclose(out, ref, atol=1e-10), (out - ref).abs().max()


@pytest.mark.parametrize("algebra", ALGEBRAS)
def test_algebra_gradcheck(algebra):
    layer = ComplexLinear(4, 6, bias=True, algebra=algebra).double()
    z = _cinput(2, 4)
    assert gradcheck(layer, (z,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("algebra", ALGEBRAS)
def test_algebra_param_count_is_identical(algebra):
    """The ablation is only fair if every tying costs exactly the same. It is the fairness gate."""
    assert count_real_params(ComplexLinear(8, 8, bias=False, algebra=algebra)) == 2 * 8 * 8


@pytest.mark.parametrize("algebra", ALGEBRAS)
def test_algebra_init_output_scale_matches(algebra):
    """The per-algebra init correction must equalise activation scale, else the ablation confounds
    the tying pattern with an init-scale difference (complexnn.layers._ALGEBRA_VAR_SCALE)."""
    torch.manual_seed(0)
    n = 512
    layer = ComplexLinear(n, n, bias=False, algebra=algebra).double()
    z = torch.complex(torch.randn(4096, n, dtype=torch.float64), torch.randn(4096, n, dtype=torch.float64))
    out = layer(z)
    total = (out.real.var() + out.imag.var()).item()
    assert 0.75 < total / 2.0 < 1.25, f"{algebra}: total output var {total:.3f}, want ~2"


def test_diag_needs_a_nonzero_imaginary_input():
    """Guard for the sweep: `diag` never mixes the streams, so a zero-initialised imaginary stream
    stays identically zero and the control silently degenerates to a real d-wide model. The sweep
    therefore forces learn_imag=True for algebra arms (experiments/sweep.py)."""
    layer = ComplexLinear(6, 6, bias=False, algebra="diag")
    z = lift_to_complex(torch.randn(2, 6))  # imaginary part exactly zero
    assert torch.all(layer(z).imag == 0)
    zz = torch.complex(torch.randn(2, 6), torch.randn(2, 6))
    assert torch.any(layer(zz).imag != 0)


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
