"""Smoke + invariant tests for the TransformerLM across all comparison conditions."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from complexnn.models import ModelConfig, TransformerLM

COMBOS = list(itertools.product(["real", "complex"], ["rope", "none", "learned"]))


@pytest.mark.parametrize("field,pe", COMBOS)
def test_forward_backward_finite(field, pe):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=23, dim=32, n_layers=2, n_heads=4, field=field, pe=pe, max_T=64)
    model = TransformerLM(cfg)
    idx = torch.randint(0, 23, (3, 16))
    logits = model(idx)
    assert logits.shape == (3, 16, 23)
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits.reshape(-1, 23), idx.reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"


def test_complex_backbone_uses_more_real_dof_at_equal_dim():
    common = dict(vocab_size=23, dim=32, n_layers=2, n_heads=4, max_T=64, pe="none")
    r = TransformerLM(ModelConfig(field="real", **common)).num_real_params()
    c = TransformerLM(ModelConfig(field="complex", **common)).num_real_params()
    assert c > r, (r, c)  # complex linears carry ~2x real DOF -> capacity-matching must account for this


def test_complex_nope_learns_position_task():
    """Feasibility: complex + NoPE must be able to learn a position-dependent mapping.

    Uses the causally-solvable sep-format copy task (answer[j] = input[j], emitted after a SEP once all
    input is visible) and memorizes a fixed tiny set. We assert real learning happened (loss drops a lot)
    and the answer span is mostly correct -- robust to the exact achievable accuracy.
    """
    from experiments.data.tasks import TaskConfig, make_batch

    torch.manual_seed(0)
    task = TaskConfig(name="copy", n_symbols=8, train_min=5, train_max=5)
    cfg = ModelConfig(vocab_size=task.vocab_size, dim=48, n_layers=2, n_heads=4,
                      field="complex", pe="none", max_T=32)
    model = TransformerLM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    X, Y, M = make_batch(task, 16, 5, 5, generator=g)  # fixed batch -> memorization
    m = M.reshape(-1)

    def loss_now():
        return F.cross_entropy(model(X).reshape(-1, task.vocab_size)[m], Y.reshape(-1)[m])

    init = loss_now().item()
    for _ in range(400):
        opt.zero_grad()
        loss = loss_now()
        loss.backward()
        opt.step()
    acc = (((model(X).argmax(-1) == Y) & M).sum() / M.sum()).item()
    assert loss.item() < 0.5 * init, f"complex-NoPE did not learn: loss {init:.3f}->{loss.item():.3f}"
    assert acc > 0.8, f"complex-NoPE answer accuracy too low on copy memorization: {acc:.3f}"
