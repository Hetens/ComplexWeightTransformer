"""Correctness gate for the synthetic tasks, and the retrieval-offset law they are built to vary.

The offset law is the pre-registered content of H3_offset (docs/hypothesis.md): a task is solvable by
a counter started at SEP iff the attention offset a model must realise is affine in the answer index
with no dependence on the input length. `revskip` and `skipcopy` are the matched pair that isolates
exactly that property, so the law is asserted here rather than trusted.
"""

import pytest
import torch

from experiments.data.tasks import ALGORITHMIC, MARK, SEP, TaskConfig, _example, make_batch

# src(j): where answer token j must be fetched from, as a function of the answer index and length.
SRC = {
    "copy":     lambda j, L: j,
    "skipcopy": lambda j, L: 2 * j,
    "reverse":  lambda j, L: L - 1 - j,
    "revskip":  lambda j, L: L - 1 - 2 * j,
}
# offset(j) = src(j) - (L + j), as documented in experiments/data/tasks.py.
OFFSET = {
    "copy":     lambda j, L: -L,
    "skipcopy": lambda j, L: j - L,
    "reverse":  lambda j, L: -(2 * j + 1),
    "revskip":  lambda j, L: -(3 * j + 1),
}
LENGTH_INDEPENDENT = {"reverse", "revskip"}


@pytest.mark.parametrize("name", sorted(SRC))
@pytest.mark.parametrize("L", [5, 6, 9, 16])
def test_retrieval_offset_law(name, L):
    """Every answer token comes from the documented source index, at the documented offset."""
    gen = torch.Generator().manual_seed(0)
    seq, alen = _example(TaskConfig(name), L, gen)
    x, ans = seq[:L], seq[L + 1:]
    assert seq[L] == SEP
    assert ans.numel() == alen
    for j in range(alen):
        src = SRC[name](j, L)
        assert 0 <= src < L
        assert int(x[src]) == int(ans[j]), f"{name} L={L} j={j}: wrong source"
        # answer token j is predicted at input index L + j after the LM shift
        assert src - (L + j) == OFFSET[name](j, L), f"{name} L={L} j={j}: wrong offset"


@pytest.mark.parametrize("name", sorted(SRC))
def test_length_independence_matches_classification(name):
    """The offset sequence is invariant to L exactly for the end-anchored tasks."""
    offs = {L: [OFFSET[name](j, L) for j in range(4)] for L in (8, 16, 32)}
    same = offs[8] == offs[16] == offs[32]
    assert same == (name in LENGTH_INDEPENDENT), f"{name}: length-independence {same}"


def test_revskip_and_skipcopy_are_a_matched_pair():
    """Same stride, same answer length, differing only in anchor. If they differed in anything else
    the comparison could not attribute a gap to length-dependence."""
    gen = torch.Generator().manual_seed(0)
    for L in (6, 7, 15, 16):
        a, alen_a = _example(TaskConfig("revskip"), L, gen)
        b, alen_b = _example(TaskConfig("skipcopy"), L, gen)
        assert alen_a == alen_b == -(-L // 2)   # ceil(L/2)
        assert a.numel() == b.numel() == L + 1 + alen_a


@pytest.mark.parametrize("name", ALGORITHMIC + ("needle",))
def test_make_batch_masks_only_answer_positions(name):
    """Loss/accuracy must be scored on answer tokens only, never on the input or the SEP."""
    gen = torch.Generator().manual_seed(0)
    X, Y, M = make_batch(TaskConfig(name), 8, 6, 12, generator=gen)
    assert M.any(dim=1).all(), "every example must contribute at least one scored token"
    for b in range(X.shape[0]):
        idx = M[b].nonzero().flatten()
        assert (idx.diff() == 1).all(), "answer positions must be contiguous"
        # the scored block ends the example: nothing after it is a target
        assert not M[b, idx[-1] + 1:].any()
        # SEP is consumed as input just before the first scored position, never scored itself
        assert (Y[b][M[b]] != SEP).all()


def test_needle_answer_is_the_marked_symbol():
    gen = torch.Generator().manual_seed(0)
    for L in (8, 32):
        seq, alen = _example(TaskConfig("needle"), L, gen)
        assert alen == 1
        d = int((seq == MARK).nonzero().flatten()[0])
        assert int(seq[-1]) == int(seq[d + 1]), "answer must be the symbol right after MARK"
