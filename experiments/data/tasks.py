"""Synthetic positional benchmarks in decoder-only format: [input] [SEP] [answer].

All input tokens precede the answer, so under causal masking every answer token *can* be produced
(unlike an in-place reverse, where early positions are mathematically unsolvable -- see docs/results.md).
Loss/accuracy are computed only on answer positions. Length generalization = train short, eval long.

Tasks:
  copy / reverse / sort  -- algorithmic; answer length == input length.
  revskip / skipcopy     -- algorithmic, stride 2; answer length == ceil(input length / 2).
  needle                 -- passkey retrieval; a MARK token precedes a secret symbol placed at a random
                            depth in a haystack of fillers; answer is that one secret. Tests whether
                            position survives distance (RULER/passkey lineage). "length" = haystack size.

Vocab: 0=PAD, 1=SEP, 2=MARK, content symbols start at 3.

## The retrieval-offset law (why these five algorithmic tasks)

With the layout [x_0..x_{L-1}] SEP [ans_0..ans_{A-1}] and the LM shift, answer token `ans_j` is
predicted at input index `L + j`. So the attention offset a model must realise to fetch `ans_j` from
its source index `src(j)` is `offset(j) = src(j) - (L + j)`:

  | task     | src(j)     | offset(j) | anchored at | length-dependent? |
  |----------|------------|-----------|-------------|-------------------|
  | copy     | j          | -L        | start       | YES               |
  | skipcopy | 2j         | j - L     | start       | YES               |
  | reverse  | L-1-j      | -(2j+1)   | end         | no                |
  | revskip  | L-1-2j     | -(3j+1)   | end         | no                |
  | sort     | content    | n/a       | n/a         | n/a               |

End-anchored tasks have an offset that is affine in the answer index `j` with *no dependence on L*, so
they are solvable by a counter started at SEP. Start-anchored tasks need the model to measure `L`.
`revskip` and `skipcopy` are the matched pair: same stride, same answer length, differing only in the
anchor, hence only in length-dependence. See docs/hypothesis.md (H3_offset).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PAD, SEP, MARK = 0, 1, 2
N_SPECIAL = 3
ALGORITHMIC = ("copy", "reverse", "sort", "revskip", "skipcopy")
TASKS = ALGORITHMIC + ("needle",)


@dataclass
class TaskConfig:
    name: str = "reverse"
    n_symbols: int = 16
    train_min: int = 4
    train_max: int = 12
    eval_lengths: tuple[int, ...] = (12, 24, 48)  # eval at train_max and beyond (extrapolation)

    @property
    def vocab_size(self) -> int:
        return N_SPECIAL + self.n_symbols


def _answer(name: str, x: Tensor) -> Tensor:
    """The target sequence for one algorithmic example. Answer length need not equal len(x)."""
    if name == "copy":
        return x.clone()
    if name == "reverse":
        return torch.flip(x, dims=[0])
    if name == "sort":
        return torch.sort(x).values
    if name == "revskip":      # end-anchored stride 2: x[L-1], x[L-3], ...   offset -(3j+1)
        return torch.flip(x, dims=[0])[::2].clone()
    if name == "skipcopy":     # start-anchored stride 2: x[0], x[2], ...     offset j-L
        return x[::2].clone()
    raise ValueError(f"unknown algorithmic task: {name}")


def _example(cfg: TaskConfig, L: int, gen: torch.Generator | None) -> tuple[Tensor, int]:
    """Return (full_sequence, answer_len) for one example of length-parameter L."""
    lo, hi = N_SPECIAL, N_SPECIAL + cfg.n_symbols
    if cfg.name == "needle":
        fillers = torch.randint(lo, hi, (L,), generator=gen)
        d = int(torch.randint(0, L, (1,), generator=gen))  # depth of the needle
        secret = fillers[d].view(1)
        inp = torch.cat([fillers[:d], torch.tensor([MARK]), fillers[d:]])  # MARK marks the secret
        seq = torch.cat([inp, torch.tensor([SEP]), secret])
        return seq, 1
    x = torch.randint(lo, hi, (L,), generator=gen)
    ans = _answer(cfg.name, x)
    seq = torch.cat([x, torch.tensor([SEP]), ans])
    return seq, int(ans.numel())


def make_batch(
    cfg: TaskConfig,
    batch_size: int,
    min_len: int,
    max_len: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (x_in, y, mask), each (B, S). x_in/y are the LM shift; mask marks answer targets."""
    lens = torch.randint(min_len, max_len + 1, (batch_size,), generator=generator)
    examples = [_example(cfg, int(lens[b]), generator) for b in range(batch_size)]

    S = max(seq.numel() for seq, _ in examples) - 1  # length after the LM shift
    X = torch.full((batch_size, S), PAD, dtype=torch.long)
    Y = torch.full((batch_size, S), PAD, dtype=torch.long)
    M = torch.zeros((batch_size, S), dtype=torch.bool)
    for b, (seq, ans_len) in enumerate(examples):
        n = seq.numel()
        X[b, : n - 1] = seq[:-1]
        Y[b, : n - 1] = seq[1:]
        M[b, n - 1 - ans_len : n - 1] = True  # last `ans_len` targets are the answer
    return X.to(device), Y.to(device), M.to(device)


@torch.no_grad()
def evaluate(model, cfg: TaskConfig, length: int, n_seqs: int, device, batch: int = 256) -> dict:
    """Token accuracy and exact-sequence accuracy on answer positions at a fixed length.

    For `needle`, answer length is 1, so token_acc == seq_acc == retrieval accuracy at haystack `length`.
    """
    model.eval()
    tok_correct = tok_total = seq_correct = seq_total = 0
    done = 0
    while done < n_seqs:
        b = min(batch, n_seqs - done)
        X, Y, M = make_batch(cfg, b, length, length, device=device)
        pred = model(X).argmax(-1)
        hit = (pred == Y) & M
        tok_correct += hit.sum().item()
        tok_total += M.sum().item()
        seq_correct += (hit.sum(dim=1) == M.sum(dim=1)).sum().item()
        seq_total += b
        done += b
    model.train()
    return {"length": length, "token_acc": tok_correct / max(tok_total, 1), "seq_acc": seq_correct / max(seq_total, 1)}
