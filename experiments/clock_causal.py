"""Round-three probe of the positional clock: is it used, and is it really fractional?

`experiments.activation_probe` found that every NoPE model carries a positional rotation in its
residual stream whose total sweep stays constant as the sequence grows, and read that as a
*fractional* code. Two things were left open, and this module tests both on the reverse checkpoints.

  1. `ablation`  Is the clock causally used? At every block input after the first, remove the
     position-dependent mean inside a 2-D plane, h(m) <- h(m) - P (hbar(m) - mu), and re-score.
     The clock plane is PCs 1-2 of the position-conditional means hbar(m); the control is PCs 3-4 of
     the same means (same rank, same construction). A model that keeps position in the clock should
     collapse under the first edit and survive the second. `mirror` additionally overwrites each
     input position's clock coordinates with those of position L-1-m.

  2. `segments`  Is the code fractional on both sides of SEP? Under a causal mask the residual at
     input position m sees tokens 0..m only, so it cannot know L, and hbar(m) for m < L must be the
     same at every evaluation length (checked: `causal_rel_diff` against a reseeded noise floor).
     An input-side code therefore cannot be fractional. The probe in activation_probe fits one line
     over the whole sequence in a plane refitted at each length, which mixes the two segments. Here
     each segment is fitted separately, in one plane fixed at the training length, so a saturating
     input code (R^2 falling with L) and a fractional answer code (R^2 holding, sweep constant) can
     be told apart.

  3. `error_profile`  Exact-position accuracy binned by answer index. reverse answer j copies input
     position L-1-j, so early answer tokens read the input positions furthest outside training.

Usage:
    uv run python -m experiments.clock_causal --match "clock_reverse_*" "long_reverse_*"
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from complexnn.models import ModelConfig, TransformerLM
from experiments.data.tasks import TaskConfig, make_batch


@dataclass
class ClockConfig:
    results_dir: str = "results"
    match: tuple[str, ...] = ("clock_reverse_*", "long_reverse_*")
    out: str = "results/_figs/clock/clock_metrics.json"
    abl_lengths: tuple[int, ...] = (16, 24)
    seg_lengths: tuple[int, ...] = (16, 24, 32, 48, 64)
    prof_lengths: tuple[int, ...] = (24, 32)
    n_seqs: int = 1024
    device: str = "auto"


def load_model(run_dir: Path, device: str) -> tuple[TransformerLM, TaskConfig, dict] | None:
    if not (run_dir / "model.pt").exists():
        return None
    d = json.loads((run_dir / "result.json").read_text())
    c = d["config"]
    task = TaskConfig(c["task"], c["n_symbols"], c["train_min"], c["train_max"], tuple(c["eval_lengths"]))
    # algebra is passed explicitly: phase_probe._model_config predates the tying arms and would
    # silently rebuild a block-diagonal checkpoint as a complex one.
    model = TransformerLM(ModelConfig(
        vocab_size=task.vocab_size, dim=c["dim"], n_layers=c["n_layers"], n_heads=c["n_heads"],
        mlp_ratio=c["mlp_ratio"], field=c["field"], pe=c["pe"], max_T=c["max_T"],
        rope_base=c["rope_base"], learn_imag=c["learn_imag"], complex_act=c["complex_act"],
        algebra=c.get("algebra") or "complex",
    ))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    return model.to(device).eval(), task, d


# --------------------------------------------------------------------------- residual helpers
def _to_r(z: torch.Tensor) -> torch.Tensor:
    return torch.cat([z.real, z.imag], -1) if z.is_complex() else z


def _from_r(x: torch.Tensor, cplx: bool) -> torch.Tensor:
    if not cplx:
        return x
    d = x.shape[-1] // 2
    return torch.complex(x[..., :d].contiguous(), x[..., d:].contiguous())


@torch.no_grad()
def block_means(model: TransformerLM, X: torch.Tensor) -> list[torch.Tensor]:
    """Position-conditional mean residual at every block input, in real [Re || Im] form: (T, W)."""
    out: list[torch.Tensor] = []
    hooks = [b.register_forward_pre_hook(lambda _m, a: out.append(_to_r(a[0]).mean(0)))
             for b in model.blocks]
    try:
        model(X)
    finally:
        for h in hooks:
            h.remove()
    return out


def _plane(H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(mu, PCs 1-2, PCs 3-4) of the rows of H."""
    mu = H.mean(0)
    _, _, Vt = torch.linalg.svd(H - mu, full_matrices=False)
    return mu, Vt[:2].T, Vt[2:4].T


def _line(P: np.ndarray) -> tuple[float, float]:
    """Slope (rad/position) and R^2 of the unwrapped in-plane angle against position."""
    ang = np.unwrap(np.arctan2(P[:, 1], P[:, 0]))
    m = np.arange(len(ang), dtype=float)
    A = np.stack([m, np.ones_like(m)], 1)
    coef, *_ = np.linalg.lstsq(A, ang, rcond=None)
    r = ang - A @ coef
    ss = float(((ang - ang.mean()) ** 2).sum())
    return float(coef[0]), 1.0 - float((r**2).sum()) / max(ss, 1e-12)


# --------------------------------------------------------------------------- 1. ablation
@torch.no_grad()
def _score(model, X, Y, M, L, edits) -> dict:
    cplx = model.cfg.field == "complex"
    hooks = []
    for b, fn in edits.items():
        hooks.append(model.blocks[b].register_forward_pre_hook(
            lambda _m, a, fn=fn: (_from_r(fn(_to_r(a[0])), cplx),)))
    try:
        pred = model(X).argmax(-1)
    finally:
        for h in hooks:
            h.remove()
    tok = ((pred == Y) & M).sum().item() / M.sum().item()
    seq = ((pred == Y) | ~M).all(1).float().mean().item()
    ans = pred[M].view(X.shape[0], -1)[:, :L]
    inp = X[:, :L]
    return {"token_acc": round(tok, 4), "seq_acc": round(seq, 4),
            "copy_match": round((ans == inp).float().mean().item(), 4)}


def ablation(model, task, L: int, n: int, device) -> dict:
    g = torch.Generator().manual_seed(1)
    X, Y, M = make_batch(task, n, L, L, device=device, generator=g)
    means = block_means(model, X)
    blocks = range(1, len(model.blocks))   # block 0's input is the embedding: no position yet

    def remove(b, which):
        H = means[b]
        mu, clock, ctrl = _plane(H)
        B = clock if which == "clock" else ctrl
        proj = (H - mu) @ B @ B.T
        return lambda x: x - proj[None]

    def mirror(b, which):
        H = means[b]
        _, clock, ctrl = _plane(H)
        B = clock if which == "clock" else ctrl
        idx = torch.arange(H.shape[0], device=H.device)
        idx[:L] = torch.arange(L - 1, -1, -1, device=H.device)
        delta = (H[idx] - H) @ B @ B.T
        return lambda x: x + delta[None]

    return {
        "baseline": _score(model, X, Y, M, L, {}),
        "remove_clock": _score(model, X, Y, M, L, {b: remove(b, "clock") for b in blocks}),
        "remove_control": _score(model, X, Y, M, L, {b: remove(b, "control") for b in blocks}),
        "mirror_clock": _score(model, X, Y, M, L, {b: mirror(b, "clock") for b in blocks}),
        "mirror_control": _score(model, X, Y, M, L, {b: mirror(b, "control") for b in blocks}),
    }


# --------------------------------------------------------------------------- 2. segments
def segments(model, task, lengths, n: int, device) -> dict:
    blk = len(model.blocks) - 1
    H = {}
    for L in lengths:
        X, _, _ = make_batch(task, n, L, L, device=device, generator=torch.Generator().manual_seed(3))
        H[L] = block_means(model, X)[blk].cpu()
    L0, Lmax = min(lengths), max(lengths)
    X2, _, _ = make_batch(task, n, L0, L0, device=device, generator=torch.Generator().manual_seed(4))
    reseed = block_means(model, X2)[blk].cpu()
    ref = H[L0][:L0]
    out = {
        "block": blk,
        # input positions 0..L0-1 at the longest length vs the shortest; equal to the noise floor
        # means the input-side states do not depend on L, as the causal mask requires
        "causal_rel_diff": round(float((H[Lmax][:L0] - ref).norm() / ref.norm()), 4),
        "noise_floor": round(float((reseed[:L0] - ref).norm() / ref.norm()), 4),
        "per_length": {},
    }
    mu0, B0, _ = _plane(H[L0])
    for L, h in H.items():
        mu, B, _ = _plane(h)
        w_all, r_all = _line(((h - mu) @ B).numpy())
        wi, ri = _line(((h[:L] - mu0) @ B0).numpy())
        wa, ra = _line(((h[L:] - mu0) @ B0).numpy())
        T, A = h.shape[0], h.shape[0] - L
        out["per_length"][str(L)] = {
            "whole_refit": {"r2": round(r_all, 4), "sweep_rad": round(abs(w_all) * (T - 1), 3)},
            "input_fixed": {"r2": round(ri, 4), "omega": round(abs(wi), 4),
                            "sweep_rad": round(abs(wi) * (L - 1), 3)},
            "answer_fixed": {"r2": round(ra, 4), "omega": round(abs(wa), 4),
                             "sweep_rad": round(abs(wa) * (A - 1), 3)},
        }
    # input-segment angle curve at the longest length, in the fixed plane: a saturating code rises
    # fast inside the training range and flattens outside it
    P = ((H[Lmax][:Lmax] - mu0) @ B0).numpy()
    ang = np.unwrap(np.arctan2(P[:, 1], P[:, 0]))
    ang = np.abs(ang - ang[0])
    out["input_angle_at_max_len"] = [round(float(a), 3) for a in ang]
    return out


# --------------------------------------------------------------------------- 3. error profile
@torch.no_grad()
def error_profile(model, task, L: int, n: int, device, bins: int = 6) -> list[float]:
    X, Y, M = make_batch(task, n, L, L, device=device, generator=torch.Generator().manual_seed(2))
    pred = model(X).argmax(-1)
    acc = (pred[M] == Y[M]).view(n, -1)[:, :L].float().mean(0).cpu().numpy()
    return [round(float(b.mean()), 4) for b in np.array_split(acc, bins)]


def main(cfg: ClockConfig) -> None:
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dirs = sorted({p for pat in cfg.match for p in glob.glob(str(Path(cfg.results_dir) / pat))})
    out: dict[str, dict] = {}
    for d in dirs:
        loaded = load_model(Path(d), device)
        if loaded is None:
            print(f"  skip {Path(d).name}: no model.pt")
            continue
        model, task, res = loaded
        torch.manual_seed(0)
        name = Path(d).name
        rec = {"config": {k: res["config"].get(k) for k in ("field", "dim", "algebra", "steps", "seed")},
               "ablation": {str(L): ablation(model, task, L, cfg.n_seqs // 2, device) for L in cfg.abl_lengths},
               "segments": segments(model, task, cfg.seg_lengths, cfg.n_seqs, device),
               "error_profile": {str(L): error_profile(model, task, L, cfg.n_seqs // 2, device)
                                 for L in cfg.prof_lengths}}
        out[name] = rec
        a = rec["ablation"][str(cfg.abl_lengths[0])]
        s = rec["segments"]["per_length"]
        print(f"  {name:48s} tok base/clock/ctrl {a['baseline']['token_acc']:.2f}/"
              f"{a['remove_clock']['token_acc']:.2f}/{a['remove_control']['token_acc']:.2f}  "
              f"input R2 {s[str(min(cfg.seg_lengths))]['input_fixed']['r2']:.2f}->"
              f"{s[str(max(cfg.seg_lengths))]['input_fixed']['r2']:.2f}  answer R2 "
              f"{s[str(min(cfg.seg_lengths))]['answer_fixed']['r2']:.2f}->"
              f"{s[str(max(cfg.seg_lengths))]['answer_fixed']['r2']:.2f}")
        del model
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out).write_text(json.dumps(out, indent=1))
    print(f"-> {cfg.out} ({len(out)} runs)")


if __name__ == "__main__":
    main(tyro.cli(ClockConfig))
