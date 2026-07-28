"""Activation-level mechanistic probe: is the inferred position in the *residual stream*?

`experiments/phase_probe.py` tests the pre-registered H1_mech claim (RoPE-like banding in the trained
QK *weights*) and returns null at three scales. That probe is looking in the wrong tensor, and the
paper says so: under NoPE the weights are position-independent, so RoPE's per-position rotation has
nowhere to be stored in them. If a NoPE model bootstraps position at all, it has to build it in the
activations, as a function of position, at run time.

This module tests that directly. Three measurements, on the reverse checkpoints that carry the one
surviving behavioural effect:

  1. `rotating_plane`  (field-agnostic, so real and complex are directly comparable)
     Take the position-conditional mean residual stream hbar(m), centre it over positions, project on
     its own top-2 principal plane, and ask whether the trajectory *turns monotonically*: unwrap the
     in-plane angle and fit alpha(m) = omega*m + c. A positional clock gives a high R^2 AND most of
     the variance in the top-2 plane. Null: shuffle the position labels, which leaves the point cloud
     and its PCA untouched and destroys only the ordering, so it isolates exactly the ordering claim.

  2. `channel_phase`   (complex conditions only)
     Per channel, the honest version of the pre-registered claim: does arg(zbar(m, j)) advance
     linearly in position? Scan the frequency grid for the omega maximising the circular resultant of
     phi(m, j) - omega*m. Reported against the same shuffled-position null.

  3. `offset_staircase` (the mechanism the offset law predicts, docs/hypothesis.md H3_offset)
     reverse needs answer token j fetched from offset -(2j+1), an arithmetic ramp stepping by 2. So a
     model that solves it by relative retrieval should own a head whose attention peak walks that
     staircase. Per (layer, head) we fit peak-offset vs j and report the slope and the exact-hit rate.
     This is behavioural, needs no complex arithmetic, and applies to real and complex alike.

Usage:
    uv run python -m experiments.activation_probe                    # default reverse run set
    uv run python -m experiments.activation_probe --lengths 16 24 --out-dir results/_figs/act
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

from complexnn.layers import to_real_concat
from complexnn.models import TransformerLM
from experiments.data.tasks import TaskConfig, make_batch
from experiments.phase_probe import load_run

FIG_TEMPLATE = "plotly_white"

# The reverse run set: the three converged complex seeds that carry the effect, the no-input-phase
# ablation, and both real controls (equal-dim and parameter-matched) at the same seeds.
DEFAULT_RUNS = [
    ("unity_reverse_complex_none_d160_s0", "C-NoPE s0"),
    ("unity_reverse_complex_none_d160_s1", "C-NoPE s1"),
    ("unity_reverse_complex_none_d160_s2", "C-NoPE s2"),
    ("unity_reverse_complex_none_d160_noimag_s0", "C-NoPE noimag s0"),
    ("unity_reverse_real_none_d160_s0", "R-NoPE d160 s0"),
    ("unity_reverse_real_none_d160_s1", "R-NoPE d160 s1"),
    ("unity_reverse_real_none_d160_s2", "R-NoPE d160 s2"),
    ("unity_reverse_real_none_d224_matched_s0", "R-NoPE d224 s0"),
    ("unity_reverse_real_none_d224_matched_s1", "R-NoPE d224 s1"),
    ("unity_reverse_real_none_d224_matched_s2", "R-NoPE d224 s2"),
    # The block-diagonal tying: same width and parameters as complex, but no complex arithmetic and
    # no cross-stream mixing anywhere, and it matches complex behaviourally (docs/results.md H2_alg).
    # If it also carries the deep positional rotation, then the dissociation this probe found is
    # between two-stream and single-stream architectures, not between complex and real. The sweep
    # ran with save_model=False, so this is an independent retrain of the same seed-0 recipe
    # (seq@24 = 0.84 locally vs 0.883 on the cluster; not bit-identical across GPUs).
    ("local_reverse_complex_none_d160_diag_s0", "C-diag s0 (block-diagonal)"),
]


@dataclass
class ProbeConfig:
    results_dir: str = "results"
    out_dir: str = "results/_figs/act"
    lengths: tuple[int, ...] = (16, 24)   # in-distribution and the 1.5x point where the effect lives
    n_seqs: int = 512
    batch: int = 128
    n_null: int = 200                     # shuffled-position resamples
    seed: int = 0
    device: str = "cpu"


# --------------------------------------------------------------------------- residual-stream capture
@torch.no_grad()
def residual_by_position(model: TransformerLM, X: torch.Tensor) -> list[np.ndarray]:
    """Mean residual stream per position, one (T, W) array per block input plus the final stream.

    Averaging over a large batch of random content marginalises content out and leaves the
    position-dependent component, the same trick `phase_probe.attention_probe` uses on logits.
    W = 2*dim for complex (a lossless [Re || Im] readout), dim for real.
    """
    streams: list[torch.Tensor] = []
    hooks = []

    def grab(_mod, args):
        streams.append(args[0].detach())

    for blk in model.blocks:
        hooks.append(blk.register_forward_pre_hook(grab))
    try:
        model(X)
    finally:
        for h in hooks:
            h.remove()

    out = []
    for z in streams:
        z = to_real_concat(z) if z.is_complex() else z
        out.append(z.mean(dim=0).cpu().numpy())   # (T, W)
    return out


@torch.no_grad()
def residual_by_position_complex(model: TransformerLM, X: torch.Tensor) -> list[np.ndarray] | None:
    """As above but keeping complex structure: one (T, dim) complex array per block. None if real."""
    if model.cfg.field != "complex":
        return None
    streams: list[torch.Tensor] = []
    hooks = []

    def grab(_mod, args):
        streams.append(args[0].detach())

    for blk in model.blocks:
        hooks.append(blk.register_forward_pre_hook(grab))
    try:
        model(X)
    finally:
        for h in hooks:
            h.remove()
    return [z.mean(dim=0).cpu().numpy() for z in streams]


# --------------------------------------------------------------------------- 1. rotating-plane test
def _lin_r2(y: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    """R^2 and slope of a least-squares line y ~ m."""
    A = np.stack([m, np.ones_like(m)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid**2).sum()) / max(ss_tot, 1e-12), float(coef[0])


def _plane_angle_fit(H: np.ndarray, order: np.ndarray) -> tuple[float, float, float, float, float]:
    """Project H (T, W) on its top-2 principal plane and fit the unwrapped in-plane angle vs `order`.

    Returns (angle-fit R2, omega rad/step, top-2 variance share, PC1-ramp R2, in-plane radius CV).
    `order` supplies the position labels, so passing a permutation gives the null.

    The last two separate the two ways a residual stream can carry position inside one plane:
    a **rotation** turns at constant radius (high angle R2, low radius CV), whereas a **ramp** slides
    along a single direction (high PC1 R2, angle barely advancing). Reporting both stops a null angle
    result from being read as "no position" when the encoding is simply not rotational.
    """
    Hc = H - H.mean(axis=0, keepdims=True)
    # economy SVD: left singular vectors scaled by S are the coordinates of the T position points
    U, S, _ = np.linalg.svd(Hc, full_matrices=False)
    var = S**2
    top2 = float(var[:2].sum() / max(var.sum(), 1e-12))
    p = (U[:, :2] * S[:2])[np.argsort(order)]      # (T, 2) coordinates, in position order
    m = np.sort(order).astype(float)
    ang = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
    r2, omega = _lin_r2(ang, m)
    ramp_r2, _ = _lin_r2(p[:, 0], m)
    rad = np.hypot(p[:, 0], p[:, 1])
    radius_cv = float(rad.std() / max(rad.mean(), 1e-12))
    return r2, omega, top2, ramp_r2, radius_cv


def rotating_plane(H: np.ndarray, n_null: int, rng: np.random.Generator) -> dict:
    T = H.shape[0]
    r2, omega, top2, ramp_r2, radius_cv = _plane_angle_fit(H, np.arange(T))
    null = np.array([_plane_angle_fit(H, rng.permutation(T))[0] for _ in range(n_null)])
    return {
        "r2": round(r2, 4),
        "omega_rad_per_step": round(omega, 4),
        "pc1_ramp_r2": round(ramp_r2, 4),      # position as a slide along one direction
        "radius_cv": round(radius_cv, 4),      # ~0 means the plane trajectory is a true circle

        # Total angle swept across the whole sequence. This is the discriminator between the two
        # readings of a positional rotation: a fixed-frequency clock (RoPE-like) holds `omega`
        # constant as T grows and sweeps proportionally further, whereas a *fractional*-position code
        # holds the total sweep constant and shrinks `omega` like 1/T. Only the latter is length
        # invariant, and only the latter can transfer to unseen lengths unchanged.
        "total_sweep_rad": round(abs(omega) * (T - 1), 4),
        "T": int(T),
        "top2_var_share": round(top2, 4),
        "null_r2_p95": round(float(np.quantile(null, 0.95)), 4),
        "null_r2_mean": round(float(null.mean()), 4),
        # empirical p: how often the shuffled ordering fits at least as well
        "p": round(float((null >= r2).mean()), 4),
    }


# --------------------------------------------------------------------------- 2. per-channel phase ramp
def channel_phase(Z: np.ndarray, n_null: int, rng: np.random.Generator, n_grid: int = 2001) -> dict:
    """Best linear phase ramp per complex channel: max over omega of |mean_m exp(i(phi(m,j)-omega*m))|.

    Z is (T, dim) complex. Returns the distribution of the per-channel resultant against the
    shuffled-position null; a genuine clock channel sits near 1 and far above the null.
    """
    T = Z.shape[0]
    phi = np.angle(Z)                                    # (T, dim)
    grid = np.linspace(-math.pi, math.pi, n_grid)[None, :, None]   # (1, G, 1)
    m = np.arange(T, dtype=float)[:, None, None]                    # (T, 1, 1)

    def best(order: np.ndarray) -> np.ndarray:
        ph = phi[order][:, None, :]                                 # (T, 1, dim)
        res = np.abs(np.exp(1j * (ph - grid * m)).mean(axis=0))     # (G, dim)
        return res.max(axis=0)                                      # (dim,)

    obs = best(np.arange(T))
    null = np.concatenate([best(rng.permutation(T)) for _ in range(max(n_null // 20, 5))])
    return {
        "resultant_mean": round(float(obs.mean()), 4),
        "resultant_max": round(float(obs.max()), 4),
        "resultant_p95": round(float(np.quantile(obs, 0.95)), 4),
        "null_mean": round(float(null.mean()), 4),
        "null_p95": round(float(np.quantile(null, 0.95)), 4),
        "frac_above_null_p95": round(float((obs > np.quantile(null, 0.95)).mean()), 4),
    }


# --------------------------------------------------------------------------- 3. offset staircase
@torch.no_grad()
def offset_staircase(model: TransformerLM, X: torch.Tensor, L: int, task_name: str) -> dict:
    """Does any head walk the retrieval staircase the task requires?

    For `reverse`, answer token j (query position L+j) must fetch input position L-1-j, i.e. offset
    -(2j+1). We recompute each block's post-softmax attention exactly as `forward` does, average over
    the batch, take the peak source position per answer query, and fit peak-offset vs j.
    """
    required = {                       # offset(j), see experiments/data/tasks.py
        "reverse": lambda j: -(2 * j + 1),
        "revskip": lambda j: -(3 * j + 1),
        "copy": lambda j: -L,
        "skipcopy": lambda j: j - L,
    }.get(task_name)
    if required is None:
        return {}

    streams: list[torch.Tensor] = []
    hooks = []

    def grab(_mod, args):
        streams.append(args[0].detach())

    for blk in model.blocks:
        hooks.append(blk.register_forward_pre_hook(grab))
    try:
        model(X)
    finally:
        for h in hooks:
            h.remove()

    T = X.shape[1]
    n_ans = min(T - L, L)                      # answer queries present at this length
    js = np.arange(n_ans)
    want = np.array([required(int(j)) for j in js], dtype=float)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    src_idx = torch.tensor((L + js) + want, dtype=torch.long)   # required source position per answer j
    per_head = []
    for li, (blk, h) in enumerate(zip(model.blocks, streams)):
        logits = blk.attn.attn_logits(blk.n1(h))           # (B, H, T, T), pre-mask pre-softmax
        w = F.softmax(logits.masked_fill(mask, float("-inf")), dim=-1).mean(dim=0)  # (H, T, T)
        for hd in range(w.shape[0]):
            rows = w[hd, L:L + n_ans, :]                   # (n_ans, T)
            peak = rows.argmax(dim=-1).cpu().numpy().astype(float)
            got = peak - (L + js)
            hit = float((got == want).mean())
            # argmax is brittle: a head can route correctly while a distractor wins the max. The mass
            # actually placed on the required source is the softer, more faithful read.
            mass = float(rows[torch.arange(n_ans), src_idx].mean())
            sharp = float(rows.max(dim=-1).values.mean())
            A = np.stack([js.astype(float), np.ones_like(js, dtype=float)], axis=1)
            coef, *_ = np.linalg.lstsq(A, got, rcond=None)
            per_head.append({"layer": li, "head": hd, "hit_rate": round(hit, 4),
                             "mass_at_required": round(mass, 4), "peak_mass": round(sharp, 4),
                             "slope": round(float(coef[0]), 4), "intercept": round(float(coef[1]), 4),
                             "peak_offsets": got.astype(int).tolist()})

    by_hit = max(per_head, key=lambda r: r["hit_rate"])
    by_mass = max(per_head, key=lambda r: r["mass_at_required"])
    req_slope = float(np.polyfit(js, want, 1)[0]) if n_ans > 1 else float("nan")
    return {
        "required_slope": round(req_slope, 4),
        "required_offsets": want.astype(int).tolist(),
        "best_head": by_hit,                    # highest exact-staircase hit rate
        "best_head_by_mass": by_mass,           # most attention mass on the required source
        "total_mass_at_required": round(float(sum(r["mass_at_required"] for r in per_head)), 4),
        "n_heads_hit_rate_above_0.9": int(sum(r["hit_rate"] > 0.9 for r in per_head)),
        "n_heads_mass_above_0.5": int(sum(r["mass_at_required"] > 0.5 for r in per_head)),
        "per_head": sorted(per_head, key=lambda r: -r["mass_at_required"])[:5],
    }


# --------------------------------------------------------------------------- driver
def probe_run(run, cfg: ProbeConfig, L: int, rng: np.random.Generator) -> dict:
    task = TaskConfig(run.task.name, run.task.n_symbols, run.task.train_min, run.task.train_max,
                      run.task.eval_lengths)
    gen = torch.Generator().manual_seed(cfg.seed)
    X, _, _ = make_batch(task, cfg.n_seqs, L, L, device=cfg.device, generator=gen)
    model = run.model.to(cfg.device).eval()

    reals = residual_by_position(model, X)
    comps = residual_by_position_complex(model, X)

    out = {
        "acc": run.acc,
        "field": model.cfg.field,
        "dim": model.cfg.dim,
        "length": L,
        # the deepest block input is the most processed residual stream before readout
        "rotating_plane": {f"L{i}": rotating_plane(H, cfg.n_null, rng) for i, H in enumerate(reals)},
        "offset_staircase": offset_staircase(model, X, L, task.name),
    }
    if comps is not None:
        out["channel_phase"] = {f"L{i}": channel_phase(Z, cfg.n_null, rng) for i, Z in enumerate(comps)}
    return out


def fig_staircase(metrics: dict, L: int, out: Path) -> None:
    """Peak attention offset vs answer index for the best head of each condition, against the law."""
    fig = go.Figure()
    ref = None
    for label, m in metrics.items():
        st = m.get(str(L), {}).get("offset_staircase", {})
        best = st.get("best_head", {})
        if "peak_offsets" not in best:
            continue
        ref = st.get("required_offsets", ref)
        y = best["peak_offsets"]
        fig.add_scatter(x=list(range(len(y))), y=y, mode="lines+markers",
                        name=f"{label} (L{best['layer']}H{best['head']}, hit={best['hit_rate']:.2f})")
    if ref is not None:
        fig.add_scatter(x=list(range(len(ref))), y=ref, mode="lines", name="required -(2j+1)",
                        line=dict(color="black", dash="dash", width=3))
    fig.update_layout(title=f"Attention peak offset vs answer index (reverse, L={L})",
                      xaxis_title="answer index j", yaxis_title="peak offset (source - query)",
                      template=FIG_TEMPLATE)
    fig.write_html(str(out / f"act_staircase_L{L}.html"))


def main(cfg: ProbeConfig) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    root = Path(cfg.results_dir)

    metrics: dict[str, dict] = {}
    for tag, label in DEFAULT_RUNS:
        run = load_run(root / tag, label) if (root / tag / "result.json").exists() else None
        if run is None or not run.trained:
            print(f"[skip] {tag}: no checkpoint")
            continue
        metrics[label] = {str(L): probe_run(run, cfg, L, rng) for L in cfg.lengths}
        for L in cfg.lengths:
            m = metrics[label][str(L)]
            deep = m["rotating_plane"][f"L{len(m['rotating_plane']) - 1}"]
            st = m["offset_staircase"].get("best_head", {})
            print(f"[{label:18s} L={L:3d}] plane R2={deep['r2']:.3f} (null p95 {deep['null_r2_p95']:.3f}, "
                  f"p={deep['p']:.3f})  staircase hit={st.get('hit_rate', float('nan')):.3f} "
                  f"slope={st.get('slope', float('nan')):+.2f}")

    (out / "activation_metrics.json").write_text(json.dumps(metrics, indent=2))
    for L in cfg.lengths:
        fig_staircase(metrics, L, out)
    print(f"\nwritten -> {out}/activation_metrics.json")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(ProbeConfig))
