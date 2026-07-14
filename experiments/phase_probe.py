"""Mechanistic probe for H1_mech (docs/hypothesis.md): do trained complex-weight QK matrices develop
emergent, RoPE-like *phase* structure -- non-uniform / banded phase spectra -- where an explicit
positional module is absent?

Pre-registered claim being tested:
    H1_mech supported iff QK phase spectra show non-uniform, frequency-structured concentration absent
    in R-NoPE (qualitative histogram + a uniformity test). F3 (no structure despite good accuracy)
    rejects H1_mech -> the capability benefit would be a confound, not the proposed mechanism.

Trabelsi init sets QK weight phases ~ U(-pi, pi), so the *init* model is the null (uniform) baseline;
CReLU breaks phase-rotation equivariance, so post-training phases are physically meaningful (not gauge).

What it computes, per condition:
  1. Pooled QK weight phase distribution (all complex q/k entries, all layers) vs the init null.
     Uniformity is quantified with three circular tests (no scipy dependency):
       - Rayleigh        -> unimodal concentration (resultant length R)
       - Kolmogorov-Smirnov vs U(-pi,pi)  -> any departure (origin-fixed)
       - Kuiper          -> any departure, rotation-invariant (robust to residual phase gauge)
  2. Eigenvalue spectrum of a square QK weight in the complex plane (rotations live on the unit circle).
  3. Behavioural attention-offset kernel: mean pre-softmax logit vs relative offset (k-q). RoPE imprints
     a relative-position profile; this asks whether complex+NoPE *recovered* one the real+NoPE lacks.

Usage:
    uv run python -m experiments.phase_probe                 # default reverse-task run set -> results/_figs/
    uv run python -m experiments.phase_probe --results-dir results
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

from complexnn.attention import ComplexSelfAttention
from complexnn.models import ModelConfig, TransformerLM
from experiments.data.tasks import TaskConfig, make_batch

FIG_TEMPLATE = "plotly_white"


# --------------------------------------------------------------------------- circular uniformity tests
def _ks_uniform(u: np.ndarray) -> tuple[float, float, float]:
    """KS against U(0,1). Returns (D, D+, D-). u is the phase mapped to [0,1)."""
    n = u.size
    us = np.sort(u)
    i = np.arange(1, n + 1)
    d_plus = float((i / n - us).max())
    d_minus = float((us - (i - 1) / n).max())
    return max(d_plus, d_minus), d_plus, d_minus


def _ks_pvalue(d: float, n: int) -> float:
    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d
    s = sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return float(min(max(2.0 * s, 0.0), 1.0))


def _kuiper_pvalue(d_plus: float, d_minus: float, n: int) -> tuple[float, float]:
    """Kuiper V (= D+ + D-) and its p-value. V is invariant to the choice of phase origin."""
    v = d_plus + d_minus
    en = math.sqrt(n)
    lam = (en + 0.155 + 0.24 / en) * v
    s = sum((4.0 * k * k * lam * lam - 1.0) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return v, float(min(max(2.0 * s, 0.0), 1.0))


def circular_stats(phases: np.ndarray) -> dict:
    """phases in radians (any range). Reports resultant length, circular variance, and uniformity p-values."""
    n = int(phases.size)
    c, s = np.cos(phases).mean(), np.sin(phases).mean()
    r = float(math.hypot(c, s))                       # mean resultant length in [0,1]
    z = n * r * r                                     # Rayleigh statistic
    rayleigh_p = float(math.exp(-z) * (1.0 + (2.0 * z - z * z) / (4.0 * n)))  # 1st-order corrected
    u = ((phases + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi)
    d, d_plus, d_minus = _ks_uniform(u)
    v, kuiper_p = _kuiper_pvalue(d_plus, d_minus, n)
    return {
        "n": n,
        "resultant_R": round(r, 4),
        "circular_var": round(1.0 - r, 4),            # 1 = uniform, 0 = a single spike
        "rayleigh_p": rayleigh_p,
        "ks_stat": round(d, 4),
        "ks_p": _ks_pvalue(d, n),
        "kuiper_V": round(v, 4),
        "kuiper_p": kuiper_p,
    }


# --------------------------------------------------------------------------- model loading / weight access
@dataclass
class Run:
    label: str
    model: TransformerLM
    cfg: ModelConfig
    task: TaskConfig
    trained: bool
    acc: float | None


def _model_config(c: dict, task: TaskConfig) -> ModelConfig:
    return ModelConfig(
        vocab_size=task.vocab_size, dim=c["dim"], n_layers=c["n_layers"], n_heads=c["n_heads"],
        mlp_ratio=c["mlp_ratio"], field=c["field"], pe=c["pe"], max_T=c["max_T"],
        rope_base=c["rope_base"], learn_imag=c["learn_imag"], complex_act=c["complex_act"],
    )


def load_run(result_dir: str | Path, label: str | None = None) -> Run | None:
    d = json.loads((Path(result_dir) / "result.json").read_text())
    c = d["config"]
    task = TaskConfig(c["task"], c["n_symbols"], c["train_min"], c["train_max"], tuple(c["eval_lengths"]))
    model = TransformerLM(_model_config(c, task))
    ckpt = Path(result_dir) / "model.pt"
    trained = ckpt.exists()
    if not trained:
        return None  # no checkpoint -> nothing to probe (re-run train.py with --save-model)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    L = str(c["train_max"])
    acc = d.get("evals", {}).get(L, {}).get("token_acc")
    return Run(label or Path(result_dir).name, model, model.cfg, task, trained, acc)


def init_twin(cfg: ModelConfig, seed: int = 0) -> TransformerLM:
    """A fresh, untrained model of the same shape -- the Trabelsi-uniform phase null."""
    torch.manual_seed(seed)
    return TransformerLM(cfg).eval()


def qk_complex_weights(model: TransformerLM) -> list[tuple[str, torch.Tensor]]:
    out = []
    for li, blk in enumerate(model.blocks):
        attn = blk.attn
        if isinstance(attn, ComplexSelfAttention):
            out.append((f"L{li}.q", attn.q.weight.detach()))
            out.append((f"L{li}.k", attn.k.weight.detach()))
    return out


def pooled_phase_mag(model: TransformerLM) -> tuple[np.ndarray, np.ndarray] | None:
    ws = qk_complex_weights(model)
    if not ws:
        return None  # real field: no complex QK phases
    ph = torch.cat([torch.angle(w).flatten() for _, w in ws]).numpy()
    mg = torch.cat([w.abs().flatten() for _, w in ws]).numpy()
    return ph, mg


def pooled_eigen(model: TransformerLM) -> tuple[np.ndarray, np.ndarray] | None:
    """Eigenvalue angles & magnitudes pooled over all square QK matrices. A pure rotation has
    eigenvalues e^{±iθ} on the unit circle, so banded eigen-angles are the cleaner RoPE signature
    than marginal entry phases (which can stay ~uniform while the operator still rotates)."""
    ws = qk_complex_weights(model)
    if not ws:
        return None
    angs, mags = [], []
    for _, w in ws:
        ev = torch.linalg.eigvals(w)
        angs.append(torch.angle(ev))
        mags.append(ev.abs())
    return torch.cat(angs).numpy(), torch.cat(mags).numpy()


def complex_metrics(model: TransformerLM, acc: float | None) -> dict:
    """Full structural metric block for one complex model (entry phases + eigen spectrum)."""
    ph, mg = pooled_phase_mag(model)
    thresh = np.quantile(mg, 0.75)
    ev_ang, ev_abs = pooled_eigen(model)
    near_unit = float(np.mean((ev_abs >= 0.9) & (ev_abs <= 1.1)))
    return {
        "acc": acc,
        "all_entries": circular_stats(ph),
        "top25pct_magnitude": circular_stats(ph[mg >= thresh]),
        "eigen_angles": circular_stats(ev_ang),
        "eigen_abs_mean": round(float(ev_abs.mean()), 4),
        "eigen_abs_max": round(float(ev_abs.max()), 4),
        "eigen_frac_near_unit": round(near_unit, 4),
    }


# --------------------------------------------------------------------------- behavioural attention probe
@torch.no_grad()
def attention_probe(model: TransformerLM, task: TaskConfig, L: int = 12, n_seqs: int = 256,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """The behavioural test of 'inferred position': how much do attention logits depend on POSITION?

    Under NoPE the weights are position-independent, so any positional dependence must emerge in the
    attention *pattern* (content -> effective position via the causal mask + depth). We feed random
    content and marginalise it out by averaging over a large batch, leaving only the position-dependent
    component M[q,k]. Per attention layer:
        pos_fraction = Var_qk(M) / Var(all logits)   -- share of logit variance that is positional.
                        Real+NoPE has no positional input -> floor ~ 1/n_seqs (content-averaging noise).
        rel_fraction = Var(offset-only reconstruction of M) / Var(M)  -- how relative-offset (RoPE-like,
                        translation-invariant) that positional structure is, vs absolute-position.
    Returns (offset deltas, mean-logit-vs-offset curve pooled over layers/heads, metrics)."""
    gen = torch.Generator().manual_seed(seed)
    X, _, _ = make_batch(task, n_seqs, L, L, generator=gen)
    attns = [blk.attn for blk in model.blocks]
    grabbed: list[torch.Tensor] = []
    handles = [a.register_forward_pre_hook(lambda m, inp: grabbed.append(inp[0].detach())) for a in attns]
    model(X)
    for h in handles:
        h.remove()

    T = X.shape[1]
    tri = torch.tril(torch.ones(T, T, dtype=torch.bool))  # causal-valid (q,k)
    didx = (torch.arange(T)[:, None] - torch.arange(T)[None, :])  # offset q-k, >=0 on/below diagonal
    pos_fracs, rel_fracs = [], []
    Msum = torch.zeros(T, T)
    for a, xin in zip(attns, grabbed):
        lg = a.attn_logits(xin)                       # (B, H, Tq, Tk)
        total_var = lg[..., tri].var(unbiased=False)  # logit variance incl. content
        M = lg.mean(dim=(0, 1))                        # (Tq, Tk), content marginalised
        Msum += M
        pos_var = M[tri].var(unbiased=False)
        # offset-only reconstruction: replace M[q,k] by the mean over its diagonal
        f = torch.zeros(T).scatter_add_(0, didx[tri], M[tri])
        cnt = torch.zeros(T).scatter_add_(0, didx[tri], torch.ones(tri.sum()))
        fbar = f / cnt.clamp_min(1)
        Mrel = fbar[didx]
        rel_var = Mrel[tri].var(unbiased=False)
        pos_fracs.append(float(pos_var / total_var.clamp_min(1e-12)))
        rel_fracs.append(float(rel_var / pos_var.clamp_min(1e-12)))

    Mbar = Msum / len(attns)
    f = torch.zeros(T).scatter_add_(0, didx[tri], Mbar[tri])
    cnt = torch.zeros(T).scatter_add_(0, didx[tri], torch.ones(tri.sum()))
    fbar = (f / cnt.clamp_min(1)).numpy()
    deltas = -np.arange(T)
    metrics = {
        "pos_fraction": round(float(np.mean(pos_fracs)), 4),   # positional share of logit variance
        "rel_fraction": round(float(np.mean(rel_fracs)), 4),   # of that, the relative-offset share
        "noise_floor": round(1.0 / n_seqs, 4),                  # real+NoPE reference (content-avg noise)
    }
    return deltas, fbar, metrics


# --------------------------------------------------------------------------- figures
def _phase_density_trace(phases: np.ndarray, name: str, nbins: int = 48):
    hist, edges = np.histogram(phases, bins=nbins, range=(-math.pi, math.pi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return go.Scatter(x=centers, y=hist, mode="lines", name=name, line_shape="hvh")


def fig_phase_density(named_phases: list[tuple[str, np.ndarray]], title: str,
                      template: str = FIG_TEMPLATE) -> go.Figure:
    fig = go.Figure([_phase_density_trace(ph, name) for name, ph in named_phases])
    fig.add_hline(y=1.0 / (2.0 * math.pi), line_dash="dot", line_color="gray",
                  annotation_text="uniform", annotation_position="top right")
    fig.update_layout(
        title=title, xaxis_title="QK weight phase  arg(A+iB)  [rad]",
        yaxis_title="density", template=template,
        xaxis=dict(tickvals=[-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi],
                   ticktext=["-π", "-π/2", "0", "π/2", "π"]),
    )
    return fig


def fig_eigenspectrum(named_models: list[tuple[str, TransformerLM]], template: str = FIG_TEMPLATE) -> go.Figure:
    fig = go.Figure()
    th = np.linspace(0, 2 * math.pi, 200)
    fig.add_scatter(x=np.cos(th), y=np.sin(th), mode="lines", name="unit circle",
                    line=dict(color="lightgray", dash="dot"), hoverinfo="skip")
    for name, model in named_models:
        ws = qk_complex_weights(model)
        if not ws:
            continue
        w = ws[0][1]  # first layer q (square dim x dim)
        ev = torch.linalg.eigvals(w).numpy()
        fig.add_scatter(x=ev.real, y=ev.imag, mode="markers", name=f"{name} ({ws[0][0]})",
                        marker=dict(size=5, opacity=0.7))
    fig.update_layout(title="QK weight eigenvalue spectrum (rotations lie on the unit circle)",
                      xaxis_title="Re(λ)", yaxis_title="Im(λ)", template=template,
                      yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def fig_offset_kernel(curves: list[tuple[str, np.ndarray, np.ndarray]], template: str = FIG_TEMPLATE) -> go.Figure:
    fig = go.Figure()
    for name, deltas, vals in curves:
        z = (vals - vals.mean()) / (vals.std() + 1e-8)  # compare shape, not scale
        order = np.argsort(deltas)
        fig.add_scatter(x=deltas[order], y=z[order], mode="lines+markers", name=name)
    fig.update_layout(
        title="Attention logit vs relative offset (z-scored) — emergent relative-position kernel",
        xaxis_title="offset  δ = key − query  (causal: δ ≤ 0)",
        yaxis_title="mean logit (z-scored per condition)", template=template,
    )
    return fig


# --------------------------------------------------------------------------- driver
# label -> result tag. Re-run these with `train.py --save-model` first (see __main__ docstring).
# d64 / 2-layer / 2000-step probe set: seeds 0/4 converged (~0.83), seeds 2/3 collapsed (~0.34).
DEFAULT_RUNS = {
    "C-NoPE (converged)": "probe_reverse_complex_none_s0",
    "C-NoPE (collapsed)": "probe_reverse_complex_none_s2",
    "C-RoPE": "probe_reverse_complex_rope_s0",
    "R-NoPE": "probe_reverse_real_none_s0",
    "R-RoPE": "probe_reverse_real_rope_s0",
}

# d128 / 4-layer / 8000-step set (scripts/probe_sweep.sh): the model clearly *solves* the task here —
# all complex-NoPE seeds converged (final loss ~1e-4, acc@16=1.0), so there is no collapsed twin. This
# is the "does emergent weight structure appear once the task is solved at scale?" test.
PROBE2_RUNS = {
    "C-NoPE (s0)": "probe2_reverse_complex_none_s0",
    "C-NoPE (s2)": "probe2_reverse_complex_none_s2",
    "C-RoPE": "probe2_reverse_complex_rope_s0",
    "R-NoPE": "probe2_reverse_real_none_s0",
    "R-RoPE": "probe2_reverse_real_rope_s0",
}

# d160 / 5-layer / 10k-step set: the Unity 5-seed sweep checkpoints (unity_sweep.sbatch, job 61568498).
# Reverse is the task where the behavioural gap actually lives at 5 seeds (complex+NoPE seq@24 = 0.86
# vs 0.41 for the param-matched real baseline), so this is the decisive probe scale: converged models,
# multiple seeds, plus the d224 param-matched real control. Checkpoints live on Unity (results/*/model.pt
# is gitignored); run this probe there or pull the reverse checkpoints first.
UNITY_D160_RUNS = {
    "C-NoPE (s0)": "unity_reverse_complex_none_d160_s0",
    "C-NoPE (s1)": "unity_reverse_complex_none_d160_s1",
    "C-NoPE (s2)": "unity_reverse_complex_none_d160_s2",
    "C-NoPE (no input phase)": "unity_reverse_complex_none_d160_noimag_s0",
    "C-RoPE": "unity_reverse_complex_rope_d160_s0",
    "R-NoPE": "unity_reverse_real_none_d160_s0",
    "R-NoPE (d224 matched)": "unity_reverse_real_none_d224_matched_s0",
    "R-RoPE": "unity_reverse_real_rope_d160_s0",
    "R-RoPE (d224 matched)": "unity_reverse_real_rope_d224_matched_s0",
}

TAGSETS = {"d64": DEFAULT_RUNS, "d128": PROBE2_RUNS, "d160": UNITY_D160_RUNS}


def build_probe_figures(results_dir: str = "results", runs: dict[str, str] | None = None,
                        template: str = FIG_TEMPLATE) -> tuple[dict[str, go.Figure], dict]:
    """Load the run set and build {name: figure} + the metrics dict, *without* writing anything.

    Shared by main() (which then writes light-mode HTML) and the marimo dashboard (which renders these
    figures directly, themed to match the notebook). Raises if no checkpoints are found."""
    runs = runs or DEFAULT_RUNS
    rdir = Path(results_dir)

    loaded: list[Run] = []
    for label, tag in runs.items():
        r = load_run(rdir / tag, label)
        if r is None:
            print(f"  skip {label}: no checkpoint at {rdir / tag} (re-run train.py --save-model)")
            continue
        loaded.append(r)
    if not loaded:
        raise SystemExit("No checkpoints found. Run the probe training set first (see module docstring).")

    metrics: dict[str, dict] = {}
    phase_curves_all: list[tuple[str, np.ndarray]] = []
    phase_curves_top: list[tuple[str, np.ndarray]] = []
    eigen_models: list[tuple[str, TransformerLM]] = []
    offset_curves: list[tuple[str, np.ndarray, np.ndarray]] = []

    # init null from the first complex run's config
    complex_runs = [r for r in loaded if r.cfg.field == "complex"]
    if complex_runs:
        twin = init_twin(complex_runs[0].cfg, seed=0)
        ph0, mg0 = pooled_phase_mag(twin)
        phase_curves_all.append(("init (Trabelsi null)", ph0))
        phase_curves_top.append(("init (Trabelsi null)", ph0[mg0 >= np.quantile(mg0, 0.75)]))
        eigen_models.append(("init", twin))
        metrics["init (Trabelsi null)"] = complex_metrics(twin, None)

    for r in loaded:
        deltas, vals, attn_m = attention_probe(r.model, r.task, L=r.task.train_max)
        offset_curves.append((r.label, deltas, vals))
        pm = pooled_phase_mag(r.model)
        if pm is None:
            metrics[r.label] = {"acc": r.acc, "field": "real", "attention": attn_m}
            continue
        ph, mg = pm
        phase_curves_all.append((r.label, ph))
        phase_curves_top.append((r.label, ph[mg >= np.quantile(mg, 0.75)]))
        eigen_models.append((r.label, r.model))
        metrics[r.label] = {**complex_metrics(r.model, r.acc), "attention": attn_m}

    figs = {
        "phase_qk_all": fig_phase_density(
            phase_curves_all, "QK weight phase spectrum — all entries (H1_mech)", template),
        "phase_qk_topmag": fig_phase_density(
            phase_curves_top, "QK weight phase spectrum — top-25% magnitude entries", template),
        "phase_eigenspectrum": fig_eigenspectrum(eigen_models, template),
        "phase_offset_kernel": fig_offset_kernel(offset_curves, template),
    }
    return figs, metrics


def main(results_dir: str = "results", runs: dict[str, str] | None = None,
         figs_subdir: str = "") -> dict:
    rdir = Path(results_dir)
    figs_dir = rdir / "_figs" / figs_subdir if figs_subdir else rdir / "_figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    figs, metrics = build_probe_figures(results_dir, runs, template=FIG_TEMPLATE)  # saved figs are light
    for name, fig in figs.items():
        fig.write_html(figs_dir / f"{name}.html")
    (figs_dir / "phase_metrics.json").write_text(json.dumps(metrics, indent=2))

    # console summary
    print("\nWEIGHT-level (H1_mech as registered): is there banding in QK weight/eigen phases?")
    print(f"  {'condition':<22} {'acc':>5} | {'entry_cv':>8} {'entry_KSp':>10} | {'eig_cv':>7} {'eig_KSp':>9}")
    for label, m in metrics.items():
        ent = m.get("all_entries")
        if ent is None:
            print(f"  {label:<22} {'(real — no complex QK phases)'}")
            continue
        eig, acc = m["eigen_angles"], ("  -  " if m["acc"] is None else f"{m['acc']:.2f}")
        print(f"  {label:<22} {acc:>5} | {ent['circular_var']:>8.3f} {ent['ks_p']:>10.2e} | "
              f"{eig['circular_var']:>7.3f} {eig['ks_p']:>9.2e}")

    print("\nBEHAVIOUR-level (where inferred position must live): attention's positional dependence")
    print(f"  pos_fraction = positional share of logit variance (real+NoPE floor ~{1/256:.4f})")
    print(f"  {'condition':<22} {'pos_fraction':>13} {'rel_fraction':>13}")
    for label, m in metrics.items():
        a = m.get("attention")
        if a:
            print(f"  {label:<22} {a['pos_fraction']:>13.4f} {a['rel_fraction']:>13.4f}")
    print(f"\nFigures + phase_metrics.json -> {figs_dir}")
    return metrics


if __name__ == "__main__":
    import tyro

    @dataclass
    class _Args:
        results_dir: str = "results"
        tagset: str = "d64"  # which run set: "d64" (default), "d128" (probe2_*), or "d160" (unity_*)

    a = tyro.cli(_Args)
    subdir = "" if a.tagset == "d64" else a.tagset
    main(a.results_dir, runs=TAGSETS[a.tagset], figs_subdir=subdir)
