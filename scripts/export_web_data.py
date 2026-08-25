"""Bundle the committed metrics into the JSON the web demo reads.

    uv run python scripts/export_web_data.py     # -> web/public/data/*.json

Everything here is a projection of files already in the repo (results/unity_*/result.json and
results/_figs/*/**.json), so the site cannot drift from the paper's evidence base.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.analysis import load_results
from experiments.stats import CANONICAL_LR, _best_lr, cond_key, welch_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "data"

TASKS = ("reverse", "copy", "sort", "needle")
# entity -> the paper's fixed hue/dash assignment (scripts/make_paper_figs.py); the site reuses it
SERIES = [
    ("complex+none", 160, "complex+NoPE", "#2a78d6", "solid"),
    ("complex+none-noimag", 160, "complex+NoPE (no input phase)", "#2a78d6", "dotted"),
    ("real+none", 160, "real+NoPE", "#eda100", "solid"),
    ("real+none", 224, "real+NoPE (d224 matched)", "#eda100", "dashed"),
    ("real+rope", 160, "real+RoPE", "#008300", "solid"),
    ("real+rope", 224, "real+RoPE (d224 matched)", "#008300", "dashed"),
    ("complex+rope", 160, "complex+RoPE", "#1baf7a", "solid"),
    ("real+learned", 160, "real+learned", "#4a3aa7", "solid"),
]


def _canonical(full):
    """Every claim except the width scan is a statement about the committed recipe, so it reads the
    canonical learning rate only. Pooling the lr grid silently shifts complex+NoPE 0.856 -> 0.838
    (experiments/stats.followup_checks says the same thing)."""
    return full[(full.lr == CANONICAL_LR) | full.lr.isna()]


def _sweep_only(full):
    """The paper's tables are the committed sweep (experiments/stats.main defaults to tag_prefix
    'unity_'). results/local_* are independent local retrains kept for the activation probe; pooling
    them in would change the seed counts the tables report."""
    return full[full.tag.str.startswith("unity_")]


def behavior() -> dict:
    df = _canonical(_sweep_only(load_results(str(ROOT / "results"))))
    df = df[df.converged]
    out = {"tasks": {}, "series": [
        {"cond": c, "dim": d, "label": lab, "color": col, "dash": dash} for c, d, lab, col, dash in SERIES]}
    for task in TASKS:
        d = df[df.task == task]
        rows = []
        for cond, dim, *_ in SERIES:
            g = d[(d.cond == cond) & (d.dim == dim)]
            if not len(g):
                continue
            pts = (g.groupby("length")["seq_acc"].agg(["mean", "std", "count"]).reset_index()
                   .sort_values("length"))
            rows.append({"cond": cond, "dim": dim, "points": [
                {"length": int(r.length), "mean": round(float(r["mean"]), 4),
                 "std": round(float(0.0 if np.isnan(r["std"]) else r["std"]), 4),
                 "n": int(r["count"])} for _, r in pts.iterrows()]})
        out["tasks"][task] = rows
    return out


def width_scan() -> list[dict]:
    """The real-width scan on reverse @24: the control that the tying, not the width, is what's cheap.

    This is the one table that must NOT be filtered to the canonical rate: its subject is the
    learning-rate grid itself (lr=2e-3 was tuned at d160 and diverges at d640). Each arm is scored at
    its own best rate over the same grid, converged seeds only, exactly as stats._best_lr does it.
    """
    full = _sweep_only(load_results(str(ROOT / "results")))
    rev = full[(full.task == "reverse") & (full.length == 24)]
    c_lr, ref = _best_lr(rev, cond="complex+none", dim=160)
    rows = []
    for cond, dim in [("complex+none", 160), ("real+none", 224), ("real+none", 320),
                      ("real+none", 448), ("real+none", 640)]:
        lr, v = _best_lr(rev, cond=cond, dim=dim)
        if not len(v):
            continue
        params = float(rev[(rev.cond == cond) & (rev.dim == dim)]["params"].mean())
        w = welch_t(ref, v)
        rows.append({"label": ("complex+NoPE d160" if cond.startswith("complex") else f"real+NoPE d{dim}"),
                     "field": "complex" if cond.startswith("complex") else "real",
                     "dim": int(dim), "params": int(params),
                     "x_complex": round(params / 3085920.0, 2), "lr": float(lr),
                     "mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4), "n": int(len(v)),
                     "delta_vs_complex": round(float(w.delta), 4), "p_vs_complex": round(float(w.p), 4)})
    return rows


def algebra() -> list[dict]:
    """The weight-tying ablation: identical params and width, only the sharing pattern differs."""
    df = _canonical(_sweep_only(load_results(str(ROOT / "results"))))
    # NoPE only: the algebra ablation is a NoPE experiment, and complex+RoPE also reports
    # algebra == "complex", so omitting this pools five zero-scoring RoPE runs into the reference arm.
    d = df[(df.task == "reverse") & (df.length == 24) & df.converged & (df.field == "complex")
           & (df.pe == "none") & (df.dim == 160) & (df.learn_imag)]
    ref = d[d.algebra == "complex"]["seq_acc"]
    names = {"complex": ("[[A,-B],[B,A]]", "complex, i²=-1"), "split": ("[[A,B],[B,A]]", "split, j²=+1"),
             "diag": ("[[A,0],[0,B]]", "none (no mixing)"), "dual": ("[[A,0],[B,A]]", "dual, ε²=0")}
    rows = []
    for alg, (tying, algebra_label) in names.items():
        g = d[d.algebra == alg]
        if not len(g):
            continue
        w = welch_t(ref, g["seq_acc"])
        rows.append({"algebra": alg, "tying": tying, "algebra_label": algebra_label,
                     "mean": round(float(g.seq_acc.mean()), 4), "std": round(float(g.seq_acc.std()), 4),
                     "n": int(len(g)), "params": int(g.params.max()),
                     "p_vs_complex": None if alg == "complex" else round(float(w.p), 4)})
    return rows


def act_probe() -> dict:
    """Slim the activation probe down to what the site plots (drop the per-head offset dumps)."""
    src = json.load(open(ROOT / "results" / "_figs" / "act" / "activation_metrics.json"))
    out = {}
    for run, by_len in src.items():
        out[run] = {}
        for L, e in by_len.items():
            st = e.get("offset_staircase", {})
            best = st.get("best_head") or {}
            out[run][L] = {
                "acc": e["acc"], "field": e["field"], "dim": e["dim"],
                "rotating_plane": e["rotating_plane"],
                "retrieval": {"hit_rate": best.get("hit_rate"), "mass": best.get("mass_at_required"),
                              "layer": best.get("layer"), "head": best.get("head"),
                              "slope": best.get("slope")},
            }
    return out


def phase_probe() -> dict:
    """Weight-level circular tests at all three scales (H3): trained complex QK phase vs uniform."""
    return {scale: json.load(open(ROOT / "results" / "_figs" / scale / "phase_metrics.json"))
            for scale in ("d64", "d128", "d160")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = {
        "behavior.json": behavior(),
        "width_scan.json": width_scan(),
        "algebra.json": algebra(),
        "act_probe.json": act_probe(),
        "phase_probe.json": phase_probe(),
    }
    for name, payload in bundle.items():
        p = OUT / name
        json.dump(payload, open(p, "w"), separators=(",", ":"))
        print(f"  {name:20s} {p.stat().st_size/1024:7.1f} KB")


if __name__ == "__main__":
    main()
