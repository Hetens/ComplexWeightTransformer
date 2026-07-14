"""Static paper figures (PDF + PNG preview) from the multi-seed sweep results.

    uv run python scripts/make_paper_figs.py            # -> paper/figs/*.pdf + *.png

Fig 1  seq-acc vs eval length, one panel per task, mean over seeds with +/- std band.
Fig 2  per-seed exact-match dots at the decision lengths (reverse @24, copy @32): the honest
       view of the reverse gap's tightness and the copy extrapolation bimodality.

Color follows the entity (condition); the d224 param-matched real baselines reuse their entity's
hue with a dashed line, and the learn_imag ablation reuses the complex hue with a dotted line, so
capacity/ablation variants never read as new conditions. Palette is the validated 5-slot
categorical set (dataviz reference palette, checked on a white surface).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.analysis import load_results

# entity -> (label, hex, dash) ; fixed assignment, never cycled
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {
    ("complex+none", 160): ("complex+NoPE", "#2a78d6", "-"),
    ("complex+none-noimag", 160): ("complex+NoPE (no input phase)", "#2a78d6", (0, (1, 1.5))),
    ("complex+rope", 160): ("complex+RoPE", "#1baf7a", "-"),
    ("real+none", 160): ("real+NoPE", "#eda100", "-"),
    ("real+none", 224): ("real+NoPE (d224 matched)", "#eda100", (0, (4, 2))),
    ("real+rope", 160): ("real+RoPE", "#008300", "-"),
    ("real+rope", 224): ("real+RoPE (d224 matched)", "#008300", (0, (4, 2))),
    ("real+learned", 160): ("real+learned", "#4a3aa7", "-"),
}
TASK_ORDER = ("reverse", "copy", "sort", "needle")
TRAIN_MAX = 16

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK2, "figure.dpi": 200, "savefig.bbox": "tight",
})


def _style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_acc_vs_length(df, metric: str = "seq_acc"):
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharey=True)
    for ax, task in zip(axes.flat, TASK_ORDER):
        d = df[df.task == task]
        for (cond, dim), (label, color, dash) in SERIES.items():
            g = (d[(d.cond == cond) & (d.dim == dim)]
                 .groupby("length")[metric].agg(["mean", "std"]).reset_index())
            if not len(g):
                continue
            ax.plot(g.length, g["mean"], color=color, linestyle=dash, linewidth=1.4,
                    marker="o", markersize=2.6, label=label)
            ax.fill_between(g.length, g["mean"] - g["std"].fillna(0),
                            g["mean"] + g["std"].fillna(0), color=color, alpha=0.12, linewidth=0)
        ax.axvline(TRAIN_MAX, color=BASE, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 24, 32, 48, 64, 96, 128])
        ax.set_xticklabels(["16", "24", "32", "48", "64", "96", "128"])
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(task, color=INK, fontweight="bold")
        _style_axis(ax)
    axes[0, 0].text(TRAIN_MAX * 1.04, 0.06, "train max", color=MUTED, fontsize=6.5, rotation=90)
    for ax in axes[1, :]:
        ax.set_xlabel("evaluation length")
    for ax in axes[:, 0]:
        ax.set_ylabel("exact-match accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06), labelcolor=INK2)
    fig.tight_layout(h_pad=1.2)
    return fig


def fig_per_seed(df):
    """Per-seed dots at the decision lengths: reverse @24 (the surviving gap, tight seeds)
    and copy @32 (large seed variance in every NoPE condition)."""
    panels = [("reverse", 24), ("copy", 32)]
    order = list(SERIES.items())
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharex=True)
    rng = np.random.default_rng(0)
    for ax, (task, L) in zip(axes, panels):
        d = df[(df.task == task) & (df.length == L)]
        ylabels = []
        for i, ((cond, dim), (label, color, dash)) in enumerate(order):
            v = d[(d.cond == cond) & (d.dim == dim)].seq_acc.values
            y = len(order) - 1 - i
            jitter = rng.uniform(-0.14, 0.14, size=len(v))
            ax.scatter(v, np.full_like(v, float(y)) + jitter, s=14, color=color, zorder=3,
                       edgecolors="white", linewidths=0.5)
            ax.scatter([v.mean()], [y], marker="|", s=130, color=INK, zorder=4, linewidths=1.2)
            ylabels.append(label)
        ax.set_yticks(range(len(order) - 1, -1, -1))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xlim(-0.04, 1.04)
        ax.set_title(f"{task} @ L={L}", color=INK, fontweight="bold")
        _style_axis(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    fig.supxlabel("exact-match accuracy (dots = seeds, bar = mean)", fontsize=8, color=INK2)
    fig.tight_layout()
    return fig


def main(results_dir: str = "results", out_dir: str = "paper/figs") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_results(results_dir)
    df = df[df.tag.str.startswith("unity_")]
    for name, fig in [("fig_acc_vs_length", fig_acc_vs_length(df)), ("fig_per_seed", fig_per_seed(df))]:
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png")
        print(f"wrote {out / name}.pdf/.png")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
