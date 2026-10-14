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

from matplotlib.lines import Line2D

from experiments.analysis import load_results

# entity -> (label, hex, dash) ; fixed assignment, never cycled
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {
    ("complex+none", 160): ("complex+NoPE", "#2a78d6", "-"),
    ("complex+none-noimag", 160): ("complex+NoPE (no input phase)", "#2a78d6", (0, (0.01, 3.0))),
    ("complex+rope", 160): ("complex+RoPE", "#1baf7a", "-"),
    ("real+none", 160): ("real+NoPE", "#eda100", "-"),
    ("real+none", 224): ("real+NoPE (d224 matched)", "#eda100", (0, (3.4, 1.5))),
    ("real+rope", 160): ("real+RoPE", "#008300", "-"),
    ("real+rope", 224): ("real+RoPE (d224 matched)", "#008300", (0, (3.4, 1.5))),
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


def _line_only(handle):
    """A marker-free copy of a plot handle, for legends.

    The legend handle is ~2-3 dash cycles long. Matplotlib centres the marker inside it, which lands
    on the gap and makes a dashed entry read as solid: the very entry whose only job is to tell the
    reader that this series is dashed."""
    from matplotlib.lines import Line2D
    # get_linestyle() returns the *style string* ("--") for a custom dash tuple and drops the pattern,
    # which silently collapses every dashed series in the legend onto one generic dash. The pattern in
    # force lives in _dash_pattern as (offset, onoffseq); copy that so the legend shows what was drawn.
    out = Line2D([], [], color=handle.get_color(), linewidth=handle.get_linewidth(),
                 dash_capstyle=handle.get_dash_capstyle())
    offset, seq = handle._dash_pattern
    out.set_linestyle("-" if seq is None else (offset, tuple(seq)))
    return out


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
                    marker="o", markersize=2.6, label=label, dash_capstyle="round")
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
    handles = [_line_only(h) for h in handles]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06), labelcolor=INK2, handlelength=3.6)
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


# ------------------------------------------------------------------ follow-up round (2026-07-26)
ALG_LABEL = {
    "complex+none": ("complex  [[A,-B],[B,A]]", "#2a78d6"),
    "complex+none-split": ("split-complex  [[A,B],[B,A]]", "#7b5bd6"),
    "complex+none-dual": ("dual  [[A,0],[B,A]]", "#c0499a"),
    "complex+none-diag": ("block-diagonal  [[A,0],[0,B]]", "#d1495b"),
}


def fig_mechanism(df, L: int = 24):
    """The two arms that ask *why* reverse wins, side by side.

    Left: every two-channel tying at identical width (320) and identical parameters (3.09M), so any
    spread is attributable to the sharing pattern alone. Right: the real quality-vs-width curve, which
    asks whether an unconstrained real model can buy the same accuracy at any budget we swept.
    """
    d = df[(df.task == "reverse") & (df.length == L)]
    algs = [(k, v) for k, v in ALG_LABEL.items() if len(d[(d.cond == k) & (d.dim == 160)])]
    wide = sorted(x for x in d[d.cond == "real+none"].dim.unique())
    if len(algs) < 2 and len(wide) < 3:
        return None

    # Two panels that each need their own y-axis furniture: the left one has long tying labels, the
    # right one a y-axis title. At the default wspace they collide, so give the pair explicit room.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={"wspace": 0.40})
    rng = np.random.default_rng(0)

    ax = axes[0]
    base = d[(d.cond == "real+none") & (d.dim == 224)].seq_acc.values
    for i, (cond, (label, color)) in enumerate(algs):
        v = d[(d.cond == cond) & (d.dim == 160)].seq_acc.values
        y = len(algs) - 1 - i
        ax.scatter(v, np.full_like(v, float(y)) + rng.uniform(-0.13, 0.13, len(v)), s=16,
                   color=color, zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter([v.mean()], [y], marker="|", s=140, color=INK, zorder=4, linewidths=1.2)
    if len(base):
        ax.axvline(base.mean(), color="#eda100", linewidth=1.2, linestyle=(0, (4, 2)), zorder=2)
        # Keep this inside the axes. Below the axes it lands on top of the x-axis label; every tying
        # scores >= 0.7 so the region left of the dashed line is empty and free for it.
        ax.annotate("real+NoPE d224\n(param-matched)", xy=(base.mean() - 0.03, 0.97),
                    xycoords=("data", "axes fraction"), fontsize=6.2, color="#a87400",
                    ha="right", va="top")
    ax.set_ylim(-0.6, len(algs) - 0.4)
    ax.set_yticks(range(len(algs) - 1, -1, -1))
    ax.set_yticklabels([lab for _, (lab, _) in algs], fontsize=6.8)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xlabel(f"exact-match accuracy @ L={L}")
    ax.set_title("weight tying (width and params fixed)", color=INK, fontweight="bold", fontsize=8)
    _style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    # Converged runs only, and the *best-tuned* learning rate per width, matching experiments/stats.py.
    # Two separate traps live here: lr=2e-3 was tuned at d160 and diverges at the widest baselines, so
    # plotting unconverged runs would draw an optimization failure as a capability curve; and pooling
    # across learning rates would average a tuned baseline with an untuned one. A baseline is compared
    # at its best, not at the rate that happens to flatter the method.
    rw = d[(d.cond == "real+none") & d.converged]
    rw_all = rw  # converged runs only; used below to detect widths that blew up at the base lr
    per_lr = (rw.groupby(["dim", "lr"])
              .agg(acc=("seq_acc", "mean"), sd=("seq_acc", "std"), n=("seq_acc", "size"),
                   params=("params", "mean")).reset_index())
    g = (per_lr.sort_values("acc").groupby("dim", as_index=False).last().sort_values("params"))
    ax.errorbar(g.params / 1e6, g.acc, yerr=g.sd.fillna(0), color="#eda100", marker="o",
                markersize=3.2, linewidth=1.4, capsize=2, label="real+NoPE")
    # Log x: the swept widths are spaced almost exactly 2x apart in parameters (1.54, 3.02, 6.16,
    # 12.06, 24.61 M), so a linear axis crushes d160 and d224 together and their labels overlap. The
    # claims in the text are ratios ("~4x the parameters") and a log axis is what makes those legible.
    ax.set_xscale("log")
    # Name each width on the axis rather than beside its point. Floating labels collided with the
    # error bars at the low end and with the complex reference line at the high end.
    ax.set_xticks(list(g.params / 1e6), minor=False)
    ax.set_xticklabels([f"d{int(r.dim)}\n{r.params / 1e6:.1f}M" for r in g.itertuples()],
                       fontsize=5.8)
    ax.set_xticks([], minor=True)
    ax.tick_params(axis="x", length=0)

    # The convergence gate, drawn. Every width has *some* learning rate that converges, so plotting
    # best-lr-per-width alone hides the failure the gate exists to catch. Mark the widest baseline at
    # the rate tuned for d160, where all five seeds diverged: that point is a training failure, and
    # scoring it as a capability limit is what would have manufactured support for our hypothesis.
    base_lr = per_lr.lr.max()
    for r in g.itertuples():
        blown = rw_all[(rw_all.dim == r.dim) & (rw_all.lr == base_lr)]
        tried = d[(d.cond == "real+none") & (d.dim == r.dim) & (d.lr == base_lr)]
        if len(tried) and not len(blown):
            ax.scatter([r.params / 1e6], [tried.seq_acc.mean()], marker="x", s=34,
                       color="#c0392b", zorder=5, linewidths=1.4)
            ax.annotate(f"all seeds diverged\nat lr {base_lr:.0e}", (r.params / 1e6, tried.seq_acc.mean()),
                        fontsize=5.5, color="#c0392b", ha="right", va="bottom",
                        xytext=(4, 7), textcoords="offset points")
    c = d[(d.cond == "complex+none") & (d.dim == 160)]
    if len(c):
        ax.axhline(c.seq_acc.mean(), color="#2a78d6", linewidth=1.3, zorder=2)
        # Anchor left, not right: the right end of this line is exactly where d640 lands, and the
        # label was printing over its marker and error bar.
        ax.text(g.params.min() / 1e6, c.seq_acc.mean() + 0.035, "complex+NoPE d160 (3.09M)",
                fontsize=6.2, color="#2a78d6", ha="left", va="bottom")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("real parameters (M)")
    ax.set_ylabel(f"exact-match accuracy @ L={L}")
    ax.set_title("real width scan", color=INK, fontweight="bold", fontsize=8)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def fig_offset_law(df):
    """The offset law, as a matched pair.

    `revskip` and `skipcopy` share stride and answer length and differ only in anchor, hence only in
    whether the retrieval offset depends on input length. If length-independence is the operative
    property, the complex-vs-real gap should open on revskip (end-anchored) and close on skipcopy
    (start-anchored). `reverse` and `copy` are the already-measured members of the same family.
    """
    panels = [("reverse", "end-anchored\noffset $-(2j{+}1)$"), ("revskip", "end-anchored\noffset $-(3j{+}1)$"),
              ("copy", "start-anchored\noffset $-L$"), ("skipcopy", "start-anchored\noffset $j{-}L$")]
    panels = [(t, s) for t, s in panels if (df.task == t).any()]
    if not any(t in ("revskip", "skipcopy") for t, _ in panels):
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(1.7 * len(panels) + 0.6, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    arms = [("complex+none", 160, "#2a78d6", "complex+NoPE"),
            ("real+none", 224, "#eda100", "real+NoPE d224")]
    for ax, (task, sub) in zip(axes, panels):
        d = df[(df.task == task) & (df.length == 24)]
        for i, (cond, dim, color, label) in enumerate(arms):
            v = d[(d.cond == cond) & (d.dim == dim)].seq_acc.values
            if not len(v):
                continue
            ax.scatter(np.full(len(v), float(i)) + rng.uniform(-0.12, 0.12, len(v)), v, s=16,
                       color=color, zorder=3, edgecolors="white", linewidths=0.5, label=label)
            ax.scatter([i], [v.mean()], marker="_", s=200, color=INK, zorder=4, linewidths=1.3)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(["complex", "real d224"], fontsize=6.5)
        ax.set_title(f"{task}\n{sub}", color=INK, fontweight="bold", fontsize=7.5)
        ax.set_ylim(-0.02, 1.05)
        _style_axis(ax)
    axes[0].set_ylabel("exact-match accuracy @ L=24")
    fig.tight_layout()
    return fig


def fig_needle_long(df):
    """Needle retrieval out to 1024, from the committed checkpoints by inference alone."""
    d = df[df.task == "needle"]
    if not len(d[d.length > 128]):
        return None
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for (cond, dim), (label, color, dash) in SERIES.items():
        g = (d[(d.cond == cond) & (d.dim == dim)]
             .groupby("length")["seq_acc"].agg(["mean", "std"]).reset_index())
        if not len(g) or g.length.max() <= 128:
            continue
        ax.plot(g.length, g["mean"], color=color, linestyle=dash, linewidth=1.4, marker="o",
                markersize=2.6, label=label, dash_capstyle="round")
        ax.fill_between(g.length, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                        color=color, alpha=0.12, linewidth=0)
    ax.axvline(TRAIN_MAX, color=BASE, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(["16", "64", "128", "256", "512", "1024"])
    ax.minorticks_off()
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("haystack length")
    ax.set_ylabel("retrieval accuracy")
    ax.set_title("needle, re-evaluated to 1024", color=INK, fontweight="bold", fontsize=8)
    _style_axis(ax)
    h, l = ax.get_legend_handles_labels()
    ax.legend([_line_only(x) for x in h], l, frameon=False, fontsize=6.2, loc="lower left",
              labelcolor=INK2, handlelength=3.6)
    fig.tight_layout()
    return fig


def fig_probe_depth(metrics_path: Path, L: int = 24):
    """Where the positional rotation survives: angle-fit R^2 against block depth, per run.

    Every NoPE model builds the rotation in its early blocks; only the complex ones still have it at
    the readout. The per-seed lines are kept rather than averaged because the real conditions split.
    """
    if not metrics_path.exists():
        return None
    import json
    m = json.loads(metrics_path.read_text())
    if not any(str(L) in per_len for per_len in m.values()):
        return None
    n_blocks = 0
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for label, per_len in m.items():
        # The RoPE runs in the metrics file are the positive control for the *frequency* statistic
        # (they are handed a fixed-frequency code, so they show omega does not fall like 1/T there).
        # This panel is about NoPE models bootstrapping position with no positional module, so
        # plotting a model that was given one would answer a different question.
        if "positive control" in label:
            continue
        if str(L) not in per_len:
            continue
        rp = per_len[str(L)]["rotating_plane"]
        r2 = [rp[f"L{i}"]["r2"] for i in range(len(rp))]
        n_blocks = max(n_blocks, len(r2))
        # block-diagonal gets its own colour: it is in the complex *field* only as a container, and
        # contains no complex arithmetic, so colouring it as complex would assert the thing the
        # ablation disproved.
        if "diag" in label:
            color, style = "#d1495b", (0, (1, 1.5))
        elif label.startswith("C-"):
            color, style = "#2a78d6", "-"
        else:
            color, style = ("#eda100" if "d160" in label else "#b06a00"), (0, (3.4, 1.5))
        ax.plot(range(len(r2)), r2, color=color, linewidth=1.3, marker="o", markersize=2.6,
                alpha=0.85, linestyle=style)
    ax.plot([], [], color="#2a78d6", linewidth=1.3, label="complex+NoPE")
    ax.plot([], [], color="#d1495b", linewidth=1.3, linestyle=(0, (1, 1.5)), label="block-diagonal")
    ax.plot([], [], color="#eda100", linewidth=1.3, linestyle=(0, (3.4, 1.5)), label="real+NoPE d160")
    ax.plot([], [], color="#b06a00", linewidth=1.3, linestyle=(0, (3.4, 1.5)), label="real+NoPE d224")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(range(n_blocks))
    ax.set_xlabel("residual stream at block input")
    ax.set_ylabel("angle-fit $R^2$ vs position")
    ax.set_title(f"positional rotation by depth (reverse, L={L})", color=INK, fontweight="bold",
                 fontsize=8)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", labelcolor=INK2, handlelength=3.6)
    fig.tight_layout()
    return fig


def fig_clock_segments(metrics_path: Path, min_r2: float = 0.75):
    """The positional clock split at SEP, from experiments.clock_causal (one line per run).

    Left: input-segment angle vs position at the longest evaluation length, in the clock plane fitted
    at the training length. Under a causal mask these states cannot depend on L, and the angle rises
    fast inside the training range and flattens outside it: a saturating absolute code.
    Right: answer-segment sweep vs evaluation length. After SEP the length is knowable, and the sweep
    stays constant: a fractional code. Answer points whose angle fit is below min_r2 are dropped.
    """
    if not metrics_path.exists():
        return None
    import json
    m = json.loads(metrics_path.read_text())
    style = {"complex": ("#2a78d6", "-", "complex+NoPE"),
             "diag": ("#d1495b", (0, (1, 1.5)), "block-diagonal"),
             "d160": ("#eda100", (0, (3.4, 1.5)), "real+NoPE d160"),
             "d224": ("#b06a00", (0, (3.4, 1.5)), "real+NoPE d224")}

    def kind(name, cfg):
        if cfg["field"] == "complex":
            return "diag" if cfg.get("algebra") == "diag" else "complex"
        return "d160" if cfg["dim"] == 160 else "d224"

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for name, rec in m.items():
        if rec["config"].get("steps", 10000) != 10000:
            continue
        color, ls, _ = style[kind(name, rec["config"])]
        seg = rec["segments"]
        ang = seg["input_angle_at_max_len"]
        axl.plot(range(len(ang)), ang, color=color, linestyle=ls, linewidth=1.0, alpha=0.75)
        xs, ys = [], []
        for L, row in sorted(seg["per_length"].items(), key=lambda kv: int(kv[0])):
            if row["answer_fixed"]["r2"] >= min_r2:
                xs.append(int(L)); ys.append(row["answer_fixed"]["sweep_rad"])
        if len(xs) >= 2:
            axr.plot(xs, ys, color=color, linestyle=ls, linewidth=1.0, alpha=0.75, marker="o",
                     markersize=2.4)
    train_max = 16
    axl.axvspan(0, train_max - 1, color=GRID, alpha=0.6, linewidth=0)
    axl.text(train_max / 2, 0.97, "trained\npositions", ha="center", va="top", fontsize=6.5,
             color=INK2, transform=axl.get_xaxis_transform())
    axl.set_xlabel("input position $m$ (evaluation length 64)")
    axl.set_ylabel("clock angle from $m=0$ (rad)")
    axl.set_title("input segment: saturating", color=INK, fontweight="bold", fontsize=8)
    axr.set_xscale("log", base=2)
    axr.set_xticks([16, 24, 32, 48, 64]); axr.set_xticklabels(["16", "24", "32", "48", "64"])
    axr.minorticks_off()
    axr.set_ylim(0, None)
    axr.set_xlabel("evaluation length")
    axr.set_ylabel("answer-segment sweep (rad)")
    axr.set_title("answer segment: fractional", color=INK, fontweight="bold", fontsize=8)
    for ax in (axl, axr):
        _style_axis(ax)
    for key in ("complex", "diag", "d160", "d224"):
        color, ls, label = style[key]
        axr.plot([], [], color=color, linestyle=ls, linewidth=1.2, label=label)
    axr.legend(frameon=False, fontsize=6.2, loc="center right", bbox_to_anchor=(1.0, 0.56),
               labelcolor=INK2, handlelength=3.4)
    fig.tight_layout()
    return fig


# Sequential blue ramp (dataviz reference palette, steps 100->700). Position is an *ordered magnitude*
# here, not an identity, so it gets a one-hue sequential ramp rather than categorical slots.
POS_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def fig_fraction_code(results_dir: str = "results", lengths=(16, 24)):
    """The rotating-plane probe drawn out: what it measures (top row), and what it finds (bottom).

    Top: each dot is one token position's mean residual stream at the deepest block, projected on
    that stream's own top-two principal plane. The probe fits the unwrapped in-plane angle against
    position; that is the whole method.
    Bottom: the same angle, unwrapped, against position. complex+NoPE sweeps the same total angle at
    both evaluation lengths and simply turns more slowly in the longer one, which is a
    fraction-of-sequence code. The real+RoPE control, handed a fixed-frequency code by construction,
    holds its rate and sweeps proportionally further. Nothing is fitted for the figure: these are the
    projections that activation_probe.py scores.
    """
    import matplotlib.colors as mcolors
    import torch
    from experiments.activation_probe import residual_by_position
    from experiments.data.tasks import TaskConfig, make_batch
    from experiments.phase_probe import load_run

    runs = (("complex+NoPE", "unity_reverse_complex_none_d160_s0", "#2a78d6"),
            ("real+RoPE (positive control)", "unity_reverse_real_rope_d160_s0", "#008300"))
    traj = {}
    for label, tag, _ in runs:
        run = load_run(Path(results_dir) / tag, label)
        if run is None:
            return None
        for L in lengths:
            task = TaskConfig(run.task.name, run.task.n_symbols, run.task.train_min,
                              run.task.train_max, run.task.eval_lengths)
            gen = torch.Generator().manual_seed(0)
            X, _, _ = make_batch(task, 512, L, L, device="cpu", generator=gen)
            H = residual_by_position(run.model.eval(), X)[-1]      # deepest block input
            Hc = H - H.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(Hc, full_matrices=False)
            pts = U[:, :2] * S[:2]
            ang = np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))
            m = np.arange(len(ang), dtype=float)
            # the least-squares line the probe actually scores; its rise over the sequence is the
            # `total_sweep_rad` quoted in the text, so the figure and the table cannot drift apart
            A = np.stack([m, np.ones_like(m)], axis=1)
            coef, *_ = np.linalg.lstsq(A, ang, rcond=None)
            traj[(label, L)] = (pts, np.abs(ang - ang[0]), np.abs((A @ coef) - coef[1]))

    cmap = mcolors.LinearSegmentedColormap.from_list("pos", POS_RAMP)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.0))

    for ax, L, tag in zip(axes[0], lengths, "ab"):
        pts = traj[("complex+NoPE", L)][0]
        T = len(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=BASE, linewidth=0.7, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(T), cmap=cmap, s=14, zorder=2,
                   edgecolor="white", linewidth=0.35)
        ax.set_title(f"({tag}) complex+NoPE, length {L}", color=INK, fontweight="bold",
                     fontsize=7.5)
        ax.set_xlabel("PC 1", labelpad=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        # Equal aspect, so a rotation reads as a rotation; explicit padded limits, so the two end
        # labels have room and cannot land on the title or the axis label.
        pad = 0.24 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        ax.set_aspect("equal", adjustable="box")
        for sp in ax.spines.values():
            sp.set_visible(False)
        # Direct labels on the two ends: only the ends of the ramp carry meaning, so a colorbar
        # would spend a legend's worth of space saying less.
        c = pts[:, :2].mean(axis=0)
        for idx in (0, T - 1):
            # push each end label radially outward, so neither lands on the trajectory or the title
            v = pts[idx] - c
            v = v / max(np.hypot(*v), 1e-9) * 13
            ax.annotate(f"pos {idx}", pts[idx], textcoords="offset points", xytext=tuple(v),
                        ha="center", va="center", fontsize=6, color=INK2)
    axes[0, 0].set_ylabel("PC 2", labelpad=1.5)

    for ax, (label, _, color), tag in zip(axes[1], runs, "cd"):
        for L, dash in zip(lengths, ["-", (0, (3.4, 1.5))]):
            _, a, fit = traj[(label, L)]
            m = np.arange(len(a))
            ax.plot(m, a, color=color, linestyle=dash, linewidth=1.4)
            ax.plot(m, fit, color=MUTED, linewidth=0.7, zorder=0)
            ax.annotate(f"{fit[-1]:.1f} rad", (m[-1], fit[-1]), textcoords="offset points",
                        xytext=(-2, 3), ha="right", fontsize=6, color=INK2)
        ax.set_xlabel("token position $m$", labelpad=1.5)
        ax.set_title(f"({tag}) {label.split(' (')[0]}", color=INK, fontweight="bold", fontsize=7.5)
        top = max(traj[(label, L)][1].max() for L in lengths)
        ax.set_ylim(0, top * 1.34)
        _style_axis(ax)
        h = [Line2D([], [], color=color, linewidth=1.4),
             Line2D([], [], color=color, linewidth=1.4, linestyle=(0, (3.4, 1.5))),
             Line2D([], [], color=MUTED, linewidth=0.7)]
        ax.legend(h, [f"eval length {lengths[0]}", f"eval length {lengths[1]}", "fitted line"],
                  frameon=False, fontsize=6, loc="upper left", labelcolor=INK2, handlelength=3.0,
                  borderpad=0.1, labelspacing=0.3)
    axes[1, 0].set_ylabel("angle swept from $m{=}0$ (rad)", labelpad=1.5)
    fig.tight_layout(h_pad=1.0, w_pad=1.4)
    return fig


def main(results_dir: str = "results", out_dir: str = "paper/figs") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_results(results_dir)
    df = df[df.tag.str.startswith("unity_")]
    figs = [
        ("fig_acc_vs_length", fig_acc_vs_length(df)),
        ("fig_per_seed", fig_per_seed(df)),
        ("fig_mechanism", fig_mechanism(df)),
        ("fig_offset_law", fig_offset_law(df)),
        ("fig_needle_long", fig_needle_long(df)),
        ("fig_probe_depth", fig_probe_depth(Path(results_dir) / "_figs/act/activation_metrics.json")),
        ("fig_fraction_code", fig_fraction_code(results_dir)),
        ("fig_clock_segments",
         fig_clock_segments(Path(results_dir) / "_figs/clock/clock_metrics_10k.json")),
    ]
    for name, fig in figs:
        if fig is None:
            print(f"skip {name}: results not present yet")
            continue
        # Pin the embedded creation timestamp. Without this, matplotlib stamps "now" into every
        # PDF/PNG, so a rerun on identical inputs produces a different file and the "figures are
        # byte-reproducible from the committed results" claim in paper/README.md is not checkable.
        fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
        fig.savefig(out / f"{name}.png", metadata={"Software": None})
        print(f"wrote {out / name}.pdf/.png")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
