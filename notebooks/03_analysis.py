"""ComplexWeights dashboard — all results + mechanistic probe + a live attention-retrieval view.

Run:  uv run marimo edit notebooks/03_analysis.py

Sections:
  1. Training results        — accuracy-vs-length, quality/params Pareto, per-seed spread, tables.
  2. Mechanistic probe (H1)  — QK phase spectrum, eigenspectrum, emergent offset kernel.
  3. Attention-retrieval flow — *the cool one*: watch a trained model reach back from each answer
                                position to the input token it copies. Press ▶ to step through it.
  4. Live training            — train a small model in-notebook and stream loss + final attention.
  5. Export (light mode)      — write all figures to results/_figs/ as light-theme HTML.

Plots follow the marimo theme (dark in dark mode) on screen; the Export cell always writes LIGHT HTML.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import plotly.graph_objects as go

    import marimo as mo
    from experiments.analysis import (
        LIGHT,
        fig_acc_vs_length,
        fig_attention_flow,
        fig_attention_heatmap,
        fig_live_loss,
        fig_pareto,
        fig_seed_spread,
        iter_training,
        load_results,
        run_example,
        stability_table,
        summary_table,
        template_for_theme,
    )
    from experiments.phase_probe import TAGSETS, build_probe_figures, load_run

    THEME = mo.app_meta().theme or "light"
    TEMPLATE = template_for_theme(THEME)  # dark on screen in dark mode; export stays LIGHT
    df = load_results()
    return (
        LIGHT,
        Path,
        TAGSETS,
        TEMPLATE,
        THEME,
        build_probe_figures,
        df,
        fig_acc_vs_length,
        fig_attention_flow,
        fig_attention_heatmap,
        fig_live_loss,
        fig_pareto,
        fig_seed_spread,
        iter_training,
        load_run,
        mo,
        run_example,
        stability_table,
        summary_table,
    )


@app.cell
def _(THEME, df, mo):
    mo.md(
        f"# ComplexWeights — dashboard  \n"
        f"**{len(df)} eval rows** · {df['cond'].nunique() if len(df) else 0} conditions · "
        f"{df['task'].nunique() if len(df) else 0} tasks · theme **{THEME}** "
        f"(plots match theme; export is light).\n\n"
        f"---\n### 1 · Training results"
        if len(df)
        else "# No results yet\nRun `uv run python -m experiments.train ...` or "
             "`bash scripts/local_rigorous_sweep.sh` first."
    )
    return


@app.cell
def _(df, mo):
    tasks = sorted(df["task"].unique()) if len(df) else []
    lengths = sorted(int(x) for x in df["length"].unique()) if len(df) else [0]
    task_sel = mo.ui.dropdown(options=tasks, value=tasks[0] if tasks else None, label="task")
    len_sel = mo.ui.dropdown(
        options={str(x): x for x in lengths}, value=str(lengths[-1]), label="eval length"
    )
    mo.hstack([task_sel, len_sel], justify="start") if len(df) else None
    return len_sel, task_sel


@app.cell
def _(TEMPLATE, df, fig_acc_vs_length, task_sel):
    fig_acc_vs_length(df, task=task_sel.value, template=TEMPLATE) if len(df) else None
    return


@app.cell
def _(TEMPLATE, df, fig_pareto, len_sel, task_sel):
    fig_pareto(df, length=len_sel.value, task=task_sel.value, template=TEMPLATE) if len(df) else None
    return


@app.cell
def _(TEMPLATE, df, fig_seed_spread, len_sel):
    # Per-seed spread — the honest view of complex+NoPE bimodality (all tasks pooled).
    fig_seed_spread(df, length=len_sel.value, task=None, template=TEMPLATE) if len(df) else None
    return


@app.cell
def _(mo):
    mo.md("""
    **Stability** — success_rate = fraction of seeds above 0.7 (the metric that exposes collapse).
    """)
    return


@app.cell
def _(df, len_sel, mo, stability_table):
    mo.ui.table(stability_table(df, length=len_sel.value)) if len(df) else None
    return


@app.cell
def _(df, mo, summary_table):
    mo.ui.table(summary_table(df)) if len(df) else None
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ### 2 · Mechanistic probe (H1_mech)
    "
        "QK weight phase spectrum + eigenspectrum (look for RoPE-like banding vs the Trabelsi init null) "
        "and the emergent attention-offset kernel. Pick which trained set to probe:
    """)
    return


@app.cell
def _(TAGSETS, mo):
    probe_tagset = mo.ui.dropdown(options=list(TAGSETS), value="d128", label="probe run set")
    probe_tagset
    return (probe_tagset,)


@app.cell
def _(TAGSETS, TEMPLATE, build_probe_figures, mo, probe_tagset):
    try:
        probe_figs, _probe_metrics = build_probe_figures(
            "results", TAGSETS[probe_tagset.value], template=TEMPLATE
        )
        probe_err = ""
    except SystemExit as e:
        probe_figs, probe_err = {}, str(e)
    mo.vstack(list(probe_figs.values())) if probe_figs else mo.md(f"⚠️ {probe_err}")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ### 3 · Attention-retrieval flow  ✨
    "
        "Top row = input tokens (keys), bottom row = answer tokens (queries). Each answer position draws "
        "edges to the input positions it attends to (width/opacity = attention probability). Press "
        "**▶ play retrieval** to step through generation order and watch it *transition* to the token it "
        "copies — **copy** → parallel diagonals, **reverse** → crossing lines, **needle** → a fan onto MARK+1.
    """)
    return


@app.cell
def _(Path, mo):
    tags = sorted(p.parent.name for p in Path("results").glob("*/model.pt"))
    default = next((t for t in tags if "complex_none" in t), tags[0] if tags else None)
    flow_tag = mo.ui.dropdown(options=tags, value=default, label="trained model")
    flow_len = mo.ui.slider(6, 48, value=12, label="example length", show_value=True)
    flow_seed = mo.ui.number(0, 999, value=3, label="example seed")
    flow_layer = mo.ui.dropdown(
        options=["last", "mean", "0", "1", "2", "3", "4", "5"], value="last", label="layer"
    )
    flow_head = mo.ui.dropdown(options=["mean", "0", "1", "2", "3"], value="mean", label="head")
    controls = mo.hstack([flow_tag, flow_len, flow_seed, flow_layer, flow_head], justify="start")
    controls if tags else mo.md("⚠️ No `model.pt` checkpoints yet — train with `--save-model` first.")
    return flow_head, flow_layer, flow_len, flow_seed, flow_tag


@app.cell
def _(
    TEMPLATE,
    fig_attention_flow,
    fig_attention_heatmap,
    flow_head,
    flow_layer,
    flow_len,
    flow_seed,
    flow_tag,
    load_run,
    mo,
    run_example,
):
    if flow_tag.value is None:
        _out = mo.md("No checkpoint selected.")
    else:
        _r = load_run(f"results/{flow_tag.value}")
        _ex = run_example(_r.model, _r.task, length=int(flow_len.value), seed=int(flow_seed.value))
        _ttl = (f"{_r.cfg.field}+{_r.cfg.pe} · {_r.task.name} · L={flow_len.value} · "
                f"token_acc={_ex['token_acc']:.2f}")
        _flow = fig_attention_flow(_ex, layer=flow_layer.value, head=flow_head.value,
                                   \
                                   title=_ttl, template=TEMPLATE)
        _heat = fig_attention_heatmap(_ex, layer=flow_layer.value, head=flow_head.value, template=TEMPLATE)
        _out = mo.vstack([_flow, _heat])
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ### 4 · Live training  ⏱️
    "
        "Train a small model in-notebook and stream the loss; the final attention-flow renders when it "
        "finishes. Small by design (so it's quick on the GTX 1050 even while a sweep runs).
    """)
    return


@app.cell
def _(mo):
    live_task = mo.ui.dropdown(options=["copy", "reverse", "sort", "needle"], value="copy", label="task")
    live_cond = mo.ui.dropdown(
        options={"complex+none": ("complex", "none"), "complex+rope": ("complex", "rope"),
                 "real+none": ("real", "none"), "real+rope": ("real", "rope")},
        value="complex+none", label="condition",
    )
    live_dim = mo.ui.slider(32, 96, value=48, step=16, label="dim", show_value=True)
    live_steps = mo.ui.slider(100, 1500, value=500, step=100, label="steps", show_value=True)
    run_live = mo.ui.run_button(label="▶ train live")
    mo.hstack([live_task, live_cond, live_dim, live_steps, run_live], justify="start")
    return live_cond, live_dim, live_steps, live_task, run_live


@app.cell
def _(
    TEMPLATE,
    fig_attention_flow,
    fig_live_loss,
    iter_training,
    live_cond,
    live_dim,
    live_steps,
    live_task,
    mo,
    run_example,
    run_live,
):
    if run_live.value:
        _field, _pe = live_cond.value
        _last = None
        for _snap in iter_training(field=_field, pe=_pe, task_name=live_task.value,
                                   dim=int(live_dim.value), steps=int(live_steps.value),
                                   train_min=4, train_max=12, eval_lengths=(12, 24), every=20):
            _last = _snap
            mo.output.replace(fig_live_loss(_snap, TEMPLATE))
        _ex = run_example(_last["model"], _last["task"], length=_last["task"].train_max, seed=0)
        _ttl = f"{_field}+{_pe} · {live_task.value} · final token_acc={_ex['token_acc']:.2f}"
        mo.output.replace(mo.vstack([
            fig_live_loss(_last, TEMPLATE),
            fig_attention_flow(_ex, title=_ttl, template=TEMPLATE),
        ]))
        _out = None
    else:
        _out = mo.md("Press **▶ train live** to start.")
    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ### 5 · Export figures (light mode)
    "
        "Writes the dashboard + probe figures to `results/_figs/` as **light-theme** HTML, regardless of "
        "the on-screen theme.
    """)
    return


@app.cell
def _(mo):
    export_btn = mo.ui.run_button(label="💾 export light-mode HTML → results/_figs/")
    export_btn
    return (export_btn,)


@app.cell
def _(
    LIGHT,
    Path,
    TAGSETS,
    build_probe_figures,
    df,
    export_btn,
    fig_acc_vs_length,
    fig_pareto,
    fig_seed_spread,
    mo,
    probe_tagset,
):
    if export_btn.value and len(df):
        _out = Path("results/_figs")
        _out.mkdir(parents=True, exist_ok=True)
        _written = []
        _longest = int(df["length"].max())
        for _task in sorted(df["task"].unique()):
            fig_acc_vs_length(df, task=_task, template=LIGHT).write_html(_out / f"acc_vs_length_{_task}.html")
            fig_pareto(df, length=_longest, task=_task, template=LIGHT).write_html(_out / f"pareto_{_task}.html")
            _written += [f"acc_vs_length_{_task}", f"pareto_{_task}"]
        fig_seed_spread(df, length=_longest, task=None, template=LIGHT).write_html(_out / "seed_spread.html")
        _sub = "" if probe_tagset.value == "d64" else probe_tagset.value
        _pdir = _out / _sub if _sub else _out
        _pdir.mkdir(parents=True, exist_ok=True)
        try:
            _pf, _ = build_probe_figures("results", TAGSETS[probe_tagset.value], template=LIGHT)
            for _n, _f in _pf.items():
                _f.write_html(_pdir / f"{_n}.html")
            _written += list(_pf)
        except SystemExit:
            pass
        _msg = mo.md(f"✅ wrote {len(_written)} light-mode figures → `{_out}/`")
    else:
        _msg = mo.md("_(idle)_")
    _msg
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
