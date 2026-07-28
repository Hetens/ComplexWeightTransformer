"""Load results/*/result.json into a DataFrame and build plotly figures.

Thin and importable so both the marimo dashboard (notebooks/03_analysis.py) and quick CLI summaries
share the same logic.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

LIGHT, DARK = "plotly_white", "plotly_dark"


def template_for_theme(theme: str | None) -> str:
    """Map a marimo theme ('light'/'dark') to a plotly template. Used so the notebook shows dark
    plots while saved HTML stays light (always pass LIGHT to the export path)."""
    return DARK if (theme or "light") == "dark" else LIGHT


def load_results(results_dir: str = "results") -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(Path(results_dir) / "*" / "result.json")):
        d = json.loads(Path(f).read_text())
        c = d["config"]
        base = dict(
            tag=Path(f).parent.name, field=c["field"], pe=c["pe"], dim=c["dim"],
            task=c["task"], seed=c["seed"], learn_imag=bool(c.get("learn_imag", False)),
            algebra=str(c.get("algebra", "complex")),  # pre-algebra runs predate the field
            lr=c.get("lr", float("nan")),
            params=d["real_params"], final_loss=d["final_loss"],
            # In-distribution convergence gate. Length generalization is only interpretable for a
            # model that actually learned the task at the trained length; averaging a diverged run
            # into a capability mean reports an optimization failure as a capability limit. The
            # committed sweep passes this for all 160 runs, so it changes nothing there, but the
            # wider real baselines do NOT all converge at the d160-tuned learning rate.
            converged=bool(d["evals"].get(str(c["train_max"]), {}).get("token_acc", 0.0) >= 0.99),
        )
        for L, ev in d["evals"].items():
            rows.append({**base, "length": int(L), "token_acc": ev["token_acc"], "seq_acc": ev["seq_acc"]})
    df = pd.DataFrame(rows)
    if len(df):
        df["cond"] = df["field"] + "+" + df["pe"]
        # learn_imag is part of the complex recipe; the ablation runs must not pool with the default
        noimag = (df["field"] == "complex") & (~df["learn_imag"])
        df.loc[noimag, "cond"] += "-noimag"
        # likewise the weight-tying algebra: split/dual/diag are separate conditions, not complex
        alg = (df["field"] == "complex") & (df["algebra"] != "complex")
        df.loc[alg, "cond"] += "-" + df.loc[alg, "algebra"]
    return df


def fig_acc_vs_length(df: pd.DataFrame, task: str | None = None, metric: str = "token_acc",
                      template: str = LIGHT) -> go.Figure:
    d = df if task is None else df[df.task == task]
    d = d.groupby(["cond", "length"], as_index=False)[metric].mean()
    fig = go.Figure()
    for cond, g in d.groupby("cond"):
        g = g.sort_values("length")
        fig.add_scatter(x=g.length, y=g[metric], mode="lines+markers", name=cond)
    fig.update_layout(
        title=f"{metric} vs eval length ({task or 'all tasks'})",
        xaxis_title="input length (train_max marks in/out of distribution)",
        yaxis_title=metric, yaxis_range=[0, 1.02], template=template,
    )
    return fig


def fig_pareto(df: pd.DataFrame, length: int, task: str | None = None, template: str = LIGHT) -> go.Figure:
    d = df[df.length == length]
    d = d if task is None else d[d.task == task]
    d = d.groupby(["cond", "dim"], as_index=False).agg(token_acc=("token_acc", "mean"), params=("params", "mean"))
    fig = go.Figure()
    for cond, g in d.groupby("cond"):
        g = g.sort_values("params")
        fig.add_scatter(x=g.params, y=g.token_acc, mode="markers+lines+text", name=cond,
                        text=[f"d{int(x)}" for x in g.dim], textposition="top center")
    fig.update_layout(
        title=f"Quality vs parameters @ length {length} ({task or 'all'})",
        xaxis_title="real parameters", yaxis_title="token accuracy",
        yaxis_range=[0, 1.02], template=template,
    )
    return fig


def fig_seed_spread(df: pd.DataFrame, length: int, task: str | None = None, template: str = LIGHT) -> go.Figure:
    """Per-seed accuracy as a box+points plot -- the honest view of complex+NoPE bimodality."""
    d = df[df.length == length]
    d = d if task is None else d[d.task == task]
    fig = go.Figure()
    for cond, g in d.groupby("cond"):
        fig.add_box(y=g.token_acc, name=cond, boxpoints="all", jitter=0.5, pointpos=0)
    fig.update_layout(
        title=f"Per-seed token accuracy @ length {length} ({task or 'all'})",
        yaxis_title="token accuracy", yaxis_range=[0, 1.02], template=template, showlegend=False,
    )
    return fig


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    t = (
        df.pivot_table(index=["task", "cond", "dim"], columns="length", values="token_acc", aggfunc="mean")
        .round(3)
        .reset_index()
    )
    # length columns come out as ints; stringify so marimo's table renderer is happy
    t.columns = [f"acc@{c}" if isinstance(c, (int, float)) else c for c in t.columns]
    return t


def stability_table(df: pd.DataFrame, length: int, threshold: float = 0.7) -> pd.DataFrame:
    """Mean/median/min/max + success_rate (fraction of seeds above threshold) per condition.

    Mean alone is misleading for bimodal complex+NoPE; success_rate exposes how often it actually converges.
    Grouped by dim as well as cond so the d160 and d224 (param-matched) real baselines never pool.
    """
    d = df[df.length == length]
    g = d.groupby(["task", "cond", "dim"])["token_acc"]
    return (
        g.agg(
            mean="mean", median="median", min="min", max="max",
            n="count", success_rate=lambda s: float((s > threshold).mean()),
        )
        .round(3)
        .reset_index()
    )


# --------------------------------------------------------------------------- attention extraction
@torch.no_grad()
def attention_weights(model, X: torch.Tensor) -> np.ndarray:
    """Post-softmax causal attention for one sequence: returns (n_layers, n_heads, S, S).

    Each attention block's `attn_logits` needs the *hidden state* it actually receives (the complex
    residual stream for complex models), not token ids -- so we grab each block's real input with a
    forward pre-hook during one forward pass (same approach as phase_probe.attention_probe), then apply
    the causal mask + softmax exactly as forward() does. Works for both real and complex fields. Uses
    example 0 if X has B>1.
    """
    model.eval()
    dev = next(model.parameters()).device
    X = X.to(dev)
    attns = [blk.attn for blk in model.blocks]
    grabbed: list[torch.Tensor] = []
    handles = [a.register_forward_pre_hook(lambda m, inp: grabbed.append(inp[0].detach())) for a in attns]
    model(X)
    for h in handles:
        h.remove()
    mats = []
    for a, xin in zip(attns, grabbed):
        lg = a.attn_logits(xin)  # (B, H, S, S)
        S = lg.shape[-1]
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=lg.device), diagonal=1)
        w = lg.masked_fill(mask, float("-inf")).softmax(dim=-1)
        mats.append(w[0].float().cpu().numpy())
    return np.stack(mats, axis=0)  # (L, H, S, S)


@torch.no_grad()
def run_example(model, task, length: int, seed: int = 0) -> dict:
    """Run one task example through the model and package everything the attention-flow figure needs:
    token ids, the SEP split, which positions are answer targets, predictions + correctness, attention."""
    from experiments.data.tasks import MARK, SEP, make_batch

    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    X, Y, M = make_batch(task, 1, length, length, generator=gen)
    pred = model(X.to(dev)).argmax(-1)[0].cpu()
    x, y, m = X[0], Y[0], M[0].bool()
    attn = attention_weights(model, X)  # (L, H, S, S)
    sep_pos = int((x == SEP).nonzero()[0].item()) if (x == SEP).any() else len(x)
    ans_pos = m.nonzero().flatten().tolist()  # query positions that emit the answer
    correct = ((pred == y) & m).tolist()
    return {
        "tokens": x.tolist(), "target": y.tolist(), "pred": pred.tolist(),
        "ans_pos": ans_pos, "sep_pos": sep_pos, "correct": correct, "attn": attn,
        "mark_pos": int((x == MARK).nonzero()[0].item()) if (x == MARK).any() else None,
        "token_acc": float(((pred == y) & m).sum() / max(int(m.sum()), 1)),
    }


def _tok_label(tid: int) -> str:
    from experiments.data.tasks import MARK, PAD, SEP

    return {PAD: "·", SEP: "SEP", MARK: "MARK"}.get(tid, str(tid))


def pool_attention(attn: np.ndarray, layer: int | str = "last", head: int | str = "mean") -> np.ndarray:
    """Reduce (L,H,S,S) to a single (S,S) attention map. layer: int | 'last' | 'mean'; head: int | 'mean'.
    Integer layer/head are clamped to the valid range so notebook dropdowns can't crash on small models."""
    L, H = attn.shape[0], attn.shape[1]
    if layer == "mean":
        a = attn.mean(0)
    else:
        li = L - 1 if layer == "last" else min(max(int(layer), 0), L - 1)
        a = attn[li]  # (H,S,S)
    return a.mean(0) if head == "mean" else a[min(max(int(head), 0), H - 1)]


def fig_attention_flow(ex: dict, layer: int | str = "last", head: int | str = "mean",
                       topk: int = 4, thresh: float = 0.04, animate: bool = True,
                       title: str = "", template: str = LIGHT) -> go.Figure:
    """Animated 'retrieval' graph: how each answer position reaches back to the input it copies.

    Two rows of nodes at their true sequence positions -- keys/input on top (y=1), answer queries on
    the bottom (y=0). An edge q->k is drawn when attention[q,k] is among q's top-k (and > thresh); its
    width/opacity encode the attention probability. The Play button steps through answer positions in
    generation order, lighting one query's edges at a time -- so you literally watch the model 'transition'
    to the token it attends to. Copy -> parallel diagonals; reverse -> crossing lines; needle -> a fan
    onto MARK+1.
    """
    A = pool_attention(ex["attn"], layer, head)  # (S,S)
    toks, sep, ans = ex["tokens"], ex["sep_pos"], ex["ans_pos"]
    S = len(toks)
    keys = list(range(sep))  # input positions are the meaningful retrieval targets

    # per-query edge list to its top-k attended (input) keys
    def edges_for(q: int):
        row = A[q, : max(sep, 1)].copy()
        order = np.argsort(row)[::-1][:topk]
        return [(q, int(k), float(row[k])) for k in order if row[k] > thresh]

    # static node layer
    key_node = go.Scatter(
        x=keys, y=[1] * len(keys), mode="markers+text",
        text=[_tok_label(toks[k]) for k in keys], textposition="top center",
        marker=dict(size=16, color="#4C78A8", line=dict(width=1, color="white")),
        name="input (keys)", hovertemplate="input pos %{x}<extra></extra>",
    )
    q_text = [f"{_tok_label(ex['pred'][q])}{'✓' if ex['correct'][q] else '✗'}" for q in ans]
    q_node = go.Scatter(
        x=ans, y=[0] * len(ans), mode="markers+text",
        text=q_text, textposition="bottom center",
        marker=dict(size=16, symbol="square",
                    color=["#54A24B" if ex["correct"][q] else "#E45756" for q in ans],
                    line=dict(width=1, color="white")),
        name="answer (queries)", hovertemplate="answer pos %{x}<extra></extra>",
    )

    def edge_traces(qs, faint):
        ex_x, ex_y, widths = [], [], []
        segs = []
        for q in qs:
            for (_, k, w) in edges_for(q):
                segs.append((q, k, w))
        # one scatter per edge so width can vary; opacity scales with weight
        traces = []
        for (q, k, w) in segs:
            traces.append(go.Scatter(
                x=[q, k], y=[0, 1], mode="lines",
                line=dict(width=1 + 5 * w, color="#888" if faint else "#F58518"),
                opacity=(0.15 if faint else 0.25 + 0.75 * w), showlegend=False,
                hovertemplate=f"attn {w:.2f}: ans {q} → in {k}<extra></extra>",
            ))
        return traces

    base = [key_node, q_node, *edge_traces(ans, faint=True)]
    fig = go.Figure(data=base)

    if animate and ans:
        frames = []
        for q in ans:
            hi = edge_traces([q], faint=False)
            cur = go.Scatter(x=[q], y=[0], mode="markers", marker=dict(size=22, symbol="square-open",
                             color="#F58518", line=dict(width=3)), showlegend=False, hoverinfo="skip")
            frames.append(go.Frame(name=str(q), data=[key_node, q_node, cur, *hi]))
        fig.frames = frames
        fig.update_layout(
            updatemenus=[dict(type="buttons", showactive=False, x=0, y=1.15, xanchor="left",
                buttons=[
                    dict(label="▶ play retrieval", method="animate",
                         args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True)]),
                    dict(label="❚❚ pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ])],
            sliders=[dict(active=0, x=0.15, len=0.85, currentvalue=dict(prefix="answer pos "),
                          steps=[dict(method="animate", label=str(q),
                                      args=[[str(q)], dict(mode="immediate",
                                            frame=dict(duration=0, redraw=True))]) for q in ans])],
        )

    fig.add_vline(x=sep - 0.5, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=title or "Attention-retrieval flow (input keys → answer queries)",
        template=template, height=460, hovermode="closest",
        yaxis=dict(showgrid=False, zeroline=False, tickvals=[0, 1],
                   ticktext=["answer", "input"], range=[-0.5, 1.5]),
        xaxis=dict(title="sequence position", showgrid=False, range=[-1, S]),
    )
    return fig


def fig_attention_heatmap(ex: dict, layer: int | str = "last", head: int | str = "mean",
                          title: str = "", template: str = LIGHT) -> go.Figure:
    """Standard query×key attention heatmap for one sequence (complements the flow graph)."""
    A = pool_attention(ex["attn"], layer, head)
    labels = [f"{i}:{_tok_label(t)}" for i, t in enumerate(ex["tokens"])]
    fig = go.Figure(go.Heatmap(z=A, x=labels, y=labels, colorscale="Viridis",
                               colorbar=dict(title="attn"), zmin=0))
    fig.add_vline(x=ex["sep_pos"] - 0.5, line_dash="dot", line_color="white")
    fig.add_hline(y=ex["sep_pos"] - 0.5, line_dash="dot", line_color="white")
    fig.update_layout(
        title=title or f"Attention heatmap (layer={layer}, head={head})",
        template=template, height=520, xaxis_title="key position", yaxis_title="query position",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# --------------------------------------------------------------------------- live training (for marimo)
def iter_training(field: str = "complex", pe: str = "none", task_name: str = "copy", dim: int = 48,
                  n_layers: int = 2, n_heads: int = 4, steps: int = 600, batch_size: int = 64,
                  lr: float = 2e-3, train_min: int = 4, train_max: int = 10,
                  eval_lengths: tuple[int, ...] = (10, 20), learn_imag: bool = True, seed: int = 0,
                  device: str = "auto", every: int = 25):
    """Generator that trains a small model and yields a progress snapshot every `every` steps.

    Pure torch (no marimo) so it stays importable/testable; the notebook consumes it and streams the
    yielded figures with mo.output.replace. Each snapshot: step/total, loss, loss-history, a quick
    in-distribution token-acc, done flag, and the live model+task (for the final attention-flow view).
    """
    import torch
    import torch.nn.functional as F

    from complexnn.models import ModelConfig, TransformerLM
    from experiments.data.tasks import TaskConfig, evaluate, make_batch

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    torch.manual_seed(seed)
    task = TaskConfig(task_name, 16, train_min, train_max, eval_lengths)
    model = TransformerLM(ModelConfig(
        vocab_size=task.vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads, field=field, pe=pe,
        learn_imag=(field == "complex" and learn_imag))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(1234 + seed)
    history: list[dict] = []
    for step in range(steps):
        X, Y, M = make_batch(task, batch_size, train_min, train_max, device=dev, generator=gen)
        logits = model(X).reshape(-1, task.vocab_size)
        m = M.reshape(-1)
        loss = F.cross_entropy(logits[m], Y.reshape(-1)[m])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % every == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.item())})
            acc = evaluate(model, task, train_max, 128, dev)["token_acc"]
            yield {"step": step, "total": steps, "loss": float(loss.item()), "history": list(history),
                   "acc": acc, "done": step == steps - 1, "model": model, "task": task}


def fig_live_loss(snap: dict, template: str = LIGHT) -> go.Figure:
    """Loss curve (+ in-dist accuracy) from a live-training snapshot."""
    h = snap["history"]
    steps = [d["step"] for d in h]
    fig = go.Figure()
    fig.add_scatter(x=steps, y=[d["loss"] for d in h], mode="lines", name="loss", line=dict(color="#F58518"))
    fig.update_layout(
        title=(f"live training — step {snap['step']}/{snap['total'] - 1}   "
               f"loss={snap['loss']:.3f}   acc@train_max={snap['acc']:.2f}"),
        template=template, height=320, xaxis_title="step", yaxis_title="train loss",
    )
    return fig
