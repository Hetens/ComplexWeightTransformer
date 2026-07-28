"""Statistical summary of the multi-seed sweep, aligned to the pre-registration in docs/hypothesis.md.

Every number in paper/main.tex Tables 1-2 and the claim verdicts in docs/results.md comes from here:

    uv run python -m experiments.stats                    # console summary
    uv run python -m experiments.stats --out-md paper/numbers.md

What it reports:
  1. mean +/- std over seeds of seq_acc and token_acc, per task x condition x dim x length
     (dim is part of the key so the d160 and d224 param-matched baselines never pool).
  2. The pre-registered H1_cap test at 2L (= 2 * train_max = 32): task-averaged
     delta(C-NoPE - R-NoPE) against the pooled seed std, plus per-task Welch t-tests
     (exact Student-t p via the regularized incomplete beta; no scipy).
  3. The named claim checks: reverse gap, needle capacity control, RoPE-enhancer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from experiments.analysis import load_results


# --------------------------------------------------------------------------- exact Welch t-test
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes 6.4)."""
    eps, fpmin, maxit = 3e-12, 1e-300, 200
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


@dataclass
class Welch:
    delta: float
    t: float
    dof: float
    p: float  # two-sided, exact Student-t
    mean_a: float
    mean_b: float


def welch_t(a, b) -> Welch:
    """Two-sided Welch t-test with exact Student-t p-value (Welch-Satterthwaite dof).

    Returns NaNs rather than raising when either arm is empty, so a report that spans conditions
    which do not all cover the same tasks degrades to a blank row instead of dying.
    """
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        nan = float("nan")
        return Welch(nan, nan, nan, nan, a.mean() if na else nan, b.mean() if nb else nan)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    if se2 == 0:
        return Welch(ma - mb, float("inf") if ma != mb else 0.0, float(na + nb - 2),
                     0.0 if ma != mb else 1.0, ma, mb)
    t = (ma - mb) / math.sqrt(se2)
    dof = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = _betai(dof / 2.0, 0.5, dof / (dof + t * t))
    return Welch(ma - mb, t, dof, p, ma, mb)


# --------------------------------------------------------------------------- report blocks
def cond_key(df: pd.DataFrame) -> pd.Series:
    return df["cond"] + "_d" + df["dim"].astype(str)


def table_mean_std(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """task x condition rows, one 'mean+/-std' string column per eval length."""
    d = df.copy()
    d["ck"] = cond_key(d)
    g = d.groupby(["task", "ck", "length"])[metric].agg(["mean", "std"])
    s = g["mean"].round(2).astype(str) + "±" + g["std"].fillna(0).round(2).astype(str)
    return s.unstack("length")


# H1_cap was registered as a task-*averaged* test over exactly these four tasks. Later rounds add
# tasks (revskip, skipcopy) that only ever ran the complex-vs-matched-real pair, so including them
# would both crash on the missing d160 arm and, worse, silently redefine a pre-registered number.
# Pinning the set keeps the registered verdict comparable across rounds.
PREREG_TASKS = ("copy", "needle", "reverse", "sort")

# Same hazard on a second axis. The condition key is (field+pe, dim) and does not mention the learning
# rate, so once the follow-up rounds swept lr, those runs began pooling into the very conditions the
# committed claims are stated over, shifting a pre-registered number by ~0.003 without any warning.
# The committed sweep is entirely at 2e-3; the primary report is pinned to it and the lr grid is
# reported only where it is the subject (followup_checks -> H2_width).
CANONICAL_LR = 2e-3


def preregistered_h1cap(df: pd.DataFrame, length: int, metric: str, baseline: str) -> list[str]:
    """H1_cap (docs/hypothesis.md): task-mean delta vs pooled seed std at the primary length point."""
    lines = []
    deltas, pooled = [], []
    for task in [t for t in PREREG_TASKS if t in set(df.task.unique())]:
        a = df[(df.task == task) & (cond_key(df) == "complex+none_d160") & (df.length == length)][metric]
        b = df[(df.task == task) & (cond_key(df) == baseline) & (df.length == length)][metric]
        w = welch_t(a, b)
        deltas.append(w.delta)
        pooled.append((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        lines.append(f"    {task:8s} delta={w.delta:+.3f}  t={w.t:6.2f} (dof {w.dof:.1f})  p={w.p:.4f}"
                     f"  (C={w.mean_a:.3f}, R={w.mean_b:.3f})")
    mean_delta = sum(deltas) / len(deltas)
    pooled_std = math.sqrt(sum(pooled) / len(pooled))
    verdict = "SUPPORTED" if mean_delta > pooled_std else "NOT SUPPORTED"
    lines.insert(0, f"  vs {baseline}: task-mean delta={mean_delta:+.3f}, pooled seed std={pooled_std:.3f}"
                    f" -> {verdict} (bar: delta > pooled std)")
    return lines


def claim_check(df: pd.DataFrame, task: str, a_key: str, b_key: str, length: int,
                metric: str = "seq_acc") -> str:
    a = df[(df.task == task) & (cond_key(df) == a_key) & (df.length == length)][metric]
    b = df[(df.task == task) & (cond_key(df) == b_key) & (df.length == length)][metric]
    w = welch_t(a, b)
    return (f"  {task} @L={length}: {a_key}={w.mean_a:.3f}±{a.std():.2f} vs {b_key}={w.mean_b:.3f}±{b.std():.2f}"
            f"  delta={w.delta:+.3f}  t={w.t:.2f}  p={w.p:.4f}")


def _best_lr(d: pd.DataFrame, **eq) -> tuple[float, pd.Series]:
    """The (lr, seq_acc values) of the best-scoring learning rate for one condition, converged only.

    Comparing a method at its tuned rate against a baseline at an untuned one is the easiest way to
    manufacture a result, so every cross-condition comparison below picks each side's own best.
    """
    sel = d[d.converged]
    for k, v in eq.items():
        sel = sel[sel[k] == v]
    cands = [(lr, sel[sel.lr == lr]["seq_acc"]) for lr in sorted(sel.lr.dropna().unique())]
    cands = [(lr, v) for lr, v in cands if len(v) >= 3]
    if not cands:
        return float("nan"), pd.Series(dtype=float)
    return max(cands, key=lambda t: t[1].mean())


def followup_checks(full: pd.DataFrame) -> list[str]:
    """The 2026-07-26 follow-up pre-registration (docs/hypothesis.md): H2_alg, H2_width, H3_offset,
    H4_needle. Each block is emitted only if its runs are present, so this stays a no-op on the
    original committed sweep.

    Takes the *unfiltered* frame because H2_width's subject is the learning-rate grid, but every other
    block is a statement about the committed recipe and so uses the canonical rate only. Mixing the
    two silently shifts complex+NoPE from its 0.856 to a cross-lr pooled 0.838.
    """
    df = full[(full.lr == CANONICAL_LR) | full.lr.isna()]
    keys = set(cond_key(df).unique())
    lines: list[str] = []
    p = lines.append

    # H2_alg: is the reverse effect C, or structured weight sharing? All tyings share width+params.
    algs = [(a, f"complex+none-{a}_d160") for a in ("split", "dual", "diag")]
    if any(k in keys for _, k in algs):
        p("\n## H2_alg: weight-tying algebra ablation on reverse (seq_acc @24)\n")
        p("  (all arms: identical activation width 320 and identical 3.09M real params)")
        for _, k in algs:
            if k in keys:
                p(claim_check(df, "reverse", "complex+none_d160", k, 24))
        for _, k in algs:
            if k in keys:
                p(claim_check(df, "reverse", k, "real+none_d224", 24))

    # H2_width: can real width buy the gap at all? The committed curve is flat from d160 to d224.
    wide = [d for d in (320, 448, 640) if f"real+none_d{d}" in set(cond_key(full).unique())]
    if wide:
        rev = full[(full.task == "reverse") & (full.length == 24)]
        p("\n## H2_width: real+NoPE width scan on reverse (seq_acc @24)\n")
        p("  Two gates, both of which change the answer:")
        p("   (a) converged only. lr=2e-3 was tuned at d160 and diverges at the widest baselines;")
        p("       scoring an optimization failure as a capability limit would support H2_width.")
        p("   (b) each arm at its OWN best lr over the same grid. Comparing a tuned method against")
        p("       an untuned baseline is the easiest way to manufacture a result.")
        c_lr, a = _best_lr(rev, cond="complex+none", dim=160)
        p(f"\n  reference: complex+none_d160 best-tuned = {a.mean():.3f} +/- {a.std():.3f} "
          f"at lr={c_lr:.0e} (3.09M params, width 320, n={len(a)})")
        p("\n  full grid (converged/total, converged mean +/- std):")
        rows = (rev[rev.cond.isin(["real+none", "complex+none"])]
                .groupby(["cond", "dim", "lr"], as_index=False)
                .agg(n=("seq_acc", "size"), params=("params", "mean")))
        for r in rows.sort_values(["cond", "dim", "lr"]).itertuples():
            ok = rev[rev.converged & (rev.cond == r.cond) & (rev.dim == r.dim)
                     & (rev.lr == r.lr)]["seq_acc"]
            mean = f"{ok.mean():.3f}" if len(ok) else "  nan"
            sd = ok.std() if len(ok) > 1 else 0.0
            p(f"    {r.cond:13s} d{int(r.dim):<4d} lr={r.lr:<7.0e} {r.params / 1e6:5.2f}M  "
              f"{len(ok)}/{r.n}  {mean} +/- {sd:.3f}")
        p("\n  best-tuned real per width vs best-tuned complex:")
        for dim in sorted(rev[rev.cond == "real+none"].dim.unique()):
            lr, b = _best_lr(rev, cond="real+none", dim=dim)
            if not len(b):
                p(f"    d{int(dim):<4d}: no learning rate produced 3+ converged seeds")
                continue
            w = welch_t(a, b)
            par = rev[(rev.cond == "real+none") & (rev.dim == dim)]["params"].mean()
            p(f"    d{int(dim):<4d} {par / 1e6:5.2f}M ({par / 3085920:4.1f}x)  lr={lr:.0e}  "
              f"{b.mean():.3f} +/- {b.std():.3f} (n={len(b)})  delta={w.delta:+.3f}  p={w.p:.4f}")

    # H3_offset: complex should win the end-anchored task and not the start-anchored one.
    for task, kind in (("revskip", "end-anchored, offset -(3j+1), length-INdependent"),
                       ("skipcopy", "start-anchored, offset j-L, length-dependent")):
        if (df.task == task).any():
            p(f"\n## H3_offset: {task} ({kind}), seq_acc\n")
            for L in (24, 32):
                if ((df.task == task) & (df.length == L)).any():
                    p(claim_check(df, task, "complex+none_d160", "real+none_d224", L))

    # H4_needle: pushed on the length axis rather than the seed axis (experiments/reeval.py).
    long_lengths = [L for L in (256, 512, 1024) if ((df.task == "needle") & (df.length == L)).any()]
    if long_lengths:
        p("\n## H4_needle: re-eval at long haystacks (seq_acc), inference only\n")
        for L in long_lengths:
            p(claim_check(df, "needle", "complex+none-noimag_d160", "real+none_d224", L))
            p(claim_check(df, "needle", "complex+none_d160", "real+none_d224", L))
    return lines


def main(results_dir: str = "results", tag_prefix: str = "unity_", train_max: int = 16,
         out_md: str = "") -> None:
    full = load_results(results_dir)
    full = full[full.tag.str.startswith(tag_prefix)]
    if not len(full):
        raise SystemExit(f"no results matching {tag_prefix}* under {results_dir}/")
    # Everything above the follow-up section is a statement about the committed recipe, so it must see
    # only the canonical learning rate. followup_checks gets `full` because the lr grid is its subject.
    df = full[(full.lr == CANONICAL_LR) | full.lr.isna()]
    two_l = 2 * train_max
    out: list[str] = []
    p = out.append

    n_runs = df.tag.nunique()
    p(f"{full.tag.nunique()} runs total; {n_runs} at the canonical lr={CANONICAL_LR:g} "
      f"(the rest are the follow-up lr grid, reported under H2_width)")
    p(f"tasks={sorted(df.task.unique())} | conditions={sorted(cond_key(df).unique())}")

    for metric in ("seq_acc", "token_acc"):
        p(f"\n## mean±std {metric} by eval length (seeds pooled per condition)\n")
        p(table_mean_std(df, metric).to_string())

    p(f"\n## Pre-registered H1_cap at 2L={two_l} (docs/hypothesis.md success threshold)\n")
    for metric in ("token_acc", "seq_acc"):
        p(f"  [{metric}]")
        for baseline in ("real+none_d160", "real+none_d224"):
            out.extend(preregistered_h1cap(df, two_l, metric, baseline))

    p("\n## Claim checks (seq_acc, Welch t)\n")
    p("  -- reverse: the surviving positive claim --")
    p(claim_check(df, "reverse", "complex+none_d160", "real+none_d160", 24))
    p(claim_check(df, "reverse", "complex+none_d160", "real+none_d224", 24))
    p(claim_check(df, "reverse", "complex+none_d160", "real+none_d224", 32))
    p("  -- sort: the dead 10x headline (parity at 5 seeds) --")
    p(claim_check(df, "sort", "complex+none_d160", "real+none_d160", 96))
    p(claim_check(df, "sort", "complex+none_d160", "real+none_d224", 96))
    p("  -- needle: complex advantage is a capacity artifact (d224 control) --")
    p(claim_check(df, "needle", "complex+none_d160", "real+none_d160", 128))
    p(claim_check(df, "needle", "complex+none_d160", "real+none_d224", 128))
    p("  -- RoPE enhancer: dead under param matching --")
    p(claim_check(df, "needle", "complex+rope_d160", "real+rope_d160", 128))
    p(claim_check(df, "needle", "complex+rope_d160", "real+rope_d224", 128))
    if (cond_key(df) == "complex+none-noimag_d160").any():
        p("  -- learn_imag ablation: is the reverse win the weights or the input phase? --")
        p(claim_check(df, "reverse", "complex+none_d160", "complex+none-noimag_d160", 24))
        p(claim_check(df, "reverse", "complex+none-noimag_d160", "real+none_d224", 24))
        p("  -- learn_imag ablation, needle side-observation: noimag is (in)distinguishable? --")
        p(claim_check(df, "needle", "complex+none-noimag_d160", "complex+none_d160", 128))
        p(claim_check(df, "needle", "complex+none-noimag_d160", "real+none_d224", 128))

    out.extend(followup_checks(full))

    text = "\n".join(out)
    print(text)
    if out_md:
        from pathlib import Path
        Path(out_md).write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten -> {out_md}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
