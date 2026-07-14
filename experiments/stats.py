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
    """Two-sided Welch t-test with exact Student-t p-value (Welch-Satterthwaite dof)."""
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
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


def preregistered_h1cap(df: pd.DataFrame, length: int, metric: str, baseline: str) -> list[str]:
    """H1_cap (docs/hypothesis.md): task-mean delta vs pooled seed std at the primary length point."""
    lines = []
    deltas, pooled = [], []
    for task in sorted(df.task.unique()):
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


def main(results_dir: str = "results", tag_prefix: str = "unity_", train_max: int = 16,
         out_md: str = "") -> None:
    df = load_results(results_dir)
    df = df[df.tag.str.startswith(tag_prefix)]
    if not len(df):
        raise SystemExit(f"no results matching {tag_prefix}* under {results_dir}/")
    two_l = 2 * train_max
    out: list[str] = []
    p = out.append

    n_runs = df.tag.nunique()
    p(f"{n_runs} runs | tasks={sorted(df.task.unique())} | conditions={sorted(cond_key(df).unique())}")

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

    text = "\n".join(out)
    print(text)
    if out_md:
        from pathlib import Path
        Path(out_md).write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten -> {out_md}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
