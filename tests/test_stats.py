"""Guards for the reporting layer in experiments/stats.py.

Two things must not break when a later round adds tasks or conditions:
  1. A pre-registered, task-averaged number must keep meaning the same thing. H1_cap was registered
     over four tasks; silently averaging in tasks added later would redefine it.
  2. The report must survive conditions that do not cover every task. Follow-up rounds run only the
     head-to-head pair on new tasks, so the equal-dimension baseline is legitimately absent there.
"""

import json
from pathlib import Path

import pytest

from experiments.stats import CANONICAL_LR, PREREG_TASKS, main, welch_t


def test_welch_t_handles_an_empty_arm():
    """Returns NaNs instead of dividing by zero, so a missing condition blanks a row not the run."""
    w = welch_t([0.1, 0.2, 0.3], [])
    assert w.delta != w.delta  # NaN
    assert w.p != w.p
    w = welch_t([], [])
    assert w.mean_a != w.mean_a and w.mean_b != w.mean_b


def test_welch_t_still_exact_on_normal_input():
    """Regression guard on the hand-rolled Welch implementation (the project avoids scipy).

    Expected values are not copied from the implementation: t and dof are the closed-form Welch
    quantities for these two samples, and p was checked against a direct numerical integration of
    the Student-t density (Simpson-free trapezoid, 4e6 points), which agrees to six decimals.
    """
    w = welch_t([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])
    assert w.delta == pytest.approx(-1.0)
    assert w.t == pytest.approx(-1.095445, abs=1e-6)
    assert w.dof == pytest.approx(6.0, abs=1e-9)
    assert w.p == pytest.approx(0.315334, abs=1e-5)


def test_prereg_task_set_is_pinned():
    """H1_cap's registered task set must stay exactly the four it was registered over."""
    assert PREREG_TASKS == ("copy", "needle", "reverse", "sort")


def test_canonical_lr_is_the_committed_sweep_rate():
    """The primary report is pinned to the rate the committed sweep used."""
    assert CANONICAL_LR == 2e-3


def test_lr_grid_runs_do_not_pool_into_the_primary_report(tmp_path, capsys):
    """A follow-up lr sweep must not shift a pre-registered number.

    The condition key is (field+pe, dim) and says nothing about the learning rate, so runs added at a
    different rate would otherwise be averaged into the very condition the committed claims are
    stated over. This reproduces that: identical runs at the canonical rate, plus deliberately
    extreme runs at another rate, must leave the reported means untouched.
    """
    for seed in range(3):
        for task in ("reverse", "copy", "sort", "needle"):
            _write_run(tmp_path, f"unity_{task}_complex_none_d160_s{seed}", task, "complex", "none",
                       160, seed, True, 0.8, lr=2e-3)
            _write_run(tmp_path, f"unity_{task}_real_none_d160_s{seed}", task, "real", "none",
                       160, seed, False, 0.4, lr=2e-3)
            _write_run(tmp_path, f"unity_{task}_real_none_d224_matched_s{seed}", task, "real", "none",
                       224, seed, False, 0.4, lr=2e-3)

    main(results_dir=str(tmp_path))
    before = capsys.readouterr().out

    for seed in range(3):
        for task in ("reverse", "copy", "sort", "needle"):
            _write_run(tmp_path, f"unity_{task}_complex_none_d160-lr5e4_s{seed}", task, "complex",
                       "none", 160, seed, True, 0.1, lr=5e-4)   # wildly different, must not pool
    main(results_dir=str(tmp_path))
    after = capsys.readouterr().out

    def prereg_block(text):
        return text.split("Pre-registered H1_cap")[1].split("## Claim checks")[0]

    assert prereg_block(before) == prereg_block(after)


def _write_run(root: Path, tag: str, task: str, field: str, pe: str, dim: int, seed: int,
               learn_imag: bool, seq: float, lr: float = 2e-3) -> None:
    cfg = dict(task=task, n_symbols=16, train_min=6, train_max=16, eval_lengths=[16, 24, 32],
               field=field, pe=pe, dim=dim, n_layers=5, n_heads=4, mlp_ratio=4.0,
               complex_act="crelu", learn_imag=learn_imag, algebra="complex", max_T=512,
               rope_base=10000.0, seed=seed, lr=lr)
    evals = {str(L): {"length": L, "token_acc": seq, "seq_acc": seq} for L in (16, 24, 32)}
    d = root / tag
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(
        {"config": cfg, "real_params": 1000, "final_loss": 0.001, "history": [], "evals": evals}))


def test_report_survives_a_task_missing_the_equal_dim_baseline(tmp_path, capsys):
    """The exact shape that broke the report: a follow-up task run only as complex vs matched real."""
    for seed in range(3):
        for task in ("reverse", "copy", "sort", "needle"):
            _write_run(tmp_path, f"unity_{task}_complex_none_d160_s{seed}", task, "complex", "none",
                       160, seed, True, 0.8)
            _write_run(tmp_path, f"unity_{task}_real_none_d160_s{seed}", task, "real", "none",
                       160, seed, False, 0.4)
            _write_run(tmp_path, f"unity_{task}_real_none_d224_matched_s{seed}", task, "real", "none",
                       224, seed, False, 0.4)
        # the follow-up task: pair only, no d160 real arm at all
        _write_run(tmp_path, f"unity_revskip_complex_none_d160_s{seed}", "revskip", "complex", "none",
                   160, seed, True, 0.3)
        _write_run(tmp_path, f"unity_revskip_real_none_d224_matched_s{seed}", "revskip", "real",
                   "none", 224, seed, False, 0.2)

    main(results_dir=str(tmp_path))            # must not raise
    text = capsys.readouterr().out
    assert "H1_cap" in text
    assert "revskip" in text                    # the new task is still reported in its own block
    # ...but it must not have been folded into the pre-registered task average
    h1 = text.split("Pre-registered H1_cap")[1].split("##")[0]
    assert "revskip" not in h1
