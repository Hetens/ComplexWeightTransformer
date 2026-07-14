"""Run the full comparison matrix (conditions x tasks x seeds) and write results/ for analysis.

The defaults here reproduce the *verified* d160 / 5-layer / 10k-step rigorous setup from
docs/results.md (train lengths 6-16, eval out to 6x train length) and scale it to the multi-seed run
that the local sweep could not afford: 5 seeds on every condition, so the headline numbers (esp. the
`sort` length-generalization gap) get variance bars instead of a single-seed read. Conditions follow
docs/hypothesis.md; `matched_real_dim` adds a capacity-matched (wider) real baseline so complex is
compared at both equal-dim and equal-real-params (d224 ~= complex d160's 3.08M real params).

The loop is seed-outer and skip-if-done, so a preempted/requeued Unity job resumes where it left off and
each completed pass adds one seed to *every* condition (best incremental value for variance estimation).

    uv run python -m experiments.sweep --tasks sort reverse --seeds 0 1 2 --steps 10000
    uv run python -m experiments.sweep --wandb --wandb-group unity-$SLURM_JOB_ID    # full A100 run
    sbatch scripts/unity_sweep.sbatch
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import tyro

from experiments.train import TrainConfig, run


@dataclass
class SweepConfig:
    # value-ordered: sort is the discriminating task (carries the headline), needle the least
    tasks: tuple[str, ...] = ("sort", "reverse", "copy", "needle")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    # model / task scale — matches the verified d160 rigorous sweep (docs/results.md)
    dim: int = 160
    n_layers: int = 5
    matched_real_dim: int = 224       # ~param-matches complex d160 (3.08M); 224/(2*n_heads) int -> RoPE ok
    train_min: int = 6
    train_max: int = 16
    eval_lengths: tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128)
    max_T: int = 512                  # fits [input][SEP][answer] = 2*128+1 at the longest eval length
    steps: int = 10000
    warmup: int = 200
    lr_real: float = 2e-3
    lr_complex: float = 2e-3          # matches the rigorous local sweep (was 3e-3 in the d64 smoke sweep)
    complex_act: str = "crelu"
    save_model: bool = True           # dump model.pt for the phase-spectrum probe (experiments.phase_probe)
    device: str = "auto"
    out_dir: str = "results"
    tag_prefix: str = "unity"         # keeps these results distinct from the local `rig_*` set
    skip_done: bool = True            # resumable: skip a run whose result.json already exists (preempt-safe)
    # Ablation: complex+NoPE with learn_imag=False. The main complex condition bundles complex weights,
    # CReLU, and a learnable imaginary token embedding; this isolates whether the reverse length-gen win
    # comes from the complex weights themselves or from the learnable input phase.
    ablate_learn_imag: bool = False
    # Weights & Biases
    wandb: bool = False
    wandb_project: str = "complexweights"
    wandb_entity: str = ""
    wandb_group: str = ""             # "" -> defaults to tag_prefix
    wandb_mode: str = "online"        # online | offline | disabled


def conditions(cfg: SweepConfig):
    """(field, pe, dim, lr, learn_imag, tag_suffix) tuples for one task/seed, core conditions first.

    Complex conditions use learn_imag=True: it gives token embeddings a learnable phase and substantially
    improves complex+NoPE optimization reliability (see docs/results.md). Real conditions ignore it.
    """
    c = [
        ("complex", "none", cfg.dim, cfg.lr_complex, True, ""),   # the primary hypothesis
        ("complex", "rope", cfg.dim, cfg.lr_complex, True, ""),   # the "RoPE enhancer" thread
        ("real", "none", cfg.dim, cfg.lr_real, False, ""),        # the NoPE baseline to beat
        ("real", "rope", cfg.dim, cfg.lr_real, False, ""),
        ("real", "learned", cfg.dim, cfg.lr_real, False, ""),
    ]
    if cfg.matched_real_dim:
        c.append(("real", "none", cfg.matched_real_dim, cfg.lr_real, False, "_matched"))
        c.append(("real", "rope", cfg.matched_real_dim, cfg.lr_real, False, "_matched"))
    if cfg.ablate_learn_imag:
        c.append(("complex", "none", cfg.dim, cfg.lr_complex, False, "_noimag"))
    return c


def main(cfg: SweepConfig):
    group = cfg.wandb_group or cfg.tag_prefix
    base = TrainConfig(
        n_layers=cfg.n_layers, train_min=cfg.train_min, train_max=cfg.train_max,
        eval_lengths=cfg.eval_lengths, max_T=cfg.max_T, steps=cfg.steps, warmup=cfg.warmup,
        complex_act=cfg.complex_act, save_model=cfg.save_model, device=cfg.device, out_dir=cfg.out_dir,
        wandb=cfg.wandb, wandb_project=cfg.wandb_project, wandb_entity=cfg.wandb_entity,
        wandb_group=group, wandb_mode=cfg.wandb_mode,
    )
    ran = skipped = 0
    for seed in cfg.seeds:            # seed-outer: each completed pass adds one seed to EVERY condition
        for task in cfg.tasks:
            for field, pe, dim, lr, learn_imag, suffix in conditions(cfg):
                tag = f"{cfg.tag_prefix}_{task}_{field}_{pe}_d{dim}{suffix}_s{seed}"
                if cfg.skip_done and (Path(cfg.out_dir) / tag / "result.json").exists():
                    skipped += 1
                    continue
                run(replace(base, task=task, seed=seed, field=field, pe=pe, dim=dim, lr=lr,
                            learn_imag=learn_imag, tag=tag))
                ran += 1
    print(f"\nsweep done: {ran} runs executed, {skipped} skipped (already present) -> {cfg.out_dir}/")


if __name__ == "__main__":
    main(tyro.cli(SweepConfig))
