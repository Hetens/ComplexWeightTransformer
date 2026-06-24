"""Train one (field, pe) condition on one synthetic positional task and log metrics to results/.

Examples:
    uv run python -m experiments.train --field complex --pe none --task reverse
    uv run python -m experiments.train --smoke                  # tiny GTX-1050 sanity run
    sbatch scripts/unity_train.sbatch --field real --pe rope --task sort --steps 8000
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro

from complexnn.models import ModelConfig, TransformerLM
from experiments.data.tasks import TaskConfig, evaluate, make_batch


@dataclass
class TrainConfig:
    # task
    task: str = "reverse"            # copy | reverse | sort
    n_symbols: int = 16
    train_min: int = 4
    train_max: int = 12
    eval_lengths: tuple[int, ...] = (12, 24, 48)
    # model
    field: str = "complex"           # real | complex
    pe: str = "none"                 # rope | none | learned
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: float = 4.0
    complex_act: str = "crelu"       # crelu | modrelu
    learn_imag: bool = False
    max_T: int = 256
    rope_base: float = 10000.0
    # optim
    steps: int = 2000
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup: int = 100
    # run
    seed: int = 0
    device: str = "auto"
    eval_seqs: int = 512
    out_dir: str = "results"
    tag: str = ""
    log_every: int = 200
    smoke: bool = False              # shrink everything for a fast local sanity run
    save_model: bool = False         # also dump model.pt (state_dict) for the phase-spectrum probe
    # Weights & Biases (optional; import is lazy so non-wandb runs need no install)
    wandb: bool = False              # log loss curve + per-length token/seq accuracy to W&B
    wandb_project: str = "complexweights"
    wandb_entity: str = ""           # team/user; "" -> wandb default
    wandb_group: str = ""            # group runs of one sweep together; "" -> ungrouped
    wandb_mode: str = "online"       # online | offline | disabled  (offline for HPC, sync later)


def _apply_smoke(cfg: TrainConfig) -> TrainConfig:
    cfg.steps, cfg.dim, cfg.eval_seqs = 250, 32, 128
    cfg.eval_lengths = (12, 24)
    return cfg


def _device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def _init_wandb(cfg: TrainConfig, tag: str):
    """Start a W&B run for this training run. Lazy import: only paid for when --wandb is set."""
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        group=cfg.wandb_group or None,
        name=tag,
        mode=cfg.wandb_mode,
        tags=[cfg.task, cfg.field, cfg.pe, f"d{cfg.dim}"],
        config=asdict(cfg),
        reinit="finish_previous",  # sweep calls run() repeatedly in-process; close any prior run first
    )


def run(cfg: TrainConfig) -> dict:
    if cfg.smoke:
        cfg = _apply_smoke(cfg)
    dev = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    tag = cfg.tag or f"{cfg.task}_{cfg.field}_{cfg.pe}_d{cfg.dim}_s{cfg.seed}"
    wb = _init_wandb(cfg, tag) if cfg.wandb else None

    task = TaskConfig(cfg.task, cfg.n_symbols, cfg.train_min, cfg.train_max, cfg.eval_lengths)
    model = TransformerLM(
        ModelConfig(
            vocab_size=task.vocab_size, dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            mlp_ratio=cfg.mlp_ratio, field=cfg.field, pe=cfg.pe, max_T=cfg.max_T,
            rope_base=cfg.rope_base, learn_imag=cfg.learn_imag, complex_act=cfg.complex_act,
        )
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_at(step):  # linear warmup -> cosine decay
        import math
        if step < cfg.warmup:
            return step / max(cfg.warmup, 1)
        p = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    gen = torch.Generator().manual_seed(1234 + cfg.seed)

    history = []
    t0 = time.time()
    for step in range(cfg.steps):
        X, Y, M = make_batch(task, cfg.batch_size, cfg.train_min, cfg.train_max, device=dev, generator=gen)
        logits = model(X).reshape(-1, task.vocab_size)
        m = M.reshape(-1)
        loss = F.cross_entropy(logits[m], Y.reshape(-1)[m])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            history.append({"step": step, "loss": float(loss.item())})
            if wb is not None:
                wb.log({"loss": float(loss.item()), "lr": sched.get_last_lr()[0]}, step=step)

    evals = {str(L): evaluate(model, task, L, cfg.eval_seqs, dev) for L in cfg.eval_lengths}
    result = {
        "config": asdict(cfg),
        "real_params": model.num_real_params(),
        "wall_sec": round(time.time() - t0, 1),
        "final_loss": float(loss.item()),
        "history": history,
        "evals": evals,
    }

    out = Path(cfg.out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    if cfg.save_model:
        torch.save(model.state_dict(), out / "model.pt")  # config lives in result.json

    if wb is not None:
        summary = {"final_loss": result["final_loss"], "real_params": result["real_params"]}
        for L in cfg.eval_lengths:
            summary[f"token_acc/{L}"] = evals[str(L)]["token_acc"]
            summary[f"seq_acc/{L}"] = evals[str(L)]["seq_acc"]  # exact-match: the honest length-gen metric
        wb.log(summary, step=cfg.steps)
        wb.summary.update(summary)
        wb.finish()

    print(f"[{tag}] params={result['real_params']} loss={result['final_loss']:.3f} "
          + " ".join(f"acc@{L}={evals[str(L)]['token_acc']:.2f}/{evals[str(L)]['seq_acc']:.2f}"
                     for L in cfg.eval_lengths)
          + f"  (tok/seq; {result['wall_sec']}s) -> {out/'result.json'}")
    return result


if __name__ == "__main__":
    run(tyro.cli(TrainConfig))
