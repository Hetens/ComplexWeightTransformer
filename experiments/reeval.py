"""Re-evaluate committed checkpoints at longer lengths, without retraining anything.

Why this exists: the `needle` reading in the 5-seed sweep is under-powered rather than null. The
no-input-phase complex model sits at 1.00 +/- 0.01 through length 128 while the parameter-matched real
baseline has already started to decay (0.92 +/- 0.18). Separating those by adding seeds would need
~80 per arm; pushing the *length* axis instead grows the effect for free, because one arm is pinned at
ceiling and the other is on a visible decay curve.

No training happens here, so this costs inference only and it reports the exact models the paper
already reports. Only `pe=none` runs are eligible: RoPE and learned-APE build a positional buffer of
size `max_T` at construction, so they cannot be evaluated past the length they were built for.

Output convention: each re-eval writes a *new* results dir `<source tag>_long/result.json` holding
**only the new lengths**, so it never overwrites committed evidence and never double-counts a length
that the source run already measured. `experiments/analysis.py` derives conditions from the config,
not the tag, so these rows pool with their source condition automatically.

    uv run python -m experiments.reeval --match "unity_needle_*_none_*"
    uv run python -m experiments.reeval --match "unity_needle_*" --lengths 256 512 --eval-batch 32
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from complexnn.models import ModelConfig, TransformerLM
from experiments.data.tasks import TaskConfig, evaluate


@dataclass
class ReevalConfig:
    results_dir: str = "results"
    match: str = "unity_needle_*_none_*"   # glob over run tags
    lengths: tuple[int, ...] = (256, 512, 1024)
    eval_seqs: int = 512
    eval_batch: int = 32                   # attention is O(batch * T^2); keep the footprint small
    device: str = "auto"
    suffix: str = "_long"
    skip_done: bool = True
    dry_run: bool = False


def _device(name: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def _rebuild(c: dict, task: TaskConfig, max_T: int) -> TransformerLM:
    return TransformerLM(ModelConfig(
        vocab_size=task.vocab_size, dim=c["dim"], n_layers=c["n_layers"], n_heads=c["n_heads"],
        mlp_ratio=c["mlp_ratio"], field=c["field"], pe=c["pe"], max_T=max_T,
        rope_base=c["rope_base"], learn_imag=c["learn_imag"], complex_act=c["complex_act"],
        algebra=c.get("algebra", "complex"),
    ))


def main(cfg: ReevalConfig) -> None:
    root = Path(cfg.results_dir)
    dev = _device(cfg.device)
    srcs = sorted(d for d in root.iterdir()
                  if d.is_dir() and fnmatch.fnmatch(d.name, cfg.match)
                  and not d.name.endswith(cfg.suffix) and (d / "result.json").exists())
    if not srcs:
        raise SystemExit(f"no runs matching {cfg.match!r} under {root}/")

    done = skipped = 0
    for src in srcs:
        tag = src.name + cfg.suffix
        out = root / tag
        if cfg.skip_done and (out / "result.json").exists():
            skipped += 1
            continue
        d = json.loads((src / "result.json").read_text())
        c = d["config"]
        if c["pe"] != "none":
            print(f"[skip] {src.name}: pe={c['pe']} builds a positional buffer at max_T, not extensible")
            skipped += 1
            continue
        ckpt = src / "model.pt"
        if not ckpt.exists():
            print(f"[skip] {src.name}: no model.pt")
            skipped += 1
            continue
        if cfg.dry_run:
            print(f"[dry]  {src.name} -> {tag}")
            continue

        task = TaskConfig(c["task"], c["n_symbols"], c["train_min"], c["train_max"], cfg.lengths)
        # `needle` builds [haystack + MARK] SEP [secret]; algorithmic tasks build 2L+1. Size the
        # positional budget for the worst case so a NoPE model is never silently clipped.
        model = _rebuild(c, task, max_T=2 * max(cfg.lengths) + 4).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.eval()

        evals = {str(L): evaluate(model, task, L, cfg.eval_seqs, dev, cfg.eval_batch) for L in cfg.lengths}
        result = {
            "config": {**c, "eval_lengths": list(cfg.lengths), "eval_seqs": cfg.eval_seqs,
                       "eval_batch": cfg.eval_batch, "tag": tag},
            "real_params": d["real_params"],
            "final_loss": d["final_loss"],
            "history": [],
            "evals": evals,
            "reeval_of": src.name,   # provenance: inference-only, weights untouched
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(f"[{tag}] " + " ".join(f"seq@{L}={evals[str(L)]['seq_acc']:.3f}" for L in cfg.lengths))
        done += 1
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    print(f"\nreeval done: {done} written, {skipped} skipped -> {root}/")


if __name__ == "__main__":
    main(tyro.cli(ReevalConfig))
