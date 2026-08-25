"""Export trained checkpoints to the flat fp16 blobs the web demo runs in the browser.

    uv run python scripts/export_web_models.py            # -> web/public/models/<id>/{weights.f16,manifest.json}

Each blob is every state-dict tensor, in order, cast to fp16 and concatenated; the manifest records
each tensor's offset and shape plus the config the runtime needs to rebuild the forward pass. fp16 is
lossless for our purposes: it leaves the argmax of every logit unchanged and moves the rotating-plane
statistics by less than 1e-4 (both checked against the PyTorch model before this script was written).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "models"

# id -> (results dir, label shown in the UI). The id is what the client fetches.
MODELS = {
    "complex-nope-d160-s0": ("unity_reverse_complex_none_d160_s0", "complex+NoPE d160"),
    "real-nope-d224-s0": ("unity_reverse_real_none_d224_matched_s0", "real+NoPE d224 (param-matched)"),
    "real-nope-d160-s0": ("unity_reverse_real_none_d160_s0", "real+NoPE d160"),
    "real-rope-d160-s0": ("unity_reverse_real_rope_d160_s0", "real+RoPE d160"),
}

CFG_KEYS = ("field", "pe", "dim", "n_layers", "n_heads", "mlp_ratio", "learn_imag", "algebra",
            "complex_act", "rope_base", "n_symbols", "task", "train_min", "train_max")


def export(run_dir: Path, model_id: str, label: str) -> dict:
    cfg = json.load(open(run_dir / "result.json"))["config"]
    evals = json.load(open(run_dir / "result.json"))["evals"]
    sd = torch.load(run_dir / "model.pt", map_location="cpu")

    out = OUT / model_id
    out.mkdir(parents=True, exist_ok=True)

    chunks, tensors, off = [], {}, 0
    for k, v in sd.items():
        a = v.numpy().astype(np.float16).ravel()
        tensors[k] = {"offset": off, "shape": list(v.shape)}
        chunks.append(a)
        off += a.size
    np.concatenate(chunks).tofile(out / "weights.f16")

    manifest = {
        "id": model_id,
        "label": label,
        "config": {k: cfg.get(k) for k in CFG_KEYS},
        "vocab_size": 3 + cfg["n_symbols"],
        "real_params": int(sum(v.numel() for v in sd.values())),
        "n_floats": int(off),
        # the paper's own teacher-forced seq_acc for this run, so the UI can show measured-vs-recorded
        "paper_seq_acc": {k: v for k, v in evals.items()},
        "tensors": tensors,
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    return {"id": model_id, "label": label, "bytes": off * 2,
            "config": manifest["config"], "vocab_size": manifest["vocab_size"],
            "real_params": manifest["real_params"], "paper_seq_acc": manifest["paper_seq_acc"]}


def main() -> None:
    index = []
    for model_id, (run, label) in MODELS.items():
        run_dir = ROOT / "results" / run
        if not (run_dir / "model.pt").exists():
            print(f"  skip {model_id}: no checkpoint at {run_dir}")
            continue
        info = export(run_dir, model_id, label)
        index.append(info)
        print(f"  {model_id:24s} {info['real_params']:>9,} params  {info['bytes']/1e6:5.2f} MB")
    json.dump(index, open(OUT / "index.json", "w"), indent=1)
    print(f"\n-> {OUT}  ({sum(i['bytes'] for i in index)/1e6:.1f} MB total)")


if __name__ == "__main__":
    main()
