#!/usr/bin/env bash
# Larger-scale mechanistic-probe run set: d128 / 4 layers / 8000 steps / reverse train_max=16.
# Tests whether emergent positional structure becomes evident at a scale where the model clearly solves
# the task (the d64/2000-step probe set showed none). Saves checkpoints for experiments.phase_probe.
#
# RESUMABLE: each condition is a separate run that writes results/<tag>/model.pt at the end. A tag that
# already has a checkpoint is skipped, so you can stop anytime (Ctrl-C, or kill the process) and just
# re-run this script to continue from where it left off. Only the in-progress run is lost and redone
# (deterministic, fixed seed). To run unattended in the background:
#     nohup bash scripts/probe_sweep.sh > results/probe_sweep.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="--task reverse --train-min 6 --train-max 16 --eval-lengths 16 32 64 --dim 128 --n-layers 4 --steps 8000 --save-model"

run_one() {
  tag="$1"; shift
  if [ -f "results/$tag/model.pt" ]; then
    echo "skip $tag (already done)"
    return
  fi
  echo "running $tag ..."
  uv run python -m experiments.train $COMMON "$@" --tag "$tag"
}

for s in 0 1 2 3; do
  run_one "probe2_reverse_complex_none_s$s" --field complex --pe none --seed "$s" --lr 2e-3 --learn-imag
done
run_one "probe2_reverse_complex_rope_s0" --field complex --pe rope --seed 0 --lr 2e-3 --learn-imag
run_one "probe2_reverse_real_none_s0"    --field real    --pe none --seed 0 --lr 2e-3
run_one "probe2_reverse_real_rope_s0"    --field real    --pe rope --seed 0 --lr 2e-3
echo "PROBE_SWEEP_DONE"
