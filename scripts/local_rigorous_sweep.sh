#!/usr/bin/env bash
# More rigorous *local* (GTX 1050) validation before the Unity A100 run.
#
# Design (per the "bigger model, longer training, fewer seeds" steer): step up from the d128/4L/8k probe
# to d160 / 5 layers / 10k steps, and instead of many seeds, get a clean per-condition read at scale.
#   - reverse + copy  : complex+NoPE gets 3 seeds (the reliability/convergence question), the stable
#                       conditions 1 seed each.
#   - sort + needle   : first real signal (currently ZERO data) — 1 seed x 4 conditions.
# Conditions: the 4 that carry the science — {complex,real} x {NoPE,RoPE}. (real+learned & param-matched
# baselines are left for the A100 matrix in scripts/unity_sweep.sbatch.)
#
# Measured cost on this GPU: ~36 min / complex run, ~7 min / real run  =>  ~8h total, but VALUE-ORDERED
# (reverse first, then copy, then sort, then needle) so the important results land first.
#
# RESUMABLE: every run writes results/<tag>/model.pt at the end; a tag that already has one is skipped.
# Stop anytime (Ctrl-C) and re-run to continue. Only the in-progress run is redone (deterministic seed).
# Run unattended + watch live in the marimo dashboard:
#     nohup bash scripts/local_rigorous_sweep.sh > results/rig_sweep.log 2>&1 &
#     uv run marimo edit notebooks/03_analysis.py     # results refresh as runs finish
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="--train-min 6 --train-max 16 --eval-lengths 16 32 64 96 --dim 160 --n-layers 5 \
        --steps 10000 --warmup 200 --lr 2e-3 --save-model"

run_one() {  # tag, then extra train.py flags
  tag="$1"; shift
  if [ -f "results/$tag/model.pt" ]; then echo "skip $tag (already done)"; return; fi
  echo "running $tag ..."
  uv run python -m experiments.train $COMMON --tag "$tag" "$@"
}

# 4 conditions for one (task, seed). complex uses learn_imag (improves NoPE reliability; ignored by real).
core_conditions() {  # task, seed
  local task="$1" s="$2"
  run_one "rig_${task}_complex_none_s${s}" --task "$task" --field complex --pe none --learn-imag --seed "$s"
  run_one "rig_${task}_complex_rope_s${s}" --task "$task" --field complex --pe rope --learn-imag --seed "$s"
  run_one "rig_${task}_real_none_s${s}"    --task "$task" --field real    --pe none --seed "$s"
  run_one "rig_${task}_real_rope_s${s}"    --task "$task" --field real    --pe rope --seed "$s"
}

# --- Phase 1+2: central tasks, complex+NoPE reliability (3 seeds), stable conditions 1 seed ---
for task in reverse copy; do
  # seed 0: all four conditions
  core_conditions "$task" 0
  # seeds 1,2: only the conditions whose reliability we're characterizing (the complex ones)
  for s in 1 2; do
    run_one "rig_${task}_complex_none_s${s}" --task "$task" --field complex --pe none --learn-imag --seed "$s"
    run_one "rig_${task}_complex_rope_s${s}" --task "$task" --field complex --pe rope --learn-imag --seed "$s"
  done
done

# --- Phase 3+4: first signal on the two un-run tasks (1 seed x 4 conditions) ---
for task in sort needle; do
  core_conditions "$task" 0
done

echo "LOCAL_RIGOROUS_SWEEP_DONE"
