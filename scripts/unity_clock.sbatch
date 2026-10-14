#!/usr/bin/env bash
# Round three on UMass Unity: is the positional clock causal, is it really fractional, and is the
# reverse gap a training-duration artifact?
# Submit:  sbatch scripts/unity_clock.sbatch
#
#   E6  checkpointed retrain      complex+NoPE d160, real+NoPE d224 (matched) and d160, and the
#                                 block-diagonal tying, 5 seeds, 10k steps, save_model=True. The
#                                 round-one checkpoints are gone; the causal probe needs weights.
#                                 Doubles as a reproduction check against the committed unity_* runs.
#   E7  training-duration control complex+NoPE d160 vs real+NoPE d224 at 30k steps (3x the budget),
#                                 5 seeds. Every committed run trained for 10k steps. If the matched
#                                 real model closes the 0.86 vs 0.41 gap with more steps, the result is
#                                 "finds the extrapolating solution faster", not parameter efficiency.
#   E8  clock probe               experiments.clock_causal over every E6/E7 checkpoint: clock-plane
#                                 ablation vs a same-rank control plane, input vs answer segment fits
#                                 in a fixed plane, and the error profile by answer index.
#
# AS RUN (job 65636868, 2026-10-13): E6 and E8 completed on an A16 node; the job was then cancelled
# before E7, which would have taken ~6 h on that GPU. E7 was run instead on an Apple-silicon GPU (MPS)
# with the identical sweep command plus --device mps --eval-lengths 16 24 32 48 64; four complex and
# three real seeds finished (results/long_*). To finish seeds here: sbatch this script again;
# skip_done skips E6, and E7 trains only what is missing from results/.
#
# Tags use their own prefixes (clock_, long_) so experiments.stats --tag-prefix unity_ never pools them
# into the committed tables.
#
# FOOTPRINT: one GPU on gpu-preempt, sequential, --requeue + skip_done (a preempted job resumes at the
# next unfinished run). Checkpoints are ~12 MB each, ~40 runs, so under 0.5 GB.

#SBATCH --job-name=cplxw-clock
#SBATCH --partition=gpu-preempt   # preemptible ONLY: never take a slot the TRM jobs could use
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-clock-%j.out
#SBATCH --requeue

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"   # uv lives here; batch shells may not source the login profile

module purge 2>/dev/null || true
module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true

echo "host=$(hostname) job=$SLURM_JOB_ID gpu=${CUDA_VISIBLE_DEVICES:-none}"
uv sync --quiet
uv run python scripts/check_env.py

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/results"
export WANDB_PROJECT="complexweights"
export PYTHONUNBUFFERED=1        # otherwise per-run summaries sit in the stdout buffer for hours

COMMON=(--tasks reverse --seeds 0 1 2 3 4 --save-model --out-dir results
        --wandb --wandb-mode "${WANDB_MODE}" --wandb-group "clock-${SLURM_JOB_ID}")

# --- E6: checkpointed retrain at the committed recipe -------------------------------------------
# core=pair is complex+NoPE d160 and real+NoPE d224; --real-width-scan 160 adds the equal-width real
# model (tagged _wide) and --algebras diag the block-diagonal control.
echo "=== E6: checkpointed reverse retrain (10k steps) ==="
uv run python -m experiments.sweep "${COMMON[@]}" --tag-prefix clock \
    --core pair --algebras diag --real-width-scan 160

# --- E8 on E6 first, so the causal probe is available even if E7 is preempted repeatedly --------
echo "=== E8: clock probe on the 10k-step checkpoints ==="
uv run python -m experiments.clock_causal --match "clock_reverse_*" \
    --out results/_figs/clock/clock_metrics_10k.json

# --- E7: training-duration control --------------------------------------------------------------
echo "=== E7: 30k-step training-duration control ==="
uv run python -m experiments.sweep "${COMMON[@]}" --tag-prefix long --core pair --steps 30000

echo "=== E8: clock probe on all checkpoints ==="
uv run python -m experiments.clock_causal --match "clock_reverse_*" "long_reverse_*"

echo "=== clock job complete ==="
