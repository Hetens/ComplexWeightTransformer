#!/usr/bin/env bash
# Full multi-seed verification sweep on UMass Unity (single A100, 32-80 GB).
# Promotes the verified d160 / 5-layer / 10k-step rigorous setup (docs/results.md) to 5 seeds on EVERY
# condition, so the headline gaps -- esp. sort length-generalization -- get variance bars, not n=1 reads.
# Submit:  sbatch scripts/unity_sweep.sbatch
# NOTE: confirm partition / constraint / module names with `sinfo` and `module avail cuda` first.
#
# PREEMPT-SAFE: --requeue + the sweep's skip_done means a killed/requeued job resumes where it stopped
# (every finished run leaves results/<tag>/result.json). Re-submitting the same script also just continues.

#SBATCH --job-name=cplxw-sweep
#SBATCH --partition=gpu        # non-preempt: gpu-preempt only guarantees 2h before eviction
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G              # host RAM; job 61568498 peaked ~4 GB on-GPU and less on-host.
                               # If SLURM OOM-kills the job, raise back toward 48G and resubmit.
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-sweep-%j.out
#SBATCH --requeue

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"   # uv lives here; batch shells may not source the login profile

# `module` may be undefined in the batch shell (env propagated from a non-login submit shell);
# the cu124 wheels bundle their CUDA runtime anyway, so a missing module is not fatal.
module purge 2>/dev/null || true
module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true

echo "host=$(hostname) job=$SLURM_JOB_ID gpu=${CUDA_VISIBLE_DEVICES:-none}"
uv sync --quiet
uv run python scripts/check_env.py

# --- Weights & Biases -------------------------------------------------------------------------------
# Default OFFLINE: Unity compute nodes usually have no outbound internet, so we log locally and sync
# from a login node afterwards.  For a LIVE dashboard instead, run `wandb login` once on a login node
# and submit with:  WANDB_MODE=online sbatch scripts/unity_sweep.sbatch
# After an offline run, upload with:   wandb sync results/wandb/offline-run-*
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/results"
export WANDB_PROJECT="complexweights"

# 4 tasks x 5 seeds x 7 conditions {complex,real}x{NoPE,RoPE} + real-learned + param-matched real.
# Scale/lengths/steps default to the verified rigorous config inside experiments/sweep.py.
# --wandb-mode must be passed explicitly: an explicit wandb.init(mode=...) argument OVERRIDES the
# WANDB_MODE env var, so exporting it alone silently leaves the config default ("online") in effect.
uv run python -m experiments.sweep \
    --seeds 0 1 2 3 4 \
    --wandb \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-group "unity-${SLURM_JOB_ID}" \
    --out-dir results
