#!/usr/bin/env bash
# learn_imag ablation + d160 phase-probe re-run on UMass Unity (single A100).
#
# The main complex condition bundles complex weights + CReLU + a learnable imaginary token embedding
# (learn_imag=True). The reverse length-gen win is therefore attributable to the bundle, not to complex
# weights per se. This job adds the missing control: complex+NoPE with learn_imag=False, 4 tasks x 5
# seeds (20 runs, ~2.5 h). skip_done skips the 140 existing unity_* results, so re-submitting resumes.
#
# It then re-runs the mechanistic phase probe on the d160 unity_reverse_* checkpoints (--tagset d160),
# restoring the probe evidence lost with the d64/d128 artifacts and adding pos_fraction on reverse,
# the task where the 5-seed behavioural gap actually lives. Figures + metrics -> results/_figs/d160/.
#
# Submit:  sbatch scripts/unity_ablation.sbatch
# PREEMPT-SAFE: --requeue + skip_done resumes where it stopped.

#SBATCH --job-name=cplxw-ablate
#SBATCH --partition=gpu        # non-preempt: gpu-preempt only guarantees 2h before eviction
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G              # host RAM; the sweep job peaked ~4 GB on-GPU and far less on-host.
                               # If SLURM OOM-kills the job (check `sacct -j <id> -o State,MaxRSS`),
                               # raise back toward 48G and resubmit; skip_done resumes.
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-ablate-%j.out
#SBATCH --requeue

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"   # uv lives here; batch shells may not source the login profile

# `module` may be undefined in the batch shell; cu124 wheels bundle their CUDA runtime, so not fatal.
module purge 2>/dev/null || true
module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true

echo "host=$(hostname) job=$SLURM_JOB_ID gpu=${CUDA_VISIBLE_DEVICES:-none}"
uv sync --quiet
uv run python scripts/check_env.py

# Offline W&B by default (compute nodes have no outbound internet); sync from a login node afterwards.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/results"
export WANDB_PROJECT="complexweights"

# --wandb-mode must be passed explicitly: an explicit wandb.init(mode=...) argument OVERRIDES the
# WANDB_MODE env var, so exporting it alone silently leaves the config default ("online") in effect.
uv run python -m experiments.sweep \
    --seeds 0 1 2 3 4 \
    --ablate-learn-imag \
    --wandb \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-group "unity-ablate-${SLURM_JOB_ID}" \
    --out-dir results

# Mechanistic probe on the d160 checkpoints (5-seed converged models + the d224 matched control).
uv run python -m experiments.phase_probe --results-dir results --tagset d160

# Fresh stats summary including the ablation condition (also the source of the paper's numbers).
uv run python -m experiments.stats --results-dir results
