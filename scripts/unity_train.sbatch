#!/usr/bin/env bash
# SLURM batch script for UMass Unity — single A100 (32 GB).
# Submit:  sbatch scripts/unity_train.sbatch --task copy --field complex --pe none
# NOTE: partition / constraint / module names below are best-guess Unity conventions —
#       confirm with `sinfo` and `module avail cuda` before the first real run.

#SBATCH --job-name=cplxw
#SBATCH --partition=gpu-preempt        # Unity preemptible GPU partition (cheap, may be requeued)
#SBATCH --constraint=a100              # request an A100 node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --requeue

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load cuda/12.4 || module load cuda || true   # match the cu124 wheel if available

echo "host=$(hostname) job=$SLURM_JOB_ID gpu=${CUDA_VISIBLE_DEVICES:-none}"
uv sync --quiet
uv run python scripts/check_env.py

# Forward all CLI args straight through to the trainer (tyro config).
uv run python -m experiments.train "$@"
