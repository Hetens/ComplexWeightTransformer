#!/usr/bin/env bash
# Symmetry fix for the H2_width comparison: sweep the *complex* arm over the same learning-rate grid
# the real baselines were swept over.
# Submit:  sbatch scripts/unity_complex_lr.sbatch   (after scripts/unity_width_lr.sbatch finishes)
#
# WHY. The width scan retuned the wide real baselines because lr=2e-3, tuned at d160, diverged at
# d448/d640. That retune turned out to change more than convergence: at d448 every seed converges at
# both 1e-3 and 5e-4, yet seq@24 is 0.659 vs 0.883. So the baseline's length generalization is
# lr-sensitive in a range where our pre-declared "select by in-distribution convergence" criterion
# cannot discriminate, because both rates converge.
#
# That leaves an asymmetry we have to close rather than choose our way out of: the real baselines
# would have been tuned over three learning rates while complex+NoPE has only ever run at 2e-3.
# Reporting best-tuned real against untuned complex is unfair to complex; reporting 1e-3 for real
# because we named the criterion first is unfair to real and happens to favour our own hypothesis.
# The honest protocol is to tune both arms over the same grid and compare best to best, so this runs
# complex+NoPE d160 at 1e-3 and 5e-4 to sit alongside the existing 2e-3 runs.
#
# Note this can only *weaken* our own claim (complex can only improve on its 0.856), which is the
# direction a fix should run when it is fixing a bias that favoured us.
#
# FOOTPRINT: one GPU on gpu-preempt, sequential, --requeue, save_model=False. 10 runs at ~7.5 min.

#SBATCH --job-name=cplxw-clr
#SBATCH --partition=gpu-preempt   # preemptible ONLY: never take a slot the TRM jobs could use
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=results/slurm-clr-%j.out
#SBATCH --requeue

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

module purge 2>/dev/null || true
module load cuda/12.4 2>/dev/null || module load cuda 2>/dev/null || true

echo "host=$(hostname) job=$SLURM_JOB_ID gpu=${CUDA_VISIBLE_DEVICES:-none}"
uv sync --quiet
uv run python scripts/check_env.py

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/results"
export WANDB_PROJECT="complexweights"
export PYTHONUNBUFFERED=1

for LR in 1e-3 5e-4; do
  NOTE="lr${LR//[.-]/}"          # 1e-3 -> lr1e3, 5e-4 -> lr5e4

  echo "=== complex+NoPE d160 at lr=${LR} (tag note: ${NOTE}) ==="
  # core=pair with matched_real_dim=0 selects the complex+NoPE arm alone (the real d224 control is
  # already measured at 2e-3 and is not what this job is for), and leaves the tag suffix empty so the
  # runs read as plain complex+NoPE d160 differing only by the lr note.
  uv run python -m experiments.sweep \
      --seeds 0 1 2 3 4 --no-save-model --out-dir results \
      --tasks reverse --core pair --matched-real-dim 0 \
      --lr-complex "${LR}" --tag-note "${NOTE}" \
      --wandb --wandb-mode "${WANDB_MODE}" --wandb-group "clr-${SLURM_JOB_ID}"

  # d320 for the same reason. It was only ever run at 2e-3 (where it converges 5/5, so nothing
  # flagged it), but "converged" is not "tuned": d448 converges 3/5 at 2e-3 and still gains +0.04
  # from a lower rate. d320 is the width that decides whether the efficiency claim is ~2x or ~8x,
  # since real is 0.542 there at 2e-3 against complex's 0.856, so it is the one narrow width worth
  # the compute. d160/d224 sit at 0.42/0.41 and are not plausibly one lr change from 0.86.
  echo "=== real+NoPE d320 at lr=${LR} (tag note: ${NOTE}) ==="
  uv run python -m experiments.sweep \
      --seeds 0 1 2 3 4 --no-save-model --out-dir results \
      --tasks reverse --core none --real-width-scan 320 \
      --lr-real "${LR}" --tag-note "${NOTE}" \
      --wandb --wandb-mode "${WANDB_MODE}" --wandb-group "clr-${SLURM_JOB_ID}"
done

echo "=== complex lr sweep complete ==="
uv run python -m experiments.stats --tag-prefix unity_ || true
