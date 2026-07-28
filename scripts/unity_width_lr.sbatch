#!/usr/bin/env bash
# Supplement to the H2_width arm: re-run the widest real+NoPE baselines at a lower learning rate.
# Submit:  sbatch scripts/unity_width_lr.sbatch
#
# WHY. The width scan inherited lr=2e-3, which was tuned at d160 and is used by every run in the
# committed sweep. It does not transfer: at lr=2e-3 the wide baselines fail to reach token_acc=1.0 at
# the *training* length (d448 converged 1/2, d640 0/1, with final losses of 0.70 and 2.15 against
# ~5e-6 for a healthy run). Averaging those into a length-generalization mean would report an
# optimization failure as a capability limit, and would do so in the direction that flatters our own
# hypothesis, since H2_width predicts the real curve stays flat. That is the exact failure mode this
# project already documented twice: the modReLU under-tuning (docs/results.md M2-M4 item 3) and the
# withdrawn ComplexFormer result it cites. We tune the baseline before concluding anything from it.
#
# Learning rates follow the usual ~1/sqrt(width) intuition from the d160 anchor: d448 is 2.8x the
# width of d160 and d640 is 4x, so 1e-3 and 5e-4 bracket the scaled value.
#
# SELECTION CRITERION, fixed in advance: per width we report the lr that converges the most seeds
# **in distribution** (token_acc = 1.0 at the training length). That criterion is measured at the
# trained length and is therefore independent of seq@24, the length-generalization quantity under
# comparison, so choosing on it cannot tilt the comparison either way. It is the same trust gate the
# committed sweep already applies to all 160 runs. The full lr grid is reported alongside the chosen
# point (experiments/stats.py breaks the width scan out by lr), so the selection is auditable.
# The 1e-3 pass runs first for both widths; if it converges cleanly the 5e-4 pass can be cancelled.
#
# tag_note keeps these distinct from the lr=2e-3 runs (without it skip_done would skip them as
# already present) and experiments/stats.py breaks the width scan out by lr.
#
# FOOTPRINT: one GPU on gpu-preempt, sequential, --requeue, save_model=False. Submit only after the
# main follow-up job has finished so this never adds a second concurrent GPU.

#SBATCH --job-name=cplxw-widthlr
#SBATCH --partition=gpu-preempt   # preemptible ONLY: never take a slot the TRM jobs could use
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G                 # d640 is 24.6M params; still small, but give it headroom
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-widthlr-%j.out
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
export PYTHONUNBUFFERED=1        # otherwise per-run summaries sit in the stdout buffer for hours

for LR in 1e-3 5e-4; do
  NOTE="lr${LR//[.-]/}"          # 1e-3 -> lr1e3, 5e-4 -> lr5e4
  echo "=== real width scan d448/d640 at lr=${LR} (tag note: ${NOTE}) ==="
  uv run python -m experiments.sweep \
      --seeds 0 1 2 3 4 --no-save-model --out-dir results \
      --tasks reverse --core none --real-width-scan 448 640 \
      --lr-real "${LR}" --tag-note "${NOTE}" \
      --wandb --wandb-mode "${WANDB_MODE}" --wandb-group "widthlr-${SLURM_JOB_ID}"
done

echo "=== width-lr supplement complete ==="
uv run python -m experiments.stats --tag-prefix unity_ || true
