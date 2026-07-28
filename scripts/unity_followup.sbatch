#!/usr/bin/env bash
# Follow-up sweep on UMass Unity: the four arms that test *why* complex+NoPE wins reverse.
# Submit:  sbatch scripts/unity_followup.sbatch
#
# The committed 5-seed sweep established one surviving positive (reverse length generalization) but
# could not say what produces it. Each arm below removes one candidate explanation:
#
#   E5  needle re-eval at 256/512/1024   inference only, no training. The needle reading was
#                                        under-powered, not null; push length instead of seeds.
#   E1  weight-tying algebra ablation    split / dual / diag at identical width and identical
#                                        parameter count. Is the effect C, or structured tying?
#   E4  offset-law tasks                 revskip vs skipcopy: same stride, same answer length,
#                                        differing only in whether the retrieval offset depends on L.
#   E2  real width-scaling curve         d320/d448/d640 on reverse. The committed real baseline is
#                                        flat from d160 to d224; can real width buy the gap at all?
#
# FOOTPRINT: this deliberately requests ONE GPU on gpu-preempt and runs the arms sequentially, so it
# never competes for the `gpu` partition where the TRM jobs sit. --requeue + skip_done make eviction
# free: a preempted job resumes at the next unfinished run. save_model=False keeps the disk cost of
# ~50 runs at a few MB of JSON instead of ~2 GB of checkpoints.

#SBATCH --job-name=cplxw-followup
#SBATCH --partition=gpu-preempt   # preemptible ONLY: never take a slot the TRM jobs could use
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-followup-%j.out
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

# Without this, Python block-buffers stdout when it is a file rather than a tty, so the per-run
# summary lines sit in the buffer and the SLURM log looks idle for hours while runs are in fact
# completing. Only the (stderr) wandb chatter gets through, which is actively misleading when you are
# checking whether a long sweep is making progress. Judge progress by results/<tag>/result.json.
export PYTHONUNBUFFERED=1

COMMON=(--seeds 0 1 2 3 4 --no-save-model --out-dir results
        --wandb --wandb-mode "${WANDB_MODE}" --wandb-group "followup-${SLURM_JOB_ID}")

# --- E5: needle at 256/512/1024, inference only over the committed checkpoints ------------------
# Cheapest arm and it answers a live question, so it runs first. eval-batch 32 keeps the O(batch*T^2)
# attention buffers near 1 GB at T~2050 instead of ~17 GB at the training default.
echo "=== E5: needle re-eval (no training) ==="
uv run python -m experiments.reeval \
    --match "unity_needle_*_none_*" --lengths 256 512 1024 --eval-seqs 512 --eval-batch 32

# --- E1: weight-tying algebra ablation ----------------------------------------------------------
# reverse is the only task with a surviving effect, so it is the only task worth the ablation.
echo "=== E1: algebra ablation (split/dual/diag) ==="
uv run python -m experiments.sweep "${COMMON[@]}" \
    --tasks reverse --core none --algebras split dual diag

# --- E4: offset-law tasks -----------------------------------------------------------------------
# core=pair gives complex+NoPE d160 vs the parameter-matched real+NoPE d224, the same head-to-head
# the main claim rests on, now on two tasks that differ only in length-dependence of the offset.
echo "=== E4: offset-law tasks (revskip vs skipcopy) ==="
uv run python -m experiments.sweep "${COMMON[@]}" \
    --tasks revskip skipcopy --core pair

# --- E2: real width-scaling curve on reverse ----------------------------------------------------
# Largest models, so it runs last: partial progress on the earlier arms is worth more.
echo "=== E2: real width scan (d320/d448/d640) ==="
uv run python -m experiments.sweep "${COMMON[@]}" \
    --tasks reverse --core none --real-width-scan 320 448 640

echo "=== followup complete ==="
uv run python -m experiments.stats --tag-prefix unity_ || true
