#!/bin/bash
#SBATCH -A {account_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:<YOUR_USERNAME>
#SBATCH --partition=cpu-batch
#SBATCH -C ARCH:X86
# Generous: the named container is node-local, so this job usually lands on a
# node that must cold-pull the NGC image (~15 GB) + pip install before the
# (seconds-long) merge. Subsequent runs on the same node reuse the container.
#SBATCH --time=00:45:00
#SBATCH --mem=8G
#SBATCH --job-name=mlsys-consolidate
#SBATCH --output=slurm-mlsys-consolidate-%j.out

# Dependent step after array_search.slurm: merge the per-task results.jsonl
# fragments in runs/$ARRAY_JOB_ID/, recompute regret.json over the merged pool,
# export analysis-ready CSVs, and push the consolidated W&B run. Submitted by
# slurm/submit.sh with --dependency=afterok:<array_id>. Consolidation is pure
# CPU work (stdlib json/csv merge), so it runs on the cpu-batch partition.
# IMPORTANT: Update the REPO_PATH default below to match your cluster checkout path.

set -euo pipefail

# Same REPO_PATH/CODE_PATH split as array_search.slurm: runs/ lives in the live
# checkout, the code comes from the snapshot submit.sh pinned (issue #49) —
# both jobs get the same values via --export=ALL, so they can't drift apart.
REPO_PATH=${REPO_PATH:-/sc/projects/<GROUP>/<Project>/<*Path>}
CODE_PATH=${CODE_PATH:-$REPO_PATH}
MOUNTS=$CODE_PATH:/workspace,$REPO_PATH/runs:/workspace/runs

# The experiment dir to consolidate; submit.sh exports it. HIDDEN must match the
# array run (submit.sh exports the same value to both jobs so the W&B head token
# can't drift). STRATEGY likewise must match the array run and is forwarded to
# `mlsys consolidate --strategy`, which derives --allow-partial itself: only
# full_eval writes both frozen+finetune rows, so any other strategy merges partial
# with no regret.json (see search/consolidate.py).
: "${ARRAY_JOB_ID:?set via --export=ALL,ARRAY_JOB_ID=<array job id>}"
HIDDEN=${HIDDEN:-256}
STRATEGY=${STRATEGY:-full_eval}
if [ "$STRATEGY" != "full_eval" ]; then
  echo "strategy=$STRATEGY (not full_eval): merging with --allow-partial, no regret.json"
fi

# The NGC image declares NVIDIA_VISIBLE_DEVICES=all, which makes enroot's GPU
# hook (98-nvidia.sh) run nvidia-container-cli at container start — a binary
# that doesn't exist on CPU nodes, so the container dies before our code runs.
# "void" is enroot's documented switch to skip the GPU hook; --container-env
# below forces our value to override the image's.
export NVIDIA_VISIBLE_DEVICES=void

echo "Consolidating runs/$ARRAY_JOB_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Code: $CODE_PATH ($(git -C "$CODE_PATH" log -1 --oneline 2>/dev/null || echo 'git unavailable'))"

# Step 1: Set up container with dependencies — same shared container + per-node
# flock as array_search.slurm (see there for the race/trade-off rationale).
echo "Setting up container with dependencies..."
CONTAINER_NAME=mlsys-$USER-pytorch2503
flock /tmp/mlsys-container-$USER.lock \
srun --container-image="nvcr.io#nvidia/pytorch:25.03-py3" \
     --container-name=$CONTAINER_NAME \
     --container-writable \
     --container-mounts=$MOUNTS \
     --container-env=NVIDIA_VISIBLE_DEVICES \
     bash -c '
set -euo pipefail

# Strip the pyarrow/dill pins from a COPY of the NGC constraint file (see
# array_search.slurm for the full rationale).
SRC="${PIP_CONSTRAINT:-/etc/pip/constraint.txt}"
if [ -f "$SRC" ]; then
  grep -ivE "^(pyarrow|dill)[<>=!~ ]" "$SRC" > /tmp/pip-constraint.txt || true
  export PIP_CONSTRAINT=/tmp/pip-constraint.txt
fi

pip install --no-cache-dir -e /workspace
'

# Step 2: Merge fragments + recompute regret + push the consolidated W&B run.
# Add --cleanup to remove the *_task_* fragment dirs after a successful merge:
#   python -m mlsys consolidate /workspace/runs/$ARRAY_JOB_ID --hidden "$HIDDEN" --strategy "$STRATEGY" --wandb --cleanup
srun --container-name=$CONTAINER_NAME \
     --container-mounts=$MOUNTS \
     --container-env=WANDB_API_KEY,ARRAY_JOB_ID,HIDDEN,STRATEGY,NVIDIA_VISIBLE_DEVICES \
     --container-workdir=/workspace \
     bash -c '
set -euo pipefail
python -m mlsys consolidate /workspace/runs/$ARRAY_JOB_ID --hidden "$HIDDEN" --strategy "$STRATEGY" --wandb
'

# Snapshot cleanup: the merge succeeded (set -e) and results live in
# $REPO_PATH/runs via the separate mount, so the pinned worktree is no longer
# needed. Only fires for snapshot runs (submit.sh sets CODE_PATH != REPO_PATH);
# --force because pip install -e leaves egg-info in the snapshot. If the array
# fails this job never runs and the snapshot survives for pinned retries.
if [ "$CODE_PATH" != "$REPO_PATH" ]; then
  git -C "$REPO_PATH" worktree remove --force "$CODE_PATH" \
    && echo "Removed code snapshot $CODE_PATH" \
    || echo "WARNING: could not remove code snapshot $CODE_PATH — remove it manually"
fi

echo "Job completed!"
echo "Output files:"
echo "  - runs/$ARRAY_JOB_ID/results.jsonl"
if [ "$STRATEGY" = "full_eval" ]; then
  echo "  - runs/$ARRAY_JOB_ID/regret.json"
fi
echo "  - runs/$ARRAY_JOB_ID/*.csv (analysis-ready)"
