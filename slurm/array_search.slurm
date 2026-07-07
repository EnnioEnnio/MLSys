#!/bin/bash
#SBATCH -A {account_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:<YOUR_USERNAME>
#SBATCH --partition=gpu-batch
#SBATCH -C ARCH:X86
#SBATCH --time=2:00:00
#SBATCH --gpus=a100:1
#SBATCH --mem=64G
#SBATCH --job-name=mlsys-array
#SBATCH --output=slurm-mlsys-array-%A_%a.out
#SBATCH --array=0-15%4

# SLURM job array: one model per task, `--strategy full_eval` each, so a slow or
# OOM-ing model only costs its own task. Fragments land in
# runs/<ARRAY_JOB_ID>/<ARRAY_JOB_ID>_task_<n>/; the dependent consolidate.slurm
# merges them and recomputes regret.json (see slurm/README.md).
#
# --array bound = pool size - 1. Don't hand-edit it: submit via `bash slurm/submit.sh`,
# which derives the bound from `mlsys list-models --count` (the same load_specs()
# order the tasks use to map index -> model). %4 throttles concurrent tasks.
#
# Retrying failed tasks: the retry job gets a NEW array job id, so pin the original
# experiment dir explicitly:
#   sbatch --array=3,7 --export=ALL,RUN_ID=<orig_id> slurm/array_search.slurm
# IMPORTANT: Update REPO_PATH below to match your cluster home directory path.
# NOTE: Excludes V100 GPUs (CUDA 7.0) since PyTorch 25.03 requires CUDA capability >= 7.5.

# Fail loudly: any unset var, failed command, or broken pipe aborts the task
# non-zero so the afterok dependency blocks consolidation.
set -euo pipefail

REPO_PATH=/sc/projects/<GROUP>/<Project>/<*Path>

# RUN_ID names the experiment dir; defaults to this array's job id. Override it
# (see retry recipe above) so retried tasks write into the original experiment.
RUN_ID=${RUN_ID:-$SLURM_ARRAY_JOB_ID}
HIDDEN=${HIDDEN:-0}
HEAD_REPEATS=${HEAD_REPEATS:-3}
OUT=/workspace/runs/${RUN_ID}/${RUN_ID}_task_${SLURM_ARRAY_TASK_ID}

echo "Starting mlsys array task"
echo "Array job: $SLURM_ARRAY_JOB_ID  task: $SLURM_ARRAY_TASK_ID  run id: $RUN_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Step 1: Set up container with dependencies (reuses container across jobs)
# The container name is SHARED with search.slurm (build-once reuse — a per-task
# name would multiply the NGC pull + pip install by N tasks in time and disk).
# Enroot named containers are node-local, so the only creation race is between
# tasks co-located on one node (bounded by the %N throttle): the per-node flock
# serializes Step 1 — the second co-located task waits, finds the container
# built, and no-ops through the install. If the cluster's pyxis/enroot still
# misbehaves under concurrent reuse, fall back to a per-task container name
# (CONTAINER_NAME=...-task$SLURM_ARRAY_TASK_ID) and accept the duplicated setup.
echo "Setting up container with dependencies..."
CONTAINER_NAME=mlsys-$USER-pytorch2503
flock /tmp/mlsys-container-$USER.lock \
srun --container-image="nvcr.io#nvidia/pytorch:25.03-py3" \
     --container-name=$CONTAINER_NAME \
     --container-writable \
     --container-mounts=$REPO_PATH:/workspace \
     bash -c '
set -euo pipefail

# NGC PyTorch >=25.03 ships /etc/pip/constraint.txt, which pins pyarrow==19 and
# dill==0.3.9 for its data-science stack. Modern datasets (>=4.1) needs pyarrow>=21,
# so those pins make datasets impossible to install. NVIDIA documents the fix as
# "remove the version constraints for the packages you want to overwrite". We do
# that on a COPY of the file (never touching the original) and point pip at the copy,
# so every OTHER NGC pin (numpy, the torch/CUDA stack, ...) stays in force.
SRC="${PIP_CONSTRAINT:-/etc/pip/constraint.txt}"
if [ -f "$SRC" ]; then
  grep -ivE "^(pyarrow|dill)[<>=!~ ]" "$SRC" > /tmp/pip-constraint.txt || true
  export PIP_CONSTRAINT=/tmp/pip-constraint.txt
fi

pip install --no-cache-dir -e /workspace
'

# Step 2: Run this task's single model
echo "Running model search for task $SLURM_ARRAY_TASK_ID..."
# expandable_segments lets the CUDA allocator grow/shrink segments instead of
# locking fixed-size blocks (see search.slurm for the full rationale).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FINETUNE_ARGS="--finetune-epochs 3 --finetune-lr 2e-5 --finetune-batch-size 64"

export OUT HIDDEN HEAD_REPEATS FINETUNE_ARGS
srun --container-name=$CONTAINER_NAME \
     --container-mounts=$REPO_PATH:/workspace \
     --container-env=WANDB_API_KEY,PYTORCH_CUDA_ALLOC_CONF,SLURM_ARRAY_TASK_ID,OUT,HIDDEN,HEAD_REPEATS,FINETUNE_ARGS \
     --container-workdir=/workspace \
     bash -c '
set -euo pipefail
# Index -> model name from the same load_specs() order submit.sh sized the array
# with. Out of range exits non-zero -> the task fails -> afterok blocks consolidation.
MODEL=$(python -m mlsys list-models --index "$SLURM_ARRAY_TASK_ID")
echo "Task $SLURM_ARRAY_TASK_ID -> $MODEL"
# results.jsonl is append-only; a retried task must not double-append its fragment.
rm -f "$OUT/results.jsonl" "$OUT/regret.json"
python -m mlsys search \
  --dataset wine_reviews \
  --strategy full_eval \
  --models "$MODEL" \
  --hidden "$HIDDEN" \
  --head-repeats "$HEAD_REPEATS" \
  --output-dir "$OUT" \
  $FINETUNE_ARGS \
  --wandb
'

echo "Task completed!"
echo "Fragment: runs/$RUN_ID/${RUN_ID}_task_${SLURM_ARRAY_TASK_ID}/results.jsonl"
