#!/bin/bash
#SBATCH -A {account_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:<YOUR_USERNAME>
#SBATCH --partition=gpu-batch
#SBATCH -C ARCH:X86
#SBATCH --time=2:00:00
#SBATCH --gpus=a100:1
#SBATCH --mem=64G
#SBATCH --job-name=mlsys-search
#SBATCH --output=slurm-mlsys-search-%j.out

# Runs `python -m mlsys search` against config/datasets.yaml + config/models.yaml.
# NGC base image + reused named container. Deps come from pyproject.toml via
# `pip install -e /workspace`, which runs every job. NGC's pip constraint file
# (see Step 1) pins pyarrow/dill in a way that blocks modern datasets, so we strip
# just those two pins before installing; all other NGC pins stay in force.
# IMPORTANT: Update REPO_PATH below to match your cluster home directory path.
# NOTE: Excludes V100 GPUs (CUDA 7.0) since PyTorch 25.03 requires CUDA capability >= 7.5.
#
# Strategy selection (override via `sbatch --export=STRATEGY=<name>`):
#   frozen     — FC head on the frozen backbone; cheap proxy ranking.
#                Fast (~30 min). Does NOT produce regret.json.
#                Example: sbatch --export=STRATEGY=frozen search.slurm
#   finetune   — Unfreeze backbone + train backbone+head jointly; ground-truth scores.
#                Expensive (~2 h). Does NOT produce regret.json.
#                Example: sbatch --export=STRATEGY=finetune search.slurm
#   full_eval  — Both passes over the pool, then writes the regret-vs-budget curve.
#                Most expensive (~4 h). Produces both results.jsonl and regret.json.
#                Example: sbatch --export=STRATEGY=full_eval search.slurm  (default)
# Adjust --time above to match the chosen strategy.

# Fail loudly: any unset var, failed command, or broken pipe aborts the job
# non-zero so SLURM/CI don't treat a failed setup as success.
set -euo pipefail

REPO_PATH=/sc/projects/<GROUP>/<Project>/<*Path>

# Default to full_eval; override at submission time with --export=STRATEGY=<name>.
STRATEGY=${STRATEGY:-full_eval}

echo "Starting mlsys search job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Step 1: Set up container with dependencies (reuses container across jobs)
echo "Setting up container with dependencies..."
CONTAINER_NAME=mlsys-$USER-pytorch2503
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

# Step 2: Run model search
echo "Running model search (strategy=$STRATEGY)..."
# expandable_segments lets the CUDA allocator grow/shrink segments instead of
# locking fixed-size blocks, so reserved memory from one backbone can be reused
# by the next differently-sized one. Without it the ~N-model loop fragments the
# GPU and OOMs on a small contiguous allocation. Forwarded into the container.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Finetune flags are only meaningful for the finetune and full_eval strategies;
# they are accepted but ignored by the frozen strategy.
FINETUNE_ARGS=""
if [[ "$STRATEGY" == "finetune" || "$STRATEGY" == "full_eval" ]]; then
  FINETUNE_ARGS="--finetune-epochs 8 --finetune-lr 2e-5 --finetune-batch-size 16"
fi

srun --container-name=$CONTAINER_NAME \
     --container-mounts=$REPO_PATH:/workspace \
     --container-env=WANDB_API_KEY,PYTORCH_CUDA_ALLOC_CONF \
     --container-workdir=/workspace \
     python -m mlsys search \
       --dataset wine_reviews \
       --strategy $STRATEGY \
       --output-dir /workspace/runs/$SLURM_JOB_ID \
       $FINETUNE_ARGS \
       --wandb

echo "Job completed!"
echo "Output files:"
echo "  - runs/$SLURM_JOB_ID/results.jsonl"
if [[ "$STRATEGY" == "full_eval" ]]; then
  echo "  - runs/$SLURM_JOB_ID/regret.json"
fi
