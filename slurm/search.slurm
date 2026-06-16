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

# Fail loudly: any unset var, failed command, or broken pipe aborts the job
# non-zero so SLURM/CI don't treat a failed setup as success.
set -euo pipefail

REPO_PATH=/sc/projects/<GROUP>/<Project>/<*Path>

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
echo "Running model search..."
# expandable_segments lets the CUDA allocator grow/shrink segments instead of
# locking fixed-size blocks, so reserved memory from one backbone can be reused
# by the next differently-sized one. Without it the ~N-model loop fragments the
# GPU and OOMs on a small contiguous allocation. Forwarded into the container.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun --container-name=$CONTAINER_NAME \
     --container-mounts=$REPO_PATH:/workspace \
     --container-env=WANDB_API_KEY,PYTORCH_CUDA_ALLOC_CONF \
     --container-workdir=/workspace \
     python -m mlsys search \
       --dataset wine_reviews \
       --strategy full_eval \
       --output-dir /workspace/runs/$SLURM_JOB_ID \
       --wandb

echo "Job completed!"
echo "Output files:"
echo "  - runs/$SLURM_JOB_ID/results.jsonl"
