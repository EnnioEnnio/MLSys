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
# Mirrors generate_embeddings.slurm: NGC base image, reused named container,
# pip install gated by a `pip list | grep -q` idiom so reruns skip the install.
# IMPORTANT: Update REPO_PATH below to match your cluster home directory path.
# NOTE: Excludes V100 GPUs (CUDA 7.0) since PyTorch 25.03 requires CUDA capability >= 7.5.

# Fail loudly: any unset var, failed command, or broken pipe aborts the job
# non-zero so SLURM/CI don't treat a failed setup as success.
set -euo pipefail

REPO_PATH=/sc/home/<YOUR_USERNAME>/mlsys

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
     bash -c 'set -euo pipefail; \
       pip list 2>/dev/null | grep -q "^sentence-transformers" || \
         pip install --no-cache-dir transformers datasets sentence-transformers model2vec \
           accelerate huggingface_hub pyyaml scipy tqdm wandb python-dotenv; \
       pip install --no-cache-dir -e /workspace'

# Step 2: Run model search
echo "Running model search..."
srun --container-name=$CONTAINER_NAME \
     --container-mounts=$REPO_PATH:/workspace \
     --container-env=WANDB_API_KEY \
     --container-workdir=/workspace \
     python -m mlsys search \
       --dataset wine_reviews \
       --strategy full_eval \
       --output-dir /workspace/runs/$SLURM_JOB_ID \
       --wandb

echo "Job completed!"
echo "Output files:"
echo "  - runs/$SLURM_JOB_ID/results.jsonl"
