#!/bin/bash
#SBATCH -A sci-rabl-mlsys2026
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:<YOUR_USERNAME>
#SBATCH --partition=gpu-batch
#SBATCH -C ARCH:X86
#SBATCH --time=12:00:00
#SBATCH --gpus=a100:1
#SBATCH --mem=64G
#SBATCH --job-name=mlsys-verify-deberta-lr-override
#SBATCH --output=slurm-mlsys-verify-deberta-lr-override-%j.out

# Re-runs `finetune` on deberta-v3-base only, across all 4 head widths, to confirm the
# per-model finetune_lr override (config/models.yaml, 1e-5 instead of the standard 2e-5)
# clears its persistent divergence. Every other previously-diverged model (deberta-v3-small,
# all-mpnet-base-v2, roberta-base, electra-base-discriminator) is already confirmed fixed
# by the LR warmup+decay schedule alone (see runs/2327067, 2327595, 2329575); deberta-v3-base
# was the sole holdout — negative/near-zero r2 at every width, grad_norm_max up to 343 at
# hidden=512 (run 2329575).
#
# The override is read automatically from config/models.yaml's finetune_lr entry, so no
# --finetune-lr flag override is needed here — finetune_candidate applies it internally
# for this model regardless of what --finetune-lr is set to on the CLI.
#
# Uses --strategy finetune only (no frozen pass, no regret.json) since we only care
# about whether the joint loop still diverges, not proxy ranking.
# IMPORTANT: Update REPO_PATH below to match your cluster home directory path.

set -euo pipefail

REPO_PATH=/sc/projects/sci-rabl/mlsys2026/jitr/MLSys

echo "Starting deberta-v3-base LR override verification job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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
SRC="${PIP_CONSTRAINT:-/etc/pip/constraint.txt}"
if [ -f "$SRC" ]; then
  grep -ivE "^(pyarrow|dill)[<>=!~ ]" "$SRC" > /tmp/pip-constraint.txt || true
  export PIP_CONSTRAINT=/tmp/pip-constraint.txt
fi
pip install --no-cache-dir -e /workspace
'

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WIDTHS=(0 128 256 512)

for WIDTH in "${WIDTHS[@]}"; do
  echo "=== hidden=$WIDTH model=deberta-v3-base ==="
  srun --container-name=$CONTAINER_NAME \
       --container-mounts=$REPO_PATH:/workspace \
       --container-env=WANDB_API_KEY,PYTORCH_CUDA_ALLOC_CONF \
       --container-workdir=/workspace \
       python -m mlsys search \
         --dataset wine_reviews \
         --strategy finetune \
         --models deberta-v3-base \
         --hidden "$WIDTH" \
         --device cuda \
         --finetune-epochs 10 \
         --finetune-lr 2e-5 \
         --finetune-batch-size 64 \
         --warmup-epochs 2 \
         --output-dir /workspace/runs/$SLURM_JOB_ID/hidden_$WIDTH \
         --wandb
done

echo "Job completed!"
echo "Output files (check the 'metrics.r2' field per row for negative values):"
for WIDTH in "${WIDTHS[@]}"; do
  echo "  - runs/$SLURM_JOB_ID/hidden_$WIDTH/results.jsonl"
done
