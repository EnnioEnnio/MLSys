#!/bin/bash
#SBATCH -A {account_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:<YOUR_USERNAME>
#SBATCH --partition=gpu-batch
#SBATCH -C ARCH:X86
#SBATCH --time=4:00:00
#SBATCH --gpus=a100:1
#SBATCH --mem=64G
#SBATCH --job-name=mlsys-verify-divergence-fix
#SBATCH --output=slurm-mlsys-verify-divergence-fix-%j.out

# Re-runs `finetune` on exactly the (model, head-width) combos that diverged (negative
# r2) in exp_wine_16, to confirm the LR warmup+decay schedule (train_full_model) clears
# the divergence map. See the issue's Findings section for the source of this list:
#   deberta-v3-small/base — all 4 head widths (FCH, MLP_128, MLP_256, MLP_512)
#   electra-base-discriminator — MLP_256, MLP_512
#   roberta-base — MLP_128
#   all-mpnet-base-v2 — FCH
#
# Uses --strategy finetune only (no frozen pass, no regret.json) since we only care
# about whether the joint loop still diverges, not proxy ranking.
# IMPORTANT: Update REPO_PATH below to match your cluster home directory path.

set -euo pipefail

REPO_PATH=/sc/projects/<GROUP>/<Project>/<*Path>

echo "Starting divergence-fix verification job"
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

# One (hidden width -> model subset) pair per diverged group above.
WIDTHS=(0 128 256 512)
MODELS_FOR_WIDTH=(
  "deberta-v3-small,deberta-v3-base,all-mpnet-base-v2"
  "deberta-v3-small,deberta-v3-base,roberta-base"
  "deberta-v3-small,deberta-v3-base,electra-base-discriminator"
  "deberta-v3-small,deberta-v3-base,electra-base-discriminator"
)

for i in "${!WIDTHS[@]}"; do
  WIDTH="${WIDTHS[$i]}"
  MODELS="${MODELS_FOR_WIDTH[$i]}"
  echo "=== hidden=$WIDTH models=$MODELS ==="
  srun --container-name=$CONTAINER_NAME \
       --container-mounts=$REPO_PATH:/workspace \
       --container-env=WANDB_API_KEY,PYTORCH_CUDA_ALLOC_CONF \
       --container-workdir=/workspace \
       python -m mlsys search \
         --dataset wine_reviews \
         --strategy finetune \
         --models "$MODELS" \
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
