#!/bin/bash
#SBATCH -A {account_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:<YOUR_USERNAME>
#SBATCH --partition=gpu-batch
#SBATCH -C ARCH:X86
#SBATCH --time=18:00:00
#SBATCH --gpus=a100:1
#SBATCH --mem=64G
#SBATCH --job-name=mlsys-search
#SBATCH --output=slurm-mlsys-search-%j.out

# Runs `python -m mlsys search` against config/datasets.yaml + config/models.yaml.
# NGC base image + reused named container. Deps come from pyproject.toml via
# `pip install -e /workspace`, which runs every job. NGC's pip constraint file
# (see Step 1) pins pyarrow/dill in a way that blocks modern datasets, so we strip
# just those two pins before installing; all other NGC pins stay in force.
# IMPORTANT: Update the REPO_PATH default below to match your cluster checkout path.
# NOTE: Excludes V100 GPUs (CUDA 7.0) since PyTorch 25.03 requires CUDA capability >= 7.5.
#
# Branch-switch safety (issue #49): a plain `sbatch slurm/search.slurm` executes
# whatever the checkout contains when the job STARTS. To pin the code at submit
# time (so you can keep working in the checkout while the job is queued):
#   CODE_PATH=$(bash slurm/snapshot.sh) sbatch --export=ALL slurm/search.slurm
# The snapshot is removed automatically after a successful run; results always
# land in $REPO_PATH/runs via a separate mount, so they outlive the snapshot.
#
# Strategy selection (edit the STRATEGY variable below):
#   frozen     — FC head on the frozen backbone; cheap proxy ranking.
#                Fast. Does NOT produce regret.json.
#   finetune   — Unfreeze backbone + train backbone+head jointly; ground-truth scores.
#                Expensive. Does NOT produce regret.json.
#   full_eval  — Both passes over the pool, then writes the regret-vs-budget curve.
#                Most expensive. Produces both results.jsonl and regret.json.
# Adjust --time above to match the chosen strategy.

# Fail loudly: any unset var, failed command, or broken pipe aborts the job
# non-zero so SLURM/CI don't treat a failed setup as success.
set -euo pipefail

# REPO_PATH is the live checkout: home of runs/ (mounted separately below so
# results outlive code snapshots). CODE_PATH is the code that actually runs —
# a worktree snapshot when submitted via the pinned recipe above, otherwise
# the live checkout.
REPO_PATH=${REPO_PATH:-/sc/projects/<GROUP>/<Project>/<*Path>}
CODE_PATH=${CODE_PATH:-$REPO_PATH}
MOUNTS=$CODE_PATH:/workspace,$REPO_PATH/runs:/workspace/runs
mkdir -p "$REPO_PATH/runs"

echo "Starting mlsys search job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Code: $CODE_PATH ($(git -C "$CODE_PATH" log -1 --oneline 2>/dev/null || echo 'git unavailable'))"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Provenance: pin the run to its exact code (issue #49 nice-to-have).
mkdir -p "$REPO_PATH/runs/$SLURM_JOB_ID"
git -C "$CODE_PATH" log -1 --format='%H %s' > "$REPO_PATH/runs/$SLURM_JOB_ID/commit.txt" \
  || echo "WARNING: could not record commit provenance"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Step 1: Set up container with dependencies (reuses container across jobs)
echo "Setting up container with dependencies..."
CONTAINER_NAME=mlsys-$USER-pytorch2503
srun --container-image="nvcr.io#nvidia/pytorch:25.03-py3" \
     --container-name=$CONTAINER_NAME \
     --container-writable \
     --container-mounts=$MOUNTS \
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
# Search strategy for this run (see header for options). Edit and re-submit to change.
STRATEGY=full_eval

# Number of head training repetitions (predictions averaged to reduce ranking variance
# on the frozen proxy pass). For full_eval the finetune pass is forced to head_repeats=1
# internally (REGRET note 2), so this setting only affects the frozen proxy pass.
HEAD_REPEATS=1

# MLP-head activation between the two head layers (relu/gelu/tanh/silu; ignored
# when --hidden 0, i.e. the linear probe). Env-overridable so single-node sweeps
# (see docs/head-activation.md) don't need to hand-edit this file per run:
#   ACTIVATION=gelu sbatch --export=ALL slurm/search.slurm
ACTIVATION=${ACTIVATION:-relu}

echo "Running model search (strategy=$STRATEGY, activation=$ACTIVATION)..."
# expandable_segments lets the CUDA allocator grow/shrink segments instead of
# locking fixed-size blocks, so reserved memory from one backbone can be reused
# by the next differently-sized one. Without it the ~N-model loop fragments the
# GPU and OOMs on a small contiguous allocation. Forwarded into the container.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Finetune flags are only meaningful for the finetune and full_eval strategies;
# they are accepted but ignored by the frozen strategy.
FINETUNE_ARGS=""
if [[ "$STRATEGY" == "finetune" || "$STRATEGY" == "full_eval" ]]; then
  FINETUNE_ARGS="--finetune-epochs 10 --finetune-lr 2e-5 --finetune-batch-size 64 --warmup-epochs 2 --grad-clipping 0"
fi

# Every value-taking flag is passed explicitly at its standard value so nothing is
# forgotten on a future run. Two flags are left unset on purpose: --models (omitted =
# whole pool; there is no explicit "all" token) and --cache-embeddings (v2 stub, no-op).
srun --container-name=$CONTAINER_NAME \
     --container-mounts=$MOUNTS \
     --container-env=WANDB_API_KEY,PYTORCH_CUDA_ALLOC_CONF \
     --container-workdir=/workspace \
     python -m mlsys search \
       --dataset wine_reviews \
       --strategy $STRATEGY \
       --epochs 30 \
       --batch-size 64 \
       --hidden 256 \
       --activation $ACTIVATION \
       --device cuda \
       --head-repeats $HEAD_REPEATS \
       --standardize-targets \
       --output-dir /workspace/runs/$SLURM_JOB_ID \
       $FINETUNE_ARGS \
       --wandb

# Snapshot cleanup: the run succeeded (set -e) and results live in
# $REPO_PATH/runs via the separate mount. Only fires for pinned runs
# (CODE_PATH != REPO_PATH); a failed job keeps its snapshot for a pinned retry
# (`sbatch --export=ALL,CODE_PATH=<snapshot> slurm/search.slurm`).
if [ "$CODE_PATH" != "$REPO_PATH" ]; then
  git -C "$REPO_PATH" worktree remove --force "$CODE_PATH" \
    && echo "Removed code snapshot $CODE_PATH" \
    || echo "WARNING: could not remove code snapshot $CODE_PATH — remove it manually"
fi

echo "Job completed!"
echo "Output files:"
echo "  - runs/$SLURM_JOB_ID/results.jsonl"
if [[ "$STRATEGY" == "full_eval" ]]; then
  echo "  - runs/$SLURM_JOB_ID/regret.json"
fi
