#!/bin/bash
# Single entry point for the cluster pipeline: job-array search (one model per
# task) + dependent consolidation.
#
#   bash slurm/submit.sh
#   DATASET=wine_reviews HIDDEN=512 FINETUNE_LR=1e-5 bash slurm/submit.sh
#
# Every knob below is an env var with the pipeline default; set any of them
# inline (as above) or export them beforehand. They are forwarded to both SLURM
# jobs via --export=ALL, so the array tasks and the consolidate step can't
# drift apart. Retries bypass this script (see slurm/README.md) — repeat any
# non-default knobs there.
#
# NOTE: HPI login nodes (lx*) kill this script at exec time ("This command is
# not allowed on the login node!") — run it from a run node instead:
# https://docs.sc.hpi.de/cluster/Resources/Run-Nodes/
set -euo pipefail

# --- Search parameters (forwarded to `python -m mlsys search` in every task) ---
export DATASET=${DATASET:-wine_reviews}                # dataset name from config/datasets.yaml
export HIDDEN=${HIDDEN:-256}                             # head hidden width; 0 = linear probe
export HEAD_REPEATS=${HEAD_REPEATS:-1}                 # frozen-pass head repeats (variance)
export EPOCHS=${EPOCHS:-30}                            # head epochs
export BATCH_SIZE=${BATCH_SIZE:-64}                    # encode/head batch size
export FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-10}          # joint-loop epochs
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-2}               # head-only warmup epochs before the joint loop; 0 = off
export FINETUNE_LR=${FINETUNE_LR:-2e-5}                # backbone learning rate
export FINETUNE_BATCH_SIZE=${FINETUNE_BATCH_SIZE:-64}  # joint-loop batch size
export GRAD_CLIPPING=${GRAD_CLIPPING:-0}               # joint-loop max grad norm; 0 = off (norm still logged)
export STANDARDIZE_TARGETS=${STANDARDIZE_TARGETS:-1}   # z-score targets on train stats; 0 = off (non-regression pilots)

# --- SLURM shape ---
THROTTLE=${THROTTLE:-4}          # max concurrent array tasks (%N)

# Run from anywhere: resolve the repo root for config/ and slurm/ paths.
cd "$(dirname "$0")/.."

# --- Code snapshot (issue #49) ---
# Tasks read the code when they RUN, not when they are submitted, so the run is
# pinned to an immutable worktree snapshot of HEAD: after this script returns
# you can `git switch` / `git pull` / edit this checkout freely without
# affecting queued or in-flight tasks. REPO_PATH (this checkout) stays the home
# of runs/ via a separate bind mount, so results outlive the snapshot, which
# consolidate.slurm removes after a successful merge.
export REPO_PATH="$(pwd)"
export CODE_PATH="$(bash slurm/snapshot.sh)"
echo "Code snapshot: $CODE_PATH ($(git -C "$CODE_PATH" log -1 --oneline))"

# Pool size, ideally from the same source of truth as the tasks' index->model
# mapping (load_specs order). HPI login nodes block python outright (the wrapper
# may even print its refusal to stdout with exit 0), so validate the output is a
# number and otherwise count models.yaml entries — same file, same order.
N=$(python -m mlsys list-models --count 2>/dev/null || true)
[[ "$N" =~ ^[0-9]+$ ]] || N=$(grep -c '^- name:' config/models.yaml)
[ "$N" -gt 0 ] || { echo "empty model pool" >&2; exit 1; }

echo "Submitting array over $N models (0-$((N - 1)), throttle %$THROTTLE)"
echo "  dataset=$DATASET hidden=$HIDDEN head_repeats=$HEAD_REPEATS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "  finetune: epochs=$FINETUNE_EPOCHS warmup_epochs=$WARMUP_EPOCHS lr=$FINETUNE_LR batch_size=$FINETUNE_BATCH_SIZE grad_clipping=$GRAD_CLIPPING"
echo "  standardize_targets=$STANDARDIZE_TARGETS"
ARRAY_ID=$(sbatch --parsable --array=0-$((N - 1))%"$THROTTLE" --export=ALL \
  slurm/array_search.slurm)
echo "Array job: $ARRAY_ID"

CONSOLIDATE_ID=$(sbatch --parsable --dependency=afterok:"$ARRAY_ID" \
  --export=ALL,ARRAY_JOB_ID="$ARRAY_ID" slurm/consolidate.slurm)
echo "Consolidate job: $CONSOLIDATE_ID (afterok:$ARRAY_ID)"

# Provenance: pin the run to its exact code (issue #49 nice-to-have).
mkdir -p "runs/$ARRAY_ID"
git -C "$CODE_PATH" log -1 --format='%H %s' > "runs/$ARRAY_ID/commit.txt"

echo "Results will land in runs/$ARRAY_ID/"
