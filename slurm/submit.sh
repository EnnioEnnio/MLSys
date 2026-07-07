#!/bin/bash
# Submit the job-array search + dependent consolidation as one pipeline:
#   bash slurm/submit.sh
# Optional env: HIDDEN=<width> (head hidden width, exported to BOTH jobs so the
# W&B head token can't drift; 0 = linear probe).
set -euo pipefail

HIDDEN=${HIDDEN:-0}

# Pool size from the same source of truth as the tasks' index->model mapping
# (load_specs order). Falls back to grep only if the package isn't importable
# on the login node.
N=$(python -m mlsys list-models --count 2>/dev/null || grep -c '^- name:' config/models.yaml)
[ "$N" -gt 0 ] || { echo "empty model pool" >&2; exit 1; }

echo "Submitting array over $N models (0-$((N - 1)), throttle %4), HIDDEN=$HIDDEN"
ARRAY_ID=$(sbatch --parsable --array=0-$((N - 1))%4 --export=ALL,HIDDEN="$HIDDEN" \
  slurm/array_search.slurm)
echo "Array job: $ARRAY_ID"

CONSOLIDATE_ID=$(sbatch --parsable --dependency=afterok:"$ARRAY_ID" \
  --export=ALL,ARRAY_JOB_ID="$ARRAY_ID",HIDDEN="$HIDDEN" slurm/consolidate.slurm)
echo "Consolidate job: $CONSOLIDATE_ID (afterok:$ARRAY_ID)"
echo "Results will land in runs/$ARRAY_ID/"
