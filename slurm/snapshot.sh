#!/bin/bash
# Create an immutable code snapshot for a cluster run and print its absolute
# path (issue #49). The snapshot is a detached `git worktree` of HEAD: cheap
# (shares the object store with the checkout), and unaffected by any later
# `git switch` / `git pull` / edits in the live checkout — so queued or
# in-flight SLURM tasks always execute the code that was submitted.
#
# submit.sh calls this automatically. For a pinned single-node run:
#   CODE_PATH=$(bash slurm/snapshot.sh) sbatch --export=ALL slurm/search.slurm
#
# Snapshots live in <repo>/.snapshots/<timestamp>-<shortsha>-<pid> and are
# removed by consolidate.slurm / search.slurm after a successful run. Stale
# ones (failed runs) are kept for pinned retries — clean up with
# `git worktree remove --force <path>` (+ `git worktree prune`).
set -euo pipefail

cd "$(dirname "$0")/.."

# Only stdout is the snapshot path (callers use command substitution) — every
# other message goes to stderr.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "WARNING: uncommitted changes will NOT be in the snapshot (it pins HEAD)" >&2
fi

git worktree prune
SNAP="$(pwd)/.snapshots/$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)-$$"
git worktree add --quiet --detach "$SNAP" HEAD >&2
echo "$SNAP"
