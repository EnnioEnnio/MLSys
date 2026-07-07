# slurm/

Cluster launch for `mlsys search`.

**One-shot setup** (all scripts): set `REPO_PATH` to your cluster checkout, `--mail-user` to
your slack handle, `-A {account_name}` to your SLURM account. For W&B, `export WANDB_API_KEY=...`
in your shell rc **before** submitting — the cluster does not read `.env`.

## Quickstart

```bash
bash slurm/submit.sh            # HIDDEN=512 bash slurm/submit.sh for an MLP head
```

Submits a **job array** (`array_search.slurm`, one model per task, `full_eval` each, `%4`
concurrent — the array bound comes from `mlsys list-models --count`) plus a dependent
consolidation job (`consolidate.slurm`, `afterok`). Each task writes a fragment to
`runs/<ARRAY_JOB_ID>/<ARRAY_JOB_ID>_task_<n>/` and streams its own W&B run live;
consolidation merges the fragments into `runs/<ARRAY_JOB_ID>/results.jsonl`, recomputes
`regret.json` as if a single-node `full_eval` had produced it, exports analysis-ready CSVs,
and pushes one consolidated W&B run named like a single-node run.

## From run to analysis

Consolidation already wrote the three CSVs in the exact filename grammar `mlsys analyze`
expects — no renaming:

```bash
scp 'cluster:<REPO_PATH>/runs/<id>/*.csv' results/<experiment>/
mlsys analyze results/<experiment>
```

Alternative road: download the `results_frozen` / `results_finetune` / regret tables from the
consolidated W&B run and append `_frozen` / `_finetune` / `_regret` — same files.

## When a task fails

The consolidate job shows `DependencyNeverSatisfied` in `squeue` and never runs (cancel it with
`scancel` if it lingers — auto-purge is scheduler-config dependent). Retry only the failed
task ids, pinning the **original** experiment dir (a retry gets a new array job id):

```bash
sbatch --array=3,7 --export=ALL,RUN_ID=<orig_id>,HIDDEN=<same_as_before> slurm/array_search.slurm
sbatch --dependency=afterok:<retry_job_id> --export=ALL,ARRAY_JOB_ID=<orig_id>,HIDDEN=<same> \
  slurm/consolidate.slurm
```

Retried tasks overwrite their own fragment (`rm -f` before the run), so no double-appending.
`mlsys consolidate` is idempotent — safe to re-run manually anytime.

## Single-node fallback

`sbatch slurm/search.slurm` runs the whole pool in one job (edit `STRATEGY` inside; results in
`runs/$SLURM_JOB_ID/`). Simpler, but the job holds a GPU for the slowest model's wall-clock and
one OOM kills the entire run — prefer the array.
