# slurm/

Cluster launch for `mlsys search`.

**One-shot setup** (all scripts): set `REPO_PATH` to your cluster checkout, `--mail-user` to
your slack handle, `-A {account_name}` to your SLURM account. For W&B, `export WANDB_API_KEY=...`
in your shell rc **before** submitting — the cluster does not read `.env`.

## Quickstart

`submit.sh` is the single entry point — every experiment parameter is an env var set there
(or inline), forwarded to all SLURM jobs via `--export=ALL`. Run it **from a
[run node](https://docs.sc.hpi.de/cluster/Resources/Run-Nodes/)** — HPI login nodes (`lx*`)
kill the script at exec time (`sbatch` itself is allowed there, but scripts aren't):

```bash
bash slurm/submit.sh
DATASET=wine_reviews HIDDEN=512 FINETUNE_LR=1e-5 bash slurm/submit.sh
```

| Knob | Default | Maps to |
|---|---|---|
| `DATASET` | `wine_reviews` | `--dataset` |
| `HIDDEN` | `256` (MLP - 0 for linear probe) | `--hidden` |
| `HEAD_REPEATS` | `1` | `--head-repeats` |
| `EPOCHS` | `30` | `--epochs` |
| `BATCH_SIZE` | `64` | `--batch-size` |
| `FINETUNE_EPOCHS` | `10` | `--finetune-epochs` |
| `WARMUP_EPOCHS` | `2` | `--warmup-epochs` (head-only LP-FT warmup before the joint loop; 0 = off) |
| `FINETUNE_LR` | `2e-5` | `--finetune-lr` |
| `FINETUNE_BATCH_SIZE` | `64` | `--finetune-batch-size` |
| `GRAD_CLIPPING` | `0` (off) | `--grad-clipping` (max global grad norm in the joint loop; 0 = off, the pre-clip norm is still logged) |
| `LR_WARMUP_RATIO` | `0.1` | `--lr-warmup-ratio` (fraction of the joint loop's steps spent on LR warmup before linear decay to 0) |
| `THROTTLE` | `4` | max concurrent array tasks (`%N`) |

The consolidate job is pure CPU work and runs on `cpu-batch`; only the array tasks occupy GPUs.

Submits a **job array** (`array_search.slurm`, one model per task, `full_eval` each — the
array bound comes from `mlsys list-models --count`) plus a dependent consolidation job
(`consolidate.slurm`, `afterok`). Each task writes a fragment to
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
`scancel` if it lingers — auto-purge is scheduler-config dependent). Find the failed task ids:

```bash
sacct -j <orig_id> --format=JobID%18,State,ExitCode   # FAILED / CANCELLED / OOM tasks
```

Retry only those ids, pinning the **original** experiment dir via `RUN_ID` (a retry gets a new
array job id — without `RUN_ID` it writes into a fresh `runs/<new_id>/`). Retries bypass
`submit.sh`, so repeat any non-default knobs on the command line:

```bash
sbatch --array=3,7 --export=ALL,RUN_ID=<orig_id>,HIDDEN=<same_as_before> slurm/array_search.slurm
sbatch --dependency=afterok:<retry_job_id> --export=ALL,ARRAY_JOB_ID=<orig_id>,HIDDEN=<same> \
  slurm/consolidate.slurm
```

The `afterok` only gates on the **retry** job — if other original tasks are still running when
it finishes, consolidate starts early, finds an incomplete pool, and fails loudly (by design);
just re-submit the consolidate line once everything is done. Retried tasks overwrite their own
fragment (`rm -f` before the run), so no double-appending. `mlsys consolidate` is idempotent —
safe to re-run manually anytime.

## Single-node fallback

`sbatch slurm/search.slurm` runs the whole pool in one job (edit `STRATEGY` inside; results in
`runs/$SLURM_JOB_ID/`). Simpler, but the job holds a GPU for the slowest model's wall-clock and
one OOM kills the entire run — prefer the array.
