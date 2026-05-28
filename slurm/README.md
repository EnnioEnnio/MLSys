# slurm/

Cluster launch for `mlsys search`.

## One-shot setup

1. Edit `slurm/search.slurm`:
   - `REPO_PATH` → your cluster checkout path (e.g. `/sc/home/<you>/mlsys`).
   - `--mail-user` → your slack handle.
   - `-A {account_name}` → your SLURM account.
2. Put your W&B key in your shell rc as `export WANDB_API_KEY=...` (the script forwards it via `--container-env`). Skip if you don't pass `--wandb`.

## Launch

```bash
sbatch slurm/search.slurm
```

Results land in `runs/$SLURM_JOB_ID/results.jsonl` inside the repo checkout. Each line is one (dataset, model) record with metrics + per-substep timing.

## Notes

- The container (`nvcr.io#nvidia/pytorch:25.01-py3`) is named `mlsys-$USER-pytorch2501` and reused across jobs — the `pip install` step skips if `sentence-transformers` is already present.
- To run a subset, edit the `python -m mlsys search` line and append `--models name1,name2`.
- Default `--time=2:00:00` is plenty for the four-model seed pool on one A100. Bump when the pool grows.
