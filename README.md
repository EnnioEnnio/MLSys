# MLSys — Model search for regression tasks

Seminar project at HPI. Take a regression dataset, build a candidate pool of HuggingFace encoders, attach a fresh fully-connected head to each frozen backbone, train the head against ground truth, rank candidates. v1 = full-evaluation baseline; v2 layers on successive halving + embedding caching.

## Setup

Requires [`uv`](https://github.com/astral-sh/uv) installed on your machine.

```bash
make setup   # uv sync + pre-commit install
```

## Common commands

| Command           | What it does                          |
|-------------------|---------------------------------------|
| `make setup`      | Install deps + pre-commit hooks       |
| `make lint`       | Run ruff linter                       |
| `make format`     | Run ruff formatter + auto-fix         |
| `make typecheck`  | Run ty type checker                   |
| `make test`       | Run pytest                            |
| `make check`      | lint + typecheck + test (same as CI)  |
| `make clean`      | Remove .venv and caches               |

## Project structure

```
src/mlsys/cli/        # python -m mlsys entrypoints
src/mlsys/datasets/   # HF dataset loaders, template rendering
src/mlsys/models/     # backbone adapters (transformers / sentence-transformers / model2vec)
src/mlsys/head/       # FC regression head + trainer
src/mlsys/search/     # search strategies (v1: full_eval), runner, timing, metrics
src/mlsys/io/         # results.jsonl writer
config/               # datasets.yaml + models.yaml
slurm/                # cluster launch scripts
tests/                # CPU-only smoke + unit tests
```

## CLI

- `python -m mlsys search` — run a search. Flags: `--dataset NAME`, `--models name1,name2` (optional, default all), `--strategy full_eval`, `--output-dir PATH` (default `runs/<unix-ts>`), `--epochs INT`, `--batch-size INT`, `--device cpu|cuda`, `--wandb`, `--cache-embeddings` (stubbed for v2).
- `python -m mlsys list-models` — dumps `config/models.yaml` entries.
- `python -m mlsys list-datasets` — dumps `config/datasets.yaml` entries.

### W&B logging (`--wandb`)

`--wandb` needs `WANDB_API_KEY`. For local runs, copy `.env.example` to `.env` and fill it in — `python -m mlsys` auto-loads `.env` via `python-dotenv`. `.env` is gitignored; never commit a real key. On the cluster the key is read from the exported shell env instead (see [Running on the cluster](#running-on-the-cluster)).

## Adding a new model

- Append one row to `config/models.yaml` with `name`, `hf_repo`, `loader`, `pooling`, `embedding_dim`, `max_length` (optional `input_prefix`).
- Pick `loader` from `sentence_transformers`, `transformers_encoder`, or `model2vec`.
- For a finetune of an existing architecture, that's literally all — share `loader`/`pooling`/`embedding_dim`/`max_length` and just bump `name` + `hf_repo`.
- For a new architecture family (different loader path, custom pooling), drop an adapter module under `src/mlsys/models/adapters/` that calls `register_adapter("your_loader", _build)` at import time. The package is auto-discovered — adding the file is the whole step; no edit to `models/registry.py` is needed.

## Adding a new dataset

- Append one row to `config/datasets.yaml` with `name`, `hf_repo`, `splits` (logical → HF split name), `target_column`, `target_type: regression`, and a `text_template` Python format string.
- Missing or `None` columns referenced in the template render as `"unknown"`, so partial-row datasets don't crash.
- If the HF dataset needs custom parsing (filtering, column synthesis), add a loader module under `src/mlsys/datasets/`.

## Running on the cluster

Edit `slurm/search.slurm` — set `REPO_PATH` to your cluster checkout, set `--mail-user`, and (if you want W&B logging) `export WANDB_API_KEY` in your shell **before** `sbatch` so `--container-env=WANDB_API_KEY` can forward it. The cluster does not read `.env` (that's local-only), so the variable must be exported. Then:

```bash
sbatch slurm/search.slurm
```

Results land in `runs/$SLURM_JOB_ID/results.jsonl` — one line per `(dataset, model)` with metrics + per-substep timing. See [slurm/README.md](slurm/README.md) for details.

## Tooling at a glance

- **uv** — package manager and task runner; deps locked in `uv.lock`
- **ruff** — linter and formatter (replaces black + flake8)
- **ty** — Astral's type checker (pre-1.0; swap to pyright with one config change if needed)
- **pytest** — smoke + critical-path tests only; no coverage gate. Tests marked `integration` (one real-model end-to-end smoke) need network + heavier deps and are skipped by default and in CI; run them with `uv run pytest -m integration`.
- **GitHub Actions** — runs `make check` on every pull request
