# MLSys — Model search for regression tasks

Seminar project at HPI. Take a regression dataset, build a candidate pool of HuggingFace encoders, attach a fresh fully-connected head to each backbone, train against ground truth, rank candidates. Three strategies share the same metrics: `frozen` (head on a frozen backbone — the cheap proxy ranking), `finetune` (unfreeze + train backbone+head jointly — the expensive ground truth), and `full_eval` (run both over the pool and report **regret** — how much quality the proxy ranking loses versus fine-tuning everything; see [REGRET.md](REGRET.md)). v2 layers on successive halving + embedding caching.

## Setup

Requires [`uv`](https://github.com/astral-sh/uv) installed on your machine.

```bash
make setup   # uv sync
```

## Common commands

| Command           | What it does                          |
|-------------------|---------------------------------------|
| `make setup`      | Install deps                          |
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
src/mlsys/finetune/   # joint backbone+head fine-tune loop (finetune/full_eval strategies)
src/mlsys/search/     # strategy dispatch (frozen/finetune/full_eval), runner, regret, timing, metrics
src/mlsys/analysis/   # turn full_eval CSVs into tables + plots + SUMMARY.md (see analysis.md)
src/mlsys/io/         # results.jsonl writer
config/               # datasets.yaml + models.yaml
slurm/                # cluster launch scripts
tests/                # CPU-only smoke + unit tests
```

## CLI

- `python -m mlsys search` — run a search. Flags: `--dataset NAME`, `--models name1,name2` (optional, default all), `--strategy {frozen,finetune,full_eval}` (default `frozen`), `--output-dir PATH` (default `runs/<unix-ts>`), `--epochs INT`, `--batch-size INT`, `--hidden WIDTH`, `--device cpu|cuda`, `--head-repeats N`, `--finetune-epochs INT`, `--finetune-lr FLOAT`, `--finetune-batch-size INT`, `--wandb`, `--cache-embeddings` (stubbed for v2).
- `python -m mlsys list-models` — dumps `config/models.yaml` entries.
- `python -m mlsys list-datasets` — dumps `config/datasets.yaml` entries.

### Strategies (`--strategy`)

- **`frozen`** (default) — train a fresh FC head on the **frozen** backbone; the cheap proxy / ranking signal.
- **`finetune`** — unfreeze the backbone and train backbone+head jointly; the expensive ground truth `t(m, D)`. Inference is fused into the joint loop, so per-row timing reports `inference_s = 0` and the training cost in `train_head_s`. Static encoders that can't be fine-tuned (model2vec) fall back to the frozen score, tagged `finetune_skipped`.
- **`full_eval`** — run both passes over the whole pool, then compute the regret-vs-budget curve (frozen r2 ranks the shortlist; finetune r2 scores it). Writes `runs/<id>/regret.json` (metric `r2`, per-budget absolute + normalized regret, the proxy ranking, and both r2 maps). Tune the joint loop with `--finetune-epochs` / `--finetune-lr` / `--finetune-batch-size`.

```bash
python -m mlsys search --dataset wine_reviews --strategy frozen        # proxy ranking only
python -m mlsys search --dataset wine_reviews --strategy finetune       # ground-truth scores
python -m mlsys search --dataset wine_reviews --strategy full_eval      # both + regret.json
```

### Head type (`--hidden`)

The head attached to each frozen backbone is a linear probe by default (`in_dim -> 1`). Pass `--hidden WIDTH` to use a 2-layer MLP instead: `in_dim -> WIDTH -> ReLU -> 1`. `WIDTH` is the size of the hidden layer — the only thing the number controls — so larger values give the head more capacity (and more parameters) to fit nonlinear structure. `--hidden 0` (or omitting the flag) keeps the linear head.

```bash
python -m mlsys search --dataset wine_reviews --hidden 256   # MLP head, 256-wide hidden layer
python -m mlsys search --dataset wine_reviews                # linear head (default)
```

### W&B logging (`--wandb`)

`--wandb` needs `WANDB_API_KEY`. For local runs, copy `.env.example` to `.env` and fill it in — `python -m mlsys` auto-loads `.env` via `python-dotenv`. `.env` is gitignored; never commit a real key. On the cluster the key is read from the exported shell env instead (see [Running on the cluster](#running-on-the-cluster)).

## Adding a new model

- Append one row to `config/models.yaml` with `name`, `hf_repo`, `loader`, `pooling`, `embedding_dim`, `max_length` (optional `input_prefix`).
- Pick `loader` from `sentence_transformers`, `transformers_encoder`, or `model2vec`.
- For a finetune of an existing architecture, that's literally all — share `loader`/`pooling`/`embedding_dim`/`max_length` and just bump `name` + `hf_repo`.
- For a new architecture family (different loader path, custom pooling), drop an adapter module under `src/mlsys/models/adapters/` that calls `register_adapter("your_loader", _build)` at import time. The package is auto-discovered — adding the file is the whole step; no edit to `models/registry.py` is needed.
- Don't forget to add the model to the pool table in this README, with notes on why it's interesting and what it adds to the variety of the search space.

## Adding a new dataset

- Append one row to `config/datasets.yaml` with `name`, `hf_repo`, `splits` (logical → HF split name), `target_column`, `target_type: regression`, and a `text_template` Python format string.
- Missing or `None` columns referenced in the template render as `"unknown"`, so partial-row datasets don't crash.
- If the HF dataset needs custom parsing (filtering, column synthesis), add a loader module under `src/mlsys/datasets/`.

## Running on the cluster

Edit `slurm/search.slurm` — set `REPO_PATH` to your cluster checkout, set `--mail-user`, and (if you want W&B logging) `export WANDB_API_KEY` in your shell **before** `sbatch` so `--container-env=WANDB_API_KEY` can forward it. The cluster does not read `.env` (that's local-only), so the variable must be exported. Then:

```bash
sbatch slurm/search.slurm
```

Results land in `runs/$SLURM_JOB_ID/results.jsonl` — one line per `(dataset, model, strategy)` with metrics + per-substep timing (plus `runs/$SLURM_JOB_ID/regret.json` under `--strategy full_eval`). See [slurm/README.md](slurm/README.md) for details.

## Analysing results

Turn a `full_eval` run's CSVs into tables, plots, and a single `SUMMARY.md` to hand to Claude for the report prose. Viz deps live in an optional group (`uv sync --group analysis`).

```bash
python -m mlsys analyze results/<experiment>                 # → results/<experiment>/analysis/SUMMARY.md
python -m mlsys regret --frozen F.csv --finetune T.csv       # standalone regret recompute (crash recovery)
```

Full plot/table catalog, the folder/filename conventions, and the crash-recovery recipe are in **[analysis.md](analysis.md)**.

## Tooling at a glance

- **uv** — package manager and task runner; deps locked in `uv.lock`
- **ruff** — linter and formatter (replaces black + flake8)
- **ty** — Astral's type checker (pre-1.0; swap to pyright with one config change if needed)
- **pytest** — smoke + critical-path tests only; no coverage gate. Tests marked `integration` (one real-model end-to-end smoke) need network + heavier deps and are skipped by default and in CI; run them with `uv run pytest -m integration`.
- **GitHub Actions** — runs `make check` on every pull request

## Model pool

| Name | Family / Architecture | Size | D/m | FT | Comments |
| :---- | :---- | :---- | :---- | :---- | :---- |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Distilled 6-layer encoder, mean-pool (built-in), 384-dim | ~23M | 260M | 900 | Most-used model on HF |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | MPNet (masking + permutation order), mean-pool (built-in), 768-dim | ~100M | 35M | 350 | Slower, more accurate sibling |
| [potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) | Static lookup table (model2vec), no attention, ignores word order, 256-dim | ~8M | 710k | 5 | Extreme low-cost anchor; ~500× faster on CPU |
| [potion-base-32M](https://huggingface.co/minishlab/potion-base-32M) | Static lookup table (model2vec), 512-dim | ~32M | 84k | 4 | Larger static variant for the cost axis |
| [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) | 6-layer distilled BERT, mean-pool, 768-dim | ~66M | ~25M | ~3k | Distilled BERT; fast CPU baseline |
| [deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small) | Disentangled attention + ELECTRA-style objective, mean-pool, 768-dim | ~44M | ~1M | ~150 | Lighter DeBERTa variant |
| [modernbert-base](https://huggingface.co/answerdotai/ModernBERT-base) | Modernized BERT (RoPE, local/global attn, 8k ctx), mean-pool, 768-dim | ~150M | 1.8M | 1200 | Current-gen encoder, huge finetune tree |
| [roberta-base](https://huggingface.co/FacebookAI/roberta-base) | BERT-family, tuned recipe, mean-pool, 768-dim | ~100M | 17M | 2300 | Plain strong encoder baseline |
| [deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) | Disentangled attention + ELECTRA-style objective, mean-pool, 768-dim | ? | 3M | 600 | Strongest base-size encoder for accuracy |
| [electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator) | Encoder trained as real/fake token detector, mean-pool, 768-dim | ? | 56M | 70 | Different pretraining game, sample-efficient |
| [albert-base-v2](https://huggingface.co/albert/albert-base-v2) | Cross-layer weight sharing + factorized embeddings, mean-pool, 768-dim | ~12M | 680k | 260 | Many effective layers, tiny footprint |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | Contrastive encoder, mean-pool, needs `query:` prefix, 768-dim | ~110M | 2M | 75 | Prefix quirk = realistic prepare-model cost (RQ2) |
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | Contrastive encoder, CLS-pool, 768-dim | ~100M | 10M | 450 | Top retrieval embedder |
| [modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base) | ModernBERT backbone, embedding-tuned + Matryoshka, mean-pool, 768-dim | ~150M | 170k | 110 | Pairs w/ modernbert-base → isolates contrastive-tuning effect |
| [sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) | T5 encoder half (encoder-decoder lineage), mean-pool (built-in), 768-dim | ~100M | 210k | 1 | Text-to-text pretraining = architectural variety |
| [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | Large contrastive encoder + Matryoshka, 1024-dim | ~300M | 5M | 55 | Large Matryoshka data point |
| [gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) | GTE encoder, long ctx + Matryoshka, 768-dim | ~130M | 725k | 830 | Richest finetune tree for variety |
| [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | Long-ctx BERT + Matryoshka, needs `search_document:` prefix, 768-dim | ~137M | 17M | 31 | Flagship Matryoshka |
