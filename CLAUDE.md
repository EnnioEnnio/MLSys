# CLAUDE.md

## What this project is

HPI Machine Learning Systems seminar project on **model search for regression tasks**. Given a regression
dataset and a pool of HuggingFace encoders, attach a fresh fully-connected head to
each backbone, train against ground truth, score on test, rank candidates. Goal: see
how model search (*SHiFT* / *Poodle* / *Alsatian*) behaves and where bottlenecks move
when the task is regression, not classification — and measure **regret** (REGRET.md):
how much quality the cheap frozen-backbone proxy ranking loses versus actually
fine-tuning every model.

`MLSYS_Proposal.md` (JIT-Model-Replacement + LLM-as-Judge) is **historical** — the
project pivoted. Current scope is the regression pipeline only.

### Strategies (`--strategy`, dispatched in `search/full_eval.py:run_strategy`)

- **`frozen`** (default) — train a fresh FC head on the **frozen** backbone (the cheap
  proxy / ranking signal). This is the original full-evaluation baseline; the name
  `full_eval` previously meant exactly this logic.
- **`finetune`** — unfreeze the backbone and train backbone+head jointly (the expensive
  ground-truth signal `t(m, D)` for regret).
- **`full_eval`** — **name reused**: run *both* passes over the whole pool, then compute
  the regret-vs-budget curve (frozen r2 ranks; finetune r2 scores) → `runs/<id>/regret.json`.

- **v1 (current):** the three strategies above over the whole pool.
- **v2 (future):** successive halving + on-disk embedding caching. `--cache-embeddings` is **stubbed** (no-op).

## Commands

```bash
make setup       # uv sync + pre-commit install
make check       # lint + typecheck + test — run before considering work done (mirrors CI)

python -m mlsys search --dataset wine_reviews                      # all models
python -m mlsys search --dataset wine_reviews --models all-MiniLM-L6-v2,potion-base-8M
python -m mlsys list-models / list-datasets
uv run pytest -m integration                                       # real-model tests (skipped by default)
```

## Analysis (report generation)

`mlsys.analysis` turns a `full_eval` run's CSV dumps into tables + plots + a single
`SUMMARY.md` to hand to Claude for the report *prose*. Viz deps (`pandas`, `matplotlib`,
`seaborn`) live in the optional **`analysis`** dependency group — kept out of
`[project.dependencies]` so the cluster's `pip install -e .` stays light, but in the default
uv dev groups so `ty`/CI resolve the lazy imports. Install with `uv sync --group analysis`;
imported lazily inside `analysis/` functions.

```bash
mlsys analyze results/<experiment>                 # → results/<experiment>/analysis/SUMMARY.md
mlsys regret --frozen F.csv --finetune T.csv [--out R.csv]   # standalone regret recompute (crash recovery)
```

Each **experiment** is a folder of CSVs (`results/<experiment>/`, e.g. `exp_wine_16`), one
`full_eval` head config per run-id named `<runid>_<strategy>_<num>_model_<HEAD>_<kind>.csv`
(`<kind>` ∈ frozen/finetune/regret; head label and width come from the filename, **not** the
`head_type` column). `analyze` writes artifacts to `results/<experiment>/analysis/` by default
(`--out-dir` overrides). Regret is **r²-only** (no `--metric`), matching the pipeline. See
`analysis.md` for the full plot/table catalog + crash-recovery recipe.

## Data flow

`cli/main.py` → `search/full_eval.py:run_strategy` → per-candidate runner in
`search/runner.py`. **frozen** uses `score_candidate`, five timed substeps:
**prepare_model** (`registry.build_backbone`) → **prepare_data** (materialise rows;
`text_template` rendering is timed here) → **inference** (`backbone.encode`) →
**train_head** (`FCHead`, AdamW+MSE, early stop) → **eval** (`RegressionMetrics`:
mse/mae/r2/spearman).

**finetune** uses `finetune_candidate` → `finetune/train_full_model` (joint AdamW over
backbone+head, via `backbone.encode_trainable`). It reuses the **same five timing
fields**: inference is *fused into the joint loop*, so `inference_s = 0` and the
training cost lands in `train_head_s`. Non-trainable backbones (`can_finetune=False`,
e.g. model2vec) fall back to the frozen `score_candidate` and are tagged
`finetune_skipped=true` (finetune score == frozen score).

Each substep is wrapped in `timing.py:Timer.section`. **Don't rename the timing
fields** (`prepare_model_s`, `prepare_data_s`, `inference_s`, `train_head_s`,
`eval_s`) — they're the RQ2 measurement and appear verbatim in `results.jsonl`.

Output: `runs/<id>/results.jsonl`, one line per `(dataset, model, strategy)` (the
`strategy` field distinguishes frozen vs finetune rows). The `full_eval` strategy also
writes `runs/<id>/regret.json` (metric=`r2`, per-budget abs+normalized regret curve,
proxy ranking, both r2 maps; `regret_estimator="point_estimate"` since finetune runs
once per model — `head_repeats=1`, see REGRET.md note 2).

## Model pool

Candidates live in `config/models.yaml`. Under the `frozen` strategy backbones stay
frozen and only the head trains; `finetune`/`full_eval` unfreeze them (a backbone opts
in via `can_finetune=True` on its adapter — model2vec is `False`). To add a model:

1. Check if an existing adapter fits (`models/adapters/`: `sentence_transformers`,
   `transformers_encoder`, `model2vec`). A finetune of an existing architecture
   reuses its `loader`/`pooling`/`embedding_dim`/`max_length` — just add a `name` +
   `hf_repo` row. New architecture family → drop a file in `adapters/` that calls
   `register_adapter(...)` at import (auto-discovered; no `registry.py` edit).
2. Append a YAML row: `name`, `hf_repo`, `loader`, `pooling`, `embedding_dim`,
   `max_length` (optional `input_prefix`). `embedding_dim` must match the
   backbone's real output width. For `transformers_encoder`, `pooling` is
   `mean`/`cls`/`last` (`builtin` is only for sentence-transformers / model2vec).

**New dataset** — append to `config/datasets.yaml`: `name`, `hf_repo`, `splits`
(`train`/`val`/`test` → HF split names, all required), `target_column`,
`target_type: regression`, `text_template`. Missing template fields render as
`"unknown"`.

## Conventions

- Python **3.12**; `from __future__ import annotations` at top of every module.
- **Lazy heavy imports:** `torch`, `transformers`, `datasets`, `wandb`, `scipy`,
  `dotenv` imported *inside functions*, never at module top — keeps config-only
  commands and CPU tests fast. A top-level `import torch` breaks tests/CI.
- Ruff (line length 100, `E,F,I,B,UP,SIM,RUF`); `ty` typecheck; CPU-only tests, no
  coverage gate. `load_specs` validates strictly — keep it loud.
- `--wandb` is opt-in (`WANDB_API_KEY`). Local: `.env` auto-loaded. Cluster:
  `export` it before `sbatch`; `slurm/search.slurm` forwards via `--container-env`.
