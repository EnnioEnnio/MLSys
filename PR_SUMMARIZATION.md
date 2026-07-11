# Pilot: extend the model-search pipeline to summarization (ROUGE)

## What & why

The pipeline ranks text **encoder** backbones as cheap proxies for an expensive fine-tune
on a **regression** task, and measures *regret* (RQ1: does the frozen ranking predict the
fine-tuned ranking; RQ2: where does compute go). This pilot answers whether that
search-and-rank concept **transfers to summarization** — a generation task scored with ROUGE.

It adds a **parallel task path** gated on `dataset.spec.target_type`, alongside the untouched
regression path, so `make check` stays green and the regression pipeline is unchanged.

## How the task type is selected

No new CLI flag — the task type is a property of the dataset. `--dataset samsum` (which carries
`target_type: summarization` in `config/datasets.yaml`) routes to the summarization path;
`--dataset wine_reviews` stays regression. `run_frozen`/`run_finetune` branch on
`dataset.spec.target_type`. `--strategy {frozen,finetune,full_eval}` is orthogonal and works
for both.

## Design decisions

- **Frozen proxy = train the LM head only** (whole seq2seq body frozen, teacher-forced
  cross-entropy) — the direct analog of "FC head on a frozen backbone". `finetune` unfreezes
  the whole model.
- **Tied weights:** T5/BART tie the LM head to the shared embedding, so name-based matching
  finds no head params. The adapter unfreezes via `model.get_output_embeddings()` — training
  the shared vocab projection while transformer blocks stay frozen. (A real-model integration
  test caught the empty-optimizer bug this fixes.)
- **Dataset = samsum** via the `knkarthick/samsum` parquet mirror (the `Samsung/samsum`
  loading script breaks under `datasets>=3.0`). **Metric = rougeL** for ranking/regret
  (rouge1/2/L all reported).

## Changes

- **Data:** `DatasetSpec.target_type` whitelist; `Row.target: float | str` with a
  `_parse_target` helper (float for regression, non-empty string for summarization).
- **Models:** new `GenerativeBackbone` Protocol + self-registering `seq2seq_lm` adapter
  (`AutoModelForSeq2SeqLM`; `teacher_forcing_loss`/`generate`/`set_trainable`). Pool:
  `t5-small`, `flan-t5-small`, `bart-base`, `distilbart`. `embedding_dim` is a nominal
  `d_model` (no FCHead built); generation length rides in `ModelSpec.extra`.
- **Metrics:** `SummarizationMetrics` + `summarization_metrics` (lazy `rouge_score`, added
  to core deps since scoring runs *in* the pipeline).
- **Runner:** new `search/summarize.py` reusing the five timing sections (`inference_s = 0.0`;
  generation lands in `eval_s`). `RunRecord.metrics` widened to a union.
- **Dispatch/regret:** `full_eval.py` dispatches on `target_type`, uses a `PRIMARY_METRIC`
  table, and writes generic `frozen_scores`/`finetune_scores` keys with `metric="rougeL"`.

## Testing

- **CPU (in `make check`):** ROUGE metrics, target-parsing, `target_type` whitelist, and a
  fake-generative-backbone smoke test asserting one row with `rouge1/2/L`, all five timing
  fields, and `inference_s == 0.0`.
- **Integration (`-m integration`):** real `t5-small` `generate()` + ROUGE; a 1-epoch
  frozen-proxy smoke run asserting `SummarizationMetrics`, `inference_s == 0.0`, `eval_s > 0.0`.

`make check` passes (ruff + ty + 95 CPU tests); both real seq2seq integration tests pass.

## How to run

```bash
python -m mlsys search --dataset samsum \
  --models t5-small,flan-t5-small,bart-base --strategy full_eval --device cpu
```

Produces `results.jsonl` (frozen + finetune rows with `rouge1/2/L` + timings) and
`regret.json` (`metric: "rougeL"` + regret-vs-budget curve). **Note:** `slurm/search.slurm`
is intentionally left pinned to `wine_reviews` and not adjusted here.

## Deferred (future work)

- Port `mlsys analyze` to a ROUGE primary score (the r²-negativity "divergence" machinery has
  no ROUGE analog; would sit behind a capability flag).
- Multi-metric selection; beam search / decoding sweeps (greedy only); summarization
  hyperparameters on the CLI; correct W&B curve labels (still `train_mse`/`val_mse`);
  `head_repeats > 1` for summarization.
