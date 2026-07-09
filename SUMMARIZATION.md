# Pilot: extend the model-search pipeline to summarization (ROUGE)

## Context

The pipeline ranks text **encoder** backbones as cheap proxies for an expensive
fine-tune on a **regression** task, and measures *regret* (RQ1: does the frozen ranking
predict the fine-tuned ranking; RQ2: where does compute go). Issue asks whether the
search-and-rank concept **transfers to summarization** — a generation task scored with
ROUGE. This is an exploratory pilot, not a full summarization framework.

Today the codebase is regression-only end-to-end: every backbone is `encode() → [B, dim]`,
the only head is `FCHead` (`Linear → scalar` + MSE), the only metric is
`RegressionMetrics(mse/mae/r2/spearman)`, and `regret.json` is built off `.metrics.r2`.
There is **zero** generative / seq2seq / ROUGE support. The *regret math itself*
(`search/regret.py`) is already metric-agnostic (higher-is-better dicts) — only its
callers hard-code r2.

The approach is a **parallel task path** gated on `dataset.spec.target_type`, added
*alongside* the untouched regression path so `make check` (ruff + ty + CPU pytest) stays
green.

### Decisions locked with the user
- **Frozen proxy = train the LM (generation) head only** (whole seq2seq body frozen,
  teacher-forced cross-entropy). Direct analog of "FC head on a frozen backbone".
  `finetune` = unfreeze the whole model.
- **Dataset = samsum** (dialogue → short abstractive summary; small, cheap ROUGE).
- **Reporting = minimal**: pipeline emits `results.jsonl` + `regret.json` (metric=`rougeL`)
  + a short manual write-up. `mlsys analyze` (tables/plots/`SUMMARY.md`) stays
  regression-only; porting it is documented as future work.

---

## 1. Data layer — admit a string target

**[datasets/registry.py](src/mlsys/datasets/registry.py)**
- `DatasetSpec.target_type: Literal["regression", "summarization"]` (line 30).
- `load_specs` (lines 46-49, 58): replace the `!= "regression"` hard-reject with a
  whitelist (`in ("regression","summarization")`, else raise) and build the spec with the
  **actual** `entry["target_type"]` instead of the hardcoded `"regression"`.
  `REQUIRED_FIELDS`/`REQUIRED_SPLITS` unchanged — for summarization `target_column` names
  the reference-summary column and `text_template` renders the source document.

**[datasets/\_\_init\_\_.py](src/mlsys/datasets/__init__.py)**
- `Row.target: float | str` (line 28). Regression rows stay `float`; summarization rows
  carry the reference summary `str`. (Overload chosen over a separate field — the
  regression consumers `_embed_split`/`train_full_model` only ever see float targets on
  their path, so nothing regresses.)
- `_SplitView.__iter__` (lines 59-70): factor a `_parse_target(target_type, raw)` helper
  and branch on `self.spec.target_type`:
  - `regression`: existing behavior byte-for-byte — `float(raw)`, skip on `None`/non-castable.
  - `summarization`: `str(raw)`, skip when `None` or empty/whitespace.
  `__len__`'s drop-counting is unaffected.

---

## 2. Model layer — a generative backbone + adapter

**[models/backbone.py](src/mlsys/models/backbone.py)** — add a third Protocol
(`runtime_checkable`, imports under `TYPE_CHECKING`), **not** extending `Backbone` (a
generator has no meaningful `encode`):
```python
class GenerativeBackbone(Protocol):
    name: str
    can_finetune: bool
    def teacher_forcing_loss(self, sources: list[str], targets: list[str]) -> torch.Tensor: ...
    def generate(self, sources: list[str]) -> list[str]: ...
    def set_trainable(self, scope: Literal["head", "full"]) -> None: ...
    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]: ...
    def train(self) -> None: ...
    def eval(self) -> None: ...
```

**New [models/adapters/seq2seq_lm.py](src/mlsys/models/adapters/seq2seq_lm.py)** — loader
`"seq2seq_lm"`, self-registers via `register_adapter`, mirrors the structure of
[transformers_encoder.py](src/mlsys/models/adapters/transformers_encoder.py) with **lazy**
`import torch` / `from transformers import AutoModelForSeq2SeqLM, AutoTokenizer` inside
`__init__`:
- Load tokenizer + `AutoModelForSeq2SeqLM.from_pretrained(..., torch_dtype=torch.float32,
  use_safetensors=True, trust_remote_code=spec.trust_remote_code)`. Store `spec.input_prefix`
  (e.g. `"summarize: "` for T5), `spec.max_length` (max **source** len),
  `spec.extra.get("max_target_length", 64)`.
- `teacher_forcing_loss`: tokenize sources (prefix + truncation), tokenize targets as
  `labels` (pad → -100), forward `model(...).loss`. No `inference_mode` (grads flow to
  whatever is unfrozen).
- `generate`: `model.generate(..., num_beams=1, max_new_tokens=max_target_length)` under
  `inference_mode`, then `tokenizer.batch_decode(skip_special_tokens=True)`.
- `set_trainable(scope)`: `for p in model.parameters(): p.requires_grad = (scope=="full")`;
  when `scope=="head"`, set `requires_grad=True` on params whose name contains
  `lm_head`/`final_logits_bias`. **Tied-weight note:** T5/BART tie the LM head to the shared
  embedding table, so "head only" trains the shared vocab projection while all transformer
  blocks stay frozen — cheap and a faithful "lightweight head" analog. Document this in the
  adapter docstring. `trainable_parameters()` returns `[p for p in model.parameters() if
  p.requires_grad]`.
- `can_finetune = True`.

**`embedding_dim` obstacle:** `models/registry.py` requires it `> 0`. Keep the validation;
put a **nominal** `d_model` in the yaml (t5/flan-t5-small=512, bart-base=768,
distilbart=1024). It's semantically ignored by the summarization runner (no FCHead built).
Document as nominal in a yaml comment. Generation params ride in `extra` (ModelSpec already
captures unknown keys there) — no schema change.

**[config/models.yaml](config/models.yaml)** — small seq2seq pool:
```yaml
- {name: t5-small,      hf_repo: google-t5/t5-small,   loader: seq2seq_lm, embedding_dim: 512, max_length: 512, input_prefix: "summarize: ", extra: {max_target_length: 64}}
- {name: flan-t5-small, hf_repo: google/flan-t5-small,  loader: seq2seq_lm, embedding_dim: 512, max_length: 512, input_prefix: "summarize: ", extra: {max_target_length: 64}}
- {name: bart-base,     hf_repo: facebook/bart-base,    loader: seq2seq_lm, embedding_dim: 768, max_length: 512, extra: {max_target_length: 64}}
- {name: distilbart,    hf_repo: sshleifer/distilbart-cnn-12-6, loader: seq2seq_lm, embedding_dim: 1024, max_length: 512, extra: {max_target_length: 64}}
```

**[config/datasets.yaml](config/datasets.yaml)** — samsum with slices:
```yaml
- name: samsum
  hf_repo: knkarthick/samsum        # script-free parquet mirror (datasets>=3.0 dropped loading scripts)
  splits: {train: "train[:2000]", val: "validation[:200]", test: "test[:200]"}
  target_column: summary
  target_type: summarization
  text_template: "{dialogue}"
```
Note: `Samsung/samsum` historically ships a loading **script** that breaks under the pinned
`datasets>=3.0`; use the `knkarthick/samsum` parquet mirror (same `dialogue`/`summary`
columns). If that repo is unavailable, the fallback is to thread `trust_remote_code` through
`load_dataset` (currently not passed).

---

## 3. Metrics — ROUGE alongside regression

**[search/metrics.py](src/mlsys/search/metrics.py)** — add next to `RegressionMetrics`:
```python
@dataclass(frozen=True)
class SummarizationMetrics:
    rouge1: float
    rouge2: float
    rougeL: float
    def to_dict(self) -> dict[str, float]: return asdict(self)

def summarization_metrics(preds: list[str], refs: list[str]) -> SummarizationMetrics:
    from rouge_score import rouge_scorer            # lazy heavy import (project convention)
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    # mean fmeasure over pairs
```
**[pyproject.toml](pyproject.toml)** — add `rouge-score` to `[project.dependencies]` (light,
pure-Python; must be in core because scoring runs *in* the pipeline, not just analysis).

---

## 4. Summarization runner — new module, reuse the 5 timing sections

**New [search/summarize.py](src/mlsys/search/summarize.py)** (keeps the regression
`runner.py` untouched; imports `RunRecord` from it):
- `SummarizeConfig` dataclass: `epochs`, `batch_size`, `head_lr`, `full_lr`,
  `early_stop_patience`, `max_target_length` — small defaults; threaded with defaults only
  (no new CLI flags per the minimal scope).
- `_train_seq2seq(backbone, rows, cfg, scope)` — near-clone of
  [train_full_model](src/mlsys/finetune/__init__.py): `backbone.set_trainable(scope)`, AdamW
  over `backbone.trainable_parameters()`, per-batch `teacher_forcing_loss`, early-stop on
  **val CE loss**, returns train/val curves + `epochs_run` (reuse `HeadTrainResult` shape or
  a light local result).
- `score_summarization_candidate(...)` — the **frozen proxy**, 5 timed sections:
  `prepare_model_s` (build backbone) · `prepare_data_s` (materialize rows) ·
  `inference_s = 0.0` (forward fused into training — matches the encoder *finetune*
  convention) · `train_head_s` (`_train_seq2seq(..., scope="head")`) ·
  `eval_s` (`backbone.generate(test_sources)` **+** `summarization_metrics` — **generation
  cost lands here**). Returns `RunRecord(metrics=SummarizationMetrics(...), strategy="frozen",
  extras={"task":"summarization","head_repeats":1})`.
- `finetune_summarization_candidate(...)` — identical but `scope="full"`,
  `strategy="finetune"`.

**[search/runner.py](src/mlsys/search/runner.py)** — only change: widen
`RunRecord.metrics` (line 29) to `RegressionMetrics | SummarizationMetrics`.
`RunRecord.to_dict` already only calls `metrics.to_dict()`, so nothing else changes.

---

## 5. Dispatch + regret generalization

**[search/full_eval.py](src/mlsys/search/full_eval.py)** — a tiny task table (single branch
point):
```python
_FROZEN_FN   = {"regression": score_candidate,     "summarization": score_summarization_candidate}
_FINETUNE_FN = {"regression": finetune_candidate,  "summarization": finetune_summarization_candidate}
PRIMARY_METRIC = {"regression": "r2", "summarization": "rougeL"}
```
- `run_frozen`/`run_finetune`: pick `candidate_fn` by `dataset.spec.target_type` (regression
  calls exactly the same functions as today). Summarization branches build a
  `SummarizeConfig` from defaults.
- `_run_pass` log line (lines 85-94): replace the hardcoded `record.metrics.r2`/`.mse` with a
  metric-agnostic string from `record.metrics.to_dict()`.
- `run_full_eval` (lines 216-222): `key = PRIMARY_METRIC[dataset.spec.target_type]`;
  `frozen_scores = {r.model: r.metrics.to_dict()[key] for r in frozen}` (finetune likewise);
  rank + `regret_curve(...)` unchanged (`regret.py` is already metric-agnostic; rougeL is
  higher-is-better).
- `_write_regret_json` (lines 290-301): parameterize `"metric": key` and write generic
  `frozen_scores`/`finetune_scores` keys (grep confirms only tests read the old `frozen_r2`
  names; the `analyze`/`regret` CLI paths read CSVs, not this JSON). `higher_is_better=True`
  already holds.

`run_strategy`, `io`, `regret.py`, and the CLI need **no** task-type changes.

---

## 6. Tests

CPU-only (default suite):
- **[tests/test_metrics.py](tests/test_metrics.py)**: `summarization_metrics(["the cat sat"],
  ["the cat sat"])` → rougeL ≈ 1.0; partial-overlap sanity. `pytest.importorskip("rouge_score")`.
- **[tests/test_registries.py](tests/test_registries.py)**: the current assertion that every
  dataset `target_type == "regression"` (line ~22) must become `in {"regression",
  "summarization"}`; add a parse test for the summarization entry; keep the
  unknown-`target_type` rejection (e.g. `classification` still raises).
- **[tests/test_full_eval_smoke.py](tests/test_full_eval_smoke.py)**: add a `_FakeGenerativeBackbone`
  (`teacher_forcing_loss` returns a small trainable-scalar loss, `generate` returns canned
  strings, `set_trainable`/`trainable_parameters`/`train`/`eval`) under a fake loader + a fake
  summarization dataset. Assert `run_frozen` writes one row with `rouge1/rouge2/rougeL` in
  `metrics`, all 5 timing keys + `peak_gpu_mem_mb`, and `inference_s == 0.0`.
- A datasets test: `_SplitView.__iter__` keeps string targets for summarization and still
  float-filters for regression.

Integration-marked (`@pytest.mark.integration`, mirrors
[tests/test_integration.py](tests/test_integration.py)):
- Real `google-t5/t5-small`: `generate(["summarize: <short doc>"])` → non-empty strings;
  `summarization_metrics` runs.
- 1-epoch frozen-proxy teacher-forced smoke on a tiny in-memory summarization dataset,
  asserting `record.metrics` is `SummarizationMetrics`, `inference_s == 0.0`, `eval_s > 0.0`.

---

## 7. Explicitly deferred (document as future work in the PR write-up)
- Porting `mlsys analyze` (tables/plots/`SUMMARY.md`) to a ROUGE primary score; the
  r2-negativity "divergence" machinery has no ROUGE analog and would be gated behind a
  capability flag (`supports_divergence` precedent already exists in `analysis/loader.py`).
- Multi-metric selection (report rouge1/2/L, rank/regret on rougeL only).
- Beam search / decoding sweeps (greedy `num_beams=1` only).
- Summarization hyperparams on the CLI (`SummarizeConfig` uses defaults; no new flags).
- W&B curve labels (`_log_curves_to_wandb` logs `train_mse`/`val_mse` keys; carry CE loss for
  summarization — cosmetic mislabel).
- `head_repeats > 1` for summarization (point estimates, same as finetune today).

---

## 8. Verification (end-to-end)

1. `make check` — ruff + ty + CPU pytest all green (regression suite unchanged; new
   fake-backbone summarization smoke passes).
2. **CPU fake-backbone smoke** (no network): the new `test_full_eval_smoke` summarization
   case proves the dispatch → `results.jsonl` → rouge-metrics path.
3. **Small real run** (integration / manual, tiny slices):
   ```
   python -m mlsys search --dataset samsum \
     --models t5-small,flan-t5-small,bart-base --strategy full_eval --device cpu
   ```
   Confirm `runs/<id>/results.jsonl` has frozen + finetune rows with `rouge1/2/L` and all 5
   timing fields, and `runs/<id>/regret.json` has `"metric": "rougeL"` + a non-increasing
   regret-vs-budget curve. Inspect: does the frozen (LM-head-only) ROUGE ranking track the
   finetuned ROUGE ranking (RQ1)? Where does compute go — `train_head_s` vs `eval_s`
   (generation) vs `prepare_*` (RQ2)?
4. **Write-up** (PR description / report appendix): RQ1 (proxy regret from `regret.json`) +
   RQ2 (timing breakdown) + "what a full summarization extension would need" (the deferred
   list above).
