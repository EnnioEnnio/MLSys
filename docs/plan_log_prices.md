# USA Real Estate: log-price target transform

## Context

`usa_real_estate` prices are heavy-tailed (a handful of multi-million-dollar
listings dominate a market of mostly $100k–$500k homes). The hypothesis is
that this heavy tail destroys regression quality (MSE dominated by a few huge
targets, R² depressed) and that regressing on `log(price)` instead will fix
it. We want to add a log-price variant of the dataset so both can be run
through the existing `frozen`/`finetune`/`full_eval` pipeline and compared
via `mlsys analyze`, with no changes to the training/metrics/regret code
itself.

Chosen approach (confirmed with user): a **new dataset-config entry**
(`usa_real_estate_log`), not a global CLI flag — this is dataset-specific
reasoning, matches the existing `_tiny`-variant convention of modeling
dataset variants as separate `config/datasets.yaml` entries, and needs no new
CLI flag or `slurm/` wiring since dataset selection already goes through
`--dataset` / `DATASET=`. Transform is natural `log`, dropping non-positive
rows (matches hedonic-pricing convention; consistent with the existing
drop-invalid-row pattern). Metrics stay in log-price space end-to-end (same
way z-scoring already works: invert whatever unit `Row.target` used, don't
introduce a second dollar-space metric set).

## Where the transform hooks in

`Row.target` is produced in exactly one place —
`_SplitView.__iter__` in [src/mlsys/datasets/__init__.py](src/mlsys/datasets/__init__.py#L59-L70) — as a plain
`float(target_val)`, with `None`/non-castable values already dropped there.
Everything downstream (`_embed_split` in `search/runner.py`, `train_full_model`
in `finetune/__init__.py`, z-scoring via `head.target_stats`, un-scaling,
`regression_metrics`, `regret.py`) treats `Row.target` as an opaque float and
never re-derives it from raw HF data — so applying `log()` at this single
point makes the transform transparent to the entire rest of the pipeline.
`results.jsonl`/`regret.json` never store raw target arrays (only scalar
metrics + `target_mean`/`target_std` extras — confirmed via `runner.py`
`RunRecord.extras`), so nothing else needs updating for the new unit space.

## Changes

1. **`src/mlsys/datasets/registry.py`**
   - Add `target_transform: Literal["identity", "log"] = "identity"` to
     `DatasetSpec` (defaulted so every existing `datasets.yaml` entry parses
     unchanged with no edits).
   - In `load_specs()`, read `entry.get("target_transform", "identity")` and
     validate it's one of `("identity", "log")`, raising `ValueError`
     otherwise — same style as the existing `target_type` check
     ([registry.py:48-51](src/mlsys/datasets/registry.py#L48-L51)).

2. **`src/mlsys/datasets/__init__.py`**
   - `import math` at top.
   - In `_SplitView.__iter__` ([__init__.py:59-70](src/mlsys/datasets/__init__.py#L59-L70)), after the existing
     `target_float = float(target_val)` cast: if
     `self.spec.target_transform == "log"`, drop the row when
     `target_float <= 0` (`continue`, joining the existing None/non-castable
     drop path so it's counted in the same dropped-row log line), else set
     `target_float = math.log(target_float)`.

3. **`config/datasets.yaml`**
   - Add `usa_real_estate_log`: clone of `usa_real_estate` (same `hf_repo`,
     `shuffle_seed: 42`, same split row counts, same `target_column: price`,
     same `text_template`) plus `target_transform: log`. Keep `shuffle_seed`
     identical so the train/val/test row membership matches `usa_real_estate`
     exactly — the only difference is the target transform, keeping the
     frozen/finetune comparison apples-to-apples.
   - Add `usa_real_estate_log_tiny` analogously (clone of
     `usa_real_estate_tiny` + `target_transform: log`) for fast local
     iteration.
   - Short comment noting these are the log-price variants for the
     heavy-tail hypothesis, next to the existing `_tiny` comment block.

4. **`CLAUDE.md`**
   - "New dataset" section: mention the optional `target_transform`
     field (`identity` default, `log` drops non-positive rows).
   - "Data flow" section: one sentence noting that when `target_transform:
     log` is set, the transform applies in `Row` construction *before*
     z-scoring, so "original units" for that dataset means log-dollars —
     z-score inversion and `regression_metrics` are unaffected/transform-agnostic,
     they just operate in whatever unit `Row.target` carries.

5. **Tests**
   - `tests/test_registries.py`: extend with a rejects-unknown-value test
     (mirroring `test_datasets_yaml_rejects_non_regression_target`) for a bad
     `target_transform`, and a parses-ok test asserting
     `specs["usa_real_estate_log"].target_transform == "log"` while an entry
     omitting the field defaults to `"identity"`.
   - New `tests/test_target_transform.py`: unit-test `_SplitView.__iter__`
     directly (construct a `DatasetSpec` with `target_transform="log"` and a
     plain list of dict rows as `hf_split` — no real HF `Dataset` needed,
     matching how `_SplitView` is already exercised) asserting: positive
     prices come out as `math.log(price)`; zero/negative prices are dropped;
     a `target_transform="identity"` spec is unaffected (regression guard
     against breaking the existing datasets).

## Out of scope (explicitly, per user's answers)

- No CLI flag, no `slurm/` wiring — dataset choice alone selects the
  transform.
- No dollar-space back-transform (`expm1`) or second metric set — R²/MSE/MAE
  for `usa_real_estate_log` are reported purely in log-price space.
- Not touching `usa_real_estate` (untransformed) — it stays as the baseline
  for comparison.

## Verification

- `make check` (lint + typecheck + test).
- `uv run pytest tests/test_registries.py tests/test_target_transform.py -v`.
- `python -m mlsys list-datasets` — confirm `usa_real_estate_log` and
  `usa_real_estate_log_tiny` show up.
- Local smoke run (needs HF network access):
  `python -m mlsys search --dataset usa_real_estate_log_tiny --models potion-base-8M --strategy frozen`
  — confirm it completes, `results.jsonl` has sane `r2`/`mse` (mse now in
  log-dollar² units, much smaller than raw-dollar mse), and
  `extras.target_mean`/`target_std` reflect log-space stats.
- Downstream research step (not part of this change): once both
  `usa_real_estate` and `usa_real_estate_log` have `full_eval` runs, compare
  their R² via `mlsys analyze` to check whether the heavy-tail hypothesis
  holds.
