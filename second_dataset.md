# USA Real Estate: match row count & split to wine_reviews

## Context

`usa_real_estate` (`config/datasets.yaml:11-19`) is already registered as a regression
dataset — `target_column: price`, `target_type: regression`, and a matching
`text_template` already exist, so that's **not** the gap. Two things are missing:

1. **Row count/split don't match wine_reviews.** Research (HF API + downloaded Parquet
   metadata):
   - `wine_reviews` (spawn99/wine-reviews): **280,901** rows total, real HF splits
     `train`/`validation`/`test` = **196,630 / 28,090 / 56,181** → exactly **70/10/20**.
   - `usa_real_estate` (jason1966/...): **2,226,382** rows, only **one** physical
     `train` split (a single CSV). The current config uses `train[:80%]` /
     `train[80%:90%]` / `train[90%:]` — that yields ~1.78M/223k/223k, far too large and
     not scaled to match wine_reviews.
   - Agreed with you: **train ≈ 200k, same as wine_reviews**, val/test scaled to the
     same 70/10/20 ratio → **train 196,630 / val 28,090 / test 56,181** (total 280,901,
     matching wine_reviews exactly).

2. **The raw data isn't i.i.d.** I streamed the CSV directly: it's grouped by
   state/date (the start is all Puerto Rico, later a contiguous block is all Florida,
   etc.). Contiguous percentage slicing (as done today) would pull train/val/test from
   essentially disjoint regions — not a fair comparison to wine_reviews' real (already
   shuffled) HF splits. Agreed with you: **shuffle before splitting** (fixed seed,
   reproducible).

Goal of this plan: give `usa_real_estate` the same row count/split structure as
wine_reviews, with a true random sample instead of a contiguous block.

## Changes

### 1. `src/mlsys/datasets/registry.py` — extend `DatasetSpec` with optional shuffle fields

Two new optional fields on `DatasetSpec` (default `None`), backward-compatible for
wine_reviews (stays unchanged):

```python
shuffle_seed: int | None = None
base_split: str | None = None
```

In `load_specs()`: both must be set together or both omitted (error otherwise). When
`shuffle_seed` is set, the values in `splits` (still `dict[str, str]`, same field as
today) must be plain row counts as strings (`"196630"` instead of `"train[:80%]"`) —
validate with a clear `ValueError`, matching the existing style
(`test_datasets_yaml_rejects_*` in `tests/test_registries.py:94-127`).

### 2. `src/mlsys/datasets/__init__.py` — branch `load_dataset()`

New path when `spec.shuffle_seed is not None`:

```python
base = hf_load_dataset(spec.hf_repo, split=spec.base_split)
base = base.shuffle(seed=spec.shuffle_seed)
offset = 0
for logical, count_str in spec.splits.items():
    count = int(count_str)
    splits[logical] = _SplitView(spec=spec, hf_split=base.select(range(offset, offset + count)))
    offset += count
```

The order of `spec.splits.items()` (dict preserves YAML insertion order) determines the
offsets — train/val/test must appear in that order in the YAML so the slices are
deterministic and non-overlapping. The existing path (string = HF split expression, no
shuffle) stays exactly as it is today for wine_reviews.

### 3. `config/datasets.yaml` — `usa_real_estate` entry

```yaml
- name: usa_real_estate
  hf_repo: jason1966/ahmedshahriarsakib_usa-real-estate-dataset
  base_split: train
  shuffle_seed: 42
  splits:
    train: "196630"
    val: "28090"
    test: "56181"
  target_column: price
  target_type: regression
  text_template: "status: {status}, bed: {bed}, bath: {bath}, acre_lot: {acre_lot}, street: {street}, city: {city}, state: {state}, zip_code: {zip_code}, house_size: {house_size}, prev_sold_date: {prev_sold_date}"
```

`target_column`/`target_type`/`text_template` stay unchanged — they were already
correct. The ~1,541 rows with missing `price` (out of 2,226,382, <0.1%) are already
dropped automatically by the existing `_SplitView` filter logic
(`datasets/__init__.py:59-70`, discards `None`/non-numeric targets) — no extra work
needed there.

### 4. Tests

- `tests/test_registries.py`: add cases following the existing
  `test_datasets_yaml_rejects_*` pattern — `shuffle_seed` without `base_split` (and
  vice versa) is rejected; a non-numeric split value while `shuffle_seed` is set is
  rejected.
- New test for the loading path itself (e.g. in `tests/test_registries.py` or a small
  new file): monkeypatch `datasets.load_dataset` with an in-memory
  `datasets.Dataset.from_dict(...)` (no network needed, consistent with "CPU-only
  tests"), verify the three splits have the expected non-overlapping row counts, and
  that two runs with the same `shuffle_seed` produce the same row order
  (determinism).

### 5. `CLAUDE.md` — doc update

In the "New dataset" section, briefly note: datasets with only one physical HF split
can optionally specify `base_split` + `shuffle_seed` instead of HF slice strings in
`splits`; in that mode `splits` values are row counts, and the loader shuffles once
with a fixed seed before splitting.

## Not needed

- No changes to `search/full_eval.py`, `search/runner.py`, the CLI, SLURM scripts, or
  W&B naming — those are already dataset-agnostic and work unchanged with
  `--dataset usa_real_estate`.
- No `src/mlsys/datasets/usa_real_estate.py` module needed (wine_reviews' counterpart
  is currently just an empty hook with no logic — the generic YAML loader already
  covers both datasets).

## Verification

1. `make check` (lint/typecheck/tests, including the new registry tests).
2. Run end-to-end once with a cheap model and check the log line from
   `_SplitView.__len__` ("Loaded X/Y rows...") to confirm train/val/test come out to
   the expected ~196,630/28,090/56,181 (minus ~0.1% missing prices):
   ```
   python -m mlsys search --dataset usa_real_estate --models potion-base-8M --strategy frozen
   ```
3. Optional: load twice with the same seed and confirm the same rows come out first
   (determinism check for reproducibility across SLURM array runs).
