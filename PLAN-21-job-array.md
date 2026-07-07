# Issue #21 — SLURM job array (one model per task) + result consolidation

## Context

`slurm/search.slurm` runs the whole 16-model pool in one job, blocking a GPU node for the
wall-clock of the slowest model, and one OOM kills the entire run. Issue #21 asks for a SLURM
job array (one model per task), a dependent consolidation step that merges the per-task
`results.jsonl` fragments and recomputes `regret.json` as if a single `full_eval` run had
produced them, plus retry-ability of individual failed tasks and docs.

User decisions (asked & answered):
- Consolidation lives in the package as an **`mlsys consolidate` subcommand** (not
  `scripts/consolidate_results.py`) — testable under `make check`, on PATH in the container.
- **W&B: combine both** — each array task keeps its own W&B run (live monitoring from phone),
  AND `mlsys consolidate --wandb` pushes one consolidated run named like a single-node
  `full_eval` run, so the existing analysis CSV-download workflow keeps working unchanged.
  The consolidated run must log **two tables** — `results_frozen` and `results_finetune` —
  exactly as a single-node `full_eval` does (plus the regret curve); see step 3.

Review feedback incorporated: two-table W&B parity, shared container + per-node flock instead
of per-task containers, `list-models --count` as single source of truth for the array bound,
`HIDDEN` passed through `submit.sh`, retry/partial-failure flow as the docs headline.

## Key design facts (verified in code)

- Run id = `--output-dir` basename; W&B run name = `_wandb_run_name(output_dir.name, dataset,
  strategy, num_models, hidden)` ([main.py:237-249](src/mlsys/cli/main.py#L237-L249)).
- `run_full_eval` ([full_eval.py:180-231](src/mlsys/search/full_eval.py#L180-L231)) works for a
  single model; regret needs only `model` + `metrics.r2` per row, split by `strategy`.
  `_write_regret_json` ([full_eval.py:286](src/mlsys/search/full_eval.py#L286)) is private.
- **PLAN-20 (live per-epoch W&B streaming, merged on this branch) does not conflict:** it only
  reworked the curve-streaming path (`_make_epoch_logger` replacing `_log_curves_to_wandb`).
  The regret block, `_log_results_table` (+ `results_frozen`/`results_finetune` names),
  `_log_regret_to_wandb`, and the `results.jsonl` schema are unchanged. Bonus: per-task array
  W&B runs now stream epoch curves live. The consolidated W&B run gets tables + regret curve
  only — no epoch-curve replay (data exists in rows, deliberately out of scope).
- **Tie-break parity:** proxy ranking = `sorted(frozen_r2, ..., reverse=True)` — stable sort, so
  ties break by dict insertion order = models.yaml order. Consolidation must order rows by
  `load_specs()` order to reproduce single-node regret exactly.
- `JsonlWriter` is **append-only** — retried tasks and re-run consolidation must not double-append.
- Cluster core install has **no pandas** — consolidation must be pure json/pathlib (do not reuse
  `analysis/regret_recompute.py`).
- Row `head_type` is only `"mlp"|"linear"` (no width) — consolidate needs `--hidden` to rebuild
  the W&B run-name head token (`FCH` / `MLP_<w>`).

## Directory & W&B naming layout

```
runs/<ARRAY_JOB_ID>/                         # experiment dir = consolidation target
  <ARRAY_JOB_ID>_task_0/results.jsonl        # fragment (+ trivial per-model regret.json)
  <ARRAY_JOB_ID>_task_1/results.jsonl
  ...
  results.jsonl                              # written by consolidate (frozen block, then finetune, registry order)
  regret.json                                # recomputed over the merged pool
```

Per-task output dir basename `<id>_task_<n>` makes the per-task W&B run name unique across
experiments (`<id>_task_<n>_wine_reviews_fulleval_1_model_FCH`) with **zero CLI changes**.
Consolidated W&B run name: `_wandb_run_name(run_dir.name, dataset, "full_eval", n_models,
hidden)` — identical grammar to a single-node run, so the analysis filename round-trip holds.

## Implementation steps

### 1. Refactor regret writing into shared helpers — `src/mlsys/search/regret.py`

Add (module stays torch/pandas-free; only `json`, `pathlib`, `mlsys.io.ensure_run_dir` added):

```python
@dataclass(frozen=True)
class RegretSummary:
    dataset: str; proxy_ranking: list[str]
    frozen_r2: dict[str, float]; finetune_r2: dict[str, float]; curve: list[RegretPoint]
    def to_payload(self) -> dict: ...
        # exact payload from current _write_regret_json INCLUDING the constant fields
        # metric="r2", higher_is_better=True, regret_estimator="point_estimate"

def summarize_regret(dataset, frozen_r2, finetune_r2) -> RegretSummary: ...
    # ranking + regret_curve; frozen_r2 insertion order is the tie-break — caller owns ordering

def write_regret_json(output_dir, summary) -> Path: ...  # json.dumps(indent=2, sort_keys=True)
```

In `run_full_eval`, replace the inline phase-3 block + `_write_regret_json` with these helpers
(pure refactor — regret.json bytes unchanged).

### 2. New module `src/mlsys/search/consolidate.py`

```python
def consolidate_run(run_dir, *, cleanup=False, allow_partial=False) -> ConsolidationResult
```

1. Glob `<run_dir>/*_task_*/results.jsonl`, sorted by numeric task index (not lexicographic).
2. No fragments: if parent `results.jsonl` exists → no-op success (idempotent after cleanup);
   else raise.
3. Parse rows; rows pass through **verbatim** (only reads `model`, `strategy`, `dataset`,
   `metrics.r2`). Raise if fragments disagree on `dataset`.
4. Dedupe on `(model, strategy)` keep-last; order frozen and finetune blocks by `load_specs()`
   index (unknown names last, warn) — tie-break parity (D7).
5. Completeness check: frozen and finetune model sets equal & non-empty, else raise listing the
   missing pairs — unless `allow_partial`, which merges but skips regret.
6. Write merged `results.jsonl` via temp file + `os.replace` (**write mode, never append** —
   idempotent), `json.dumps(row, sort_keys=True)` per line (same serialization as JsonlWriter).
7. If complete: build r2 dicts in registry order → `summarize_regret` → `write_regret_json`.
8. **Analysis-ready CSV export (default on):** write `<runname>_{frozen,finetune,regret}.csv`
   next to `results.jsonl`, where `<runname>` = `_wandb_run_name(run_dir.name, dataset,
   "full_eval", n_models, hidden)` — exactly the filename grammar `mlsys analyze` expects, so
   prep-for-analysis = copy `runs/<id>/*.csv` into `results/<experiment>/`. Stdlib `csv` only
   (no pandas — cluster-safe). frozen/finetune columns = W&B table layout (columns from first
   row, flattened `metrics`/`timing`, curves dropped) so local CSVs are byte-compatible with
   the W&B-download path; regret CSV = `budget,regret,normalized_regret` (same as
   `mlsys regret --out`). Needs `hidden` passed down (CLI already has `--hidden`).
9. `cleanup=True`: `shutil.rmtree` the `*_task_*` dirs only after all writes succeed.

### 3. CLI wiring — `src/mlsys/cli/main.py`

- `list-models --index N`: print bare model name at registry position N; out-of-range → stderr +
  non-zero exit (a bad `--array` bound then fails loudly and `afterok` blocks consolidation).
- `list-models --count`: print the pool size from `load_specs()` — single source of truth for
  the `--array` bound in `submit.sh` (same ordering as `--index`, no grep drift).
- New `consolidate` subcommand: positional `run_dir`; `--cleanup`; `--allow-partial`;
  `--wandb` + `--hidden` (default `HeadTrainConfig.hidden`, used only for the W&B run name).
  `_run_consolidate` lazy-imports `consolidate_run`; with `--wandb` it opens
  `wandb.init(entity="HPI_MLSys", project="mlsys-model-search", name=_wandb_run_name(...))` and:
  - splits merged rows by `strategy` and logs **two tables, `results_frozen` and
    `results_finetune`** — same names as single-node `run_frozen`/`run_finetune`
    (`_log_results_table` via `full_eval.py:108`), since the analysis round-trip downloads each
    `_<kind>` table separately. Column derivation matches `_log_results_table`: columns from the
    first row (dict-based equivalent of `_result_row`: flatten `metrics`/`timing`, drop curves).
    Divergent extras (e.g. `finetune_skipped`) stay ragged exactly as single-node — do not "fix".
  - logs the regret curve via the existing `_log_regret_to_wandb` pattern (already takes
    `RegretPoint`s).

### 4. `slurm/array_search.slurm`

Clone `search.slurm` structure (container dance, pip-constraint fix, `set -euo pipefail`,
`PYTORCH_CUDA_ALLOC_CONF`) with:
- `#SBATCH --array=0-15%4` (comment: bound = pool size − 1, use `submit.sh` which derives it
  from `list-models --count`; `%4` throttles concurrency), `--job-name=mlsys-array`,
  `--output=slurm-mlsys-array-%A_%a.out`, `--time=2:00:00`.
- **Shared container name** (`mlsys-$USER-pytorch2503`, same as `search.slurm`) so the
  build-once pip install is reused, NOT a per-task name (which would multiply the NGC pull +
  `pip install -e` by N tasks in time and disk). Enroot named containers are node-local, so the
  creation race only exists between tasks co-located on one node (bounded by `%4`): serialize
  Step 1 with a **per-node flock** — the array task's batch script runs on its allocated node,
  so `flock /tmp/mlsys-container-$USER.lock srun --container-image=... --container-name=$CONTAINER_NAME ... pip install ...`
  makes the second co-located task wait, find the container built, and no-op through the
  install. Step 2 uses the same `$CONTAINER_NAME` (one variable — Step 1/Step 2 can't drift).
  Script comment documents the trade-off and the fallback (per-task suffix) if the cluster's
  pyxis/enroot still misbehaves under concurrent reuse.
- `RUN_ID=${RUN_ID:-$SLURM_ARRAY_JOB_ID}`; `HIDDEN=${HIDDEN:-0}`;
  `OUT=/workspace/runs/${RUN_ID}/${RUN_ID}_task_${SLURM_ARRAY_TASK_ID}`.
- Step 2 as one `srun ... bash -c` block (forward `WANDB_API_KEY` etc. via `--container-env`):
  ```bash
  MODEL=$(python -m mlsys list-models --index "$SLURM_ARRAY_TASK_ID")
  rm -f "$OUT/results.jsonl" "$OUT/regret.json"   # retried task must not double-append
  python -m mlsys search --dataset wine_reviews --strategy full_eval --models "$MODEL" \
    --hidden "$HIDDEN" --head-repeats "$HEAD_REPEATS" --output-dir "$OUT" $FINETUNE_ARGS --wandb
  ```

### 5. `slurm/consolidate.slurm`

No `--gpus`, `--mem=8G`, `--time=00:20:00`. Requires
`: "${ARRAY_JOB_ID:?...}"` from `--export`; `HIDDEN=${HIDDEN:-0}` (same var as the array script —
`submit.sh` exports it to both jobs so the W&B head token can't drift). Same container/pip
Step 1 (same flock guard), then:
`python -m mlsys consolidate /workspace/runs/$ARRAY_JOB_ID --hidden "$HIDDEN" --wandb`
(commented `--cleanup` variant).

### 6. `slurm/submit.sh`

```bash
HIDDEN=${HIDDEN:-0}
# Pool size from the same source of truth as the tasks' index→model mapping (load_specs order).
# Falls back to grep only if the package isn't importable on the login node.
N=$(python -m mlsys list-models --count 2>/dev/null || grep -c '^- name:' config/models.yaml)
[ "$N" -gt 0 ] || { echo "empty model pool" >&2; exit 1; }
ARRAY_ID=$(sbatch --parsable --array=0-$((N-1))%4 --export=ALL,HIDDEN="$HIDDEN" slurm/array_search.slurm)
sbatch --dependency=afterok:"$ARRAY_ID" --export=ALL,ARRAY_JOB_ID="$ARRAY_ID",HIDDEN="$HIDDEN" slurm/consolidate.slurm
```

Retry recipe (docs): `sbatch --array=3,7 --export=ALL,RUN_ID=<orig_id> slurm/array_search.slurm`,
then re-submit consolidate with `ARRAY_JOB_ID=<orig_id>` depending on the retry job.

### 7. Tests

**New `tests/test_consolidate.py`** (CPU-only; monkeypatch `load_specs` on
`sys.modules["mlsys.search.consolidate"]`, pattern from `test_full_eval_smoke.py`):
- Merge correctness: hand-built fragments → frozen-then-finetune blocks in registry order, rows
  verbatim, expected `proxy_ranking`/`curve` in regret.json.
- Tie-break parity: equal frozen r2 fed in reversed task order → ties break by registry order.
- End-to-end parity via the fake-adapter harness (2 fake models): `run_full_eval` into `dir_a`
  vs per-model runs into `dir_b/<id>_task_*` + `consolidate_run(dir_b)` → `regret.json` bytes
  equal; `results.jsonl` rows equal modulo `timing`.
- CSV export: three files named per the `<runname>_<kind>.csv` grammar; frozen/finetune columns
  match the flat W&B-table layout; filename round-trips through the analysis loader's parser.
- Idempotency (run twice; keep-last dedupe; no-fragments-but-consolidated → no-op).
- Failure modes: no fragments → raises; frozen-without-finetune → raises (with `allow_partial`:
  merged results, `regret_path is None`); mixed datasets → raises.
- Cleanup: dirs removed only on success.

**Extend `tests/test_cli.py`:** `list-models --index` (valid + out-of-range exit code) and
`--count`; `consolidate` routing (monkeypatch `consolidate_run`, assert flags forwarded —
mirrors `test_strategy_routes_to_run_strategy`); with `--wandb`, the two logged tables are
named `results_frozen`/`results_finetune` (fake wandb module, pattern already used in tests).

### 8. Docs

- Rewrite `slurm/README.md` — **short and concise**, four sections in this order:
  1. **Quickstart:** `bash slurm/submit.sh` — what it submits (array, one model/task + dependent
     consolidate), where output lands (`runs/<id>/`).
  2. **From run to analysis** (the user's primary workflow — must be findable immediately):
     `scp cluster:.../runs/<id>/*.csv results/<experiment>/ && mlsys analyze results/<experiment>`
     — consolidate already wrote analysis-ready CSVs in the correct filename grammar; W&B table
     download remains as the alternative road (same files).
  3. **When a task fails:** consolidate job goes `DependencyNeverSatisfied` (check `squeue`,
     cancel if stuck — purge behavior is scheduler-config dependent); retry recipe: resubmit
     failed ids with `RUN_ID=<orig>`, then manually submit consolidate depending on the retry.
  4. **Single-node fallback:** `search.slurm`, one paragraph.
- `README.md`: add `consolidate` + `list-models --index/--count` to CLI list; update SLURM section.
- `CLAUDE.md`: add `mlsys consolidate` to commands; update slurm note.
- Sync the revised plan into the repo as `PLAN-21-job-array.md` (repo convention, file already
  opened by the user) as the first implementation step.

## Files

| File | Change |
|---|---|
| `src/mlsys/search/regret.py` | + `RegretSummary`, `summarize_regret`, `write_regret_json` |
| `src/mlsys/search/full_eval.py` | use shared helpers; drop `_write_regret_json` |
| `src/mlsys/search/consolidate.py` | **new** — merge + regret recompute + cleanup |
| `src/mlsys/cli/main.py` | `consolidate` subcommand; `list-models --index/--count`; two-table W&B push |
| `slurm/array_search.slurm`, `slurm/consolidate.slurm`, `slurm/submit.sh` | **new** |
| `tests/test_consolidate.py` (new), `tests/test_cli.py` | tests |
| `slurm/README.md`, `README.md`, `CLAUDE.md` | docs |

Commit sequence: (1) regret refactor, (2) consolidate module + CLI + tests, (3) slurm scripts + docs.

## Verification

1. `make check` (ruff + ty + pytest + format check) — all gates green.
2. End-to-end parity test is the acceptance-criterion proof: seeded fake backbones make
   single-node vs array-consolidated `regret.json` byte-identical (production runs differ
   anyway because head seeds come from unseeded torch RNG — parity is structural).
3. Local smoke without SLURM: run `python -m mlsys search --models <m> --strategy full_eval
   --output-dir /tmp/x/x_task_0` for two small models (or the fake harness), then
   `mlsys consolidate /tmp/x`; inspect `results.jsonl` + `regret.json`, copy the exported CSVs
   into a `results/<experiment>/` folder and run `mlsys analyze` on it — proves the run→analysis
   path end-to-end with zero manual renaming.
4. On-cluster (user-run): `bash slurm/submit.sh`, verify per-task W&B runs appear live, then
   the consolidated W&B run + `runs/<id>/regret.json`; kill one task and exercise the retry recipe.

## Risks / edge cases

- Enroot container race on shared nodes → shared container name + per-node flock around Step 1
  (build-once reuse preserved); fallback to per-task suffix documented in the script comment if
  concurrent reuse still misbehaves.
- Retried array gets a new job id → `RUN_ID` override is mandatory for retries; prominent in docs.
- `--array` bound drift vs models.yaml → bound comes from `list-models --count` (same
  `load_specs()` source as the index→model mapping); `--index` past the pool exits non-zero, so
  the task fails and `afterok` blocks consolidation (loud, correct).
- Trivial per-task `regret.json` fragments are ignored by consolidation and removed by `--cleanup`.
- Rows pass through verbatim (e.g. `finetune_skipped`) — consolidation never re-shapes rows.
