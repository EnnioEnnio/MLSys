# Plan: per-candidate failure isolation + custom-arch model root-cause

Status as of 2026-06-17: `gte-base-en-v1.5` and `nomic-embed-text-v1.5` are
**commented out** in `config/models.yaml` to unblock a clean sweep of the other 16
models. This document is the follow-up work to (1) make the sweep robust so one bad
model never kills the whole run again, and (2) actually root-cause the two
custom-architecture checkpoints so they can be re-enabled.

Do these **after** we have at least one clean run of the 16-model pool.

---

## Background: why one model takes down the whole sweep

`search/full_eval.py` iterates candidates and calls `score_candidate` in a single
long-lived process, reclaiming GPU memory between models (`release_gpu_memory`).
There is **no per-candidate error boundary**, so any exception aborts the run and we
lose every not-yet-run model's results.

Failures we have hit, each of which killed the run:

| Model(s)            | Failure                                            | Class |
|---------------------|----------------------------------------------------|-------|
| sentence-t5-base    | apex FusedRMSNorm rejects fp16 (fixed)             | Python exception |
| bge-base-en-v1.5    | cls-pool view OOM (fixed)                          | Python exception |
| gte / nomic         | `trust_remote_code` load refusal (fixed)          | Python exception |
| gte / nomic         | forward-pass CUDA device-side assert (open)       | **CUDA assert** |

The last class is the important one: a **CUDA device-side assert poisons the CUDA
context**. After it fires, *every* subsequent CUDA call in the same process raises
`device-side assert triggered` — so the next model can't run either. A plain
`try/except` around `score_candidate` therefore **cannot** recover from it; the
process itself is unusable. This is why model 18 never ran after model 17 (gte)
asserted.

---

## Work item 1 — Isolate per-candidate failures

Goal: the sweep always attempts all N candidates and emits N rows in
`results.jsonl` (a result row on success, an error row on failure), regardless of
how any single model fails.

Two tiers; ship Tier 1 first (cheap, covers most failures), then Tier 2 (covers the
unrecoverable CUDA-assert / segfault class).

### Tier 1 — in-process try/except (catches Python exceptions)

- Wrap the `score_candidate` call in `full_eval`'s loop in `try/except Exception`.
- On failure, write an error row instead of a result row, e.g.
  `{"dataset": ..., "model": ..., "status": "error", "error_type": "...", "error": "<msg>"}`,
  log it, call `release_gpu_memory`, and continue to the next candidate.
- Successful rows gain `"status": "ok"` (or keep current shape + an absent/`ok`
  status) so downstream analysis can filter.
- **Does not** rescue CUDA device-side asserts — after one fires, the loop's
  remaining models will also error out. That's acceptable as a first step but is
  exactly what Tier 2 fixes.

Effort: small. One function, plus an error-row schema decision.

### Tier 2 — subprocess per candidate (catches everything)

Run each candidate in its **own process** so a crash (CUDA assert, segfault, OOM
kill) is contained and the next candidate starts with a **fresh CUDA context**.

- New single-candidate entrypoint: `python -m mlsys score-one --dataset X --model NAME
  --output-dir DIR [--batch-size ... --device ... head-config flags]`. It runs
  exactly one `score_candidate` and writes its one JSONL row, then exits.
- Orchestrator (a new strategy, e.g. `full_eval_isolated`, or a flag on `full_eval`)
  loops over specs and `subprocess.run([... score-one ...], timeout=...)` per model:
  - exit 0 → the worker already appended its result row (or the orchestrator reads
    the worker's row file and appends it itself — pick one owner of the writer).
  - non-zero exit / timeout / signal → orchestrator appends an error row
    (`status: "crashed"`, include exit code / signal) and moves on.
- Each worker gets a clean process → a device-side assert in one worker cannot
  affect the next. This is the only thing that actually survives the gte-class crash.

Design considerations to settle when implementing:
- **Who owns `results.jsonl`.** Cleanest: orchestrator owns the writer; worker prints
  its row as JSON to stdout (or a temp file) and the orchestrator appends. Avoids two
  processes writing the same file.
- **Timing semantics unchanged.** Each candidate is already timed independently
  inside its own `Timer`; the `prepare_model_s … eval_s` fields are computed
  per-candidate and are unaffected by running in a subprocess. Process startup
  (interpreter + heavy imports) is **outside** the timed sections, so RQ2 numbers
  stay comparable. Note the added wall-clock overhead per model (~imports + container
  exec) in the run log, but it does not enter the measured substeps.
- **W&B.** A shared single run across N subprocesses needs care: either each worker
  attaches to the same run via `WANDB_RUN_ID`/`WANDB_RESUME`, or workers stay offline
  and the orchestrator logs the aggregated results table at the end (preferred —
  matches today's `_log_results_table`).
- **CPU tests stay in-process.** Keep the current in-process loop as the default path
  (fast, simple, what the CPU test-suite exercises); gate the subprocess path behind
  a flag/strategy so tests don't pay process-spawn cost.
- **SLURM.** Subprocess-per-model runs inside the existing container exec; no extra
  `srun` per model needed — just `subprocess` within the one `srun`.

Acceptance: re-enable gte (or inject a deliberately-crashing fake model); the sweep
finishes all other models and `results.jsonl` has one row per candidate, with the
crasher recorded as an error/crashed row.

---

## Work item 2 — Root-cause gte / nomic forward-pass assert

Symptom: both custom-arch checkpoints **load** (weights 135/135) but their forward
pass throws `IndexKernel.cu:93 ... index out of bounds` — an embedding/index gather
fed an index outside its table. The Python traceback (it points at `batch_to_device`)
is misleading because CUDA kernels are async; the assert surfaces at the next CUDA
call, not the offending one.

### Step 1 — get an accurate trace

Rerun a single model with the assert forced synchronous:

```bash
CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 \
  python -m mlsys search --dataset wine_reviews --models gte-base-en-v1.5
```

(temporarily re-enable the row, or run a tiny standalone script that builds the
SentenceTransformer and encodes a couple of `wine_reviews` rows). With
`CUDA_LAUNCH_BLOCKING=1` the traceback points at the **real** kernel/line. Capture:
the failing op, and the offending tensor's shape / dtype / max value.

### Step 2 — hypotheses (index-out-of-bounds in a gather)

1. **`token_type_ids` vs `type_vocab_size`** — new-impl's token-type embedding may be
   sized 0/1 while sentence-transformers passes `token_type_ids=0`. Check
   `model.config.type_vocab_size` vs the ids actually passed.
2. **Sequence-length / position handling** — confirm truncation actually applies
   (`max_seq_length=512`) and the rotary/abs position indexing stays in range. In
   particular check whether the model uses its `unpad_inputs` /
   `use_memory_efficient_attention` fast paths, which do custom index gathers and are
   the most-reported source of this exact gte-v1.5 crash on non-matching stacks.
3. **`input_ids` ≥ `vocab_size`** — tokenizer vs embedding-table width mismatch.
   Compare `tokenizer.vocab_size` to `model.get_input_embeddings().num_embeddings`.

### Step 3 — likely fixes (try in order, smallest first)

1. Load with config overrides `unpad_inputs=False`, `use_memory_efficient_attention=False`
   for gte new-impl (disables the custom gather fast path). This is the leading
   candidate.
2. Don't pass `token_type_ids`, or clamp to zeros within range.
3. Pin a known-good `transformers` / model repo `revision`.

### Step 4 — make the fix declarative

These fixes are per-model loader tweaks, like `trust_remote_code` already is. Add a
per-model `config_overrides` (or `model_kwargs`) field in `config/models.yaml`,
plumbed through the adapters the same way `trust_remote_code` is (we already collect
unknown keys into `ModelSpec.extra`). Then the gte/nomic re-enable is a YAML change,
not adapter code, and the workaround is auditable next to the model row.

Acceptance: gte and nomic produce valid embeddings + a result row on the cluster, and
are re-enabled in `config/models.yaml`.

---

## Suggested order

1. Land Work item 1 / Tier 1 (in-process try/except) — cheap insurance on the next run.
2. Get one clean 16-model run.
3. Work item 2 root-cause (needs a GPU debug run).
4. Work item 1 / Tier 2 (subprocess isolation) — the durable fix for the CUDA-assert
   class; do alongside or after re-enabling gte/nomic so the sweep can never again be
   killed by a single model.
