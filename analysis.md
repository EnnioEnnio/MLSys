# Analysis & report generation

Turn a `full_eval` run's CSV dumps into tables, plots, and a single `SUMMARY.md` you hand to
Claude for the *prose* of the seminar report. The tool generates every artifact
deterministically — same tables, same plots, every time — so the writing step never silently
drops a figure or recomputes a number.

## Setup

The plotting/dataframe deps live in an optional dependency group (kept out of the core
package so the cluster's `pip install -e .` stays light):

```bash
uv sync --group analysis     # installs pandas, matplotlib, seaborn
```

(They're also in the default dev groups, so a plain `uv sync` / `make setup` pulls them in
too. They are imported lazily inside `mlsys.analysis.*`, so `list-models` etc. stay fast.)

## Folder-per-experiment convention

One **experiment** = one `full_eval` sweep over a model pool at several head widths. Put its
CSVs in a single folder and run `analyze` on that folder:

```
results/exp_wine_16/                         # the input CSVs (12 for a 4-head sweep)
    2296332_fulleval_16_model_FCH_frozen.csv
    2296332_fulleval_16_model_FCH_finetune.csv
    2296332_fulleval_16_model_FCH_regret.csv
    2296333_fulleval_16_model_MLP_128_frozen.csv
    ...
results/exp_wine_16/analysis/                # generated (default --out-dir)
    SUMMARY.md                               # tables + embedded plots, in fixed section order
    FCH/  MLP_128/  MLP_256/  MLP_512/        # per-head tables + plots
    comparison/                              # cross-head tables + plots
```

The folder name (`exp_wine_16` here — wine dataset, 16-model pool) is arbitrary; `analyze`
only uses it as the `SUMMARY.md` title. Name experiments however you like.

### Filename grammar (how head labels are recovered)

```
<runid>_<strategy>_<num>_model_<HEAD>_<kind>.csv
```

- `<kind>` is the last token: `frozen`, `finetune`, or `regret`.
- `<HEAD>` is every token between the literal `model` and `<kind>` — so both `FCH` and
  `MLP_512` round-trip.
- Files are grouped into a **triple** by `<runid>` (one head config per run-id).

The CSV's own `head_type` column is only `linear`/`mlp` and does **not** encode MLP width —
width comes from the filename. A filename that doesn't match the grammar raises a clear error
naming the expected pattern (no silent guessing, no CLI override).

## Commands

### `mlsys analyze <experiment_dir> [--out-dir DIR]`

Discovers the triples, builds every per-head + cross-head artifact, and writes `SUMMARY.md`.

```bash
mlsys analyze results/exp_wine_16
mlsys analyze results/exp_wine_16 --out-dir /tmp/wine_report
```

- **No `--metric` flag.** The whole regret pipeline is r²-only (proxy ranking = frozen-r²
  desc, ground truth = finetune r²; `regret.json` hard-codes `"metric": "r2"`). mse/mae still
  appear in the per-model quality tables for context — just not as the regret basis.
- **No `--label` flag.** Head labels come from the filename grammar.
- If a triple's `*_regret.csv` is missing, it is **recomputed on the fly** and written back
  into the experiment folder (see crash recovery).

### `mlsys regret --frozen F.csv --finetune T.csv [--out R.csv] [--json J.json]`

Standalone regret-curve recompute — the "finetune crashed, recover regret without re-running
`full_eval`" path. Emits the curve in the exact `budget,regret,normalized_regret` format
(one row per integer budget `B = 1..|M|`, no gaps). With `--json` it also writes a
`regret.json`-shaped payload.

```bash
mlsys regret \
  --frozen   results/exp_wine_16/2296332_fulleval_16_model_FCH_frozen.csv \
  --finetune results/exp_wine_16/2296332_fulleval_16_model_FCH_finetune.csv \
  --out      results/exp_wine_16/2296332_fulleval_16_model_FCH_regret.csv
```

The recomputed curve is byte-identical to what `full_eval` would have written: it reuses
`mlsys.search.regret.regret_curve` and builds the proxy ranking the same way the pipeline
does (frozen r² desc, CSV row order as the stable tie-break).

## Crash-recovery recipe (step by step)

The finetune pass died mid-run, so you have the frozen + (partial) finetune CSVs but no
regret curve:

1. Keep the partial `*_finetune.csv` (and its `*_frozen.csv`).
2. Recompute the curve:
   ```bash
   mlsys regret \
     --frozen   <runid>_..._frozen.csv \
     --finetune <runid>_..._finetune.csv \
     --out      <runid>_..._regret.csv
   ```
3. Drop the recovered `*_regret.csv` back into the experiment folder.
4. Re-run `mlsys analyze <experiment_dir>`.

`analyze` is crash-tolerant by design — you don't even strictly need step 2/3: if a
`*_regret.csv` is absent it recomputes and writes it for you. It also degrades gracefully
when a whole head is missing:

- **Missing `*_regret.csv`** → recomputed and written back.
- **A whole head config missing** (e.g. the MLP_512 slurm job died) → that head is skipped
  with a warning; every surviving head still gets its folder and all cross-head artifacts.
- **One of frozen/finetune missing for a head** → just that head's folder is skipped, warned.

It never crashes the whole run because one file is absent.

## Plot / table catalog

Filenames are fixed slugs (referenced verbatim by `report.py` and `SUMMARY.md`). RQ1 =
adapting model search to regression; RQ2 = how bottlenecks shift.

### Per-head (`analysis/<head>/`)

| File | Shows | Serves |
| --- | --- | --- |
| `tables.csv` / `tables.md` | per-model frozen/finetune r²,mse,mae,spearman, Δ, skipped/diverged flags, epochs, costs | RQ1 + RQ2 |
| `r2_frozen_vs_finetune.png` | grouped bars, frozen vs finetune r² per model | §1/§2 |
| `proxy_scatter.png` | frozen-r² (proxy) vs finetune-r² with y=x; skipped/diverged coloured | RQ1 |
| `r2_delta.png` | diverging bars of Δ = finetune − frozen r² | §3 |
| `regret_curve.png` | regret + normalized_regret vs budget B | §4 |
| `finetune_spearman_vs_r2.png` | finetune Spearman vs r² — "rank kept, scale broken" | §2 / RQ1 |
| `timing_stacked.png` | stacked substep timing, frozen vs finetune side-by-side | RQ2 (R1) |
| `peak_gpu_mem.png` | peak GPU memory, frozen vs finetune | RQ2 (R2) |
| `frozen_time_breakdown.png` | frozen `inference_s` vs `train_head_s` share | RQ2 (R3) |

### Comparison (`analysis/comparison/`)

| File | Shows | Serves |
| --- | --- | --- |
| `per_head_summary.csv/.md` | one row per head: best r², regret@1, budget-to-zero, rank-ρ, counts | all |
| `frozen_r2_matrix.*` / `finetune_r2_matrix.*` | model × head r² tables | §3 |
| `diverged_models.*` | every diverged model's frozen→finetune r² + finetune ρ per head | §6 / RQ1 |
| `cost_table.*` | per-(head,model) frozen total, finetune total, peak mem, epochs | RQ2 |
| `regret_curves_by_head.png` | overlaid regret curves, one line per head | §4 |
| `regret_at1_vs_head.png` | regret@1 / AUC / budget-to-zero vs head width | §4 |
| `best_r2_vs_head.png` | best frozen & best finetune r² vs head width | §3 |
| `heatmap_frozen_r2.png` / `heatmap_finetune_r2.png` | model × head r² heatmaps | §3 |
| `divergence_map.png` | binary model × head map of `diverged` (finetune r² < 0) | RQ1 |
| `proxy_rank_spearman_vs_head.png` | Spearman(frozen rank, finetune rank) per head | RQ1 |
| `cost_vs_head.png` | mean finetune `train_head_s` & `peak_gpu_mem_mb` vs head width | RQ2 (R4) |

### RQ2 timing semantics (read the stacked bars correctly)

The five `*_s` fields are the RQ2 measurement. In the **frozen** pass cost splits across
`inference_s` (encode) + `train_head_s` (head fit). In the **finetune** pass inference is
fused into the joint loop, so `inference_s = 0` and `train_head_s` is the **end-to-end**
finetune cost. The plots/labels say this explicitly.

## The generate → hand to Claude workflow

1. `mlsys analyze results/<experiment>` — generates all artifacts + `SUMMARY.md`.
2. `SUMMARY.md` has a **fixed section order**: 0 metadata → 1 frozen → 2 finetune → 3
   frozen-vs-finetune → 4 regret → 5 RQ2 bottlenecks → 6 RQ1/RQ2 synthesis stubs. Section 6
   is *templated*: every number is pre-filled as `**metric:** <value>  <!-- prose: -->`, plus
   a diverged-model table, so Claude writes narrative over given numbers rather than
   re-deriving them from plots (the exact failure mode this tool prevents).
3. Hand the whole `analysis/` folder (or just `SUMMARY.md`, which `![](...)`s the plots) to
   Claude and ask for the report prose. Claude fills the `<!-- prose: -->` slots; it should
   **not** regenerate artifacts.
