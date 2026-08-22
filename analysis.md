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
results/first_fulleval_wine_16_outdated/                                       # the input CSVs (12 for a 4-head sweep)
    2296332_wine_reviews_fulleval_16_model_FCH_frozen.csv
    2296332_wine_reviews_fulleval_16_model_FCH_finetune.csv
    2296332_wine_reviews_fulleval_16_model_FCH_regret.csv
    2296333_wine_reviews_fulleval_16_model_MLP_128_frozen.csv
    ...
results/first_fulleval_wine_16_outdated/analysis/                # generated (default --out-dir)
    SUMMARY.md                               # tables + embedded plots, in fixed section order
    FCH/  MLP_128/  MLP_256/  MLP_512/        # per-head tables + plots
    comparison/                              # cross-head tables + plots
```

The folder name (`first_fulleval_wine_16_outdated` here — wine dataset, 16-model pool) is arbitrary; `analyze`
only uses it as the `SUMMARY.md` title. Name experiments however you like.

### Filename grammar (how head labels are recovered)

```
<runid>_<dataset>_<strategy>_<num>_model_<HEAD>_<kind>.csv
```

The stem up to `_<kind>` is **exactly the W&B run name** the pipeline sets (see
`_wandb_run_name` in `src/mlsys/cli/main.py`, documented in the README). So the workflow is:
download a run's results CSV from W&B, append `_frozen` / `_finetune` / `_regret`, and drop it
into the experiment folder — no manual renaming.

- `<runid>` is the cluster/job id (W&B run name prefix); `<dataset>` may itself contain
  underscores (e.g. `wine_reviews`) and round-trips fine.
- `<strategy>` is the underscore-stripped strategy (`full_eval` → `fulleval`); `<num>` is the
  model-pool size.
- `<kind>` is the last token: `frozen`, `finetune`, or `regret`.
- `<HEAD>` is every token between the literal `model` and `<kind>` — so both `FCH` (linear,
  bare) and `MLP_512` round-trip.
- Files are grouped into a **triple** by `<runid>` (one head config per run-id).

The CSV's own `head_type` column is only `linear`/`mlp` and does **not** encode MLP width —
width comes from the filename. A filename that doesn't match the grammar raises a clear error
naming the expected pattern (no silent guessing, no CLI override).

## Commands

### `mlsys analyze <experiment_dir> [--out-dir DIR] [--paper]`

Discovers the triples, builds every per-head + cross-head artifact, and writes `SUMMARY.md`.

```bash
mlsys analyze results/first_fulleval_wine_16_outdated
mlsys analyze results/first_fulleval_wine_16_outdated --out-dir /tmp/wine_report
mlsys analyze results/first_fulleval_wine_16_outdated --paper     # figures for the LaTeX paper
```

- **No `--metric` flag.** The whole regret pipeline is r²-only (proxy ranking = frozen-r²
  desc, ground truth = finetune r²; `regret.json` hard-codes `"metric": "r2"`). mse/mae still
  appear in the per-model quality tables for context — just not as the regret basis.
- **No `--label` flag.** Head labels come from the filename grammar.
- If a triple's `*_regret.csv` is missing, it is **recomputed on the fly** and written back
  into the experiment folder (see crash recovery).

### `--paper` — the print preset

Default output is tuned for reading `SUMMARY.md` on screen. `--paper` retunes it for inclusion
in the LaTeX report, where figures are shrunk to column width and 12 pt matplotlib text ends up
at 4–6 pt on the page. The same flag exists on `scripts/noise_report.py` and
`scripts/cross_dataset_report.py`.

| | default (`SUMMARY.md`) | `--paper` |
|---|---|---|
| in-plot title | yes | **no** — the LaTeX `\caption` carries it |
| text size | seaborn `notebook` (11–12 pt) | **9 pt everywhere** (acmart body/caption size) |
| figure size | dynamic, `max(8, len(models))` | **fixed** at the final printed size |
| output | `.png` @ 120/150 dpi | **`.pdf`**, `pdf.fonttype=42` |
| `SUMMARY.md` | embeds the PNGs | *links* the PDFs (Markdown cannot inline a PDF) |

**Include the PDFs at exactly their authored width — do not rescale.** The sizes are chosen so
LaTeX applies scale 1.0 and 9 pt in the file stays 9 pt on the page:

```latex
3.335 in  ->  \includegraphics[width=\columnwidth]{...}      % single-column plots
3.43  in  ->  \includegraphics[width=0.49\textwidth]{...}    % side-by-side panels
7.0   in  ->  \includegraphics[width=\textwidth]{...}        % full-width, inside figure*
```

Type 42 font embedding is not optional: matplotlib's PDF backend defaults to Type 3, which
ACM/VLDB/IEEE PDF checkers reject, and a PDF/A build additionally requires every font embedded.

Implementation lives in `analysis/theme.py` (`apply_theme(paper=True)` plus the `theme.title()`
/ `theme.size()` / `theme.annot_kws()` / `theme.fig_ext()` helpers). It is a module-level
switch, so every plot function picks it up without a per-call argument.

### `mlsys regret --frozen F.csv --finetune T.csv [--out R.csv] [--json J.json]`

Standalone regret-curve recompute — the "finetune crashed, recover regret without re-running
`full_eval`" path. Emits the curve in the exact `budget,regret,normalized_regret` format
(one row per integer budget `B = 1..|M|`, no gaps). With `--json` it also writes a
`regret.json`-shaped payload.

```bash
mlsys regret \
  --frozen   results/first_fulleval_wine_16_outdated/2296332_wine_reviews_fulleval_16_model_FCH_frozen.csv \
  --finetune results/first_fulleval_wine_16_outdated/2296332_wine_reviews_fulleval_16_model_FCH_finetune.csv \
  --out      results/first_fulleval_wine_16_outdated/2296332_wine_reviews_fulleval_16_model_FCH_regret.csv
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
| `tables.csv` / `tables.md` | per-model proxy/reference r²,mse,mae,spearman, Δ, skipped/diverged flags, epochs, costs | RQ1 + RQ2 |
| `r2_proxy_vs_reference.png` | grouped bars, proxy vs reference r² per model | §1/§2 |
| `proxy_scatter.png` | proxy-r² vs reference-r² with y=x; skipped/diverged coloured | RQ1 |
| `r2_delta.png` | diverging bars of Δ = reference − proxy r² | §3 |
| `regret_curve.png` | regret + normalized_regret vs budget B | §4 |
| `ref_spearman_vs_r2.png` | reference Spearman vs r² — "rank kept, scale broken" | §2 / RQ1 |
| `timing_stacked.png` | stacked substep timing, proxy vs reference side-by-side | RQ2 (R1) |
| `peak_gpu_mem.png` | peak GPU memory, proxy vs reference | RQ2 (R2) |
| `proxy_time_breakdown.png` | proxy `inference_s` vs `train_head_s` share | RQ2 (R3) |

### Comparison (`analysis/comparison/`)

| File | Shows | Serves |
| --- | --- | --- |
| `per_head_summary.csv/.md` | one row per head: best r², regret@1, budget-to-zero, rank-ρ, counts | all |
| `proxy_r2_matrix.*` / `reference_r2_matrix.*` | model × head r² tables | §3 |
| `diverged_models.*` | every diverged model's proxy→reference r² + reference ρ per head | §6 / RQ1 |
| `cost_table.*` | per-(head,model) proxy total, reference total, peak mem, epochs | RQ2 |
| `proxy_distribution.*` | per-head proxy r² mean/std/min/max/n_negative — spread-collapse story | §7.1 |
| `head_gain.*` | per-model Δ proxy r² narrowest→widest head — biggest gainers | §7.2 |
| `epochs_table.*` | per-head mean proxy/reference epochs, n-at-cap, cap | §7.3 |
| `head_rank_agreement.*` | head × head Spearman ρ over proxy r² | §7.4 |
| `proxy_timing_share.*` | per-head % share of each timing substep | §7.5 |
| `value_frontier.*` | proxy inference_s vs r² at the widest head | §7.6 |
| `regret_curves_by_head.png` | overlaid regret curves, one line per head | §4 |
| `regret_at1_vs_head.png` | regret@1 / AUC / budget-to-zero vs head width | §4 |
| `best_r2_vs_head.png` | best proxy & best reference r² vs head width | §3 |
| `heatmap_proxy_r2.png` / `heatmap_ref_r2.png` | model × head r² heatmaps | §3 |
| `divergence_map.png` | binary model × head map of `diverged` (reference r² < 0) | RQ1 |
| `proxy_rank_spearman_vs_head.png` | Spearman(proxy rank, reference rank) per head | RQ1 |
| `cost_vs_head.png` | mean reference `train_head_s` & `peak_gpu_mem_mb` vs head width | RQ2 (R4) |
| `epochs_vs_head.png` | mean proxy vs reference epochs per head; annotated proxy cap | §7.3 |
| `head_rank_agreement.png` | heatmap of head × head Spearman ρ matrix | §7.4 |
| `proxy_timing_share.png` | horizontal 100%-stacked bar: substep % share per head | §7.5 |
| `value_frontier.png` | scatter: proxy inference_s vs r² at the widest head, annotated | §7.6 |

### RQ2 timing semantics (read the stacked bars correctly)

The five `*_s` fields are the RQ2 measurement. In the **frozen** pass cost splits across
`inference_s` (encode) + `train_head_s` (head fit). In the **finetune** pass inference is
fused into the joint loop, so `inference_s = 0` and `train_head_s` is the **end-to-end**
finetune cost. The plots/labels say this explicitly.

## The generate → hand to Claude workflow

1. `mlsys analyze results/<experiment>` — generates all artifacts + `SUMMARY.md`.
2. `SUMMARY.md` has a **fixed section order**: 0 metadata → 1 frozen → 2 finetune → 3
   frozen-vs-finetune → 4 regret → 5 RQ2 bottlenecks → 6 RQ1/RQ2 synthesis stubs → 7
   distribution, ranking stability & cost. Section 6 is *templated*: every number is
   pre-filled as `**metric:** <value>  <!-- prose: -->`, plus a diverged-model table, so
   Claude writes narrative over given numbers rather than re-deriving them from plots (the
   exact failure mode this tool prevents). Section 7 adds the six analysis gaps from the
   intermediate deck: frozen r² spread collapse, per-model head-gain, early-stopping epochs,
   head × head rank agreement, frozen timing-substep share, and the inference value-frontier.
3. Hand the whole `analysis/` folder (or just `SUMMARY.md`, which `![](...)`s the plots) to
   Claude and ask for the report prose. Claude fills the `<!-- prose: -->` slots; it should
   **not** regenerate artifacts.
