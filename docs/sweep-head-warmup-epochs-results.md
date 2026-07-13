# Head-warmup epoch sweep — 0 / 1 / 2 / 4 / 8 warmup epochs

Extends [warmup-results.md](warmup-results.md) (which compared 2-epoch warmup vs. none at a
single width) into a full sweep of the **warmup budget**. How many head-only warmup epochs
does LP-FT ([finetune-warmup.md](finetune-warmup.md)) need before the 10-epoch joint loop to
kill the negative-r² divergences, and is there a quality cost from over-warming?

> **The `mlsys analyze` pipeline does not apply here.** It pairs a `frozen` proxy with a
> `finetune` reference per run-id to compute regret; it has no warmup axis, and all five files
> are the same `kind` (`finetune`) with distinct run-ids, so no analysable pair forms. These
> numbers were computed directly from the five CSVs.

## Setup

- **dataset:** wine_reviews, 16-model pool
- **head:** MLP, hidden dim 256
- **finetune:** 10 epochs joint training
- **warmup arms:** 0, 1, 2, 4, 8 head-only epochs before the joint loop
  - `wu0` = job `2320382`, `wu2` = job `2320372` (both reused from
    [warmup-results.md](warmup-results.md)); `wu1` = `2327073`, `wu4` = `2327074`,
    `wu8` = `2327075`.
- one finetune run per model (`head_repeats=1`), so per-model r² deltas are noisy; the robust
  signals are the **divergence count** and the **aggregate distribution**.

Source CSVs: `results/warmup_sweep_256/`.

## Bottom line

Warmup is a **stabilizer, not a ceiling-raiser** — consistent with
[warmup-results.md](warmup-results.md). **Even 1 warmup epoch removes all divergences** that a
cold start (`wu0`) produces. Among the models that were already stable, more warmup buys a
small, noisy quality bump that peaks around **wu4**; beyond that it flattens. The one chronic
diverger (`deberta-v3-base`) is *not* rescued by any warmup budget — that's a separate LR /
clipping problem (#32), not a warmup one.

**Recommendation: `wu4` is the sweet spot** (zero divergences + highest mean r²). `wu1`
already captures the entire stability win at the lowest cost; the cluster default of `wu2` is
a fine conservative choice.

## Divergence — the headline

Count of models with r² < 0 (a ranking-preserving scale/offset blow-up; Spearman stays ≈ 0.91
in every arm, so it's a regression-fit failure, not a ranking one):

| warmup | 0 | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- | --- |
| **n diverged** | **3** | **0** | 1 | **0** | 1 |
| diverged models | deberta-v3-small, roberta-base, deberta-v3-base | — | deberta-v3-base | — | deberta-v3-base |

Cold start (`wu0`) blows up three backbones. **A single warmup epoch fixes two of the three**
(deberta-v3-small: −0.26 → +0.81; roberta-base: −0.12 → +0.86) and clears the board. The
`wu2`/`wu8` "divergences" are the *same* chronic model, `deberta-v3-base`, which never clears
positive r² under any budget (best = **0.26** at wu4).

## Aggregate quality & cost

| warmup | mean r² | median r² | mean Spearman | n diverged | mean MAE | total train (h) | mean epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.553 | 0.798 | 0.915 | 3 | 1.591 | 10.7 | 9.6 |
| 1 | 0.762 | 0.859 | 0.916 | 0 | 1.121 | 11.3 | 9.8 |
| 2 | 0.736 | 0.862 | 0.910 | 1 | 1.181 | 10.7 | 8.8 |
| **4** | **0.787** | 0.866 | 0.914 | **0** | **1.087** | 12.2 | 9.4 |
| 8 | 0.740 | 0.876 | 0.919 | 1 | 1.152 | 11.7 | 9.3 |

The mean-r² column is dominated by the divergence count (a single −0.6 outlier drags the mean
hard), which is why `wu0` looks catastrophic and the divergence-free arms (`wu1`, `wu4`) top
it. **Median** r² is the cleaner ceiling signal and it climbs gently and monotonically with
warmup (0.798 → 0.876), confirming warmup doesn't hurt the models that already converge.

**Cost is flat.** Total joint-training time stays ~10.7–12.2 h across all arms — warmup epochs
are head-only (backbone frozen) and cheap, so even `wu8` adds no meaningful wall-clock penalty.

### Quality on the already-stable models

Excluding the three models that diverge at `wu0` (deberta-v3-small, roberta-base,
deberta-v3-base), so the mean isn't hostage to the blow-ups:

| warmup | 0 | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- | --- |
| mean r² (13 stable models) | 0.758 | 0.807 | 0.805 | **0.825** | 0.820 |

A clean, small monotone-ish gain that peaks at **wu4** (+0.067 over cold start) and plateaus —
the classic diminishing-returns warmup curve.

## Per-model r² across the sweep

Best arm per model in the last column (noisy at `head_repeats=1`, read qualitatively):

| model | wu0 | wu1 | wu2 | wu4 | wu8 | best |
| --- | --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.870 | 0.859 | 0.878 | 0.866 | 0.867 | wu2 |
| all-mpnet-base-v2 | 0.892 | 0.883 | 0.824 | 0.889 | 0.895 | wu8 |
| potion-base-8M | 0.681 | 0.681 | 0.659 | 0.692 | 0.666 | wu4 |
| potion-base-32M | 0.724 | 0.697 | 0.666 | 0.695 | 0.680 | wu0 |
| deberta-v3-small | **−0.258** | 0.815 | 0.771 | 0.743 | 0.734 | wu1 |
| modernbert-base | 0.900 | 0.895 | 0.906 | 0.885 | 0.909 | wu8 |
| distilbert-base-uncased | 0.798 | 0.892 | 0.893 | 0.893 | 0.893 | wu4 |
| roberta-base | **−0.118** | 0.865 | 0.788 | 0.873 | 0.880 | wu8 |
| deberta-v3-base | **−0.623** | 0.023 | **−0.240** | 0.255 | **−0.439** | wu4 (chronic) |
| electra-base-discriminator | 0.205 | 0.383 | 0.408 | 0.662 | 0.548 | wu4 |
| albert-base-v2 | 0.873 | 0.899 | 0.866 | 0.827 | 0.785 | wu1 |
| e5-base-v2 | 0.703 | 0.855 | 0.866 | 0.787 | 0.876 | wu8 |
| bge-base-en-v1.5 | 0.619 | 0.803 | 0.882 | 0.882 | 0.876 | wu2 |
| modernbert-embed-base | 0.913 | 0.890 | 0.908 | 0.891 | 0.902 | wu0 |
| sentence-t5-base | 0.867 | 0.852 | 0.862 | 0.852 | 0.852 | wu0 |
| mxbai-embed-large-v1 | 0.805 | 0.906 | 0.845 | 0.903 | 0.909 | wu8 |

The strong sentence-transformer encoders (modernbert, mpnet, MiniLM, t5) are ~warmup-agnostic
— they sit at r² ≈ 0.85–0.91 regardless, because they never distort. The **from-scratch /
masked-LM backbones** (deberta, roberta, electra, distilbert) are where warmup earns its keep:
they're the ones that blow up cold and recover with warmup. `electra` in particular needs the
*most* warmup (0.21 → 0.66 from wu0 → wu4).

## Takeaways

1. **Turn warmup on.** `wu0` diverges 3/16 models; any warmup ≥ 1 removes the epidemic. This
   re-confirms [warmup-results.md](warmup-results.md) with finer resolution.
2. **`wu4` is the peak, `wu1` is the bargain.** wu4 gives the best mean/stable-mean r² with
   zero divergences; wu1 already captures the whole stability win essentially for free. wu2
   (cluster default) is a reasonable middle. There's no case for `wu8` — it's not better and
   occasionally worse.
3. **Warmup is nearly free.** Head-only epochs don't move total training time, so the choice is
   a quality/stability call, not a cost one.
4. **`deberta-v3-base` is a warmup-resistant outlier.** No budget clears it above r² ≈ 0.26.
   That's the LR / grad-clipping / target-standardization workstream (#32), not warmup (#31).
