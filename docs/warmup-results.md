# Head warmup results — 2-epoch warmup vs. no warmup

Follow-up experiment for the [head-warmup (LP-FT)](finetune-warmup.md) change
([issue #38](https://github.com/EnnioEnnio/MLSys/issues/38)). Does head-only warmup
eliminate the negative-r² finetune divergences, and at what quality cost on the models
that already converged?

## Setup

- **dataset:** wine_reviews, 16-model pool
- **head:** MLP, hidden dim 256
- **finetune:** 10 epochs joint training
- **A — warmup:** 2 epochs head-only warmup before the joint loop (array job `2320372`)
- **B — no warmup:** straight to joint training (array job `2320382`)
- one finetune run per model (`head_repeats=1`), so per-model r² deltas are noisy; the
  robust signal is the **divergence count**.

Artifacts: `results/full_eval_warmup_vs_no_warmup_256/analysis/`.

## Bottom line

Head warmup does what LP-FT predicts: it **stabilizes** fine-tuning. It does not raise the
ceiling (best model is `modernbert-embed-base` at r² ≈ 0.91 in both runs) — it raises the
**floor** by stopping backbones from diverging when the joint loop starts from a
randomly-initialised head.

## Divergence (the headline)

| | warmup (2ep) | no warmup |
| --- | --- | --- |
| diverged finetunes (r² < 0) | **1** | **3** |
| mean finetune r² (all 16 models) | **0.736** | 0.553 |

Warmup rescued two backbones that blew up without it:

| model | no-warmup ref_r² | warmup ref_r² |
| --- | --- | --- |
| deberta-v3-small | **−0.258** | **+0.771** |
| roberta-base | **−0.118** | **+0.788** |
| deberta-v3-base | −0.623 | −0.240 (still diverged) |

`deberta-v3-base` is fragile in both runs — warmup halves its damage but doesn't cure it,
so it's an honest control that warmup is not a blanket fix. As in `exp_wine_16`, the diverged
runs keep a healthy Spearman (0.86–0.93): a ranking-preserving scale/offset blow-up
consistent with head-driven feature distortion, exactly the failure mode warmup targets.

Several already-converging models also improved (bge-base 0.62→0.88, e5-base 0.70→0.87,
distilbert 0.80→0.89); a couple slipped slightly (all-mpnet 0.89→0.82) — within noise.

## Regret

| head | regret@1 | norm. regret@1 | diverged |
| --- | --- | --- | --- |
| warmup | **0.043** | **0.047** | 1 |
| no warmup | 0.210 | 0.230 | 3 |

Warmup cuts regret@1 ~5×. Caveat: no-warmup regret@1 is inflated *because* the diverged
models corrupt the finetune ground truth. The `budget_to_zero` (warmup=7, nowa=2) and
`rank_spearman` (both ≈0) numbers are artifacts of *which* models diverged, not a
warmup-vs-nowa signal — don't lean on them.

## Caveats

- **Noise floor.** The frozen (proxy) pass should be identical between the two runs (warmup
  only touches finetuning), but it isn't — e.g. bge-base frozen r² 0.708 vs 0.602. That gap
  is pure run-to-run variance between the two SLURM jobs and sets the noise floor:
  single-model differences below ~0.05–0.10 are not real.
- **n=1 per config.** The divergence-reduction story (3→1) is the robust conclusion;
  individual r² deltas are noisy.

## Conclusion

Keep the cluster default of `--warmup-epochs 2`. It cut divergences 3→1 and lifted mean
finetune r² from 0.55 to 0.74 at the cost of two extra head-only epochs, with no loss at the
top of the pool. The remaining `deberta-v3-base` divergence is the LR / gradient-clipping /
target-standardization half of the fix (issue #32).
