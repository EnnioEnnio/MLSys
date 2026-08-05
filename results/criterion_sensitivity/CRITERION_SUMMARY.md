# Criterion sensitivity (does the ranking metric change measured proxy fidelity?)

## Headline

- housing rank correlation spans -0.044 to +0.450 across criteria on identical runs — the criterion, not the proxy, decides
- wine spans only +0.179..+0.219 (5-seed means): no tail, so the criterion barely matters
- housing regret@1 is at or above the random null under EVERY criterion; under MAE/Spearman the proxy's top pick is the non-finetunable model2vec embedder (Result 5's trap, independent of the metric)
- wine rows are 5 seed repeats under the final recipe (Result 5); housing is a single run, so its rank correlations are point estimates (n=16, SE ~ 0.26) and its regret@1 carries the [0.000, 0.132] single-run spread Result 5 measured

## Per-run, per-criterion

| group | criterion | rank_corr_mean | rank_corr_min | rank_corr_max | ref_min_trainable | ref_max_trainable | regret_at_1_mean | regret_at_1_max | null_regret_at_1 | budget_to_zero | proxy_top_1 | top_1_finetunable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wine (5 seeds) | r2 | 0.2030 | 0.0680 | 0.3000 | 0.8717 | 0.9176 | 0.1002 | 0.1323 | 0.0348 | 2 | potion-base-32M | · |
| wine (5 seeds) | mae | 0.1790 | 0.0790 | 0.2910 | 0.5938 | 0.8718 | 0.4204 | 0.5687 | 0.1749 | 14 | potion-base-32M | · |
| wine (5 seeds) | spearman | 0.2190 | 0.1650 | 0.2590 | 0.9324 | 0.9557 | 0.0544 | 0.0722 | 0.0179 | 2 | potion-base-32M | · |
| housing (1 run) | r2 | -0.0440 | -0.0440 | -0.0440 | -0.0000 | 0.0307 | 0.0131 | 0.0131 | 0.0090 | 10 | roberta-base | ✓ |
| housing (1 run) | mae | 0.4500 | 0.4500 | 0.4500 | 260787.2569 | 437642.1758 | 54177.1357 | 54177.1357 | 47372.8095 | 11 | potion-base-32M | · |
| housing (1 run) | spearman | 0.1910 | 0.1910 | 0.1910 | -0.0467 | 0.7756 | 0.1271 | 0.1271 | 0.1275 | 12 | potion-base-32M | · |
