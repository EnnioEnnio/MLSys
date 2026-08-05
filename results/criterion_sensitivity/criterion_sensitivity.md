| group | criterion | rank_corr_mean | rank_corr_min | rank_corr_max | ref_min_trainable | ref_max_trainable | regret_at_1_mean | regret_at_1_max | null_regret_at_1 | budget_to_zero | proxy_top_1 | top_1_finetunable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wine (5 seeds) | r2 | 0.2030 | 0.0680 | 0.3000 | 0.8717 | 0.9176 | 0.1002 | 0.1323 | 0.0348 | 2 | potion-base-32M | · |
| wine (5 seeds) | mae | 0.1790 | 0.0790 | 0.2910 | 0.5938 | 0.8718 | 0.4204 | 0.5687 | 0.1749 | 14 | potion-base-32M | · |
| wine (5 seeds) | spearman | 0.2190 | 0.1650 | 0.2590 | 0.9324 | 0.9557 | 0.0544 | 0.0722 | 0.0179 | 2 | potion-base-32M | · |
| housing (1 run) | r2 | -0.0440 | -0.0440 | -0.0440 | -0.0000 | 0.0307 | 0.0131 | 0.0131 | 0.0090 | 10 | roberta-base | ✓ |
| housing (1 run) | mae | 0.4500 | 0.4500 | 0.4500 | 260787.2569 | 437642.1758 | 54177.1357 | 54177.1357 | 47372.8095 | 11 | potion-base-32M | · |
| housing (1 run) | spearman | 0.1910 | 0.1910 | 0.1910 | -0.0467 | 0.7756 | 0.1271 | 0.1271 | 0.1275 | 12 | potion-base-32M | · |
