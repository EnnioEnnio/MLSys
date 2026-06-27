| head | n_models | best_frozen_model | best_frozen_r2 | best_finetune_model | best_finetune_r2 | regret_at_1 | normalized_regret_at_1 | budget_to_zero | regret_auc | rank_spearman | n_diverged | n_finetune_skipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCH | 16 | e5-base-v2 | 0.6820 | modernbert-embed-base | 0.8660 | 0.2740 | 0.3164 | 9 | 0.0509 | 0.4294 | 3 | 2 |
| MLP_128 | 16 | e5-base-v2 | 0.7095 | modernbert-embed-base | 0.8472 | 0.2487 | 0.2935 | 6 | 0.0450 | -0.0382 | 3 | 2 |
| MLP_256 | 16 | potion-base-32M | 0.7173 | modernbert-embed-base | 0.8597 | 0.1762 | 0.2049 | 3 | 0.0220 | 0.0324 | 3 | 2 |
| MLP_512 | 16 | mxbai-embed-large-v1 | 0.7494 | modernbert-embed-base | 0.8544 | 0.0662 | 0.0775 | 2 | 0.0041 | 0.1765 | 3 | 2 |
