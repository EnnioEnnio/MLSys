# Dataset comparison report (fixed MLP_256 final recipe)

## Headline

- housing_full: finetune r² in [-0.000, 0.031] while prediction-vs-target rho reaches 0.78 — ordering is learnable, the squared error is not (long-tailed target; affine z-scoring cannot fix this)
- wine rank survival at 2k rows: frozen↔frozen rho = 0.88

## Per-dataset summary

| dataset | n_models | fz_r2_mean | fz_r2_min | fz_r2_max | ft_r2_mean | ft_r2_min | ft_r2_max | ft_band_trainable | ft_sigma_between | ft_pred_spearman_mean | n_diverged | top1_frozen | top1_finetune | regret_at_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wine_full | 16 | 0.7339 | 0.5334 | 0.7927 | 0.8835 | 0.7474 | 0.9176 | 0.0459 | 0.0132 | 0.9475 | 0 | potion-base-32M | modernbert-embed-base | 0.1002 |
| wine_tiny | 16 | 0.5754 | 0.3160 | 0.6763 | 0.6786 | 0.3920 | 0.7419 | 0.3499 | 0.0886 | 0.8305 | 0 | modernbert-embed-base | all-mpnet-base-v2 | 0.0121 |
| housing_full | 16 | 0.0143 | -0.0001 | 0.0226 | 0.0217 | -0.0000 | 0.0307 | 0.0307 | 0.0089 | 0.6537 | 1 | roberta-base | deberta-v3-small | 0.0131 |
| housing_tiny | 16 | 0.0796 | 0.0351 | 0.1391 | 0.1809 | 0.0100 | 0.2563 | 0.2463 | 0.0614 | 0.6264 | 0 | mxbai-embed-large-v1 | modernbert-base | 0.0468 |

## Rank survival (full pool, 16 models)

| dataset | tiny_fz_vs_full_fz | tiny_fz_vs_full_ft | tiny_ft_vs_full_ft | tiny_fz_vs_tiny_ft | full_fz_vs_full_ft | tiny_regret_at_1 | full_regret_at_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wine | 0.8794 | 0.1441 | 0.6235 | 0.4941 | 0.2029 | 0.0121 | 0.1002 |
| housing | 0.3412 | 0.3147 | 0.5588 | 0.3882 | -0.0441 | 0.0468 | 0.0131 |
