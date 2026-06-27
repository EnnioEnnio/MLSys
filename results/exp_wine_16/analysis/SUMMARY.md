# Analysis — exp_wine_16

## 0. Experiment metadata

- **dataset:** wine_reviews

- **pool size:** 16 models

- **heads found:** FCH, MLP_128, MLP_256, MLP_512

- **heads skipped:** none


## 1. Frozen results (cheap proxy)

### Head FCH

| model | frozen_r2 | frozen_mse | frozen_mae | frozen_spearman |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | -0.2435 | 12.4546 | 2.7684 | 0.4425 |
| all-mpnet-base-v2 | 0.1169 | 8.8446 | 2.3219 | 0.5570 |
| potion-base-8M | -0.1124 | 11.1415 | 2.6270 | 0.5120 |
| potion-base-32M | 0.0870 | 9.1443 | 2.3766 | 0.5785 |
| deberta-v3-small | 0.6189 | 3.8165 | 1.5461 | 0.7845 |
| modernbert-base | 0.6297 | 3.7084 | 1.5239 | 0.7892 |
| distilbert-base-uncased | 0.6540 | 3.4657 | 1.4690 | 0.8050 |
| roberta-base | 0.6724 | 3.2812 | 1.4313 | 0.8180 |
| deberta-v3-base | 0.3965 | 6.0441 | 1.9503 | 0.6299 |
| electra-base-discriminator | 0.6611 | 3.3938 | 1.4573 | 0.8097 |
| albert-base-v2 | 0.6627 | 3.3786 | 1.4538 | 0.8108 |
| e5-base-v2 | 0.6820 | 3.1847 | 1.4086 | 0.8230 |
| bge-base-en-v1.5 | 0.6227 | 3.7787 | 1.5366 | 0.7873 |
| modernbert-embed-base | 0.6291 | 3.7149 | 1.5251 | 0.7970 |
| sentence-t5-base | 0.6505 | 3.5003 | 1.4769 | 0.8057 |
| mxbai-embed-large-v1 | 0.6653 | 3.3522 | 1.4462 | 0.8142 |

![](FCH/r2_frozen_vs_finetune.png)

![](FCH/proxy_scatter.png)

### Head MLP_128

| model | frozen_r2 | frozen_mse | frozen_mae | frozen_spearman |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.5722 | 4.2846 | 1.6355 | 0.7495 |
| all-mpnet-base-v2 | 0.6266 | 3.7393 | 1.5274 | 0.7882 |
| potion-base-8M | 0.6694 | 3.3106 | 1.4294 | 0.8129 |
| potion-base-32M | 0.6878 | 3.1272 | 1.3941 | 0.8256 |
| deberta-v3-small | 0.6478 | 3.5279 | 1.4844 | 0.7985 |
| modernbert-base | 0.6407 | 3.5985 | 1.4993 | 0.7951 |
| distilbert-base-uncased | 0.6647 | 3.3580 | 1.4448 | 0.8120 |
| roberta-base | 0.6841 | 3.1638 | 1.4050 | 0.8256 |
| deberta-v3-base | 0.4523 | 5.4855 | 1.8593 | 0.6677 |
| electra-base-discriminator | 0.6893 | 3.1115 | 1.3922 | 0.8245 |
| albert-base-v2 | 0.6674 | 3.3313 | 1.4414 | 0.8126 |
| e5-base-v2 | 0.7095 | 2.9094 | 1.3413 | 0.8367 |
| bge-base-en-v1.5 | 0.6250 | 3.7561 | 1.5325 | 0.7894 |
| modernbert-embed-base | 0.6730 | 3.2746 | 1.4308 | 0.8211 |
| sentence-t5-base | 0.6873 | 3.1321 | 1.3916 | 0.8254 |
| mxbai-embed-large-v1 | 0.6684 | 3.3209 | 1.4374 | 0.8153 |

![](MLP_128/r2_frozen_vs_finetune.png)

![](MLP_128/proxy_scatter.png)

### Head MLP_256

| model | frozen_r2 | frozen_mse | frozen_mae | frozen_spearman |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.6104 | 3.9019 | 1.5562 | 0.7751 |
| all-mpnet-base-v2 | 0.6268 | 3.7378 | 1.5296 | 0.7887 |
| potion-base-8M | 0.6954 | 3.0508 | 1.3694 | 0.8291 |
| potion-base-32M | 0.7173 | 2.8317 | 1.3206 | 0.8427 |
| deberta-v3-small | 0.6489 | 3.5161 | 1.4805 | 0.7987 |
| modernbert-base | 0.6377 | 3.6289 | 1.5067 | 0.7954 |
| distilbert-base-uncased | 0.6728 | 3.2774 | 1.4283 | 0.8174 |
| roberta-base | 0.6943 | 3.0612 | 1.3799 | 0.8309 |
| deberta-v3-base | 0.4880 | 5.1275 | 1.7965 | 0.6916 |
| electra-base-discriminator | 0.6943 | 3.0619 | 1.3806 | 0.8270 |
| albert-base-v2 | 0.6904 | 3.1011 | 1.3867 | 0.8249 |
| e5-base-v2 | 0.7169 | 2.8353 | 1.3265 | 0.8419 |
| bge-base-en-v1.5 | 0.6826 | 3.1792 | 1.4059 | 0.8206 |
| modernbert-embed-base | 0.7142 | 2.8621 | 1.3340 | 0.8413 |
| sentence-t5-base | 0.6967 | 3.0382 | 1.3683 | 0.8302 |
| mxbai-embed-large-v1 | 0.7101 | 2.9038 | 1.3428 | 0.8385 |

![](MLP_256/r2_frozen_vs_finetune.png)

![](MLP_256/proxy_scatter.png)

### Head MLP_512

| model | frozen_r2 | frozen_mse | frozen_mae | frozen_spearman |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.6400 | 3.6053 | 1.4931 | 0.7930 |
| all-mpnet-base-v2 | 0.6432 | 3.5733 | 1.4915 | 0.7976 |
| potion-base-8M | 0.7211 | 2.7932 | 1.3079 | 0.8444 |
| potion-base-32M | 0.7419 | 2.5849 | 1.2591 | 0.8575 |
| deberta-v3-small | 0.6249 | 3.7564 | 1.5345 | 0.7844 |
| modernbert-base | 0.6877 | 3.1273 | 1.3952 | 0.8241 |
| distilbert-base-uncased | 0.6935 | 3.0695 | 1.3790 | 0.8276 |
| roberta-base | 0.6984 | 3.0209 | 1.3713 | 0.8326 |
| deberta-v3-base | 0.4918 | 5.0895 | 1.7902 | 0.6937 |
| electra-base-discriminator | 0.6976 | 3.0286 | 1.3729 | 0.8291 |
| albert-base-v2 | 0.6919 | 3.0854 | 1.3827 | 0.8264 |
| e5-base-v2 | 0.7433 | 2.5707 | 1.2574 | 0.8561 |
| bge-base-en-v1.5 | 0.7215 | 2.7895 | 1.3100 | 0.8446 |
| modernbert-embed-base | 0.7450 | 2.5536 | 1.2594 | 0.8599 |
| sentence-t5-base | 0.6901 | 3.1036 | 1.3852 | 0.8269 |
| mxbai-embed-large-v1 | 0.7494 | 2.5097 | 1.2432 | 0.8606 |

![](MLP_512/r2_frozen_vs_finetune.png)

![](MLP_512/proxy_scatter.png)


## 2. Finetune results (ground truth)

### Head FCH

| model | finetune_r2 | finetune_spearman | diverged | finetune_skipped | finetune_epochs |
| --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.7165 | 0.8676 | · | · | 3 |
| all-mpnet-base-v2 | -0.0007 | 0.4143 | ✓ | · | 3 |
| potion-base-8M | -0.1145 | 0.5113 | · | ✓ | 30 |
| potion-base-32M | 0.0875 | 0.5783 | · | ✓ | 30 |
| deberta-v3-small | -0.0296 | 0.8886 | ✓ | · | 3 |
| modernbert-base | 0.8357 | 0.9110 | · | · | 3 |
| distilbert-base-uncased | 0.7984 | 0.8966 | · | · | 3 |
| roberta-base | 0.4782 | 0.8947 | · | · | 3 |
| deberta-v3-base | -0.3981 | 0.8727 | ✓ | · | 3 |
| electra-base-discriminator | 0.2457 | 0.9038 | · | · | 3 |
| albert-base-v2 | 0.8249 | 0.9094 | · | · | 3 |
| e5-base-v2 | 0.5920 | 0.9190 | · | · | 3 |
| bge-base-en-v1.5 | 0.7039 | 0.9274 | · | · | 3 |
| modernbert-embed-base | 0.8660 | 0.9281 | · | · | 3 |
| sentence-t5-base | 0.7477 | 0.8791 | · | · | 3 |
| mxbai-embed-large-v1 | 0.7936 | 0.9356 | · | · | 3 |

![](FCH/finetune_spearman_vs_r2.png)

### Head MLP_128

| model | finetune_r2 | finetune_spearman | diverged | finetune_skipped | finetune_epochs |
| --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.8275 | 0.9065 | · | · | 3 |
| all-mpnet-base-v2 | 0.8219 | 0.9104 | · | · | 3 |
| potion-base-8M | 0.6408 | 0.7956 | · | ✓ | 18 |
| potion-base-32M | 0.6715 | 0.8173 | · | ✓ | 23 |
| deberta-v3-small | -0.4730 | 0.9078 | ✓ | · | 3 |
| modernbert-base | 0.8420 | 0.9149 | · | · | 3 |
| distilbert-base-uncased | 0.6739 | 0.8908 | · | · | 3 |
| roberta-base | -0.1178 | 0.9219 | ✓ | · | 3 |
| deberta-v3-base | -0.4896 | 0.9061 | ✓ | · | 3 |
| electra-base-discriminator | 0.1504 | 0.9155 | · | · | 3 |
| albert-base-v2 | 0.8219 | 0.9023 | · | · | 3 |
| e5-base-v2 | 0.5985 | 0.9297 | · | · | 3 |
| bge-base-en-v1.5 | 0.5404 | 0.9030 | · | · | 3 |
| modernbert-embed-base | 0.8472 | 0.9168 | · | · | 3 |
| sentence-t5-base | 0.8235 | 0.9059 | · | · | 3 |
| mxbai-embed-large-v1 | 0.7112 | 0.9079 | · | · | 3 |

![](MLP_128/finetune_spearman_vs_r2.png)

### Head MLP_256

| model | finetune_r2 | finetune_spearman | diverged | finetune_skipped | finetune_epochs |
| --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.8267 | 0.9068 | · | · | 3 |
| all-mpnet-base-v2 | 0.8432 | 0.9166 | · | · | 3 |
| potion-base-8M | 0.6957 | 0.8286 | · | ✓ | 22 |
| potion-base-32M | 0.6835 | 0.8236 | · | ✓ | 17 |
| deberta-v3-small | -1.1842 | 0.8943 | ✓ | · | 3 |
| modernbert-base | 0.7861 | 0.8989 | · | · | 3 |
| distilbert-base-uncased | 0.8070 | 0.9055 | · | · | 3 |
| roberta-base | 0.1261 | 0.8953 | · | · | 3 |
| deberta-v3-base | -0.7111 | 0.9146 | ✓ | · | 3 |
| electra-base-discriminator | -0.0289 | 0.8870 | ✓ | · | 3 |
| albert-base-v2 | 0.8069 | 0.8979 | · | · | 3 |
| e5-base-v2 | 0.5861 | 0.9174 | · | · | 3 |
| bge-base-en-v1.5 | 0.7472 | 0.9161 | · | · | 3 |
| modernbert-embed-base | 0.8597 | 0.9233 | · | · | 3 |
| sentence-t5-base | 0.8283 | 0.9073 | · | · | 3 |
| mxbai-embed-large-v1 | 0.8035 | 0.9245 | · | · | 3 |

![](MLP_256/finetune_spearman_vs_r2.png)

### Head MLP_512

| model | finetune_r2 | finetune_spearman | diverged | finetune_skipped | finetune_epochs |
| --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.8181 | 0.8994 | · | · | 3 |
| all-mpnet-base-v2 | 0.8292 | 0.9226 | · | · | 3 |
| potion-base-8M | 0.7105 | 0.8390 | · | ✓ | 22 |
| potion-base-32M | 0.7284 | 0.8485 | · | ✓ | 18 |
| deberta-v3-small | -0.5057 | 0.8928 | ✓ | · | 3 |
| modernbert-base | 0.7864 | 0.8966 | · | · | 3 |
| distilbert-base-uncased | 0.4470 | 0.8895 | · | · | 3 |
| roberta-base | 0.4098 | 0.9198 | · | · | 3 |
| deberta-v3-base | -1.3162 | 0.9114 | ✓ | · | 3 |
| electra-base-discriminator | -0.0849 | 0.9114 | ✓ | · | 3 |
| albert-base-v2 | 0.8144 | 0.9034 | · | · | 3 |
| e5-base-v2 | 0.4927 | 0.9002 | · | · | 3 |
| bge-base-en-v1.5 | 0.5653 | 0.9017 | · | · | 3 |
| modernbert-embed-base | 0.8544 | 0.9217 | · | · | 3 |
| sentence-t5-base | 0.8132 | 0.8995 | · | · | 3 |
| mxbai-embed-large-v1 | 0.7882 | 0.9312 | · | · | 3 |

![](MLP_512/finetune_spearman_vs_r2.png)


## 3. Frozen vs finetune comparison

### Frozen r² (model x head)

| model | FCH | MLP_128 | MLP_256 | MLP_512 |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | -0.2435 | 0.5722 | 0.6104 | 0.6400 |
| all-mpnet-base-v2 | 0.1169 | 0.6266 | 0.6268 | 0.6432 |
| potion-base-8M | -0.1124 | 0.6694 | 0.6954 | 0.7211 |
| potion-base-32M | 0.0870 | 0.6878 | 0.7173 | 0.7419 |
| deberta-v3-small | 0.6189 | 0.6478 | 0.6489 | 0.6249 |
| modernbert-base | 0.6297 | 0.6407 | 0.6377 | 0.6877 |
| distilbert-base-uncased | 0.6540 | 0.6647 | 0.6728 | 0.6935 |
| roberta-base | 0.6724 | 0.6841 | 0.6943 | 0.6984 |
| deberta-v3-base | 0.3965 | 0.4523 | 0.4880 | 0.4918 |
| electra-base-discriminator | 0.6611 | 0.6893 | 0.6943 | 0.6976 |
| albert-base-v2 | 0.6627 | 0.6674 | 0.6904 | 0.6919 |
| e5-base-v2 | 0.6820 | 0.7095 | 0.7169 | 0.7433 |
| bge-base-en-v1.5 | 0.6227 | 0.6250 | 0.6826 | 0.7215 |
| modernbert-embed-base | 0.6291 | 0.6730 | 0.7142 | 0.7450 |
| sentence-t5-base | 0.6505 | 0.6873 | 0.6967 | 0.6901 |
| mxbai-embed-large-v1 | 0.6653 | 0.6684 | 0.7101 | 0.7494 |

### Finetune r² (model x head)

| model | FCH | MLP_128 | MLP_256 | MLP_512 |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.7165 | 0.8275 | 0.8267 | 0.8181 |
| all-mpnet-base-v2 | -0.0007 | 0.8219 | 0.8432 | 0.8292 |
| potion-base-8M | -0.1145 | 0.6408 | 0.6957 | 0.7105 |
| potion-base-32M | 0.0875 | 0.6715 | 0.6835 | 0.7284 |
| deberta-v3-small | -0.0296 | -0.4730 | -1.1842 | -0.5057 |
| modernbert-base | 0.8357 | 0.8420 | 0.7861 | 0.7864 |
| distilbert-base-uncased | 0.7984 | 0.6739 | 0.8070 | 0.4470 |
| roberta-base | 0.4782 | -0.1178 | 0.1261 | 0.4098 |
| deberta-v3-base | -0.3981 | -0.4896 | -0.7111 | -1.3162 |
| electra-base-discriminator | 0.2457 | 0.1504 | -0.0289 | -0.0849 |
| albert-base-v2 | 0.8249 | 0.8219 | 0.8069 | 0.8144 |
| e5-base-v2 | 0.5920 | 0.5985 | 0.5861 | 0.4927 |
| bge-base-en-v1.5 | 0.7039 | 0.5404 | 0.7472 | 0.5653 |
| modernbert-embed-base | 0.8660 | 0.8472 | 0.8597 | 0.8544 |
| sentence-t5-base | 0.7477 | 0.8235 | 0.8283 | 0.8132 |
| mxbai-embed-large-v1 | 0.7936 | 0.7112 | 0.8035 | 0.7882 |

![](comparison/heatmap_frozen_r2.png)

![](comparison/heatmap_finetune_r2.png)

![](comparison/divergence_map.png)

![](comparison/best_r2_vs_head.png)

![](FCH/r2_delta.png)

![](MLP_128/r2_delta.png)

![](MLP_256/r2_delta.png)

![](MLP_512/r2_delta.png)


## 4. Regret

| head | n_models | best_frozen_model | best_frozen_r2 | best_finetune_model | best_finetune_r2 | regret_at_1 | normalized_regret_at_1 | budget_to_zero | regret_auc | rank_spearman | n_diverged | n_finetune_skipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCH | 16 | e5-base-v2 | 0.6820 | modernbert-embed-base | 0.8660 | 0.2740 | 0.3164 | 9 | 0.0509 | 0.4294 | 3 | 2 |
| MLP_128 | 16 | e5-base-v2 | 0.7095 | modernbert-embed-base | 0.8472 | 0.2487 | 0.2935 | 6 | 0.0450 | -0.0382 | 3 | 2 |
| MLP_256 | 16 | potion-base-32M | 0.7173 | modernbert-embed-base | 0.8597 | 0.1762 | 0.2049 | 3 | 0.0220 | 0.0324 | 3 | 2 |
| MLP_512 | 16 | mxbai-embed-large-v1 | 0.7494 | modernbert-embed-base | 0.8544 | 0.0662 | 0.0775 | 2 | 0.0041 | 0.1765 | 3 | 2 |

![](comparison/regret_curves_by_head.png)

![](comparison/regret_at1_vs_head.png)

![](comparison/proxy_rank_spearman_vs_head.png)

![](FCH/regret_curve.png)

![](MLP_128/regret_curve.png)

![](MLP_256/regret_curve.png)

![](MLP_512/regret_curve.png)


## 5. RQ2 — bottlenecks (timing + GPU memory)

Frozen cost splits across `inference_s` (encode) + `train_head_s` (head fit); finetune fuses inference into the joint loop so `inference_s = 0` and `train_head_s` is the end-to-end finetune cost.

![](comparison/cost_vs_head.png)

### Head FCH

![](FCH/timing_stacked.png)

![](FCH/peak_gpu_mem.png)

![](FCH/frozen_time_breakdown.png)

### Head MLP_128

![](MLP_128/timing_stacked.png)

![](MLP_128/peak_gpu_mem.png)

![](MLP_128/frozen_time_breakdown.png)

### Head MLP_256

![](MLP_256/timing_stacked.png)

![](MLP_256/peak_gpu_mem.png)

![](MLP_256/frozen_time_breakdown.png)

### Head MLP_512

![](MLP_512/timing_stacked.png)

![](MLP_512/peak_gpu_mem.png)

![](MLP_512/frozen_time_breakdown.png)


## 6. Synthesis (numbers filled in; prose for the writer)

### RQ1 — adapting model search to regression

#### Head FCH

- **regret@1:** 0.2740  <!-- prose: -->
- **normalized regret@1:** 0.3164  <!-- prose: -->
- **budget-to-zero:** 9  <!-- prose: -->
- **best frozen r²:** 0.6820 (e5-base-v2)  <!-- prose: -->
- **best finetune r²:** 0.8660 (modernbert-embed-base)  <!-- prose: -->
- **diverged models:** 3  <!-- prose: -->
- **proxy rank Spearman:** 0.4294  <!-- prose: -->

#### Head MLP_128

- **regret@1:** 0.2487  <!-- prose: -->
- **normalized regret@1:** 0.2935  <!-- prose: -->
- **budget-to-zero:** 6  <!-- prose: -->
- **best frozen r²:** 0.7095 (e5-base-v2)  <!-- prose: -->
- **best finetune r²:** 0.8472 (modernbert-embed-base)  <!-- prose: -->
- **diverged models:** 3  <!-- prose: -->
- **proxy rank Spearman:** -0.0382  <!-- prose: -->

#### Head MLP_256

- **regret@1:** 0.1762  <!-- prose: -->
- **normalized regret@1:** 0.2049  <!-- prose: -->
- **budget-to-zero:** 3  <!-- prose: -->
- **best frozen r²:** 0.7173 (potion-base-32M)  <!-- prose: -->
- **best finetune r²:** 0.8597 (modernbert-embed-base)  <!-- prose: -->
- **diverged models:** 3  <!-- prose: -->
- **proxy rank Spearman:** 0.0324  <!-- prose: -->

#### Head MLP_512

- **regret@1:** 0.0662  <!-- prose: -->
- **normalized regret@1:** 0.0775  <!-- prose: -->
- **budget-to-zero:** 2  <!-- prose: -->
- **best frozen r²:** 0.7494 (mxbai-embed-large-v1)  <!-- prose: -->
- **best finetune r²:** 0.8544 (modernbert-embed-base)  <!-- prose: -->
- **diverged models:** 3  <!-- prose: -->
- **proxy rank Spearman:** 0.1765  <!-- prose: -->

- **regret@1 vs head width:** FCH=0.2740, MLP_128=0.2487, MLP_256=0.1762, MLP_512=0.0662 (decreasing with width)  <!-- prose: -->

#### Diverged-model story (rank kept, scale broken)

| model | FCH_frozen_r2 | FCH_finetune_r2 | FCH_finetune_spearman | MLP_128_frozen_r2 | MLP_128_finetune_r2 | MLP_128_finetune_spearman | MLP_256_frozen_r2 | MLP_256_finetune_r2 | MLP_256_finetune_spearman | MLP_512_frozen_r2 | MLP_512_finetune_r2 | MLP_512_finetune_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all-mpnet-base-v2 | 0.1169 | -0.0007 | 0.4143 | 0.6266 | 0.8219 | 0.9104 | 0.6268 | 0.8432 | 0.9166 | 0.6432 | 0.8292 | 0.9226 |
| deberta-v3-base | 0.3965 | -0.3981 | 0.8727 | 0.4523 | -0.4896 | 0.9061 | 0.4880 | -0.7111 | 0.9146 | 0.4918 | -1.3162 | 0.9114 |
| deberta-v3-small | 0.6189 | -0.0296 | 0.8886 | 0.6478 | -0.4730 | 0.9078 | 0.6489 | -1.1842 | 0.8943 | 0.6249 | -0.5057 | 0.8928 |
| electra-base-discriminator | 0.6611 | 0.2457 | 0.9038 | 0.6893 | 0.1504 | 0.9155 | 0.6943 | -0.0289 | 0.8870 | 0.6976 | -0.0849 | 0.9114 |
| roberta-base | 0.6724 | 0.4782 | 0.8947 | 0.6841 | -0.1178 | 0.9219 | 0.6943 | 0.1261 | 0.8953 | 0.6984 | 0.4098 | 0.9198 |

### RQ2 — where do the bottlenecks shift?

For the best model (**modernbert-embed-base**, head MLP_512):

- **finetune/frozen train cost ratio:** 2.8x (1400s vs 508s)  <!-- prose: -->
- **finetune/frozen peak GPU mem ratio:** 9.4x (17068MB vs 1812MB)  <!-- prose: -->

- **backbone encode-cost spread (head MLP_512, frozen inference_s):** 9.0x — cheapest potion-base-8M 30s vs priciest mxbai-embed-large-v1 265s  <!-- prose: -->
