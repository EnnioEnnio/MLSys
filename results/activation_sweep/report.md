# Activation sweep report (results/activation_sweep)

## Summary

| activation | mean_r2 | std_r2 | mean_spearman | proxy_top1 |
| --- | --- | --- | --- | --- |
| gelu | 0.6853 | 0.0799 | 0.8230 | roberta-base |
| relu | 0.6547 | 0.0604 | 0.8043 | mxbai-embed-large-v1 |
| silu | 0.6717 | 0.0900 | 0.8152 | e5-base-v2 |
| tanh | 0.7159 | 0.0613 | 0.8396 | modernbert-embed-base |

## Per-model ranking (1 = best r2 within that activation)

| model | gelu_r2 | gelu_rank | relu_r2 | relu_rank | silu_r2 | silu_rank | tanh_r2 | tanh_rank | mean_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mxbai-embed-large-v1 | 0.7218 | 5 | 0.7325 | 1 | 0.7448 | 2 | 0.7653 | 3 | 2.7500 |
| e5-base-v2 | 0.7379 | 3 | 0.6972 | 3 | 0.7632 | 1 | 0.7506 | 7 | 3.5000 |
| roberta-base | 0.7559 | 1 | 0.6793 | 7 | 0.7303 | 3 | 0.7585 | 4 | 3.7500 |
| potion-base-32M | 0.7422 | 2 | 0.7036 | 2 | 0.6522 | 13 | 0.7529 | 5 | 5.5000 |
| electra-base-discriminator | 0.7365 | 4 | 0.6858 | 5 | 0.6564 | 12 | 0.7506 | 6 | 6.7500 |
| modernbert-embed-base | 0.6856 | 12 | 0.6585 | 10 | 0.7268 | 4 | 0.7724 | 1 | 6.7500 |
| bge-base-en-v1.5 | 0.7020 | 8 | 0.6328 | 14 | 0.7252 | 5 | 0.7666 | 2 | 7.2500 |
| albert-base-v2 | 0.7092 | 6 | 0.6889 | 4 | 0.7148 | 7 | 0.7009 | 13 | 7.5000 |
| distilbert-base-uncased | 0.6949 | 10 | 0.6751 | 8 | 0.7137 | 8 | 0.7432 | 8 | 8.5000 |
| deberta-v3-small | 0.7043 | 7 | 0.6431 | 11 | 0.7117 | 9 | 0.7258 | 9 | 9.0000 |
| modernbert-base | 0.6981 | 9 | 0.6399 | 12 | 0.7230 | 6 | 0.6772 | 14 | 10.2500 |
| sentence-t5-base | 0.6786 | 13 | 0.6721 | 9 | 0.6667 | 10 | 0.7115 | 10 | 10.5000 |
| all-mpnet-base-v2 | 0.6904 | 11 | 0.6334 | 13 | 0.6646 | 11 | 0.7062 | 11 | 11.5000 |
| potion-base-8M | 0.6652 | 14 | 0.6815 | 6 | 0.6029 | 14 | 0.7015 | 12 | 11.5000 |
| all-MiniLM-L6-v2 | 0.6319 | 15 | 0.5797 | 15 | 0.5376 | 15 | 0.6403 | 15 | 15.0000 |
| deberta-v3-base | 0.4094 | 16 | 0.4720 | 16 | 0.4126 | 16 | 0.5318 | 16 | 16.0000 |

## Pairwise Spearman (model r2 rank agreement)

|  | gelu | relu | silu | tanh |
| --- | --- | --- | --- | --- |
| gelu | 1.000 | 0.691 | 0.556 | 0.568 |
| relu | 0.691 | 1.000 | 0.374 | 0.421 |
| silu | 0.556 | 0.374 | 1.000 | 0.618 |
| tanh | 0.568 | 0.421 | 0.618 | 1.000 |

pairwise Spearman rho over per-model r2 ranks ranges [0.374, 0.691]; proxy top-1 **changes** across activations: gelu=roberta-base, relu=mxbai-embed-large-v1, silu=e5-base-v2, tanh=modernbert-embed-base.

