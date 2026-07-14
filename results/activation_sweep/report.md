# Activation sweep report (results/activation_sweep)

## Summary

| activation | mean_r2 | std_r2 | mean_spearman | proxy_top1 |
| --- | --- | --- | --- | --- |
| relu | 0.6547 | 0.0604 | 0.8043 | mxbai-embed-large-v1 |
| silu | 0.6717 | 0.0900 | 0.8152 | e5-base-v2 |
| tanh | 0.7159 | 0.0613 | 0.8396 | modernbert-embed-base |

## Per-model ranking (1 = best r2 within that activation)

| model | relu_r2 | relu_rank | silu_r2 | silu_rank | tanh_r2 | tanh_rank | mean_rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mxbai-embed-large-v1 | 0.7325 | 1 | 0.7448 | 2 | 0.7653 | 3 | 2.0000 |
| e5-base-v2 | 0.6972 | 3 | 0.7632 | 1 | 0.7506 | 7 | 3.6667 |
| roberta-base | 0.6793 | 7 | 0.7303 | 3 | 0.7585 | 4 | 4.6667 |
| modernbert-embed-base | 0.6585 | 10 | 0.7268 | 4 | 0.7724 | 1 | 5.0000 |
| potion-base-32M | 0.7036 | 2 | 0.6522 | 13 | 0.7529 | 5 | 6.6667 |
| bge-base-en-v1.5 | 0.6328 | 14 | 0.7252 | 5 | 0.7666 | 2 | 7.0000 |
| electra-base-discriminator | 0.6858 | 5 | 0.6564 | 12 | 0.7506 | 6 | 7.6667 |
| distilbert-base-uncased | 0.6751 | 8 | 0.7137 | 8 | 0.7432 | 8 | 8.0000 |
| albert-base-v2 | 0.6889 | 4 | 0.7148 | 7 | 0.7009 | 13 | 8.0000 |
| deberta-v3-small | 0.6431 | 11 | 0.7117 | 9 | 0.7258 | 9 | 9.6667 |
| sentence-t5-base | 0.6721 | 9 | 0.6667 | 10 | 0.7115 | 10 | 9.6667 |
| potion-base-8M | 0.6815 | 6 | 0.6029 | 14 | 0.7015 | 12 | 10.6667 |
| modernbert-base | 0.6399 | 12 | 0.7230 | 6 | 0.6772 | 14 | 10.6667 |
| all-mpnet-base-v2 | 0.6334 | 13 | 0.6646 | 11 | 0.7062 | 11 | 11.6667 |
| all-MiniLM-L6-v2 | 0.5797 | 15 | 0.5376 | 15 | 0.6403 | 15 | 15.0000 |
| deberta-v3-base | 0.4720 | 16 | 0.4126 | 16 | 0.5318 | 16 | 16.0000 |

## Pairwise Spearman (model r2 rank agreement)

|  | relu | silu | tanh |
| --- | --- | --- | --- |
| relu | 1.000 | 0.374 | 0.421 |
| silu | 0.374 | 1.000 | 0.618 |
| tanh | 0.421 | 0.618 | 1.000 |

pairwise Spearman rho over per-model r2 ranks ranges [0.374, 0.618]; proxy top-1 **changes** across activations: relu=mxbai-embed-large-v1, silu=e5-base-v2, tanh=modernbert-embed-base.

