# Full-eval results — tanh activation

Follow-up to the frozen-only activation sweep ([head-activation.md](head-activation.md)), which found
`tanh` leads on mean r² (0.7159) and mean spearman (0.8396) at `hidden=256`, and recommended
switching the cluster default away from `relu` — pending confirmation via a real frozen/finetune
pair (regret), since the sweep itself couldn't compute one (four frozen-only CSVs, no finetune
pass). This run closes that gap: `full_eval` (frozen + finetune) with `--activation tanh`, across
four head widths.

## Setup

- **dataset:** wine_reviews, 16-model pool
- **activation:** `tanh` (head only; inert on the linear probe)
- **strategy:** `full_eval` (frozen proxy + finetune reference, both passes)
- **heads:** `FCH` (linear), `MLP_128`, `MLP_256`, `MLP_512`

| head | job |
| --- | --- |
| FCH | 2375369 |
| MLP_128 | 2444207 |
| MLP_256 | 2375093 |
| MLP_512 | 2375205 |

Artifacts: `results/full_eval_tanh/analysis/` (`mlsys analyze results/full_eval_tanh`).

## Bottom line

**Regret is ~0 at every head width, and `modernbert-embed-base` — tanh's proxy top-1 from the
original sweep — is also the finetune (ground-truth) top-1 in 3 of 4 configs.** That's the
strongest possible outcome for a proxy: not just low regret, but the actual top pick surviving
fine-tuning as the actual best model.

| head | best_proxy_model | best_proxy_r2 | best_ref_model | best_ref_r2 | regret@1 | normalized regret@1 | budget-to-zero | rank Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCH | modernbert-embed-base | 0.7206 | modernbert-embed-base | 0.9168 | 0.0000 | 0.0000 | 1 | 0.5471 |
| MLP_128 | modernbert-embed-base | 0.7805 | modernbert-embed-base | 0.9178 | 0.0000 | 0.0000 | 1 | 0.5676 |
| MLP_256 | modernbert-embed-base | 0.7858 | modernbert-embed-base | 0.9171 | 0.0000 | 0.0000 | 1 | 0.6618 |
| MLP_512 | e5-base-v2 | 0.7941 | modernbert-embed-base | 0.9158 | 0.0097 | 0.0106 | 2 | 0.5941 |

Only at `MLP_512` does the proxy top pick flip — to `e5-base-v2`, which is itself the reference
runner-up (0.9060 vs 0.9158) — so the miss costs almost nothing (regret@1 0.0097) and is fixed at
budget 2. No model diverged (finetune r² < 0) in any head; the two model2vec backbones
(`potion-base-8M`, `potion-base-32M`) are `can_finetune=False` and correctly show up as
`ref_skipped` (2/16) rather than diverged.

## Reproducibility check against the frozen-only sweep

The `MLP_256` arm here is the same cell the activation sweep already ran (dataset, pool, head
width, activation), just as part of a frozen+finetune pair instead of frozen-only:

| source | mean_proxy_r2 | std_proxy_r2 | proxy top-1 |
| --- | --- | --- | --- |
| [head-activation.md](head-activation.md) sweep (tanh, hidden=256) | 0.7159 | 0.0613 | modernbert-embed-base |
| this run, `MLP_256` | 0.7059 | 0.0784 | modernbert-embed-base |

Close enough to be run-to-run noise, and the proxy top-1 model matches exactly — the sweep's
finding reproduces under a fresh run.

## Ranking stability across head width

Proxy rank Spearman between head widths stays high (min off-diagonal 0.75, FCH vs MLP_256):

| head | FCH | MLP_128 | MLP_256 | MLP_512 |
| --- | --- | --- | --- | --- |
| FCH | 1.0000 | 0.8500 | 0.7529 | 0.8353 |
| MLP_128 | 0.8500 | 1.0000 | 0.9471 | 0.9382 |
| MLP_256 | 0.7529 | 0.9471 | 1.0000 | 0.9529 |
| MLP_512 | 0.8353 | 0.9382 | 0.9529 | 1.0000 |

Proxy-vs-reference rank Spearman itself is highest at `MLP_256` (0.6618) and lowest at `FCH`
(0.5471) — the linear probe is the noisiest ranker of the four, though it still lands at
regret@1 = 0.0000 here because its top pick happens to also win the reference.

## RQ2 — cost, unaffected by activation

Consistent with every other run on this pool: inference (backbone encode) dominates proxy cost
(57–69% of proxy wall-clock, 72–83% once head training is folded in as "feature extraction"), and
finetuning is far more expensive than the frozen pass — for the best model
(`modernbert-embed-base`, `MLP_512`): **20.8x** wall-clock (4790s vs 231s) and **9.8x** peak GPU
memory (17726MB vs 1812MB) versus frozen. `tanh` doesn't change this picture; it's a head-only
knob and RQ2's bottleneck story is backbone-bound regardless of activation.

Early-stopping epochs stay close to the `relu` baseline pattern (mean proxy epochs 14–16 across
heads, reference typically hits the 10-epoch default before the 30-epoch cap for all but the two
model2vec backbones, which run long — 30 epochs — at `MLP_128`/`MLP_512`).

## Conclusion

This is the confirmation [head-activation.md](head-activation.md) asked for: a real frozen/finetune
pair, not just a frozen-only sweep. Regret stays at or near zero across all four head widths, the
proxy top-1 model matches the frozen-only sweep's finding, and the one non-zero regret cell
(`MLP_512`) is cheap. **Recommendation stands: `tanh` is at least as good as `relu` as the cluster
default on wine_reviews**, and this run adds the piece the original sweep couldn't produce — actual
regret, not just proxy-side r²/spearman. The original sweep's caveat (confirm on a second dataset)
is still open.
