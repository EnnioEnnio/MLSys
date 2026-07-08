# Head-width sweep under warmup finetuning — FCH / MLP 128 / 256 / 512

Full `full_eval` sweep over all four head configs, with every finetune run using the
[head-warmup (LP-FT)](finetune-warmup.md) schedule (2 epochs head-only warmup before the
10-epoch joint loop). Sister experiment to
[warmup-results.md](warmup-results.md), which isolated *warmup vs. no-warmup* at a single
width; this one holds warmup fixed and asks **what head width buys you**.
Tracks [issue #39](https://github.com/EnnioEnnio/MLSys/issues/39).

## Setup

- **dataset:** wine_reviews, 16-model pool
- **heads:** FCH (linear), MLP_128, MLP_256, MLP_512 (array jobs `2320368 / 2320370 / 2320372 / 2320374`)
- **finetune:** 2 epochs head-only warmup → 10 epochs joint training
- one finetune run per model (`head_repeats=1`), so per-model r² deltas are noisy; the
  robust signals are the **aggregate distribution** and the **divergence count**.

Artifacts: `results/full_eval_10_epoch_warmup_2/analysis/`.

## Bottom line

Head width is a **proxy-side lever, not a reference-side one.** Widening the head barely
moves the fine-tuned (reference) ceiling — the strong encoders converge to r² ≈ 0.90
regardless — but it dramatically changes the **frozen proxy**: the linear head (FCH) is an
unreliable, slow proxy, and *any* MLP collapses the proxy spread and removes the negative-r²
blow-ups. **MLP_128 captures most of the benefit; 256/512 add little.**

## Reference (finetune) quality — the ceiling is flat

| head | mean ref r² | best model | best ref r² | n_diverged |
| --- | --- | --- | --- | --- |
| FCH | 0.586 | modernbert-base | 0.911 | 2 |
| MLP_128 | 0.706 | modernbert-base | 0.905 | 1 |
| MLP_256 | 0.736 | modernbert-embed-base | 0.908 | 1 |
| MLP_512 | 0.737 | modernbert-embed-base | 0.901 | 1 |

The **best** achievable r² is ~0.90 at every width — once you fine-tune the backbone, head
capacity is washed out for the models that converge. The rising *mean* is not a real
head-capacity effect: it is driven by (a) the two `model2vec` models (`potion-*`), which are
`finetune_skipped` so their "reference" is just the frozen linear head — terrible at FCH,
fine with an MLP — and (b) the FCH divergences below.

## The proxy is where head width matters

| head | mean proxy r² | std | min | n_negative | mean proxy epochs | n at 30-epoch cap |
| --- | --- | --- | --- | --- | --- | --- |
| FCH | 0.461 | 0.305 | −0.248 | 2 | 29.1 | **12 / 16** |
| MLP_128 | 0.646 | 0.051 | 0.490 | 0 | 15.6 | 0 |
| MLP_256 | 0.657 | 0.057 | 0.478 | 0 | 14.7 | 0 |
| MLP_512 | 0.669 | 0.069 | 0.430 | 0 | 13.8 | 0 |

The linear proxy is **doubly bad**: it produces negative r² for `all-MiniLM-L6-v2` (−0.25)
and `potion-base-8M` (−0.11), has 6× the spread of any MLP, **and** it is the slowest to
train — 12 of 16 models hit the 30-epoch early-stopping cap without converging. Adding a
single hidden layer (MLP_128) collapses the spread to σ ≈ 0.05, eliminates every negative,
and converges in ~15 epochs. Going wider (256 → 512) is marginal.

The models that gain the most from a wider head are exactly the ones a linear probe cannot
read off frozen embeddings:

| model | FCH proxy r² | MLP_512 proxy r² | gain |
| --- | --- | --- | --- |
| all-MiniLM-L6-v2 | −0.248 | 0.611 | +0.860 |
| potion-base-8M | −0.113 | 0.710 | +0.824 |
| potion-base-32M | 0.087 | 0.733 | +0.646 |
| all-mpnet-base-v2 | 0.116 | 0.629 | +0.513 |

Everything already ≥ 0.62 at FCH gains < 0.10 — the widening only rescues the frozen-hostile
encoders.

## Ranking & regret — read with care

| head | regret@1 | norm. regret@1 | budget_to_zero | rank ρ (proxy vs ref) |
| --- | --- | --- | --- | --- |
| FCH | 0.081 | 0.088 | 8 | +0.12 |
| MLP_128 | 0.045 | 0.049 | 13 | −0.18 |
| MLP_256 | 0.043 | 0.047 | 7 | −0.03 |
| MLP_512 | 0.176 | 0.195 | 4 | −0.11 |

**The frozen proxy is a weak ranker for fine-tuned quality on this task at every width**
(Spearman ρ ≈ 0 throughout). The reason is ceiling compression: most models fine-tune into a
narrow 0.85–0.91 band, so the reference ranking within that band is dominated by n=1 noise,
and no proxy — however well-calibrated — can recover it. The low regret@1 at MLP_128/256 and
the jump at MLP_512 are **not** a monotonic "wider is better/worse" trend; with a single
finetune run per model they are largely a function of *which* model happened to land on top.
Don't over-read `budget_to_zero` or the regret ordering across heads.

## Divergence — head capacity does not fix instability

`deberta-v3-base` fine-tunes to **negative r² at every head width** (−0.25 / −0.56 / −0.24 /
−0.00), even with warmup applied. Head capacity is orthogonal to the blow-up: widening the
head cannot stabilise a backbone whose features the joint loop distorts. `deberta-v3-small`
diverges only at FCH (−0.07) and recovers with an MLP. As elsewhere, the diverged runs keep a
healthy Spearman (0.86–0.94) — a ranking-preserving scale blow-up, the LP-FT failure mode.
The residual `deberta-v3-base` case is the LR / gradient-clipping / target-standardization
half of the fix ([issue #32](https://github.com/EnnioEnnio/MLSys/issues/32)), not something
more head capacity or warmup will solve.

## Cost (RQ2)

- **Finetuning costs 10–30× the frozen proxy** per model in wall-clock (`ref_total_s` ~1k–8k s
  vs `proxy_total_s` ~80–300 s) and **5–10× the peak GPU memory** (up to ~28 GB for
  `mxbai-embed-large-v1` vs ~2.8 GB frozen). This is the core regret trade-off: the proxy is
  cheap, but here it barely ranks the expensive signal.
- The **FCH proxy is the worst of both worlds** — lowest quality *and* slowest (epoch-cap),
  so there is no cost argument for the linear head either. The proxy sweet spot is
  **MLP_128 / MLP_256**.
- `model2vec` (`potion-*`) can't finetune, so its reference cost equals its (tiny) proxy cost.

## Conclusion

- **Use an MLP head for the frozen proxy — `MLP_128` or `MLP_256`.** It removes the linear
  head's negative-r² blow-ups, cuts proxy variance ~6×, and converges twice as fast. Wider
  than 256 is not worth it.
- **Head width does not move the fine-tuned ceiling** (~0.90 everywhere); it only rescues
  frozen-hostile encoders in the *proxy* pass.
- **The frozen proxy ranks fine-tuned quality poorly here** (ρ ≈ 0 at all widths) because the
  reference r² is compressed at the top — a regret/RQ1 caveat, not a head-width knob.
- **Warmup holds divergence to the single stubborn `deberta-v3-base`**, which needs the
  optimizer-side fix ([issue #32](https://github.com/EnnioEnnio/MLSys/issues/32)).
