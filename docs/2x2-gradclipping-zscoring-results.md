# 2×2 results — grad clipping × target z-scoring (on top of warmup-2)

Follow-up to the grad-clipping (#50/#59) and target-standardisation (#32/#58) work, run as a
2×2 so the two can be read *conditionally* rather than as two separate wins over the baseline
(see `METHOD.md`, "one-factor-at-a-time results do not compose").

## Setup

- **dataset:** wine_reviews, 16-model pool, MLP_256 head, ReLU
- **fixed:** 2-epoch head warmup in every cell (settled by #38/#40 — not a factor here)
- **factors:** grad clipping (norm 1000) × target z-scoring
- one finetune run per model (`head_repeats=1`)

| cell | label | job |
| --- | --- | --- |
| warmup-2 only | `MLP_256` | 2331889 |
| + grad clipping | `MLP_256gc` | 2329658 |
| + z-scoring | `MLP_256zsc` | 2333233 |
| + both | `MLP_256_zsgc` | 2333558 |

Artifacts: `results/full_eval_2x2_gradclipping_zscoring_256/analysis/`.

## Bottom line

**Z-scoring is the only intervention that matters. Grad clipping is a no-op for where training
ends up, and adding it on top of z-scoring buys nothing.**

That settles the recipe. But the run's more important result is what z-scoring *reveals* once it
stops the divergences: **the reference collapses into a band so narrow that regret on wine is
close to meaningless**, and the frozen proxy turns out to be measuring something systematically
different from what the reference rewards. Sections 3–5 are the ones that matter for the report.

---

## 1. Divergence — the recipe question

| cell | diverged (ref r² < 0) | deberta-v3-base | deberta-v3-small | electra-base |
| --- | --- | --- | --- | --- |
| warmup-2 only | 1 | **−0.489** | 0.622 | 0.428 |
| + grad clipping | 1 | **−0.508** | 0.589 | 0.407 |
| + z-scoring | **0** | **0.890** | **0.901** | **0.906** |
| + both | **0** | 0.885 | 0.897 | 0.899 |

Z-scoring rescues the entire unstable trio — not just deberta-v3-base (the one that went
negative) but the two that were *quietly crippled* at r² 0.4–0.6 while the rest of the pool sat
at 0.9. Those two never showed up as "divergences" and would have been missed by a
divergence-count-only reading.

**Clipping treats a symptom; z-scoring treats the cause.** The clip *fired* on most of the pool,
mostly in the first joint epoch — raw targets make the initial loss huge, so the first gradients
spike (deberta-v3-base: mean norm ~200, spikes to ~30000). But clipping the spike does not fix
the thing producing it. Endpoints are unchanged (−0.489 → −0.508), and once z-scoring removes the
target-scale problem, there is no spike left to clip: the `+ both` cell is within noise of
`+ z-scoring` on every model.

Clipping does have one real effect: mid-training on the unstable family is visibly smoother
(without it, electra's val MSE swings hard between epochs). It damps missteps even where it
doesn't change the destination. That is not worth a mandatory place in the recipe.

**→ Final recipe: warmup-2 + target z-scoring. Grad clipping off, LR decay subsumed.**

Note z-scoring is not only a reference-side fix — mean **proxy** r² also rises, 0.651 → 0.735.
It is applied to both passes, so the regret comparison stays apples-to-apples.

---

## 2. The confound to fix before the final runs

Under z-scoring, **13 of 14 trainable models run to the 10-epoch finetune cap** (vs 5/14 in the
baseline cell). Standardized targets keep val MSE declining slowly, so early stopping never
trips.

Two consequences, and the second one is not in `METHOD.md` yet:

1. Part of the r² gain in the z-scored cells is "trained longer", not "stabilized". (The
   zero-divergence claim is unaffected — a diverged model does not become good by training longer.)
2. **The epoch cap is now the binding constraint on the reference.** Every model is being scored
   at the same arbitrary cutoff, while still improving. The reference is a *lower bound*, and the
   narrow band in §3 may partly be an artifact of stopping everyone mid-climb.

**Recommendation:** raise the finetune epoch cap (or loosen patience) before #67 and #69, and
report how many models still hit the cap. If the band widens with more epochs, the discriminativeness
story changes materially.

---

## 3. The band — why regret on wine is nearly meaningless

Reference r² spread, trainable models only (the two `potion` model2vec backbones are
`can_finetune=False`, so their "reference" is just their frozen score — excluding them is the
honest comparison):

| cell | min | max | spread | std |
| --- | --- | --- | --- | --- |
| warmup-2 only | −0.489 | 0.907 | 1.396 | 0.372 |
| + z-scoring | **0.839** | **0.917** | **0.078** | **0.021** |
| + both | 0.871 | 0.917 | 0.046 | 0.012 |

Before z-scoring, the reference "spread" was almost entirely divergence noise. After it, **14
trainable backbones spanning 8M–300M parameters land within 0.078 r² of each other**, and 11 of
them within ~0.03.

That is a genuine and useful finding — *on wine, backbone choice barely matters once you finetune;
take the cheapest adequate model.* But it is also a direct threat to RQ1, and the two must not be
conflated:

> If the reference is nearly constant across the pool, **any** proxy ranking — including a random
> one — incurs near-zero regret. Low regret would mean the task is too easy to fail at, not that
> the proxy ranks well.

An indicative noise estimate is available from this run for free: grad clipping is a near-no-op
under z-scoring, so the `zsc` → `zsgc` reference deltas approximate run-to-run wobble. Median
|Δ| = **0.003**, mean **0.007**, but **max 0.061** (albert-base-v2, which early-stopped at 5
epochs in one cell and 10 in the other). Against a between-model std of 0.012–0.021, the
signal-to-noise ratio is roughly **2–4×** — thin, and dominated by exactly the early-stopping
instability §2 describes.

This is not a substitute for the seed study (#67); clipping is an intervention, not a reseed. But
it is enough to say the seed study is **necessary**, not precautionary.

---

## 4. The proxy is blind to finetuning headroom

Proxy rank Spearman against the reference, per cell:

| cell | rank Spearman (all 16) | regret@1 | budget-to-zero |
| --- | --- | --- | --- |
| warmup-2 only | **−0.179** | 0.093 | 14 |
| + grad clipping | **−0.159** | 0.035 | 14 |
| + z-scoring | **0.385** | 0.000 | 1 |
| + both | **0.191** | 0.117 | 3 |

Even in the best cell the proxy barely ranks the reference. The mechanism is visible in the
**headroom** (ref r² − proxy r², z-scored cell):

| model | proxy r² | ref r² | headroom |
| --- | --- | --- | --- |
| deberta-v3-base | 0.517 | 0.890 | **+0.373** |
| modernbert-base | 0.701 | 0.917 | **+0.216** |
| all-MiniLM-L6-v2 | 0.691 | 0.888 | +0.197 |
| … | | | |
| albert-base-v2 | 0.731 | 0.839 | +0.108 |
| **potion-base-32M** | 0.783 | 0.790 | **+0.007** |
| **potion-base-8M** | 0.742 | 0.747 | **+0.005** |

Two systematic failures:

**(a) Headroom is anti-correlated with proxy score** (Spearman −0.54). The models the proxy likes
least are the ones finetuning helps most — deberta-v3-base has the *worst* frozen features in the
pool and the *largest* gain. Since the reference is nearly constant (§3), headroom is essentially
`const − proxy`, so the proxy's variance is close to pure noise with respect to the reference.

**(b) The proxy systematically over-ranks static embeddings.** The two model2vec backbones are
strong *frozen* encoders (proxy rank 3 and 5 of 16) and bottom-of-pool references (rank 15 and 16)
— they have **zero finetuning headroom** by construction. The frozen proxy measures feature
quality as-is; the reference measures feature quality *after adaptation*. Static models max out
the first and cannot move on the second.

The cost of (b) is measurable: restricting to trainable models lifts proxy↔reference Spearman from
**0.385 → 0.596**. A meaningful chunk of the proxy's apparent ranking failure is a single
architectural blind spot, and it is fixable — either exclude non-finetunable candidates from the
regret pool, or report both numbers.

---

## 5. regret@1 is knife-edge — do not report it from one seed

The cleanest illustration in the whole run:

- `+ z-scoring`: proxy top-1 = **modernbert-embed-base** (0.7910) → **regret@1 = 0.000**
- `+ both`: proxy top-1 = **potion-base-32M** (0.7918) → **regret@1 = 0.117**

The two cells differ by grad clipping, which §1 established is a **no-op on the reference**. The
reference rankings are effectively identical. What changed is a **0.006 wobble in proxy r²** that
flipped first place from modernbert-embed-base to a model2vec backbone with no finetuning headroom
— and regret@1 swung from perfect to the worst value in the experiment.

Regret@1 on wine is decided by the top of a crowded, near-flat proxy band, and the gaps up there
(0.7910 / 0.7896 / 0.7859 / 0.7833) are smaller than the noise. **Any single-seed regret@1 on this
dataset is a coin flip.** Report regret with cross-seed error bars, alongside `regret_auc` and
budget-to-zero, and always against the random-ranking null (#68).

---

## 6. RQ2 — unaffected, and clean

Nothing here disturbs the bottleneck picture; it holds across all four cells.

- **Feature extraction dominates proxy cost:** `inference_s` is 53–63% of the proxy's time, and
  encode + data prep is **67–76%**. Consistent with Alsatian's classification finding — it carries
  over to regression unchanged.
- **Finetuning is not cheap:** for modernbert-base, finetune vs frozen is **21.4×** wall-clock
  (4918s vs 229s) and **9.8×** peak GPU memory (17.7GB vs 1.8GB).
- **Backbone encode cost spans 8.4×** across the pool (potion-base-8M 32s → mxbai-embed-large-v1
  267s).

That cost gap is the entire reason regret matters — and §3–§5 are the reason we cannot yet say
whether the proxy earns it.

---

## What this changes

1. **Recipe settled:** warmup-2 + z-scoring, MLP_256, ReLU. Clipping and LR decay are reported as
   *tested and found redundant* — a negative result, stated plainly.
2. **Raise the finetune epoch cap** before #67/#69. The reference is currently undertrained and
   the cap is binding on 13/14 models (§2).
3. **The seed study (#67) is load-bearing, not precautionary.** Signal-to-noise on the reference
   band is ~2–4× (§3).
4. **The random-ranking null (#68) is non-optional.** With a 0.078-wide reference band, low regret
   proves nothing on its own (§3).
5. **Decide how model2vec enters the regret pool** (§4b). It is currently costing 0.2 Spearman and
   it caused the single worst regret@1 in the experiment. Either exclude non-finetunable candidates
   or report trainable-only regret alongside the full-pool number.
6. **`usa_real_estate` (#69) now has a sharp job:** find a task where the pool actually separates.
   Wine does not.
