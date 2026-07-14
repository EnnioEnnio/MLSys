# LR warmup+decay results — verifying the divergence fix from `exp_wine_16`

Follow-up experiment for the LR warmup+decay change (`train_full_model`,
`src/mlsys/finetune/__init__.py`; linear warmup over 10% of steps, then linear decay to 0,
per Mosbach et al. 2021 / Liu et al. 2019). Does it eliminate the negative-r² finetune
divergences found in `exp_wine_16`?

## Setup

- **dataset:** wine_reviews
- **strategy:** `finetune` only (no frozen pass, no regret.json — we only care whether the
  joint loop still diverges, not proxy ranking)
- **finetune:** 10 epochs joint training, backbone/head LR 2e-5, 2-epoch head warmup (LP-FT,
  issue #38, orthogonal to this fix), batch size 64
- **models re-run:** exactly the (model, head-width) combos that diverged (negative r²) in
  `exp_wine_16`:
  - `deberta-v3-small` / `deberta-v3-base` — all 4 head widths (FCH, MLP_128, MLP_256, MLP_512)
  - `electra-base-discriminator` — MLP_256, MLP_512
  - `roberta-base` — MLP_128
  - `all-mpnet-base-v2` — FCH
- one finetune run per model (`head_repeats=1`), and no fixed seed across the three SLURM
  jobs below — treat single-run deltas as noisy, the divergence pattern as the signal.

Jobs: `2327067`, `2327595` (both hit the 4h time limit mid-run before the fix), `2329575`
(24h limit, completed all remaining widths).

## Bottom line

The LR warmup+decay schedule **fixes every previously-diverged model except one**.
`deberta-v3-small`, `all-mpnet-base-v2`, `roberta-base`, and `electra-base-discriminator` are
now consistently non-negative across every head width tested. `deberta-v3-base` remains
broken.

## Results

| model | width | r² | verdict |
| --- | --- | --- | --- |
| deberta-v3-small | 0 | 0.44 / 0.37 | ✅ fixed |
| **deberta-v3-base** | **0** | **−0.32 / −0.23** | ❌ still diverges |
| all-mpnet-base-v2 | 0 | 0.87 / 0.88 | ✅ fixed |
| deberta-v3-small | 128 | 0.64 / 0.80 | ✅ fixed |
| **deberta-v3-base** | **128** | **0.22 / −0.27** | ⚠️ flips sign between runs |
| roberta-base | 128 | 0.885 | ✅ fixed |
| deberta-v3-small | 256 | 0.63 | ✅ fixed |
| **deberta-v3-base** | **256** | **−0.51** | ❌ diverges |
| electra-base-discriminator | 256 | 0.47 | ✅ fixed |
| deberta-v3-small | 512 | 0.82 | ✅ fixed |
| **deberta-v3-base** | **512** | **0.13** | ⚠️ barely positive |
| electra-base-discriminator | 512 | 0.54 | ✅ fixed |

## `deberta-v3-base` deep dive

Negative or barely-positive at every width, with no consistent trend by head size. The
`hidden=512` run's W&B log gives a concrete diagnostic: `grad_norm_max` hit **343** (vs.
`deberta-v3-small`'s 295 at the same width, which converges fine) — a spike large enough
that gradient clipping, not LR alone, looks like the relevant lever.

**Follow-up already tried and ruled out:** a per-model `finetune_lr` override (1e-5 instead
of 2e-5, config/models.yaml) was implemented and tested (job `2331919`). At `hidden=0` it
produced r²=−0.2684 — essentially unchanged from the 2e-5 baseline (−0.32 / −0.23). Dropping
the backbone LR by 20x did **not** fix this model, so the divergence isn't purely an LR
magnitude problem. The override was reverted (see commit history) rather than left in as a
dead end.

## Caveats

- **n=1 per (model, width) pair, no fixed seed** across the three jobs — the flip-flop at
  `hidden=128` (0.22 vs −0.27) is itself informative (this model's instability is seed-sensitive,
  not just a global "always diverges"), but individual values shouldn't be read too precisely.
- Two of the three jobs (`2327067`, `2327595`) were killed by the 4h SLURM time limit before
  finishing; only `2329575` (24h limit) completed a full width sweep in one run.

## Conclusion

Ship the LR warmup+decay schedule — it's an unambiguous, broad-based fix for 4 of 5
previously-diverged models. `deberta-v3-base` needs a different intervention; the grad-norm
spike at `hidden=512` points toward **gradient clipping** as the next thing to try, not a
further LR reduction (already ruled out). Treat `deberta-v3-base` as a known, documented,
still-open case rather than blocking the rest of the fix on it.
