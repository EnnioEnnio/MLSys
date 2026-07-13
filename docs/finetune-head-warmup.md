# Head warmup for finetune (LP-FT)

## What it does

Before the `finetune` / `full_eval` joint loop unfreezes the backbone, `train_full_model`
can first train **only the head** for a few epochs against the *frozen* backbone, then
unfreeze and run the existing joint loop from that warmed-up head. This is **LP-FT**
(linear-probe-then-fine-tune, Kumar et al. 2022, *"Fine-Tuning can Distort Pretrained
Features and Underperform Out-of-Distribution"*; the two-phase idea traces back to
ULMFiT's gradual unfreezing).

Enable it with `--warmup-epochs N` (default `0` = off, straight to joint training):

```bash
python -m mlsys search --dataset wine_reviews --strategy finetune --warmup-epochs 2
```

Cluster runs use `2` (`WARMUP_EPOCHS` in `slurm/submit.sh`).

## Why

With a **freshly random-initialised** head, the first joint steps backprop large,
feature-distorting gradients from the untrained head into the pretrained backbone,
wrecking the very features the backbone brings. In `first_fulleval_wine_16_outdated`, 12/64 finetunes
diverged to negative r² (Spearman stayed intact — a ranking-preserving scale/offset blow-up
consistent with head-driven distortion). Warming the head up first means the backbone only
ever sees gradients from an already-sensible head.

This is the #31 half of the fix; the complementary LR / gradient-clipping /
target-standardization work is tracked in #32.

## Timing semantics

The warmup runs inside the same `timer.section("train_head_s")` that wraps the whole
`train_full_model` call in `search/runner.py:finetune_candidate`. So the warmup cost —
including its frozen embedding pass — lands in **`train_head_s`**, and **`inference_s`
stays `0`** for finetune rows (inference is fused into the joint loop, per CLAUDE.md's
timing contract). No timing fields change. Warmup epochs are **not** merged into the
returned train/val curves and don't get an `epoch_callback`, so W&B joint-loop epoch
numbering is unchanged.

Non-trainable backbones (`can_finetune=False`, e.g. model2vec) fall back to the frozen
`score_candidate` and never reach `train_full_model`, so warmup does not apply to them.

## Results

See [warmup-results.md](warmup-results.md) for the with/without-warmup comparison
(wine_reviews, MLP-256 head, 10-epoch finetune). Short version: 2-epoch warmup cut finetune
divergences from **3→1** and lifted mean finetune r² from **0.55→0.74**, with no loss at the
top of the pool. It's a floor-raiser, not a ceiling-raiser; the residual `deberta-v3-base`
divergence is the LR / clipping / target-standardization half of the fix (#32).
