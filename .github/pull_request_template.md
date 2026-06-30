## Summary

<!-- One or two sentences: what does this PR do and why? -->

## Changes

<!-- Bullet the notable changes. -->
-

## Testing

<!-- How was this verified? Commands run, results. -->
-

## Checklist

- [ ] `make check` passes (lint + typecheck + tests)
- [ ] Tests added/updated for the change
- [ ] Docs updated (`README.md` / `analysis.md` / `CLAUDE.md`) if behavior or conventions changed
- [ ] No heavy imports at module top-level (`torch`, `transformers`, `datasets`, ... stay lazy)
- [ ] Timing field names unchanged (`prepare_model_s`, `prepare_data_s`, `inference_s`, `train_head_s`, `eval_s`)
