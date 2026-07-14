# Head activation (`--activation`)

The MLP head (`--hidden WIDTH`) is `in_dim -> WIDTH -> ACT -> 1`; `--activation` picks `ACT` from
`relu` (default), `gelu`, `tanh`, `silu` (`mlsys.head.ACTIVATIONS`). The **linear probe**
(`--hidden 0`, or the flag omitted) has no nonlinearity, so the value is inert there. Cluster
default is `relu` (`ACTIVATION` in `slurm/submit.sh`); the chosen name is recorded as an
`activation` column on every result row (frozen *and* finetune) and in the W&B config, but does
**not** appear in the run name.

```bash
python -m mlsys search --dataset wine_reviews --hidden 256 --activation gelu
```

## Why

`relu` was never a considered choice, just what `FCHead` happened to build. The question that
matters for the regret story (REGRET.md) isn't which activation gives the best r² — expected to be
a wash on a 2-layer head — but whether it changes **which backbone the frozen proxy ranks first**.
If the ranking doesn't move, `relu` stays the default and one degree of freedom is closed off.

## Sweep

**dataset:** wine_reviews, 16-model pool · **strategy:** `frozen` (activation is a head knob; the
finetune ceiling is flat in head capacity, see
[warmup-head-sweep-results.md](warmup-head-sweep-results.md)) · **head:** MLP, `hidden=256` ·
**arms:** `relu` / `gelu` / `tanh` / `silu`

`mlsys analyze` does **not** apply to this sweep: it pairs one frozen pass with one finetune pass
per run-id to compute regret, and four frozen-only CSVs with distinct run-ids form no analysable
pair. The table below is computed straight from the CSVs in `results/activation_sweep/`.

| activation | mean r² | std | mean spearman | n_negative | mean epochs_run | proxy top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| gelu | 0.6853 | 0.0799 | 0.8230 | 0 | 18.25 | roberta-base |
| relu | 0.6547 | 0.0604 | 0.8043 | 0 | 16.25 | mxbai-embed-large-v1 |
| silu | 0.6717 | 0.0900 | 0.8152 | 0 | 14.12 | e5-base-v2 |
| tanh | 0.7159 | 0.0613 | 0.8396 | 0 | 23.00 | modernbert-embed-base |

**Does the ranking move?** pairwise Spearman rho over per-model r2 ranks ranges [0.374, 0.691]; proxy top-1 **changes** across activations: gelu=roberta-base, relu=mxbai-embed-large-v1, silu=e5-base-v2, tanh=modernbert-embed-base.

## Conclusion

The ranking does move (proxy top-1 differs for all four arms, pairwise Spearman only
0.374–0.691), so the "if it doesn't move, keep `relu`" bar from [Why](#why) isn't met. `tanh` also
leads on both mean r² (0.7159) and mean spearman (0.8396) in this sweep, and `n_negative` is 0 for
every arm so it isn't a tie-breaker here. **Recommendation:** switch the cluster default away from
`relu` to `tanh` where feasible, but confirm first on a second dataset — note `tanh` also runs the
most epochs before early-stop on average (23.0 vs. `silu`'s 14.1), so the r²/spearman gain comes at
extra `train_head_s` per candidate, not for free.
