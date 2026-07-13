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

<!-- Fill in from results/activation_sweep/ once the four arms have run. -->

**dataset:** wine_reviews, 16-model pool · **strategy:** `frozen` (activation is a head knob; the
finetune ceiling is flat in head capacity, see
[warmup-head-sweep-results.md](warmup-head-sweep-results.md)) · **head:** MLP, `hidden=256` ·
**arms:** `relu` / `gelu` / `tanh` / `silu`

`mlsys analyze` does **not** apply to this sweep: it pairs one frozen pass with one finetune pass
per run-id to compute regret, and four frozen-only CSVs with distinct run-ids form no analysable
pair. The table below is computed straight from the CSVs in `results/activation_sweep/`.

| activation | mean r² | std | mean spearman | proxy top-1 |
| --- | --- | --- | --- | --- |
| relu | | | | |
| gelu | | | | |
| tanh | | | | |
| silu | | | | |

**Does the ranking move?** TODO — pairwise Spearman correlation between the arms' proxy rankings,
and whether the top-1/top-3 picks change.

## Conclusion

TODO — keep `relu` as the default, or switch.
