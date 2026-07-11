# Head activation (`--activation`)

## What it does

The MLP head (`--hidden WIDTH`) is `in_dim -> WIDTH -> ACT -> 1`. `--activation` picks `ACT`
from `relu` (default), `gelu`, `tanh`, `silu` — the four entries of `mlsys.head.ACTIVATIONS`.
The **linear probe** (`--hidden 0`, or the flag omitted) has no nonlinearity at all, so the
value is inert there.

```bash
python -m mlsys search --dataset wine_reviews --hidden 256 --activation gelu
```

Cluster runs use `relu` (`ACTIVATION` in `slurm/submit.sh`). The chosen name is written to
every result row as an `activation` column (frozen *and* finetune passes, so the CSV/W&B
schemas stay identical) and to the W&B run config; it does **not** appear in the run name.

## Why

`relu` was never a considered choice — it was the only thing `FCHead` could build. For a
2-layer regression head on frozen sentence embeddings the nonlinearity is one of the few
knobs that could change *which* backbone the proxy ranks first, and a proxy ranking that is
an artifact of an arbitrary default is a threat to the regret story (REGRET.md). So the
question is not "which activation gives the best r²" — the differences there are expected to
be small — but **"does the proxy's ranking move at all?"** If it doesn't, `relu` is a safe
default and one degree of freedom is closed off.

## Sweep

<!-- Fill in from results/activation_sweep/ once the four arms have run. -->

**Note:** `mlsys analyze` does **not** apply to this sweep. It pairs one frozen pass with one
finetune pass per run-id to compute regret and has no activation axis; four frozen-only CSVs
with distinct run-ids form no analysable pair. The tables below are computed straight from
the CSVs.

### Setup

- **dataset:** wine_reviews, 16-model pool
- **strategy:** `frozen` (proxy pass only — the activation is a head knob, and the finetune
  ceiling is known to be flat in head capacity, see [warmup-head-sweep-results.md](warmup-head-sweep-results.md))
- **head:** MLP, `hidden=256`, `head_repeats=1`, 30 epochs w/ early stop
- **arms:** `relu` / `gelu` / `tanh` / `silu` (array jobs `TODO / TODO / TODO / TODO`)

Artifacts: `results/activation_sweep/`.

### Bottom line

TODO — one paragraph: does the activation move r² / spearman, and does it move the ranking?

### Proxy quality per arm

| activation | mean r² | std | min | n_negative | mean epochs_run | mean spearman |
| --- | --- | --- | --- | --- | --- | --- |
| relu | | | | | | |
| gelu | | | | | | |
| tanh | | | | | | |
| silu | | | | | | |

### Does the ranking move?

Pairwise rank correlation between the arms' **proxy rankings** (the thing the search
actually consumes):

| | relu | gelu | tanh | silu |
| --- | --- | --- | --- | --- |
| relu | 1.00 | | | |
| gelu | | 1.00 | | |
| tanh | | | 1.00 | |
| silu | | | | 1.00 |

Top-1 / top-3 picks per arm:

| activation | top-1 model | top-3 models |
| --- | --- | --- |
| relu | | |
| gelu | | |
| tanh | | |
| silu | | |

### Cost

TODO — `train_head_s` per arm (all four share the same embedding pass, so only the head
training cost can differ; expect it to be noise).

## Conclusion

TODO — keep `relu` as the default, or switch.
