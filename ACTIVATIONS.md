# Configurable MLP-head activation + frozen-proxy sweep

## Context

`FCHead` hard-codes `nn.ReLU()` as the only activation on the MLP path
([head/__init__.py:33](src/mlsys/head/__init__.py#L33)). ReLU is a reasonable default, but for a
small 2-layer regression head, GELU / Tanh / SiLU may fit better — or, more importantly for this
project, may **rank candidates differently**, which would make the frozen proxy's ranking an
artifact of an arbitrary choice. There is currently no way to select another activation from
`HeadTrainConfig` or the CLI.

The goal: make the activation a first-class knob, sweep `{relu, gelu, tanh, silu}` at `hidden=256`
over the whole pool under the **frozen** strategy, check whether r²/spearman and the proxy ranking
move, and decide whether to change the default. The linear-probe path (`hidden` None/≤0) has no
activation and stays untouched.

## Decisions taken (with the user)

- **Activation set:** `relu` (default), `gelu`, `tanh`, `silu`. No `leaky_relu`.
- **Run naming:** the activation is recorded as an **extras column** on every result row. The
  `_wandb_run_name` grammar and the `analysis/loader.py` filename parser are **not** touched — arms
  are distinguished by run-id, and sweep CSVs get a hand-inserted arm token, exactly like
  `results/warmup_sweep/` did (`..._MLP_256_gelu_frozen.csv`).
- **Sweep execution:** the user runs the arms on the cluster. I implement the flag, wire SLURM, and
  leave a `docs/head-activation.md` skeleton to fill in from the resulting CSVs.

## Implementation

### 1. `src/mlsys/head/__init__.py` — the activation itself

- Module-level lookup table next to `FCHead` (torch is already a top-level import in *this* module,
  so no lazy-import gymnastics needed here — see the docstring at line 1):

  ```python
  ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
      "relu": nn.ReLU,
      "gelu": nn.GELU,
      "tanh": nn.Tanh,
      "silu": nn.SiLU,
  }
  ```

- `FCHead.__init__(self, in_dim, hidden=None, out_dim=1, activation="relu")` — build
  `ACTIVATIONS[activation]()` into the `nn.Sequential`; raise `ValueError` with the sorted known
  keys on an unknown name (the parser's `choices=` catches CLI typos, but `FCHead` is also
  constructed directly in tests and in `runner.py`). Leave the `hidden is None or hidden <= 0`
  linear branch alone — it takes no activation.
- `HeadTrainConfig`: add `activation: str = "relu"` after `hidden`. Add a `__post_init__` that
  validates against `ACTIVATIONS`, mirroring `FinetuneConfig.__post_init__`
  ([finetune/__init__.py:28-50](src/mlsys/finetune/__init__.py#L28-L50)).
- `train_head` line 106: `FCHead(..., hidden=config.hidden, activation=config.activation)`. The
  docstring at lines 88-90 already says a caller-supplied `head` makes `config.hidden` ignored —
  extend that sentence to cover `config.activation`.

`finetune/__init__.py:109` builds the warmup config via `dataclasses.replace`, so the new field
rides along with no change there.

### 2. `src/mlsys/cli/main.py` — the flag

- New `search` arg, modelled on `--strategy`'s `choices=` and `--hidden`'s help style
  ([cli/main.py:83-94](src/mlsys/cli/main.py#L83-L94)):

  ```python
  search.add_argument(
      "--activation",
      default=HeadTrainConfig.activation,
      choices=sorted(ACTIVATIONS),
      help="activation between the two layers of the MLP head "
           "(ignored for the linear probe, i.e. --hidden 0/omitted; default: %(default)s)",
  )
  ```
  Default comes from the dataclass attribute, never a literal — that's the house convention.
- [cli/main.py:401](src/mlsys/cli/main.py#L401): `HeadTrainConfig(..., activation=args.activation)`.
- W&B config dict (~line 421-434): add `"activation": head_cfg.activation` next to `"hidden"`.

### 3. `src/mlsys/search/runner.py` — record it

Add `"activation": head_config.activation` to the `extras` dict in **both**
[`score_candidate`](src/mlsys/search/runner.py#L180-L185) and
[`finetune_candidate`](src/mlsys/search/runner.py#L296-L315), so the frozen and finetune passes keep
identical CSV/W&B column schemas. Extras flow through `_result_row` and `consolidate.flatten_row`
untouched, so nothing else needs a change to get the column into the CSVs.

While there: the frozen path inlines the linear/mlp expression that `_head_type`
([runner.py:189-191](src/mlsys/search/runner.py#L189-L191)) already encapsulates — call the helper
in both spots.

### 4. `slurm/` — the mandated wiring (CLAUDE.md's last bullet)

Follow the `GRAD_CLIPPING` pattern end-to-end:

- `slurm/submit.sh`: `export ACTIVATION=${ACTIVATION:-relu}` in the search-parameters block; add to
  the banner echo.
- `slurm/array_search.slurm`: `ACTIVATION=${ACTIVATION:-relu}` default; add to the `export` list;
  **add to the `--container-env=` list**; pass `--activation "$ACTIVATION"` to the CLI.
- `slurm/search.slurm`: add `--activation relu` to the hardcoded head flags (~line 100-104).
- `slurm/README.md`: one row in the knob table.

`consolidate` needs **no** change — the run name doesn't carry the activation (decision above).

### 5. Docs

- `docs/head-activation.md` — new. Follow `docs/finetune-warmup.md`'s shape (*What it does* / *Why* /
  flag + cluster default), then a `## Sweep` section with the Setup/Bottom-line/tables layout of
  `docs/warmup-head-sweep-results.md`, left as a skeleton for the user's numbers. Carry the same
  caveat that doc does: **`mlsys analyze` does not apply to this sweep** — it pairs one frozen pass
  with one finetune pass per run-id to compute regret, and has no activation axis; four frozen-only
  CSVs with distinct run-ids form no analysable pair, so the tables are computed straight from the
  CSVs.
- `README.md:60-65` (`### Head type (--hidden)`) and the flag list at line 43: the
  `in_dim -> WIDTH -> ReLU -> 1` sketch becomes `-> ACT ->`; mention `--activation`.
- `cli/main.py:88-94`'s `--hidden` help has the same literal `-> ReLU -> 1` — update it too.

### 6. Tests (`tests/`, flat dir)

- `tests/test_head_training.py` (guards `torch = pytest.importorskip("torch")` before importing):
  - parametrize over the four names, assert `FCHead(in_dim=8, hidden=16, activation=name).net[1]`
    is the expected `nn.Module` subclass — there is currently *no* test asserting on `head.net`'s
    structure at all, so this also backfills the linear-vs-MLP assertion.
  - `FCHead(..., hidden=16, activation="swish")` raises `ValueError`; `HeadTrainConfig(activation="swish")` raises.
  - linear probe (`hidden=None`) ignores `activation` — still a bare `nn.Linear`.
- `tests/test_cli.py`: clone `test_grad_clipping_reaches_finetune_config` (lines 140-168) into
  `test_activation_reaches_head_config`, capturing `kwargs["head_config"].activation`; and a
  `--activation swish` → `SystemExit` test on the `choices=` rejection, like
  `test_unknown_strategy_exits_nonzero`.

## Sweep prep (handed to the user, not run here)

Four arms, `hidden=256`, frozen strategy, full pool:

```bash
for ACT in relu gelu tanh silu; do
  ACTIVATION=$ACT HIDDEN=256 DATASET=wine_reviews STRATEGY=frozen bash slurm/submit.sh
done
```

Then drop the consolidated CSVs into `results/activation_sweep/` with the arm token appended
(`<runid>_wine_reviews_frozen_16_model_MLP_256_<act>_frozen.csv`) and I'll fill in
`docs/head-activation.md`: per-arm mean/std/min r², spearman, n_negative, epochs_run, and — the
question that actually matters — the pairwise Spearman/Kendall correlation between the arms' **proxy
rankings**, plus whether the top-k picks change.

## Verification

- `make check` (lint + `ty` + tests — mirrors CI).
- End-to-end on CPU, both head paths and a rejection:
  ```bash
  python -m mlsys search --dataset wine_reviews --models all-MiniLM-L6-v2,potion-base-8M \
      --strategy frozen --hidden 256 --activation gelu --epochs 2 --head-repeats 1 --device cpu
  python -m mlsys search ... --hidden 0 --activation tanh   # linear probe: activation ignored, no crash
  python -m mlsys search ... --activation swish             # exits non-zero, argparse choices error
  ```
  Confirm `runs/<id>/results.jsonl` carries `"activation": "gelu"` on every row and that the frozen
  and finetune passes agree on the column set.
- `python -m mlsys consolidate runs/<id> --hidden 256` on a `full_eval` run to check the exported
  CSV headers include `activation` and the run-name grammar is unchanged.
