"""Discover + load a ``full_eval`` experiment's CSV dumps into pandas frames.

An **experiment** is a folder of CSVs, one ``full_eval`` head config per run-id, each
contributing up to three ``kind`` files:

    <runid>_<strategy>_<num>_model_<HEAD>_<kind>.csv

e.g. ``2296342_fulleval_16_model_MLP_512_frozen.csv`` → run-id ``2296342``, head
``MLP_512``, kind ``frozen``. The head label is *only* recoverable from the filename — the
CSV ``head_type`` column is just ``linear``/``mlp`` and does **not** encode MLP width.

Grouping is by **run-id**: the three (frozen / finetune / regret) files that share a run-id
form one :class:`Triple`. ``*_regret.csv`` may be missing (recomputed downstream); the
frozen/finetune pair is the minimum a triple needs.

All heavy imports (``pandas``) are lazy so config-only CLI paths stay fast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)

KINDS = ("frozen", "finetune", "regret")

_FILENAME_GRAMMAR = "<runid>_<strategy>_<num>_model_<HEAD>_<kind>.csv"


@dataclass(frozen=True)
class ParsedName:
    """The fields lifted out of a result filename (see :data:`_FILENAME_GRAMMAR`)."""

    run_id: str
    strategy: str
    num: str
    head: str
    kind: str


def parse_filename(path: str | Path) -> ParsedName:
    """Parse one result filename. Raises ``ValueError`` naming the grammar on a mismatch.

    The head label is every token between the literal ``model`` marker and the trailing
    ``kind`` token, re-joined with ``_`` — so both ``FCH`` and ``MLP_512`` round-trip.
    """
    stem = Path(path).stem
    tokens = stem.split("_")
    # Need at least: runid, strategy, num, "model", <head>, kind  → 6 tokens.
    if len(tokens) < 6 or "model" not in tokens:
        raise ValueError(
            f"filename {Path(path).name!r} does not match the expected grammar "
            f"{_FILENAME_GRAMMAR!r} (e.g. 2296342_fulleval_16_model_MLP_512_frozen.csv)"
        )
    model_idx = tokens.index("model")
    kind = tokens[-1]
    if kind not in KINDS:
        raise ValueError(
            f"filename {Path(path).name!r} ends in kind {kind!r}; expected one of {KINDS} "
            f"per the grammar {_FILENAME_GRAMMAR!r}"
        )
    head_tokens = tokens[model_idx + 1 : -1]
    if not head_tokens:
        raise ValueError(
            f"filename {Path(path).name!r} has no head label between 'model' and {kind!r}; "
            f"expected the grammar {_FILENAME_GRAMMAR!r}"
        )
    return ParsedName(
        run_id=tokens[0],
        strategy=tokens[1],
        num="_".join(tokens[2:model_idx]),
        head="_".join(head_tokens),
        kind=kind,
    )


def head_sort_key(head: str) -> tuple[int, int]:
    """Order heads by capacity: ``FCH`` (linear, width 0) first, then ``MLP_<width>`` asc."""
    if head.upper().startswith("MLP_"):
        try:
            return (1, int(head.split("_", 1)[1]))
        except ValueError:
            return (1, 0)
    return (0, 0)


@dataclass
class Triple:
    """One head config's loaded frames + the per-model flags derived from them.

    ``regret`` is ``None`` when no ``*_regret.csv`` was present; callers recompute it.
    ``diverged`` / ``finetune_skipped`` are model→bool maps keyed on the model name.
    """

    run_id: str
    head: str
    frozen: pd.DataFrame
    finetune: pd.DataFrame
    regret: pd.DataFrame | None
    diverged: dict[str, bool]
    finetune_skipped: dict[str, bool]

    @property
    def models(self) -> list[str]:
        """Models in frozen-CSV row order (the stable tie-break basis for the proxy rank)."""
        return list(self.frozen["model"])


@dataclass
class _TripleFiles:
    run_id: str
    head: str
    paths: dict[str, Path]  # kind -> path


def discover_triples(experiment_dir: str | Path) -> list[_TripleFiles]:
    """Group an experiment folder's CSVs by run-id, sorted by head capacity.

    A pure grouping primitive: it does **not** judge completeness. Every run-id with at least
    one grammar-conforming CSV is returned; deciding what is analysable (the frozen+finetune
    requirement) and reporting which heads were skipped is the caller's job (see
    ``report._load_surviving``), so it can surface skip reasons in the SUMMARY metadata. Files
    that don't match the grammar are skipped with a warning rather than crashing — the folder
    may hold unrelated artifacts.
    """
    root = Path(experiment_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"experiment dir not found: {root}")

    by_run: dict[str, _TripleFiles] = {}
    for csv in sorted(root.glob("*.csv")):
        try:
            parsed = parse_filename(csv)
        except ValueError as exc:
            log.warning("skipping non-conforming file %s: %s", csv.name, exc)
            continue
        tf = by_run.setdefault(
            parsed.run_id, _TripleFiles(run_id=parsed.run_id, head=parsed.head, paths={})
        )
        tf.paths[parsed.kind] = csv

    return sorted(by_run.values(), key=lambda tf: head_sort_key(tf.head))


def _flag_diverged(finetune: pd.DataFrame, skipped: dict[str, bool]) -> dict[str, bool]:
    """A *real* finetune is *diverged* when its r² went negative (rank kept, scale broken).

    ``finetune_skipped`` models (model2vec ``can_finetune=False``) are **never** diverged: they
    were never fine-tuned — their "finetune" row is the frozen ``score_candidate`` reused — so a
    negative r² there is frozen-head underfitting, not the finetune blow-up this flag tracks.
    Excluding them keeps the divergence story (deberta/electra/roberta) free of model2vec noise
    across the table, the divergence map, and the ``n_diverged`` count.
    """
    return {
        str(model): bool(r2 < 0) and not skipped.get(str(model), False)
        for model, r2 in zip(finetune["model"], finetune["r2"], strict=True)
    }


def _flag_finetune_skipped(finetune: pd.DataFrame, head: str) -> dict[str, bool]:
    """Detect model2vec ``can_finetune=False`` fallbacks (frozen score reused as finetune).

    Primary, construction-invariant signal: a *real* finetune fuses inference into the joint
    loop so ``inference_s == 0``; the fallback runs the frozen ``score_candidate`` and so
    records a nonzero encode time. We corroborate with ``epochs_run``: real finetunes run
    exactly the configured budget (inferred here as the mode of the non-skipped rows), while
    the fallback runs the early-stopping frozen loop. Disagreement between the two signals is
    a warning, not an error — the ``inference_s`` signal wins.
    """
    inference = finetune["inference_s"] if "inference_s" in finetune.columns else None
    skipped = {
        str(model): bool(inf > 0.0)
        for model, inf in zip(
            finetune["model"],
            inference if inference is not None else [0.0] * len(finetune),
            strict=True,
        )
    }
    if "epochs_run" in finetune.columns:
        real_epochs = finetune.loc[~finetune["model"].map(skipped), "epochs_run"]
        budget = int(real_epochs.mode().iloc[0]) if not real_epochs.empty else None
        if budget is not None:
            for model, epochs_raw in zip(finetune["model"], finetune["epochs_run"], strict=True):
                model = str(model)
                epochs = int(epochs_raw)
                if skipped[model] and epochs == budget:
                    log.warning(
                        "%s/%s: inference_s>0 (looks skipped) but epochs_run==finetune budget "
                        "%d — signals disagree; trusting inference_s",
                        head,
                        model,
                        budget,
                    )
                elif not skipped[model] and epochs != budget:
                    log.warning(
                        "%s/%s: inference_s==0 (looks finetuned) but epochs_run=%d != budget %d "
                        "— signals disagree; trusting inference_s",
                        head,
                        model,
                        epochs,
                        budget,
                    )
    return skipped


def _crosscheck_head_type(df: pd.DataFrame, head: str, kind: str) -> None:
    """Warn if the CSV ``head_type`` contradicts the filename head label (FCH↔linear)."""
    if "head_type" not in df.columns:
        return
    expect = "linear" if head.upper() == "FCH" else "mlp"
    bad = set(df["head_type"].unique()) - {expect}
    if bad:
        log.warning(
            "%s %s: filename head %r implies head_type=%r but CSV has %s",
            head,
            kind,
            head,
            expect,
            sorted(bad),
        )


def load_triple(tf: _TripleFiles) -> Triple:
    """Load a discovered triple's frames + derive the per-model flags.

    Requires both frozen and finetune frames (the per-triple analysis needs the pair). Use
    :func:`discover_triples` ahead of this; callers that tolerate a half-present head should
    check ``"frozen" in tf.paths and "finetune" in tf.paths`` first.
    """
    import pandas as pd

    if "frozen" not in tf.paths or "finetune" not in tf.paths:
        missing = [k for k in ("frozen", "finetune") if k not in tf.paths]
        raise FileNotFoundError(
            f"run-id {tf.run_id} (head {tf.head}) is missing {missing}; cannot load triple"
        )

    frozen = pd.read_csv(tf.paths["frozen"])
    finetune = pd.read_csv(tf.paths["finetune"])
    regret = pd.read_csv(tf.paths["regret"]) if "regret" in tf.paths else None

    _crosscheck_head_type(frozen, tf.head, "frozen")
    _crosscheck_head_type(finetune, tf.head, "finetune")

    skipped = _flag_finetune_skipped(finetune, tf.head)
    return Triple(
        run_id=tf.run_id,
        head=tf.head,
        frozen=frozen,
        finetune=finetune,
        regret=regret,
        diverged=_flag_diverged(finetune, skipped),
        finetune_skipped=skipped,
    )
