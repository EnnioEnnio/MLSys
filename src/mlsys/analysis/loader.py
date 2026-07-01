"""Discover + load a ``full_eval`` experiment's CSV dumps into pandas frames.

An **experiment** is a folder of CSVs, one ``full_eval`` head config per run-id, each
contributing up to three ``kind`` files:

    <runid>_<dataset>_<strategy>_<num>_model_<HEAD>_<kind>.csv

e.g. ``2296342_wine_reviews_fulleval_16_model_MLP_512_frozen.csv`` → run-id ``2296342``,
dataset ``wine_reviews``, head ``MLP_512``, kind ``frozen``. The stem (everything before
``_<kind>``) is exactly the W&B run name emitted by ``mlsys.cli.main._wandb_run_name`` — you
download a run's CSV, append ``_<kind>``, and drop it in. The head label is *only* recoverable
from the filename — the CSV ``head_type`` column is just ``linear``/``mlp`` and does **not**
encode MLP width.

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

KINDS = ("frozen", "finetune", "regret", "r1", "r3")

# Maps (proxy_kind, truth_kind) → (proxy_label, truth_label).
# proxy = cheap pass used to build the ranking; truth = expensive pass used to score regret.
# Add new pairs here to support further comparison modes without touching load_triple.
_ROLE_PAIRS: dict[tuple[str, str], tuple[str, str]] = {
    ("frozen", "finetune"): ("frozen", "finetune"),
    ("r1", "r3"): ("repeat=1", "repeat=3"),
}

_FILENAME_GRAMMAR = "<runid>_<dataset>_<strategy>_<num>_model_<HEAD>_<kind>.csv"
_EXAMPLE_NAME = "2296342_wine_reviews_fulleval_16_model_MLP_512_frozen.csv"


@dataclass(frozen=True)
class ParsedName:
    """The fields lifted out of a result filename (see :data:`_FILENAME_GRAMMAR`)."""

    run_id: str
    dataset: str
    strategy: str
    num: str
    head: str
    kind: str


def parse_filename(path: str | Path) -> ParsedName:
    """Parse one result filename. Raises ``ValueError`` naming the grammar on a mismatch.

    The literal ``model`` token is the anchor: ``strategy`` and ``num`` are the two tokens
    immediately before it, ``dataset`` is everything between the run-id and the strategy
    (so a multi-token dataset like ``wine_reviews`` round-trips), and the head label is every
    token between ``model`` and the trailing ``kind`` (so both ``FCH`` and ``MLP_512`` work).
    """
    stem = Path(path).stem
    tokens = stem.split("_")
    if "model" not in tokens:
        raise ValueError(
            f"filename {Path(path).name!r} does not match the expected grammar "
            f"{_FILENAME_GRAMMAR!r} (e.g. {_EXAMPLE_NAME})"
        )
    model_idx = tokens.index("model")
    # Before "model": runid (1) + dataset (>=1) + strategy (1) + num (1)  → model_idx >= 4.
    # After  "model": head (>=1) + kind (1)                               → >= 2 more tokens.
    if model_idx < 4 or len(tokens) < model_idx + 3:
        raise ValueError(
            f"filename {Path(path).name!r} does not match the expected grammar "
            f"{_FILENAME_GRAMMAR!r} (e.g. {_EXAMPLE_NAME})"
        )
    kind = tokens[-1]
    if kind not in KINDS:
        raise ValueError(
            f"filename {Path(path).name!r} ends in kind {kind!r}; expected one of {KINDS} "
            f"per the grammar {_FILENAME_GRAMMAR!r}"
        )
    return ParsedName(
        run_id=tokens[0],
        dataset="_".join(tokens[1 : model_idx - 2]),
        strategy=tokens[model_idx - 2],
        num=tokens[model_idx - 1],
        head="_".join(tokens[model_idx + 1 : -1]),
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
    ``proxy_label`` / ``truth_label`` are display names for the two passes (e.g. "frozen" /
    "finetune" for the standard full_eval, or "repeat=1" / "repeat=3" for a repeat comparison).
    Internally the DataFrames are always accessed as ``.frozen`` (proxy) / ``.finetune`` (truth).
    """

    run_id: str
    head: str
    frozen: pd.DataFrame
    finetune: pd.DataFrame
    regret: pd.DataFrame | None
    diverged: dict[str, bool]
    finetune_skipped: dict[str, bool]
    proxy_label: str = "frozen"
    truth_label: str = "finetune"

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


def _flag_finetune_skipped(finetune: pd.DataFrame) -> dict[str, bool]:
    """Detect model2vec ``can_finetune=False`` fallbacks (frozen score reused as finetune).

    Construction-invariant signal: a *real* finetune fuses inference into the joint loop so
    ``inference_s == 0``; the fallback runs the frozen ``score_candidate`` and so records a
    nonzero encode time. ``epochs_run`` cannot corroborate this — ``train_full_model`` has the
    same ``early_stop_patience`` early-stopping as the frozen head, so a real finetune does
    *not* run a fixed budget and any epoch-count check would just flag every early-stopped
    model. ``inference_s`` alone is the reliable signal.
    """
    inference = finetune["inference_s"] if "inference_s" in finetune.columns else None
    return {
        str(model): bool(inf > 0.0)
        for model, inf in zip(
            finetune["model"],
            inference if inference is not None else [0.0] * len(finetune),
            strict=True,
        )
    }


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


def resolve_role_pair(paths: dict[str, Path]) -> tuple[str, str, str, str] | None:
    """Return ``(proxy_kind, truth_kind, proxy_label, truth_label)`` for the pair present in
    ``paths``, or ``None`` when no recognised proxy/truth combination is found.

    Iterates :data:`_ROLE_PAIRS` in insertion order so the first matching pair wins.
    """
    for (pk, tk), (pl, tl) in _ROLE_PAIRS.items():
        if pk in paths and tk in paths:
            return pk, tk, pl, tl
    return None


def load_triple(tf: _TripleFiles) -> Triple:
    """Load a discovered triple's frames + derive the per-model flags.

    Accepts any recognised proxy/truth kind pair (see :data:`_ROLE_PAIRS`): the standard
    ``frozen`` + ``finetune`` pair for a ``full_eval`` run, or ``r1`` + ``r3`` for a
    frozen repeat-count comparison. Internally the DataFrames are always stored as
    ``.frozen`` (proxy) and ``.finetune`` (truth); display labels come from
    ``Triple.proxy_label`` / ``Triple.truth_label``.

    Use :func:`discover_triples` ahead of this; callers that tolerate missing heads should
    check ``resolve_role_pair(tf.paths) is not None`` first.
    """
    import pandas as pd

    role = resolve_role_pair(tf.paths)
    if role is None:
        available = sorted(tf.paths)
        raise FileNotFoundError(
            f"run-id {tf.run_id} (head {tf.head}) has kinds {available} but no recognised "
            f"proxy/truth pair; expected one of {list(_ROLE_PAIRS)}"
        )
    proxy_kind, truth_kind, proxy_label, truth_label = role

    frozen = pd.read_csv(tf.paths[proxy_kind])
    finetune = pd.read_csv(tf.paths[truth_kind])
    regret = pd.read_csv(tf.paths["regret"]) if "regret" in tf.paths else None

    _crosscheck_head_type(frozen, tf.head, proxy_kind)
    _crosscheck_head_type(finetune, tf.head, truth_kind)

    # finetune_skipped detects model2vec fallbacks via inference_s > 0; that signal only
    # applies to a real finetune pass (inference fused → 0).  For frozen-vs-frozen both
    # passes always have inference_s > 0, so the flag is meaningless — leave it empty.
    skipped = _flag_finetune_skipped(finetune) if truth_kind == "finetune" else {}
    return Triple(
        run_id=tf.run_id,
        head=tf.head,
        frozen=frozen,
        finetune=finetune,
        regret=regret,
        diverged=_flag_diverged(finetune, skipped),
        finetune_skipped=skipped,
        proxy_label=proxy_label,
        truth_label=truth_label,
    )
