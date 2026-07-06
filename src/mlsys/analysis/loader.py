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


@dataclass(frozen=True)
class RolePair:
    """Metadata for a (proxy, reference) pass pair used in a two-pass analysis run.

    ``supports_regret`` is True only for genuine proxy-vs-ground-truth pairs (frozen/finetune):
    regret presupposes different strategies, so computing it for same-strategy pairs like
    (r1, r3) produces all-zero curves by construction.

    ``supports_divergence`` is True only when the reference pass is a finetune run: backbone
    divergence (r² < 0 due to training instability) cannot occur during a frozen head-fit.
    """

    proxy_kind: str
    reference_kind: str
    proxy_label: str
    reference_label: str
    supports_regret: bool
    supports_divergence: bool


# First matching pair wins (insertion order). Add new pairs here; no other file needs editing.
# ("frozen", "finetune") is the only genuine proxy-vs-ground-truth pair.
# ("r1", "r3") is a same-strategy variance comparison — both passes are frozen proxies;
# r3 is a lower-variance estimate of the same signal as r1, NOT ground truth.
_ROLE_PAIRS: list[RolePair] = [
    RolePair(
        "frozen",
        "finetune",
        "frozen",
        "finetune",
        supports_regret=True,
        supports_divergence=True,
    ),
    RolePair(
        "r1",
        "r3",
        "repeat=1",
        "repeat=3",
        supports_regret=False,
        supports_divergence=False,
    ),
]

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
    ``diverged`` / ``ref_skipped`` are model→bool maps keyed on the model name; both are empty
    for non-finetune pairs (backbone divergence and finetune-skip detection only apply to a real
    finetune pass).
    ``role_pair`` carries display labels and capability flags for the two passes.  Use
    ``triple.proxy_label`` / ``triple.reference_label`` for display; use
    ``triple.role_pair.supports_regret`` / ``triple.role_pair.supports_divergence`` to gate
    metrics that only make sense for genuine proxy-vs-finetune pairs.
    ``.reference`` is NOT always ground truth — for same-strategy pairs like (r1, r3) it is a
    lower-variance frozen proxy, not a finetune signal.
    """

    run_id: str
    head: str
    proxy: pd.DataFrame
    reference: pd.DataFrame
    regret: pd.DataFrame | None
    diverged: dict[str, bool]
    ref_skipped: dict[str, bool]
    role_pair: RolePair

    @property
    def proxy_label(self) -> str:
        return self.role_pair.proxy_label

    @property
    def reference_label(self) -> str:
        return self.role_pair.reference_label

    @property
    def models(self) -> list[str]:
        """Models in proxy-CSV row order (the stable tie-break basis for the proxy rank)."""
        return list(self.proxy["model"])


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


def _flag_ref_skipped(ref_df: pd.DataFrame) -> dict[str, bool]:
    """Detect model2vec ``can_finetune=False`` fallbacks (frozen score reused as finetune).

    Construction-invariant signal: a *real* finetune fuses inference into the joint loop so
    ``inference_s == 0``; the fallback runs the frozen ``score_candidate`` and so records a
    nonzero encode time. ``epochs_run`` cannot corroborate this — ``train_full_model`` has the
    same ``early_stop_patience`` early-stopping as the frozen head, so a real finetune does
    *not* run a fixed budget and any epoch-count check would just flag every early-stopped
    model. ``inference_s`` alone is the reliable signal.

    Only meaningful for a genuine finetune reference pass (``role_pair.supports_divergence``).
    For frozen-vs-frozen pairs both passes always have ``inference_s > 0``, so the flag would
    fire for every model — callers must not call this for non-finetune pairs.
    """
    inference = ref_df["inference_s"] if "inference_s" in ref_df.columns else None
    return {
        str(model): bool(inf > 0.0)
        for model, inf in zip(
            ref_df["model"],
            inference if inference is not None else [0.0] * len(ref_df),
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


def resolve_role_pair(paths: dict[str, Path]) -> RolePair | None:
    """Return the :class:`RolePair` for the pair present in ``paths``, or ``None``.

    Iterates :data:`_ROLE_PAIRS` in insertion order so the first matching pair wins.
    """
    for role in _ROLE_PAIRS:
        if role.proxy_kind in paths and role.reference_kind in paths:
            return role
    return None


def load_triple(tf: _TripleFiles) -> Triple:
    """Load a discovered triple's frames + derive the per-model flags.

    Accepts any recognised proxy/reference kind pair (see :data:`_ROLE_PAIRS`): the standard
    ``frozen`` + ``finetune`` pair for a ``full_eval`` run, or ``r1`` + ``r3`` for a
    frozen repeat-count comparison. Internally the DataFrames are always stored as
    ``.proxy`` (cheap pass) and ``.reference`` (second pass — only ground truth for the
    finetune kind); display labels and capability flags come from ``Triple.role_pair``.

    Use :func:`discover_triples` ahead of this; callers that tolerate missing heads should
    check ``resolve_role_pair(tf.paths) is not None`` first.
    """
    import pandas as pd

    role = resolve_role_pair(tf.paths)
    if role is None:
        available = sorted(tf.paths)
        raise FileNotFoundError(
            f"run-id {tf.run_id} (head {tf.head}) has kinds {available} but no recognised "
            f"proxy/reference pair; expected one of "
            f"{[(r.proxy_kind, r.reference_kind) for r in _ROLE_PAIRS]}"
        )

    proxy_df = pd.read_csv(tf.paths[role.proxy_kind])
    ref_df = pd.read_csv(tf.paths[role.reference_kind])
    regret = pd.read_csv(tf.paths["regret"]) if "regret" in tf.paths else None

    _crosscheck_head_type(proxy_df, tf.head, role.proxy_kind)
    _crosscheck_head_type(ref_df, tf.head, role.reference_kind)

    # Both flags are finetune-specific: ref_skipped detects model2vec fallbacks via
    # inference_s > 0 (only meaningful when inference is fused → 0 in a real finetune);
    # diverged detects backbone training instability (r2 < 0 after fine-tuning).
    # For frozen-vs-frozen pairs (r1/r3) both signals are meaningless — leave them empty.
    skipped = _flag_ref_skipped(ref_df) if role.supports_divergence else {}
    diverged = _flag_diverged(ref_df, skipped) if role.supports_divergence else {}
    return Triple(
        run_id=tf.run_id,
        head=tf.head,
        proxy=proxy_df,
        reference=ref_df,
        regret=regret,
        diverged=diverged,
        ref_skipped=skipped,
        role_pair=role,
    )
