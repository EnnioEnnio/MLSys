"""Recompute the regret curve from frozen + finetune CSVs.

This is the crash-recovery path: if a ``full_eval`` run wrote its frozen/finetune CSVs but
died before emitting ``*_regret.csv`` (or you only kept the partial dumps), this rebuilds the
curve **byte-identically** to what the pipeline would have written.

Parity is guaranteed by reusing :func:`mlsys.search.regret.regret_curve` and building the
proxy ranking exactly as ``full_eval.run_full_eval`` does (``full_eval.py:209``): frozen r²
descending, with the frozen-CSV **row order** as the stable tie-break. Python's ``sorted`` is
stable, so feeding it the models in row order reproduces the pipeline's ordering on ties.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mlsys.search.regret import regret_curve

if TYPE_CHECKING:
    import pandas as pd

    from mlsys.analysis.loader import Triple

_REGRET_COLUMNS = ("budget", "regret", "normalized_regret")


def triple_regret_df(triple: Triple) -> pd.DataFrame:
    """The triple's regret curve — the committed one if present, else recomputed.

    Shared by ``plots.py`` and ``tables.py`` (which can't import from each other) so the
    committed-vs-recompute fallback lives in exactly one place.
    """
    if triple.regret is not None:
        return triple.regret
    return recompute_regret(triple.proxy, triple.gt)


def _r2_map(frame: pd.DataFrame) -> dict[str, float]:
    """``{model: r2}`` preserving CSV row order (the stable tie-break basis)."""
    return {str(model): float(r2) for model, r2 in zip(frame["model"], frame["r2"], strict=True)}


def proxy_ranking_from_frozen(frozen: pd.DataFrame) -> list[str]:
    """Frozen r² desc, ties broken by CSV row order — matches ``full_eval.py:209`` exactly."""
    # dict preserves row order; sorted() is stable, so equal r² keep their row order.
    frozen_r2 = _r2_map(frozen)
    return sorted(frozen_r2, key=lambda m: frozen_r2[m], reverse=True)


def recompute_regret(frozen: pd.DataFrame, finetune: pd.DataFrame) -> pd.DataFrame:
    """Frozen+finetune frames → a regret-curve DataFrame (one row per budget ``1..|M|``)."""
    import pandas as pd

    proxy_ranking = proxy_ranking_from_frozen(frozen)
    curve = regret_curve(proxy_ranking, _r2_map(finetune))
    return pd.DataFrame([p.to_dict() for p in curve], columns=list(_REGRET_COLUMNS))


def regret_json_payload(
    frozen: pd.DataFrame, finetune: pd.DataFrame, dataset: str | None = None
) -> dict[str, object]:
    """Mirror ``full_eval._write_regret_json``'s payload for the standalone ``--json`` path."""
    proxy_ranking = proxy_ranking_from_frozen(frozen)
    frozen_r2 = _r2_map(frozen)
    finetune_r2 = _r2_map(finetune)
    curve = regret_curve(proxy_ranking, finetune_r2)
    if dataset is None and "dataset" in frozen.columns and len(frozen):
        dataset = str(frozen["dataset"].iloc[0])
    return {
        "dataset": dataset,
        "metric": "r2",
        "higher_is_better": True,
        "regret_estimator": "point_estimate",
        "proxy_ranking": proxy_ranking,
        "frozen_r2": frozen_r2,
        "finetune_r2": finetune_r2,
        "curve": [p.to_dict() for p in curve],
    }


def recompute_to_files(
    frozen_csv: str | Path,
    finetune_csv: str | Path,
    out_csv: str | Path | None = None,
    json_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read two CSVs, write the recovered ``budget,regret,normalized_regret`` curve.

    Returns the curve DataFrame; optionally also writes a ``regret.json``-shaped file.
    """
    import json

    import pandas as pd

    frozen = pd.read_csv(frozen_csv)
    finetune = pd.read_csv(finetune_csv)
    curve = recompute_regret(frozen, finetune)
    if out_csv is not None:
        curve.to_csv(out_csv, index=False)
    if json_path is not None:
        payload = regret_json_payload(frozen, finetune)
        Path(json_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return curve
