"""Regression metrics for ranking candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    mae: float
    r2: float
    spearman: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    # Use scipy when available (sharper tie handling); fall back to manual rho.
    try:
        import warnings

        from scipy.stats import ConstantInputWarning, spearmanr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            rho = spearmanr(x, y).correlation
        return float(rho) if rho == rho else 0.0  # NaN guard
    except ImportError:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
        if denom == 0.0:
            return 0.0
        return float((rx * ry).sum() / denom)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    diff = y_pred - y_true
    mse = float((diff**2).mean())
    mae = float(np.abs(diff).mean())
    var = float(y_true.var())
    r2 = float(1.0 - mse / var) if var > 0 else 0.0
    sp = _spearman(y_true, y_pred)
    return RegressionMetrics(mse=mse, mae=mae, r2=r2, spearman=sp)


@dataclass(frozen=True)
class SummarizationMetrics:
    """ROUGE F-measures for summarization ranking (all higher-is-better)."""

    rouge1: float
    rouge2: float
    rougeL: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def summarization_metrics(preds: list[str], refs: list[str]) -> SummarizationMetrics:
    """Mean ROUGE-1/2/L F-measure over ``(pred, ref)`` pairs."""
    from rouge_score import rouge_scorer  # lazy heavy import (project convention)

    if len(preds) != len(refs):
        raise ValueError(
            f"summarization_metrics needs equal-length preds/refs; "
            f"got {len(preds)} preds and {len(refs)} refs"
        )
    keys = ("rouge1", "rouge2", "rougeL")
    scorer = rouge_scorer.RougeScorer(list(keys), use_stemmer=True)
    if not preds:
        return SummarizationMetrics(rouge1=0.0, rouge2=0.0, rougeL=0.0)
    sums = dict.fromkeys(keys, 0.0)
    for pred, ref in zip(preds, refs, strict=True):
        scores = scorer.score(ref, pred)
        for k in keys:
            sums[k] += scores[k].fmeasure
    n = len(preds)
    return SummarizationMetrics(
        rouge1=sums["rouge1"] / n,
        rouge2=sums["rouge2"] / n,
        rougeL=sums["rougeL"] / n,
    )
