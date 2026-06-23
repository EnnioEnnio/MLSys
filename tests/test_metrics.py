"""regression_metrics: hand-computed values + R²/Spearman edge cases + scipy fallback."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from src.mlsys.search.metrics import regression_metrics


def test_mse_mae_hand_computed() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])  # error only on the last element: +2
    m = regression_metrics(y_true, y_pred)
    assert m.mse == pytest.approx(4.0 / 3.0)
    assert m.mae == pytest.approx(2.0 / 3.0)


def test_r2_is_one_on_perfect_predictions() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y.copy())
    assert m.r2 == pytest.approx(1.0)


def test_r2_is_zero_on_mean_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.full_like(y, y.mean())  # predicting the mean → R² == 0
    m = regression_metrics(y, pred)
    assert abs(m.r2) < 1e-9


def test_r2_zero_variance_guard() -> None:
    y = np.array([5.0, 5.0, 5.0])  # var == 0 → guarded to 0.0, no ZeroDivision
    m = regression_metrics(y, np.array([1.0, 2.0, 3.0]))
    assert m.r2 == 0.0


def test_spearman_one_on_monotonic_map() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    m = regression_metrics(y_true, y_pred)
    assert m.spearman == pytest.approx(1.0)


def test_spearman_zero_variance_guard() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([7.0, 7.0, 7.0])  # constant → undefined corr → guarded to 0
    m = regression_metrics(y_true, y_pred)
    assert m.spearman == 0.0


def test_spearman_fallback_without_scipy(monkeypatch) -> None:
    # Force the manual-rho path: make `from scipy.stats import spearmanr` raise.
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 4.0, 9.0, 16.0])  # monotonic increasing → rho == 1
    m = regression_metrics(y_true, y_pred)
    assert m.spearman == pytest.approx(1.0)
