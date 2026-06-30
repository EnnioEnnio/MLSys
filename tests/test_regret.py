"""Regret metric: REGRET.md reference behaviour + curve invariants. CPU-only, no torch."""

from __future__ import annotations

from itertools import pairwise

import pytest

from mlsys.search.regret import regret, regret_curve


def test_regret_reference_example() -> None:
    # Point-estimate regret = max over pool - max over shortlist.
    scores = {"a": 0.9, "b": 0.7, "c": 0.4, "d": 0.1}
    assert regret(scores, ["a"]) == pytest.approx(0.0)  # shortlist has the global best
    assert regret(scores, ["b"]) == pytest.approx(0.2)  # 0.9 - 0.7
    assert regret(scores, ["c", "d"]) == pytest.approx(0.5)  # 0.9 - max(0.4, 0.1)
    assert regret(scores, ["a", "b", "c", "d"]) == pytest.approx(0.0)  # whole pool


def test_regret_rejects_empty() -> None:
    with pytest.raises(ValueError):
        regret({}, ["a"])
    with pytest.raises(ValueError):
        regret({"a": 1.0}, [])


def test_regret_curve_non_increasing_and_zero_at_full_budget() -> None:
    finetune_scores = {"a": 0.9, "b": 0.7, "c": 0.4, "d": 0.1}
    # A proxy ranking that is NOT the finetune order, to exercise real regret.
    proxy_ranking = ["c", "b", "a", "d"]
    curve = regret_curve(proxy_ranking, finetune_scores)

    assert [p.budget for p in curve] == [1, 2, 3, 4]
    regrets = [p.regret for p in curve]
    # Non-increasing in B.
    assert all(later <= earlier for earlier, later in pairwise(regrets))
    # B=1 picks "c" (0.4) -> regret 0.5; full budget -> 0.
    assert regrets[0] == pytest.approx(0.5)
    assert regrets[-1] == pytest.approx(0.0)


def test_regret_curve_normalized() -> None:
    finetune_scores = {"a": 0.8, "b": 0.4, "c": 0.2}
    curve = regret_curve(["b", "a", "c"], finetune_scores)
    # B=1 picks "b" (0.4): abs regret 0.4, normalized 0.4 / 0.8 = 0.5.
    assert curve[0].regret == pytest.approx(0.4)
    assert curve[0].normalized_regret == pytest.approx(0.5)
    assert curve[-1].normalized_regret == pytest.approx(0.0)


def test_regret_curve_handles_nonpositive_best() -> None:
    # When the best achievable r2 is <= 0, normalization is ill-defined -> reported as 0.
    finetune_scores = {"a": -0.1, "b": -0.5}
    curve = regret_curve(["b", "a"], finetune_scores)
    assert curve[0].regret == pytest.approx(0.4)  # -0.1 - (-0.5)
    assert curve[0].normalized_regret == 0.0


def test_regret_curve_rejects_ranking_not_in_scores() -> None:
    with pytest.raises(ValueError, match="missing"):
        regret_curve(["a", "z"], {"a": 0.5})
