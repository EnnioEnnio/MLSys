"""_SplitView.__iter__ parses/filters targets per target_type (float vs. string)."""

from __future__ import annotations

from typing import Literal

from mlsys.datasets import _parse_target, _SplitView
from mlsys.datasets.registry import DatasetSpec


def _spec(target_type: Literal["regression", "summarization"]) -> DatasetSpec:
    return DatasetSpec(
        name="d",
        hf_repo="local/fake",
        splits={"train": "train", "val": "val", "test": "test"},
        target_column="y",
        target_type=target_type,
        text_template="{text}",
    )


def test_parse_target_regression() -> None:
    assert _parse_target("regression", "3.5") == 3.5
    assert _parse_target("regression", 2) == 2.0
    assert _parse_target("regression", None) is None
    assert _parse_target("regression", "not a number") is None


def test_parse_target_summarization() -> None:
    assert _parse_target("summarization", "a summary") == "a summary"
    assert _parse_target("summarization", 42) == "42"  # coerced to str, kept
    assert _parse_target("summarization", None) is None
    assert _parse_target("summarization", "   ") is None  # whitespace-only dropped
    assert _parse_target("summarization", "") is None


def test_split_view_summarization_keeps_string_targets() -> None:
    hf_split = [
        {"text": "dialogue one", "y": "summary one"},
        {"text": "dialogue two", "y": None},  # dropped
        {"text": "dialogue three", "y": "  "},  # dropped (whitespace)
        {"text": "dialogue four", "y": "summary four"},
    ]
    view = _SplitView(spec=_spec("summarization"), hf_split=hf_split)
    rows = list(view)
    assert [r.target for r in rows] == ["summary one", "summary four"]
    assert all(isinstance(r.target, str) for r in rows)
    assert len(view) == 2


def test_split_view_regression_float_filters() -> None:
    hf_split = [
        {"text": "a", "y": "1.0"},
        {"text": "b", "y": None},  # dropped
        {"text": "c", "y": "oops"},  # dropped (non-castable)
        {"text": "d", "y": 3},
    ]
    view = _SplitView(spec=_spec("regression"), hf_split=hf_split)
    rows = list(view)
    assert [r.target for r in rows] == [1.0, 3.0]
    assert all(isinstance(r.target, float) for r in rows)
