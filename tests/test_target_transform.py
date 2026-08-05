"""_SplitView.__iter__ applies DatasetSpec.target_transform to Row.target."""

from __future__ import annotations

import math
from typing import Literal

from mlsys.datasets import Row, _SplitView
from mlsys.datasets.registry import DatasetSpec


def _spec(target_transform: Literal["identity", "log"]) -> DatasetSpec:
    return DatasetSpec(
        name="d",
        hf_repo="y/z",
        splits={"train": "train", "val": "validation", "test": "test"},
        target_column="price",
        target_type="regression",
        text_template="{price}",
        target_transform=target_transform,
    )


def test_log_transform_applies_log_to_positive_targets() -> None:
    rows = [{"price": 100.0}, {"price": 2.5}]
    view = _SplitView(spec=_spec("log"), hf_split=rows)
    out = list(view)
    assert out == [Row(text="100.0", target=math.log(100.0)), Row(text="2.5", target=math.log(2.5))]


def test_log_transform_drops_nonpositive_targets() -> None:
    rows = [{"price": 100.0}, {"price": 0.0}, {"price": -5.0}]
    view = _SplitView(spec=_spec("log"), hf_split=rows)
    out = list(view)
    assert len(out) == 1
    assert out[0].target == math.log(100.0)


def test_identity_transform_is_unaffected() -> None:
    rows = [{"price": 100.0}, {"price": -5.0}]
    view = _SplitView(spec=_spec("identity"), hf_split=rows)
    out = list(view)
    assert [row.target for row in out] == [100.0, -5.0]


def test_nonfinite_targets_are_dropped_on_both_transforms() -> None:
    # A NaN clears both `is None` and `float()`; one of them poisons the train-split
    # mean/std that every downstream metric is z-scored against.
    rows = [{"price": 100.0}, {"price": float("nan")}, {"price": float("inf")}]
    for transform in ("identity", "log"):
        out = list(_SplitView(spec=_spec(transform), hf_split=rows))
        assert len(out) == 1, transform
