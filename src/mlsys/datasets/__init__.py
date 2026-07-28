"""HF dataset loading + per-row text-template rendering."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from mlsys.datasets.registry import REQUIRED_SPLITS, DatasetSpec, get_spec, load_specs

log = logging.getLogger(__name__)

__all__ = [
    "DatasetSpec",
    "LoadedDataset",
    "Row",
    "get_spec",
    "load_dataset",
    "load_specs",
    "render_template",
]


@dataclass(frozen=True)
class Row:
    text: str
    target: float


class _SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "unknown"


def render_template(template: str, row: dict[str, Any]) -> str:
    """Render `template` against `row`, substituting `"unknown"` for missing/None values."""
    cleaned = {k: ("unknown" if v is None else v) for k, v in row.items()}
    return template.format_map(_SafeDict(cleaned))


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    splits: dict[str, _SplitView]

    def split(self, name: str) -> _SplitView:
        if name not in self.splits:
            raise KeyError(f"unknown split {name!r}; have {sorted(self.splits)}")
        return self.splits[name]


@dataclass
class _SplitView:
    spec: DatasetSpec
    hf_split: Any  # datasets.Dataset; kept untyped to avoid hard import at module load
    _filtered_len: int | None = field(default=None, init=False, repr=False)

    def __iter__(self) -> Iterator[Row]:
        template = self.spec.text_template
        target_col = self.spec.target_column
        for row in self.hf_split:
            target_val = row[target_col]
            if target_val is None:
                continue
            try:
                target_float = float(target_val)
            except (ValueError, TypeError):
                continue
            if self.spec.target_transform == "log":
                if target_float <= 0:
                    continue
                target_float = math.log(target_float)
            yield Row(text=render_template(template, row), target=target_float)

    def __len__(self) -> int:
        if self._filtered_len is None:
            total = len(self.hf_split)
            filtered = sum(1 for _ in self)
            dropped = total - filtered
            log.info(
                "Loaded %d/%d rows (%d dropped due to missing/invalid targets)",
                filtered,
                total,
                dropped,
            )
            if dropped == total:
                log.warning(
                    "All %d rows were dropped — check target_column %r",
                    total,
                    self.spec.target_column,
                )
            self._filtered_len = filtered
        return self._filtered_len

    def batched(self, batch_size: int) -> Iterable[list[Row]]:
        batch: list[Row] = []
        for row in self:
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def load_dataset(name: str) -> LoadedDataset:
    """Open the HF dataset for `name` and expose per-split iterators."""
    from datasets import load_dataset as hf_load_dataset

    spec = get_spec(name)
    splits: dict[str, _SplitView] = {}
    if spec.shuffle_seed is not None:
        assert spec.base_split is not None  # enforced together by load_specs()
        base = hf_load_dataset(spec.hf_repo, split=spec.base_split)
        base = base.shuffle(seed=spec.shuffle_seed)
        total = sum(int(spec.splits[s]) for s in REQUIRED_SPLITS)
        if total > len(base):
            raise ValueError(
                f"dataset {spec.name!r}: split row counts sum to {total} but base split "
                f"{spec.base_split!r} has only {len(base)} rows"
            )
        offset = 0
        # Slice in REQUIRED_SPLITS order (not YAML key order) so split membership
        # can't change if the YAML keys are reordered.
        for logical in REQUIRED_SPLITS:
            count = int(spec.splits[logical])
            # flatten_indices() rewrites the slice contiguously; without it every
            # per-candidate iteration random-accesses the full shuffled base table.
            splits[logical] = _SplitView(
                spec=spec,
                hf_split=base.select(range(offset, offset + count)).flatten_indices(),
            )
            offset += count
    else:
        for logical, hf_split_name in spec.splits.items():
            hf_split = hf_load_dataset(spec.hf_repo, split=hf_split_name)
            splits[logical] = _SplitView(spec=spec, hf_split=hf_split)
    return LoadedDataset(spec=spec, splits=splits)
