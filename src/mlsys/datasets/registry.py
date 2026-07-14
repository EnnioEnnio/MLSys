"""Parse config/datasets.yaml into typed DatasetSpec records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

REQUIRED_FIELDS = (
    "name",
    "hf_repo",
    "splits",
    "target_column",
    "target_type",
    "text_template",
)
# The logical splits score_candidate() always requests. Validated here at load
# time and referenced from mlsys.search.runner so the two can't drift.
REQUIRED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_repo: str
    splits: dict[str, str]
    target_column: str
    target_type: Literal["regression"]
    text_template: str
    shuffle_seed: int | None = None
    base_split: str | None = None


def _config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "datasets.yaml"


def load_specs(path: Path | None = None) -> dict[str, DatasetSpec]:
    cfg_path = path or _config_path()
    raw = yaml.safe_load(cfg_path.read_text()) or []
    specs: dict[str, DatasetSpec] = {}
    for entry in raw:
        for field in REQUIRED_FIELDS:
            if field not in entry:
                raise ValueError(f"datasets.yaml entry missing required field {field!r}: {entry}")
        if entry["target_type"] != "regression":
            raise ValueError(
                f"v1 supports target_type=regression only; got {entry['target_type']!r}"
            )
        missing = [s for s in REQUIRED_SPLITS if s not in entry["splits"]]
        if missing:
            raise ValueError(f"datasets.yaml entry {entry['name']!r}: missing split(s) {missing}")
        shuffle_seed = entry.get("shuffle_seed")
        base_split = entry.get("base_split")
        if (shuffle_seed is None) != (base_split is None):
            raise ValueError(
                f"datasets.yaml entry {entry['name']!r}: shuffle_seed and base_split "
                "must both be set or both omitted"
            )
        if shuffle_seed is not None:
            for logical, count_str in entry["splits"].items():
                if not str(count_str).isdigit():
                    raise ValueError(
                        f"datasets.yaml entry {entry['name']!r}: splits.{logical} must be a "
                        f"plain row count when shuffle_seed is set, got {count_str!r}"
                    )
        spec = DatasetSpec(
            name=entry["name"],
            hf_repo=entry["hf_repo"],
            splits=dict(entry["splits"]),
            target_column=entry["target_column"],
            target_type="regression",
            text_template=entry["text_template"],
            shuffle_seed=shuffle_seed,
            base_split=base_split,
        )
        if spec.name in specs:
            raise ValueError(f"duplicate dataset name in datasets.yaml: {spec.name}")
        specs[spec.name] = spec
    return specs


def get_spec(name: str, path: Path | None = None) -> DatasetSpec:
    specs = load_specs(path)
    if name not in specs:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(specs)}")
    return specs[name]
