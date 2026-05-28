"""Both YAML registries parse, every entry has required fields, names are unique."""

from __future__ import annotations

import pytest

from mlsys.datasets.registry import load_specs as load_dataset_specs
from mlsys.models.registry import KNOWN_LOADERS
from mlsys.models.registry import load_specs as load_model_specs


def test_datasets_yaml_parses() -> None:
    specs = load_dataset_specs()
    assert specs, "datasets.yaml must contain at least one entry"
    for spec in specs.values():
        assert spec.target_type == "regression"
        assert {"train", "val", "test"} <= set(spec.splits)
        assert "{" in spec.text_template


def test_dataset_names_unique() -> None:
    specs = load_dataset_specs()
    assert len(specs) == len({s.name for s in specs.values()})


def test_models_yaml_parses() -> None:
    specs = load_model_specs()
    assert specs, "models.yaml must contain at least one entry"
    for spec in specs.values():
        assert spec.loader in KNOWN_LOADERS
        assert spec.embedding_dim > 0


def test_model_names_unique() -> None:
    specs = load_model_specs()
    assert len(specs) == len({s.name for s in specs.values()})


def test_seed_pool_contains_expected_models() -> None:
    specs = load_model_specs()
    expected = {"all-MiniLM-L6-v2", "all-mpnet-base-v2", "potion-base-8M", "potion-base-32M"}
    assert expected <= set(specs), f"missing seed-pool models: {expected - set(specs)}"


def test_models_yaml_rejects_unknown_loader(tmp_path) -> None:
    bad = tmp_path / "models.yaml"
    bad.write_text("- name: x\n  hf_repo: y/z\n  loader: not_a_real_loader\n  embedding_dim: 8\n")
    with pytest.raises(ValueError, match="unknown loader"):
        load_model_specs(bad)
