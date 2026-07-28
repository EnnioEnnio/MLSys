"""Both YAML registries parse, every entry has required fields, names are unique."""

from __future__ import annotations

from typing import Never

import pytest

from mlsys.datasets.registry import load_specs as load_dataset_specs
from mlsys.models.registry import (
    _ADAPTERS,
    KNOWN_LOADERS,
    _ensure_adapters_registered,
    register_adapter,
)
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


def test_models_yaml_rejects_nonpositive_embedding_dim(tmp_path) -> None:
    bad = tmp_path / "models.yaml"
    bad.write_text(
        "- name: x\n  hf_repo: y/z\n  loader: sentence_transformers\n  embedding_dim: 0\n"
    )
    with pytest.raises(ValueError, match="embedding_dim must be > 0"):
        load_model_specs(bad)


def test_every_models_yaml_loader_has_a_registered_adapter() -> None:
    # Regression guard: each loader named in models.yaml must resolve to an adapter.
    _ensure_adapters_registered()
    for spec in load_model_specs().values():
        assert spec.loader in _ADAPTERS, f"{spec.name}: no adapter for loader {spec.loader!r}"


def test_load_specs_accepts_newly_registered_loader(tmp_path) -> None:
    # Extension-workflow regression guard: a freshly registered adapter's loader
    # must pass load_specs() validation with no edit to registry.py (matches the
    # README's "dropping the file is the whole step" claim).
    def _unused_builder(spec, device) -> Never:  # load_specs validates, never builds
        raise AssertionError("load_specs must not instantiate adapters")

    register_adapter("dummy_loader_xyz", _unused_builder)
    try:
        cfg = tmp_path / "models.yaml"
        cfg.write_text(
            "- name: x\n  hf_repo: y/z\n  loader: dummy_loader_xyz\n  embedding_dim: 8\n"
        )
        specs = load_model_specs(cfg)
        assert specs["x"].loader == "dummy_loader_xyz"
    finally:
        _ADAPTERS.pop("dummy_loader_xyz", None)


def test_datasets_yaml_rejects_missing_field(tmp_path) -> None:
    bad = tmp_path / "datasets.yaml"
    # text_template omitted
    bad.write_text(
        "- name: d\n  hf_repo: y/z\n"
        "  splits:\n    train: train\n    val: validation\n    test: test\n"
        "  target_column: y\n  target_type: regression\n"
    )
    with pytest.raises(ValueError, match="missing required field"):
        load_dataset_specs(bad)


def test_datasets_yaml_rejects_non_regression_target(tmp_path) -> None:
    bad = tmp_path / "datasets.yaml"
    bad.write_text(
        "- name: d\n  hf_repo: y/z\n"
        "  splits:\n    train: train\n    val: validation\n    test: test\n"
        '  target_column: y\n  target_type: classification\n  text_template: "{x}"\n'
    )
    with pytest.raises(ValueError, match="regression"):
        load_dataset_specs(bad)


def test_datasets_yaml_rejects_missing_split(tmp_path) -> None:
    bad = tmp_path / "datasets.yaml"
    # val split omitted
    bad.write_text(
        "- name: d\n  hf_repo: y/z\n"
        "  splits:\n    train: train\n    test: test\n"
        '  target_column: y\n  target_type: regression\n  text_template: "{x}"\n'
    )
    with pytest.raises(ValueError, match="missing split"):
        load_dataset_specs(bad)


def test_datasets_yaml_rejects_shuffle_seed_without_base_split(tmp_path) -> None:
    bad = tmp_path / "datasets.yaml"
    bad.write_text(
        "- name: d\n  hf_repo: y/z\n  shuffle_seed: 42\n"
        "  splits:\n    train: '100'\n    val: '10'\n    test: '10'\n"
        '  target_column: y\n  target_type: regression\n  text_template: "{x}"\n'
    )
    with pytest.raises(ValueError, match="shuffle_seed and base_split"):
        load_dataset_specs(bad)


def test_datasets_yaml_rejects_base_split_without_shuffle_seed(tmp_path) -> None:
    bad = tmp_path / "datasets.yaml"
    bad.write_text(
        "- name: d\n  hf_repo: y/z\n  base_split: train\n"
        "  splits:\n    train: '100'\n    val: '10'\n    test: '10'\n"
        '  target_column: y\n  target_type: regression\n  text_template: "{x}"\n'
    )
    with pytest.raises(ValueError, match="shuffle_seed and base_split"):
        load_dataset_specs(bad)


def test_datasets_yaml_rejects_non_numeric_split_with_shuffle_seed(tmp_path) -> None:
    bad = tmp_path / "datasets.yaml"
    bad.write_text(
        "- name: d\n  hf_repo: y/z\n  base_split: train\n  shuffle_seed: 42\n"
        "  splits:\n    train: 'train[:80%]'\n    val: '10'\n    test: '10'\n"
        '  target_column: y\n  target_type: regression\n  text_template: "{x}"\n'
    )
    with pytest.raises(ValueError, match="plain row count"):
        load_dataset_specs(bad)


def test_datasets_yaml_accepts_shuffle_seed_and_base_split(tmp_path) -> None:
    good = tmp_path / "datasets.yaml"
    good.write_text(
        "- name: d\n  hf_repo: y/z\n  base_split: train\n  shuffle_seed: 42\n"
        "  splits:\n    train: '100'\n    val: '10'\n    test: '10'\n"
        '  target_column: y\n  target_type: regression\n  text_template: "{x}"\n'
    )
    specs = load_dataset_specs(good)
    assert specs["d"].shuffle_seed == 42
    assert specs["d"].base_split == "train"


def test_datasets_yaml_rejects_unknown_target_transform(tmp_path) -> None:
    bad = tmp_path / "datasets.yaml"
    bad.write_text(
        "- name: d\n  hf_repo: y/z\n  target_transform: square\n"
        "  splits:\n    train: train\n    val: validation\n    test: test\n"
        '  target_column: y\n  target_type: regression\n  text_template: "{x}"\n'
    )
    with pytest.raises(ValueError, match="target_transform"):
        load_dataset_specs(bad)


def test_datasets_yaml_target_transform_defaults_and_parses() -> None:
    specs = load_dataset_specs()
    assert specs["usa_real_estate_log"].target_transform == "log"
    assert specs["usa_real_estate"].target_transform == "identity"
