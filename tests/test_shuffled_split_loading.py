"""load_dataset() shuffle+row-count path: non-overlapping splits, deterministic order."""

from __future__ import annotations

from pathlib import Path

import pytest

import mlsys.datasets as datasets_module

N_ROWS = 1000


def _fake_dataset():
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "text": [f"row-{i}" for i in range(N_ROWS)],
            "y": [float(i) for i in range(N_ROWS)],
        }
    )


@pytest.fixture
def shuffled_spec_path(tmp_path: Path) -> Path:
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        "- name: shuffled\n  hf_repo: fake/repo\n  base_split: train\n  shuffle_seed: 7\n"
        "  splits:\n    train: '700'\n    val: '100'\n    test: '200'\n"
        '  target_column: y\n  target_type: regression\n  text_template: "{text}"\n'
    )
    return cfg


def test_shuffled_splits_are_non_overlapping_and_correctly_sized(
    monkeypatch: pytest.MonkeyPatch, shuffled_spec_path: Path
) -> None:
    def _get_spec(name: str) -> datasets_module.DatasetSpec:
        return datasets_module.load_specs(shuffled_spec_path)[name]

    monkeypatch.setattr(datasets_module, "get_spec", _get_spec)
    monkeypatch.setattr("datasets.load_dataset", lambda repo, split: _fake_dataset())

    loaded = datasets_module.load_dataset("shuffled")

    assert len(loaded.split("train").hf_split) == 700
    assert len(loaded.split("val").hf_split) == 100
    assert len(loaded.split("test").hf_split) == 200

    train_texts = {row["text"] for row in loaded.split("train").hf_split}
    val_texts = {row["text"] for row in loaded.split("val").hf_split}
    test_texts = {row["text"] for row in loaded.split("test").hf_split}
    assert train_texts.isdisjoint(val_texts)
    assert train_texts.isdisjoint(test_texts)
    assert val_texts.isdisjoint(test_texts)
    assert train_texts | val_texts | test_texts == {f"row-{i}" for i in range(N_ROWS)}


def test_shuffled_splits_are_deterministic_for_same_seed(
    monkeypatch: pytest.MonkeyPatch, shuffled_spec_path: Path
) -> None:
    def _get_spec(name: str) -> datasets_module.DatasetSpec:
        return datasets_module.load_specs(shuffled_spec_path)[name]

    monkeypatch.setattr(datasets_module, "get_spec", _get_spec)
    monkeypatch.setattr("datasets.load_dataset", lambda repo, split: _fake_dataset())

    first = datasets_module.load_dataset("shuffled")
    second = datasets_module.load_dataset("shuffled")

    first_train_order = [row["text"] for row in first.split("train").hf_split]
    second_train_order = [row["text"] for row in second.split("train").hf_split]
    assert first_train_order == second_train_order
