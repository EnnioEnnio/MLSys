"""End-to-end smoke test: synthetic dataset + a fake backbone via a custom adapter.

Real wine_reviews + potion-base-8M would need network access and bring in
heavy deps (datasets, model2vec). The runner contract is what we want to lock
down here, so we register a fake-backbone adapter under a custom loader name
and feed it a tiny in-memory dataset that mimics the LoadedDataset shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

torch = pytest.importorskip("torch")

from mlsys.datasets import LoadedDataset, Row  # noqa: E402
from mlsys.datasets.registry import DatasetSpec  # noqa: E402
from mlsys.head import HeadTrainConfig  # noqa: E402
from mlsys.models.registry import _ADAPTERS, ModelSpec, register_adapter  # noqa: E402
from mlsys.search.full_eval import full_eval  # noqa: E402


class _FakeBackbone:
    name: str
    embedding_dim: int

    def __init__(self, spec: ModelSpec, device: str) -> None:
        self.name = spec.name
        self.embedding_dim = spec.embedding_dim
        self._device = device
        torch.manual_seed(0)
        self._proj = torch.randn(64, spec.embedding_dim)

    def encode(self, texts: list[str]) -> torch.Tensor:
        # Map each text to a deterministic vector via a hash → one-hot → projection.
        rows = []
        for t in texts:
            h = abs(hash(t)) % 64
            v = torch.zeros(64)
            v[h] = 1.0
            rows.append(v)
        x = torch.stack(rows, dim=0) @ self._proj
        return x.to(self._device)


class _FakeSplit:
    def __init__(self, rows: list[Row]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


class _FakeDataset:
    def __init__(self, name: str) -> None:
        torch.manual_seed(1)
        n_per_split = 20
        self.spec = DatasetSpec(
            name=name,
            hf_repo="local/fake",
            splits={"train": "train", "val": "val", "test": "test"},
            target_column="y",
            target_type="regression",
            text_template="{text}",
        )
        self.splits = {
            "train": _FakeSplit([Row(text=f"t{i}", target=float(i)) for i in range(n_per_split)]),
            "val": _FakeSplit(
                [Row(text=f"v{i}", target=float(i)) for i in range(n_per_split // 2)]
            ),
            "test": _FakeSplit(
                [Row(text=f"x{i}", target=float(i)) for i in range(n_per_split // 2)]
            ),
        }

    def split(self, name: str):
        return self.splits[name]


@pytest.fixture
def fake_loader_registered():
    register_adapter("fake_loader", lambda spec, device: _FakeBackbone(spec, device))
    yield
    _ADAPTERS.pop("fake_loader", None)


def test_full_eval_writes_one_row_per_model(
    tmp_path: Path, fake_loader_registered, monkeypatch
) -> None:
    # Patch the registry's load_specs so the runner sees only our fake model.
    # The module name `mlsys.search.full_eval` collides with the re-exported
    # function attribute on the package, so reach through sys.modules.
    import sys

    fake_spec = ModelSpec(
        name="fake-model",
        hf_repo="local/fake",
        loader="fake_loader",
        embedding_dim=8,
    )
    fe_module = sys.modules["mlsys.search.full_eval"]
    monkeypatch.setattr(fe_module, "load_specs", lambda: {"fake-model": fake_spec})

    dataset = _FakeDataset(name="synthetic")
    out_dir = tmp_path / "run"
    records = full_eval(
        cast(LoadedDataset, dataset),
        output_dir=out_dir,
        device="cpu",
        batch_size=4,
        head_config=HeadTrainConfig(epochs=3, batch_size=4, lr=1e-2),
    )

    assert len(records) == 1
    results_jsonl = out_dir / "results.jsonl"
    assert results_jsonl.exists()
    rows = [json.loads(line) for line in results_jsonl.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "synthetic"
    assert row["model"] == "fake-model"
    for key in ("mse", "mae", "r2", "spearman"):
        assert key in row["metrics"]
    for key in (
        "prepare_model_s",
        "prepare_data_s",
        "inference_s",
        "train_head_s",
        "eval_s",
        "peak_gpu_mem_mb",
    ):
        assert key in row["timing"]
    assert row["epochs_run"] >= 1
