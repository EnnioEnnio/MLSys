"""End-to-end smoke test: synthetic dataset + a fake backbone via a custom adapter.

Real wine_reviews + potion-base-8M would need network access and bring in
heavy deps (datasets, model2vec). The runner contract is what we want to lock
down here, so we register a fake-backbone adapter under a custom loader name
and feed it a tiny in-memory dataset that mimics the LoadedDataset shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

torch = pytest.importorskip("torch")

from mlsys.datasets import LoadedDataset, Row  # noqa: E402
from mlsys.datasets.registry import DatasetSpec  # noqa: E402
from mlsys.head import HeadTrainConfig  # noqa: E402
from mlsys.models.registry import _ADAPTERS, ModelSpec, register_adapter  # noqa: E402
from mlsys.search.full_eval import run_frozen  # noqa: E402

if TYPE_CHECKING:
    from mlsys.models.backbone import Backbone


class _FakeBackbone:
    name: str
    embedding_dim: int

    def __init__(self, spec: ModelSpec, device: str) -> None:
        self.name = spec.name
        self.embedding_dim = spec.embedding_dim
        self._device = device
        torch.manual_seed(0)
        self._proj = torch.randn(64, spec.embedding_dim)

    def encode(self, texts: list[str]):
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


class _FakeGenerativeBackbone:
    name: str
    can_finetune: bool = True

    def __init__(self, spec: ModelSpec, device: str) -> None:
        self.name = spec.name
        self._device = device
        # A real parameter so teacher_forcing_loss is differentiable and AdamW can step.
        self._param = torch.nn.Parameter(torch.zeros(1))
        self._trainable = [self._param]

    def teacher_forcing_loss(self, sources: list[str], targets: list[str]):
        # Trivial trainable scalar loss (depends on the param so backward populates grad).
        return (self._param**2).sum() + float(len(sources))

    def generate(self, sources: list[str]) -> list[str]:
        return ["a canned summary" for _ in sources]

    def set_trainable(self, scope: str) -> None:
        self._trainable = [self._param]

    def trainable_parameters(self):
        return iter(self._trainable)

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


class _FakeSummarizationDataset:
    def __init__(self, name: str) -> None:
        n_per_split = 12
        self.spec = DatasetSpec(
            name=name,
            hf_repo="local/fake",
            splits={"train": "train", "val": "val", "test": "test"},
            target_column="summary",
            target_type="summarization",
            text_template="{dialogue}",
        )
        self.splits = {
            "train": _FakeSplit(
                [Row(text=f"dialogue {i}", target=f"summary {i}") for i in range(n_per_split)]
            ),
            "val": _FakeSplit(
                [Row(text=f"vd {i}", target=f"vs {i}") for i in range(n_per_split // 2)]
            ),
            "test": _FakeSplit(
                [Row(text=f"xd {i}", target=f"xs {i}") for i in range(n_per_split // 2)]
            ),
        }

    def split(self, name: str):
        return self.splits[name]


@pytest.fixture
def fake_loader_registered():
    register_adapter("fake_loader", lambda spec, device: _FakeBackbone(spec, device))
    yield
    _ADAPTERS.pop("fake_loader", None)


@pytest.fixture
def fake_generative_loader_registered():
    # GenerativeBackbone isn't a (regression) Backbone; the registry is typed for the
    # encoder surface, so cast at the boundary (matches the real seq2seq adapter).
    register_adapter(
        "fake_gen_loader",
        lambda spec, device: cast("Backbone", _FakeGenerativeBackbone(spec, device)),
    )
    yield
    _ADAPTERS.pop("fake_gen_loader", None)


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
    records = run_frozen(
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


def test_full_eval_writes_summarization_row(
    tmp_path: Path, fake_generative_loader_registered, monkeypatch
) -> None:
    pytest.importorskip("rouge_score")
    import sys

    fake_spec = ModelSpec(
        name="fake-gen",
        hf_repo="local/fake",
        loader="fake_gen_loader",
        embedding_dim=8,  # nominal; ignored by the summarization runner
    )
    fe_module = sys.modules["mlsys.search.full_eval"]
    monkeypatch.setattr(fe_module, "load_specs", lambda: {"fake-gen": fake_spec})

    dataset = _FakeSummarizationDataset(name="synthetic_sum")
    out_dir = tmp_path / "run"
    records = run_frozen(cast(LoadedDataset, dataset), output_dir=out_dir, device="cpu")

    assert len(records) == 1
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "synthetic_sum"
    assert row["model"] == "fake-gen"
    assert row["strategy"] == "frozen"
    assert row["task"] == "summarization"
    for key in ("rouge1", "rouge2", "rougeL"):
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
    assert row["timing"]["inference_s"] == 0.0
