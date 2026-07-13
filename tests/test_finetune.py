"""Joint finetune loop trains the backbone; finetune_candidate produces a RunRecord;
non-trainable backbones fall back to the frozen score. CPU-only, tiny dummy backbone."""

from __future__ import annotations

from typing import cast

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlsys.datasets import LoadedDataset, Row  # noqa: E402
from mlsys.datasets.registry import DatasetSpec  # noqa: E402
from mlsys.finetune import FinetuneConfig, train_full_model  # noqa: E402
from mlsys.head import FCHead, HeadTrainConfig  # noqa: E402
from mlsys.models.backbone import TrainableBackbone  # noqa: E402
from mlsys.models.registry import _ADAPTERS, ModelSpec, register_adapter  # noqa: E402
from mlsys.search.metrics import RegressionMetrics  # noqa: E402
from mlsys.search.runner import finetune_candidate  # noqa: E402


def _featurize(texts: list[str], device: str) -> torch.Tensor:
    rows = []
    for t in texts:
        v = torch.zeros(64)
        v[abs(hash(t)) % 64] = 1.0
        rows.append(v)
    return torch.stack(rows).to(device)


class _TrainableFakeBackbone(nn.Module):
    """A tiny real nn.Module so backbone params genuinely receive gradients."""

    can_finetune = True

    def __init__(self, spec: ModelSpec, device: str) -> None:
        super().__init__()
        self.name = spec.name
        self.embedding_dim = spec.embedding_dim
        self._device = device
        self.proj = nn.Linear(64, spec.embedding_dim)
        self.to(device)

    def encode(self, texts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            return self.proj(_featurize(texts, self._device))

    def encode_trainable(self, texts: list[str]) -> torch.Tensor:
        return self.proj(_featurize(texts, self._device))


class _StaticFakeBackbone:
    """model2vec-style: no trainable weights -> falls back to the frozen score."""

    can_finetune = False

    def __init__(self, spec: ModelSpec, device: str) -> None:
        self.name = spec.name
        self.embedding_dim = spec.embedding_dim
        self._device = device
        torch.manual_seed(0)
        self._proj = torch.randn(64, spec.embedding_dim)

    def encode(self, texts: list[str]) -> torch.Tensor:
        return (_featurize(texts, self._device) @ self._proj).to(self._device)

    def encode_trainable(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def parameters(self) -> list[object]:
        return []

    def train(self) -> None: ...

    def eval(self) -> None: ...


class _FakeSplit:
    def __init__(self, rows: list[Row]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


class _FakeDataset:
    def __init__(self, name: str = "synthetic") -> None:
        self.spec = DatasetSpec(
            name=name,
            hf_repo="local/fake",
            splits={"train": "train", "val": "val", "test": "test"},
            target_column="y",
            target_type="regression",
            text_template="{text}",
        )
        self.splits = {
            "train": _FakeSplit([Row(text=f"t{i}", target=float(i % 5)) for i in range(20)]),
            "val": _FakeSplit([Row(text=f"v{i}", target=float(i % 5)) for i in range(8)]),
            "test": _FakeSplit([Row(text=f"x{i}", target=float(i % 5)) for i in range(8)]),
        }

    def split(self, name: str):
        return self.splits[name]


def _rows(ds: _FakeDataset) -> dict[str, list[Row]]:
    return {s: list(ds.split(s)) for s in ("train", "val", "test")}


def test_lr_schedule_warms_up_then_decays() -> None:
    # 3 warmup steps (10% of 30) then linear decay to ~0 by the last step.
    torch.manual_seed(0)
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="x", embedding_dim=8)
    backbone = _TrainableFakeBackbone(spec, "cpu")
    head = FCHead(in_dim=8, hidden=None)

    lrs: list[float] = []
    orig_step = torch.optim.AdamW.step

    def _record_and_step(self: torch.optim.AdamW, *args: object, **kwargs: object) -> object:
        lrs.append(self.param_groups[0]["lr"])
        return orig_step(self, *args, **kwargs)

    torch.optim.AdamW.step = _record_and_step  # type: ignore[method-assign]
    try:
        train_full_model(
            cast(TrainableBackbone, backbone),
            head,
            _rows(_FakeDataset()),
            HeadTrainConfig(),
            FinetuneConfig(epochs=6, batch_size=4, backbone_lr=2e-5, head_lr=2e-5),
            "cpu",
        )
    finally:
        torch.optim.AdamW.step = orig_step  # type: ignore[method-assign]

    # 20 train rows / batch_size 4 = 5 steps/epoch * 6 epochs = 30 steps total.
    assert len(lrs) == 30
    assert lrs[0] < lrs[2] < lrs[3], "LR should ramp up during warmup"
    assert lrs[-1] < lrs[3], "LR should decay after warmup"
    assert lrs[-1] < 1e-6, "LR should decay close to 0 by the final step"


def test_train_full_model_updates_backbone_params() -> None:
    torch.manual_seed(0)
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="x", embedding_dim=8)
    backbone = _TrainableFakeBackbone(spec, "cpu")
    head = FCHead(in_dim=8, hidden=None)
    before = backbone.proj.weight.detach().clone()

    result = train_full_model(
        cast(TrainableBackbone, backbone),
        head,
        _rows(_FakeDataset()),
        HeadTrainConfig(),
        FinetuneConfig(epochs=5, batch_size=4, backbone_lr=1e-2, head_lr=1e-2),
        "cpu",
    )

    assert not torch.allclose(before, backbone.proj.weight), "backbone params did not update"
    assert result.epochs_run >= 1
    assert len(result.train_curve) == len(result.val_curve)


def test_warmup_trains_head_but_not_backbone() -> None:
    # warmup_epochs > 0 with no joint epochs: the head moves, the backbone stays frozen.
    torch.manual_seed(0)
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="x", embedding_dim=8)
    backbone = _TrainableFakeBackbone(spec, "cpu")
    head = FCHead(in_dim=8, hidden=None)
    backbone_before = backbone.proj.weight.detach().clone()
    head_before = [p.detach().clone() for p in head.parameters()]

    train_full_model(
        cast(TrainableBackbone, backbone),
        head,
        _rows(_FakeDataset()),
        HeadTrainConfig(),
        FinetuneConfig(epochs=0, warmup_epochs=2, batch_size=4),
        "cpu",
    )

    assert torch.allclose(backbone_before, backbone.proj.weight), (
        "backbone changed during head-only warmup"
    )
    assert any(
        not torch.equal(b, p) for b, p in zip(head_before, head.parameters(), strict=True)
    ), "head params did not change during warmup"


@pytest.fixture
def trainable_loader():
    register_adapter("trainable_fake", lambda spec, device: _TrainableFakeBackbone(spec, device))
    yield
    _ADAPTERS.pop("trainable_fake", None)


@pytest.fixture
def static_loader():
    register_adapter("static_fake", lambda spec, device: _StaticFakeBackbone(spec, device))
    yield
    _ADAPTERS.pop("static_fake", None)


def test_finetune_candidate_produces_record(trainable_loader) -> None:
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="trainable_fake", embedding_dim=8)
    record = finetune_candidate(
        cast(LoadedDataset, _FakeDataset()),
        spec,
        device="cpu",
        finetune_config=FinetuneConfig(epochs=2, batch_size=4),
    )
    assert record.strategy == "finetune"
    assert isinstance(record.metrics, RegressionMetrics)
    # Finetune fuses inference into the training loop, so inference_s stays 0.
    assert record.timing["inference_s"] == 0.0
    assert record.timing["train_head_s"] > 0.0
    assert record.extras["head_repeats"] == 1


def test_finetune_candidate_seed_produces_identical_curves(trainable_loader) -> None:
    # Real backbones are pretrained (not randomly initialized), so torch.manual_seed(0)
    # here stands in for that fixed starting point; _TrainableFakeBackbone's nn.Linear
    # would otherwise draw a different random init on each finetune_candidate call.
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="trainable_fake", embedding_dim=8)
    finetune_config = FinetuneConfig(epochs=3, batch_size=4)

    torch.manual_seed(0)
    record1 = finetune_candidate(
        cast(LoadedDataset, _FakeDataset()),
        spec,
        device="cpu",
        finetune_config=finetune_config,
        seed=42,
    )
    torch.manual_seed(0)
    record2 = finetune_candidate(
        cast(LoadedDataset, _FakeDataset()),
        spec,
        device="cpu",
        finetune_config=finetune_config,
        seed=42,
    )

    assert record1.head_train_curve == record2.head_train_curve
    assert record1.head_val_curve == record2.head_val_curve


def test_finetune_candidate_with_warmup_keeps_timing_contract(trainable_loader) -> None:
    # Warmup cost lands in train_head_s (inference still fused into the joint loop -> 0).
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="trainable_fake", embedding_dim=8)
    record = finetune_candidate(
        cast(LoadedDataset, _FakeDataset()),
        spec,
        device="cpu",
        finetune_config=FinetuneConfig(epochs=2, batch_size=4, warmup_epochs=2),
    )
    assert record.timing["inference_s"] == 0.0
    assert record.timing["train_head_s"] > 0.0


def test_finetune_candidate_falls_back_for_static(static_loader) -> None:
    spec = ModelSpec(name="static", hf_repo="local/fake", loader="static_fake", embedding_dim=8)
    record = finetune_candidate(
        cast(LoadedDataset, _FakeDataset()),
        spec,
        device="cpu",
        head_config=HeadTrainConfig(epochs=2, batch_size=4),
    )
    assert record.strategy == "finetune"
    assert record.extras.get("finetune_skipped") is True
    assert isinstance(record.metrics, RegressionMetrics)
