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


def test_grad_norms_measured_even_without_clipping() -> None:
    # grad_clipping=0 disables rescaling but still measures the pre-clip norm each
    # step, so the per-epoch curves are populated for threshold tuning.
    torch.manual_seed(0)
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="x", embedding_dim=8)
    backbone = _TrainableFakeBackbone(spec, "cpu")
    head = FCHead(in_dim=8, hidden=None)

    result = train_full_model(
        cast(TrainableBackbone, backbone),
        head,
        _rows(_FakeDataset()),
        HeadTrainConfig(),
        FinetuneConfig(epochs=3, batch_size=4, grad_clipping=0.0),
        "cpu",
    )

    assert len(result.grad_norm_curve) == len(result.train_curve)
    assert len(result.grad_norm_max_curve) == len(result.train_curve)
    assert all(n > 0 for n in result.grad_norm_curve)
    assert all(
        mx >= mean
        for mean, mx in zip(result.grad_norm_curve, result.grad_norm_max_curve, strict=True)
    )


def test_negative_grad_clipping_rejected() -> None:
    # load-loud: a negative threshold is nonsensical and must fail rather than silently
    # degrade to measure-only (the >0 gate would otherwise treat it as inf).
    with pytest.raises(ValueError, match="grad_clipping"):
        FinetuneConfig(grad_clipping=-1.0)


@pytest.mark.parametrize("grad_clipping,expected_max_norm", [(0.5, 0.5), (0.0, float("inf"))])
def test_grad_clipping_forwards_max_norm(grad_clipping, expected_max_norm, monkeypatch) -> None:
    # The configured threshold reaches clip_grad_norm_ verbatim; 0 degrades to inf
    # (measure-only, no rescaling).
    torch.manual_seed(0)
    seen: list[float] = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *args, **kwargs):
        seen.append(float(max_norm))
        return real(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="x", embedding_dim=8)

    train_full_model(
        cast(TrainableBackbone, _TrainableFakeBackbone(spec, "cpu")),
        FCHead(in_dim=8, hidden=None),
        _rows(_FakeDataset()),
        HeadTrainConfig(),
        FinetuneConfig(epochs=1, batch_size=4, grad_clipping=grad_clipping),
        "cpu",
    )

    assert seen and set(seen) == {expected_max_norm}


def test_epoch_callback_receives_grad_norms() -> None:
    # The joint loop attaches per-epoch grad norms as callback extras (streamed to W&B).
    torch.manual_seed(0)
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="x", embedding_dim=8)
    seen: list[dict[str, float]] = []

    def cb(epoch: int, train_mse: float, val_mse: float, **extras: float) -> None:
        seen.append(extras)

    train_full_model(
        cast(TrainableBackbone, _TrainableFakeBackbone(spec, "cpu")),
        FCHead(in_dim=8, hidden=None),
        _rows(_FakeDataset()),
        HeadTrainConfig(),
        FinetuneConfig(epochs=2, batch_size=4),
        "cpu",
        epoch_callback=cb,
    )

    assert seen
    assert all(e["grad_norm"] > 0 and e["grad_norm_max"] >= e["grad_norm"] for e in seen)


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


def test_finetune_candidate_reports_grad_stats(trainable_loader) -> None:
    # Scalar grad-norm summaries land in extras (results table / CSV); the per-epoch
    # curves land on the record itself (results.jsonl).
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="trainable_fake", embedding_dim=8)
    record = finetune_candidate(
        cast(LoadedDataset, _FakeDataset()),
        spec,
        device="cpu",
        finetune_config=FinetuneConfig(epochs=2, batch_size=4, grad_clipping=1.0),
    )
    assert record.extras["grad_clipping"] == 1.0
    assert record.extras["grad_norm_mean"] > 0
    assert record.extras["grad_norm_max_overall"] >= record.extras["grad_norm_mean"]
    assert record.grad_norm_curve and record.grad_norm_max_curve
    assert record.to_dict()["grad_norm_curve"] == record.grad_norm_curve


def test_train_full_model_returns_target_stats() -> None:
    # The joint loop z-scores targets on train stats (issue #32) and hands the stats
    # back so callers can map predictions to original units.
    torch.manual_seed(0)
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="x", embedding_dim=8)

    result = train_full_model(
        cast(TrainableBackbone, _TrainableFakeBackbone(spec, "cpu")),
        FCHead(in_dim=8, hidden=None),
        _rows(_FakeDataset()),
        HeadTrainConfig(),
        FinetuneConfig(epochs=1, batch_size=4),
        "cpu",
    )

    y = torch.tensor([float(i % 5) for i in range(20)])
    assert result.target_mean == pytest.approx(float(y.mean()))
    assert result.target_std == pytest.approx(float(y.std()))


def test_train_full_model_standardization_opt_out() -> None:
    # standardize_targets=False (summarization pilot / pipeline reuse) keeps the
    # identity transform: raw targets in the loss, (0, 1) stats on the result.
    torch.manual_seed(0)
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="x", embedding_dim=8)

    result = train_full_model(
        cast(TrainableBackbone, _TrainableFakeBackbone(spec, "cpu")),
        FCHead(in_dim=8, hidden=None),
        _rows(_FakeDataset()),
        HeadTrainConfig(standardize_targets=False),
        FinetuneConfig(epochs=1, batch_size=4),
        "cpu",
    )

    assert result.target_mean == 0.0
    assert result.target_std == 1.0


def test_finetune_candidate_unscales_predictions(trainable_loader) -> None:
    # Wine-style targets (~85-89): the joint loop trains in z-space, so a missing
    # inversion would leave predictions near 0 and mse at ~87^2 ≈ 7.5e3; in original
    # units even a mean-predicting head stays within the target variance.
    torch.manual_seed(0)
    spec = ModelSpec(name="fake", hf_repo="local/fake", loader="trainable_fake", embedding_dim=8)
    ds = _FakeDataset()
    ds.splits = {
        name: _FakeSplit([Row(text=r.text, target=85.0 + r.target) for r in split])
        for name, split in ds.splits.items()
    }

    record = finetune_candidate(
        cast(LoadedDataset, ds),
        spec,
        device="cpu",
        finetune_config=FinetuneConfig(epochs=2, batch_size=4),
    )

    assert record.metrics.mse < 100.0
    assert record.extras["target_mean"] == pytest.approx(87.0, abs=0.5)
    assert record.extras["target_std"] > 0.0


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
    # Grad-stat / target-stat columns stay schema-stable across skipped and trained
    # finetune rows (the frozen fallback records its own z-scoring stats).
    assert record.extras["grad_norm_mean"] is None
    assert record.extras["grad_clipping"] == FinetuneConfig().grad_clipping
    assert record.extras["target_std"] > 0.0
