"""Per-candidate runner: extract embeddings, train head, score on test."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mlsys.datasets.registry import REQUIRED_SPLITS
from mlsys.head import HeadTrainConfig, train_head
from mlsys.models.registry import ModelSpec, build_backbone
from mlsys.search.metrics import RegressionMetrics, regression_metrics
from mlsys.search.timing import Timer, reset_peak_gpu_memory

if TYPE_CHECKING:
    from mlsys.datasets import LoadedDataset, Row
    from mlsys.models.backbone import Backbone


@dataclass
class RunRecord:
    dataset: str
    model: str
    metrics: RegressionMetrics
    timing: dict[str, float]
    head_train_curve: list[float]
    head_val_curve: list[float]
    epochs_run: int
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dataset": self.dataset,
            "model": self.model,
            "metrics": self.metrics.to_dict(),
            "timing": self.timing,
            "head_train_curve": self.head_train_curve,
            "head_val_curve": self.head_val_curve,
            "epochs_run": self.epochs_run,
        }
        out.update(self.extras)
        return out


def _embed_split(
    backbone: Backbone,
    split: Iterable[Row],
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    feats: list[torch.Tensor] = []
    targets: list[float] = []
    batch_texts: list[str] = []
    batch_targets: list[float] = []
    for row in split:
        batch_texts.append(row.text)
        batch_targets.append(row.target)
        if len(batch_texts) == batch_size:
            with torch.inference_mode():
                emb = backbone.encode(batch_texts)
            feats.append(emb.detach().to(device, dtype=torch.float32))
            targets.extend(batch_targets)
            batch_texts.clear()
            batch_targets.clear()
    if batch_texts:
        with torch.inference_mode():
            emb = backbone.encode(batch_texts)
        feats.append(emb.detach().to(device, dtype=torch.float32))
        targets.extend(batch_targets)
    if not feats:
        return (
            torch.zeros((0, backbone.embedding_dim), device=device),
            torch.zeros((0,), device=device),
        )
    return torch.cat(feats, dim=0), torch.tensor(targets, dtype=torch.float32, device=device)


def release_gpu_memory(device: str) -> None:
    """Drop dereferenced GPU tensors and return cached blocks to the allocator.

    Called between candidates so a freed backbone's CUDA memory doesn't linger
    as reserved/fragmented blocks across the ~N-model loop. Without this the
    caching allocator accumulates per-model fragments and eventually OOMs on a
    contiguous allocation even though each candidate's live footprint is small.
    """
    import gc

    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _average_curves(curves: list[list[float]]) -> list[float]:
    """Average training curves of different lengths by padding each with its last value."""
    if not curves:
        return []
    max_len = max(len(c) for c in curves)
    padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
    return padded.mean(axis=0).tolist()


def score_candidate(
    dataset: LoadedDataset,
    spec: ModelSpec,
    *,
    device: str = "cpu",
    batch_size: int = 64,
    head_config: HeadTrainConfig | None = None,
    head_repeats: int = 3,
) -> RunRecord:
    """Train an FC head on `dataset` using embeddings from `spec`'s backbone, score on test.

    The head is trained `head_repeats` times from different random initialisations and the
    test predictions are averaged, reducing variance from the small (~50-sample) train split.
    """
    head_config = head_config or HeadTrainConfig()
    reset_peak_gpu_memory()
    timer = Timer()

    with timer.section("prepare_model_s"):
        backbone = build_backbone(spec, device=device)

    with timer.section("prepare_data_s"):
        # Materialise here so per-row text_template rendering (lazy in
        # _SplitView.__iter__) is attributed to prepare_data_s, not inference_s.
        rows = {split: list(dataset.split(split)) for split in REQUIRED_SPLITS}

    with timer.section("inference_s"):
        x_train, y_train = _embed_split(backbone, rows["train"], batch_size, device)
        x_val, y_val = _embed_split(backbone, rows["val"], batch_size, device)
        x_test, y_test = _embed_split(backbone, rows["test"], batch_size, device)

    with timer.section("train_head_s"):
        head_results = [
            train_head(x_train, y_train, x_val, y_val, head_config)
            for _ in range(head_repeats)
        ]

    with timer.section("eval_s"):
        with torch.inference_mode():
            all_preds = np.mean(
                [r.head(x_test).detach().cpu().numpy() for r in head_results],
                axis=0,
            )
        metrics = regression_metrics(y_test.detach().cpu().numpy(), all_preds)

    timer.record_peak_gpu_mb()

    return RunRecord(
        dataset=dataset.spec.name,
        model=spec.name,
        metrics=metrics,
        timing=timer.breakdown.to_dict(),
        head_train_curve=_average_curves([r.train_curve for r in head_results]),
        head_val_curve=_average_curves([r.val_curve for r in head_results]),
        epochs_run=round(sum(r.epochs_run for r in head_results) / len(head_results)),
        extras={
            "embedding_dim": spec.embedding_dim,
            # Mirror FCHead's own linear/mlp decision (hidden None *or* <= 0 -> linear).
            "head_type": "mlp" if head_config.hidden and head_config.hidden > 0 else "linear",
            "head_repeats": head_repeats,
        },
    )
