"""Per-candidate runner: extract embeddings, train head, score on test."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from mlsys.datasets.registry import REQUIRED_SPLITS
from mlsys.finetune import FinetuneConfig, train_full_model
from mlsys.head import EpochCallback, FCHead, HeadTrainConfig, target_stats, train_head
from mlsys.models.backbone import TrainableBackbone
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
    strategy: str = "frozen"
    extras: dict[str, Any] = field(default_factory=dict)
    # Per-epoch pre-clip grad-norm curves (mean/max of step norms); finetune rows only.
    grad_norm_curve: list[float] = field(default_factory=list)
    grad_norm_max_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dataset": self.dataset,
            "model": self.model,
            "strategy": self.strategy,
            "metrics": self.metrics.to_dict(),
            "timing": self.timing,
            "head_train_curve": self.head_train_curve,
            "head_val_curve": self.head_val_curve,
            "epochs_run": self.epochs_run,
        }
        if self.grad_norm_curve:
            out["grad_norm_curve"] = self.grad_norm_curve
            out["grad_norm_max_curve"] = self.grad_norm_max_curve
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


def make_seeds(length: int) -> list[int]:
    return [int(torch.randint(0, 2**32 - 1, (1,)).item()) for _ in range(length)]


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
    seeds: Sequence[int | None] | None = None,
    epoch_callback: EpochCallback | None = None,
) -> RunRecord:
    """Train an FC head on `dataset` using embeddings from `spec`'s backbone, score on test.

    The head is trained `head_repeats` times with the provided seeds (or random ones if not given)
    and the test predictions are averaged, reducing variance.
    """
    head_config = head_config or HeadTrainConfig()
    reset_peak_gpu_memory()
    if seeds is None:
        seeds = make_seeds(head_repeats)
    timer = Timer(label=f"frozen:{spec.name}")

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
        # Z-score targets on train-split stats (issue #32): heads train against a
        # zero-mean/unit-variance target; predictions are mapped back before metrics
        # so mse/mae stay in original units (r2/spearman are affine-invariant).
        # Train/val curves in results.jsonl are therefore in standardized units.
        # `standardize_targets=False` keeps the identity transform (raw targets).
        target_mean, target_std = (
            target_stats(y_train) if head_config.standardize_targets else (0.0, 1.0)
        )
        y_train_z = (y_train - target_mean) / target_std
        y_val_z = (y_val - target_mean) / target_std
        head_results = [
            train_head(
                x_train,
                y_train_z,
                x_val,
                y_val_z,
                head_config,
                seeds[i],
                epoch_callback=epoch_callback,
            )
            for i in range(head_repeats)
        ]

    with timer.section("eval_s"):
        with torch.inference_mode():
            all_preds = np.mean(
                [r.head(x_test).detach().cpu().numpy() for r in head_results],
                axis=0,
            )
        all_preds = all_preds * target_std + target_mean
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
            "head_type": _head_type(head_config),
            "head_repeats": head_repeats,
            "target_mean": target_mean,
            "target_std": target_std,
            # Inert on the linear path (no activation there), but recorded on every
            # row so frozen and finetune keep one column schema.
            "activation": head_config.activation,
        },
    )


def _head_type(head_config: HeadTrainConfig) -> str:
    # Mirror FCHead's own linear/mlp decision (hidden None *or* <= 0 -> linear).
    return "mlp" if head_config.hidden and head_config.hidden > 0 else "linear"


def finetune_candidate(
    dataset: LoadedDataset,
    spec: ModelSpec,
    *,
    device: str = "cpu",
    batch_size: int = 64,
    head_config: HeadTrainConfig | None = None,
    finetune_config: FinetuneConfig | None = None,
    head_repeats: int = 1,
    epoch_callback: EpochCallback | None = None,
) -> RunRecord:
    """Unfreeze the backbone and train backbone+head jointly, then score on test.

    This is the expensive ground-truth signal ``t(m, D)`` for the regret metric. Inference
    is fused into training (``train_head_s``), so ``inference_s`` stays 0. ``head_repeats``
    defaults to 1: re-fine-tuning a full backbone N times is too costly, so regret's
    expectations collapse to point estimates (see REGRET.md note 2).

    Static / non-trainable backbones (``can_finetune=False``, e.g. model2vec) can't be
    fine-tuned, so we fall back to the frozen :func:`score_candidate` score and tag the row
    with ``finetune_skipped=True`` (finetune score == frozen score for them).
    """
    assert head_repeats == 1, (
        "finetune runs once per model; increase head_repeats only after updating REGRET.md note 2"
    )
    head_config = head_config or HeadTrainConfig()
    finetune_config = finetune_config or FinetuneConfig()
    reset_peak_gpu_memory()
    timer = Timer(label=f"finetune:{spec.name}")

    with timer.section("prepare_model_s"):
        backbone = build_backbone(spec, device=device)

    if not getattr(backbone, "can_finetune", False):
        # Non-trainable backbone: finetune is a no-op, so reuse the frozen path.
        release_gpu_memory(device)
        record = score_candidate(
            dataset,
            spec,
            device=device,
            batch_size=batch_size,
            head_config=head_config,
            head_repeats=head_repeats,
            epoch_callback=epoch_callback,
        )
        record.strategy = "finetune"
        record.extras["finetune_skipped"] = True
        # Keep the grad-stat columns present on every finetune row so the pass's
        # W&B table / CSV schema doesn't depend on which model comes first.
        record.extras.update(
            {
                "grad_clipping": finetune_config.grad_clipping,
                "grad_norm_mean": None,
                "grad_norm_max_overall": None,
            }
        )
        return record

    # Past the guard, the backbone is fine-tunable; narrow the type for the trainer.
    trainable = cast(TrainableBackbone, backbone)

    with timer.section("prepare_data_s"):
        rows = {split: list(dataset.split(split)) for split in REQUIRED_SPLITS}

    head = FCHead(
        in_dim=backbone.embedding_dim,
        hidden=head_config.hidden,
        activation=head_config.activation,
    ).to(device)

    with timer.section("train_head_s"):
        # Inference is fused into the joint loop, so inference_s stays 0 for finetune.
        result = train_full_model(
            trainable, head, rows, head_config, finetune_config, device, epoch_callback
        )

    with timer.section("eval_s"):
        trainable.eval()
        result.head.eval()
        bs = finetune_config.batch_size
        test_rows = rows["test"]
        with torch.inference_mode():
            preds = [
                result.head(backbone.encode([r.text for r in test_rows[s : s + bs]]))
                .detach()
                .cpu()
                .numpy()
                for s in range(0, len(test_rows), bs)
            ]
        # The joint loop trained in z-space (issue #32); map predictions back to
        # original units before metrics.
        all_preds = np.concatenate(preds) if preds else np.zeros((0,))
        all_preds = all_preds * result.target_std + result.target_mean
        y_test = np.array([r.target for r in test_rows], dtype=np.float64)
        metrics = regression_metrics(y_test, all_preds)

    timer.record_peak_gpu_mb()

    return RunRecord(
        dataset=dataset.spec.name,
        model=spec.name,
        metrics=metrics,
        timing=timer.breakdown.to_dict(),
        head_train_curve=result.train_curve,
        head_val_curve=result.val_curve,
        epochs_run=result.epochs_run,
        strategy="finetune",
        grad_norm_curve=result.grad_norm_curve,
        grad_norm_max_curve=result.grad_norm_max_curve,
        extras={
            "embedding_dim": spec.embedding_dim,
            "head_type": _head_type(head_config),
            "head_repeats": head_repeats,
            "target_mean": result.target_mean,
            "target_std": result.target_std,
            "activation": head_config.activation,
            # Scalar grad-norm summary for the results table / CSV; the per-epoch
            # curves stay in results.jsonl (and stream live via the epoch callback).
            # ``grad_norm_mean`` averages the per-epoch step-norm means; since every
            # epoch runs the same number of steps this equals the mean over all steps.
            # ``grad_norm_max_overall`` is the max across every epoch (vs. the live
            # per-epoch ``grad_norm_max`` streamed to W&B).
            "grad_clipping": finetune_config.grad_clipping,
            "grad_norm_mean": (
                sum(result.grad_norm_curve) / len(result.grad_norm_curve)
                if result.grad_norm_curve
                else None
            ),
            "grad_norm_max_overall": (
                max(result.grad_norm_max_curve) if result.grad_norm_max_curve else None
            ),
        },
    )
