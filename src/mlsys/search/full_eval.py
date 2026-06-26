"""v1 strategy: train and evaluate the head on every candidate model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from mlsys.head import HeadTrainConfig
from mlsys.io import JsonlWriter, results_path
from mlsys.models.registry import ModelSpec, load_specs
from mlsys.search.runner import RunRecord, release_gpu_memory, score_candidate

if TYPE_CHECKING:
    from mlsys.datasets import LoadedDataset


def _resolve_models(model_names: Iterable[str] | None) -> list[ModelSpec]:
    specs = load_specs()
    if model_names is None:
        return list(specs.values())
    selected: list[ModelSpec] = []
    for name in model_names:
        if name not in specs:
            raise KeyError(f"unknown model {name!r}; known: {sorted(specs)}")
        selected.append(specs[name])
    return selected


def full_eval(
    dataset: LoadedDataset,
    output_dir: str | Path,
    *,
    model_names: Iterable[str] | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    head_config: HeadTrainConfig | None = None,
    head_repeats: int = 3,
    wandb_run: object | None = None,
) -> list[RunRecord]:
    """Score every (resolved) candidate; append each row to results.jsonl."""
    specs = _resolve_models(model_names)
    out_path = results_path(output_dir)
    records: list[RunRecord] = []
    with JsonlWriter(out_path) as writer:
        for spec in specs:
            record = score_candidate(
                dataset,
                spec,
                device=device,
                batch_size=batch_size,
                head_config=head_config,
                head_repeats=head_repeats,
            )
            writer.write(record.to_dict())
            records.append(record)
            if wandb_run is not None:
                _log_curves_to_wandb(record)
            # score_candidate's GPU tensors (backbone, embeddings, head) are now
            # dereferenced; reclaim them so the next candidate starts clean.
            release_gpu_memory(device)
    if wandb_run is not None:
        _log_results_table(records)
    return records


def _result_row(record: RunRecord) -> dict[str, object]:
    return {
        "model": record.model,
        "dataset": record.dataset,
        **record.metrics.to_dict(),
        **record.timing,
        "epochs_run": record.epochs_run,
        **record.extras,
    }


def _log_curves_to_wandb(record: RunRecord) -> None:
    # Imported lazily so the unused-without-flag path stays free of the dep.
    import wandb  # type: ignore[import-not-found]

    for epoch, (tr, va) in enumerate(
        zip(record.head_train_curve, record.head_val_curve, strict=False)
    ):
        wandb.log(
            {
                f"{record.model}/train_mse": tr,
                f"{record.model}/val_mse": va,
                "epoch": epoch,
            }
        )


def _log_results_table(records: list[RunRecord]) -> None:
    """Log one results table for the whole run (one row per model), not N tables."""
    if not records:
        return
    import wandb  # type: ignore[import-not-found]

    columns = list(_result_row(records[0]).keys())
    data = [[_result_row(r)[c] for c in columns] for r in records]
    wandb.log({"results": wandb.Table(columns=columns, data=data)})
