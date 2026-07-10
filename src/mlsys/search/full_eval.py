"""Strategy dispatch for model search.

Three strategies, all sharing the same ``RegressionMetrics`` and ``results.jsonl`` layout:

- ``frozen``   — train a fresh FC head on the **frozen** backbone (the cheap proxy / ranking
  signal). This is the original full-evaluation baseline.
- ``finetune`` — unfreeze the backbone and train backbone+head jointly (the expensive
  ground truth ``t(m, D)`` for regret).
- ``full_eval``— run both passes over the whole pool, then compute the regret-vs-budget
  curve (frozen r2 ranks; finetune r2 scores) and write ``regret.json``.

Each ``(dataset, model, strategy)`` is one row in ``results.jsonl`` (the ``strategy`` field
distinguishes frozen vs finetune rows in a full_eval run).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from mlsys.finetune import FinetuneConfig
from mlsys.head import EpochCallback, HeadTrainConfig
from mlsys.io import JsonlWriter, results_path
from mlsys.models.registry import ModelSpec, load_specs
from mlsys.search.regret import summarize_regret, write_regret_json
from mlsys.search.runner import (
    RunRecord,
    finetune_candidate,
    make_seeds,
    release_gpu_memory,
    score_candidate,
)
from mlsys.search.summarize import (
    SummarizeConfig,
    finetune_summarization_candidate,
    score_summarization_candidate,
)

if TYPE_CHECKING:
    from mlsys.datasets import LoadedDataset

log = logging.getLogger(__name__)

STRATEGIES = ("frozen", "finetune", "full_eval")

# The single ranking/scoring metric per task type (both higher-is-better). Used to build
# the regret curve and label regret.json; the regret math itself is metric-agnostic.
PRIMARY_METRIC = {"regression": "r2", "summarization": "rougeL"}

# Wall-clock substep fields summed for the per-candidate "done in Ns" log line.
_TIMING_KEYS = (
    "prepare_model_s",
    "prepare_data_s",
    "inference_s",
    "train_head_s",
    "eval_s",
)


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


def _run_pass(
    specs: list[ModelSpec],
    *,
    out_path: Path,
    candidate_fn: Callable[[ModelSpec, EpochCallback | None], RunRecord],
    device: str,
    wandb_run: object | None,
    table_name: str,
    pass_name: str,
) -> list[RunRecord]:
    """Score every candidate with ``candidate_fn``, appending each row to ``out_path``."""
    total = len(specs)
    log.info("%s pass: %d model(s) on %s", pass_name, total, device)
    records: list[RunRecord] = []
    with JsonlWriter(out_path) as writer:
        for i, spec in enumerate(specs, start=1):
            log.info("[%s %d/%d] %s — starting", pass_name, i, total, spec.name)
            # Stream per-epoch curves live to W&B (no-op when W&B is off).
            cb = (
                _make_epoch_logger(spec.name, pass_name, wandb_run)
                if wandb_run is not None
                else None
            )
            record = candidate_fn(spec, cb)
            total_s = sum(record.timing.get(k, 0.0) for k in _TIMING_KEYS)
            metrics_str = " ".join(f"{k}={v:.4f}" for k, v in record.metrics.to_dict().items())
            log.info(
                "[%s %d/%d] %s — done: %s in %.1fs",
                pass_name,
                i,
                total,
                spec.name,
                metrics_str,
                total_s,
            )
            writer.write(record.to_dict())
            records.append(record)
            # Per-candidate GPU tensors are now dereferenced; reclaim them so the
            # next candidate starts clean.
            release_gpu_memory(device)
    log.info("%s pass complete: %d row(s) written", pass_name, len(records))
    if wandb_run is not None:
        _log_results_table(records, name=table_name)
    return records


def run_frozen(
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
    """Frozen-proxy pass over every candidate; append to results.jsonl.

    Regression trains a fresh FC head on the frozen encoder; summarization trains only the
    LM/generation head (whole seq2seq body frozen). Dispatched on ``dataset.spec.target_type``.
    """
    specs = _resolve_models(model_names)

    if dataset.spec.target_type == "summarization":
        cfg = SummarizeConfig()

        # Summarization candidates don't stream per-epoch curves yet, so the callback is
        # accepted (to match _run_pass's signature) but unused.
        def candidate_fn(spec: ModelSpec, _cb: EpochCallback | None) -> RunRecord:
            return score_summarization_candidate(dataset, spec, device=device, config=cfg)
    else:
        seeds = make_seeds(head_repeats)

        def candidate_fn(spec: ModelSpec, _cb: EpochCallback | None) -> RunRecord:
            return score_candidate(
                dataset,
                spec,
                device=device,
                batch_size=batch_size,
                head_config=head_config,
                head_repeats=head_repeats,
                seeds=seeds,
                epoch_callback=_cb,
            )

    return _run_pass(
        specs,
        out_path=results_path(output_dir),
        device=device,
        wandb_run=wandb_run,
        table_name="results_frozen",
        pass_name="frozen",
        candidate_fn=candidate_fn,
    )


def run_finetune(
    dataset: LoadedDataset,
    output_dir: str | Path,
    *,
    model_names: Iterable[str] | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    head_config: HeadTrainConfig | None = None,
    finetune_config: FinetuneConfig | None = None,
    head_repeats: int = 1,
    wandb_run: object | None = None,
) -> list[RunRecord]:
    """Finetune pass: unfreeze + jointly train every candidate; append to results.jsonl.

    Regression unfreezes the encoder + head; summarization unfreezes the whole seq2seq model.
    Dispatched on ``dataset.spec.target_type``.
    """
    specs = _resolve_models(model_names)

    if dataset.spec.target_type == "summarization":
        cfg = SummarizeConfig()

        def candidate_fn(spec: ModelSpec, _cb: EpochCallback | None) -> RunRecord:
            return finetune_summarization_candidate(dataset, spec, device=device, config=cfg)
    else:

        def candidate_fn(spec: ModelSpec, _cb: EpochCallback | None) -> RunRecord:
            return finetune_candidate(
                dataset,
                spec,
                device=device,
                batch_size=batch_size,
                head_config=head_config,
                finetune_config=finetune_config,
                head_repeats=head_repeats,
                epoch_callback=_cb,
            )

    return _run_pass(
        specs,
        out_path=results_path(output_dir),
        device=device,
        wandb_run=wandb_run,
        table_name="results_finetune",
        pass_name="finetune",
        candidate_fn=candidate_fn,
    )


def run_full_eval(
    dataset: LoadedDataset,
    output_dir: str | Path,
    *,
    model_names: Iterable[str] | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    head_config: HeadTrainConfig | None = None,
    finetune_config: FinetuneConfig | None = None,
    head_repeats: int = 3,
    wandb_run: object | None = None,
) -> list[RunRecord]:
    """Run both passes over the pool, then compute regret and write ``regret.json``.

    The proxy ranking is the frozen r2 (desc); the ground-truth scores are the finetune r2.
    Returns frozen rows followed by finetune rows.
    """
    log.info("full_eval on %s: phase 1/3 — frozen (proxy ranking)", dataset.spec.name)
    frozen = run_frozen(
        dataset,
        output_dir,
        model_names=model_names,
        device=device,
        batch_size=batch_size,
        head_config=head_config,
        head_repeats=head_repeats,
        wandb_run=wandb_run,
    )
    log.info("full_eval on %s: phase 2/3 — finetune (ground truth)", dataset.spec.name)
    finetune = run_finetune(
        dataset,
        output_dir,
        model_names=model_names,
        device=device,
        batch_size=batch_size,
        head_config=head_config,
        finetune_config=finetune_config,
        head_repeats=1,
        wandb_run=wandb_run,
    )

    log.info("full_eval on %s: phase 3/3 — computing regret curve", dataset.spec.name)
    # Metric-agnostic: pick the higher-is-better score per task type (r2 / rougeL) and read
    # it from each record's to_dict() so the same code path serves both tasks.
    key = PRIMARY_METRIC[dataset.spec.target_type]
    frozen_scores = {r.model: r.metrics.to_dict()[key] for r in frozen}
    finetune_scores = {r.model: r.metrics.to_dict()[key] for r in finetune}
    summary = summarize_regret(dataset.spec.name, frozen_scores, finetune_scores, metric=key)
    log.info("proxy ranking (frozen %s desc): %s", key, ", ".join(summary.proxy_ranking))

    write_regret_json(output_dir, summary)
    if wandb_run is not None:
        _log_regret_to_wandb(summary.curve)
    return frozen + finetune


def run_strategy(
    strategy: str,
    dataset: LoadedDataset,
    output_dir: str | Path,
    *,
    model_names: Iterable[str] | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    head_config: HeadTrainConfig | None = None,
    finetune_config: FinetuneConfig | None = None,
    head_repeats: int = 3,
    wandb_run: object | None = None,
) -> list[RunRecord]:
    """Dispatch to the named strategy. ``strategy`` must be one of :data:`STRATEGIES`."""
    if strategy == "frozen":
        return run_frozen(
            dataset,
            output_dir,
            model_names=model_names,
            device=device,
            batch_size=batch_size,
            head_config=head_config,
            head_repeats=head_repeats,
            wandb_run=wandb_run,
        )
    if strategy == "finetune":
        return run_finetune(
            dataset,
            output_dir,
            model_names=model_names,
            device=device,
            batch_size=batch_size,
            head_config=head_config,
            finetune_config=finetune_config,
            head_repeats=1,
            wandb_run=wandb_run,
        )
    if strategy == "full_eval":
        return run_full_eval(
            dataset,
            output_dir,
            model_names=model_names,
            device=device,
            batch_size=batch_size,
            head_config=head_config,
            finetune_config=finetune_config,
            head_repeats=head_repeats,
            wandb_run=wandb_run,
        )
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")


def _result_row(record: RunRecord) -> dict[str, object]:
    return {
        "model": record.model,
        "dataset": record.dataset,
        "strategy": record.strategy,
        **record.metrics.to_dict(),
        **record.timing,
        "epochs_run": record.epochs_run,
        **record.extras,
    }


def _make_epoch_logger(model: str, strategy: str, wandb_run: object) -> EpochCallback:
    """Build a per-epoch callback that streams train/val MSE live to W&B.

    Keyed by ``<model>/<strategy>/{train,val}_mse`` so passes (frozen vs finetune) land on
    distinct series on the same run. For frozen's repeated heads all repeats stream under the
    same keys — the curves are a visual anchor; the averaged results table is authoritative.

    Each model/pass gets its own ``<model>/<strategy>/epoch`` step-metric so its MSE curves
    plot against that model's own 0-based epoch. Without this the charts default to W&B's
    run-global auto-step, which keeps incrementing across models, so the next model's curve
    would start where the previous one left off (e.g. step 25) instead of resetting to 0. The
    plain ``epoch`` scalar is still logged for the single global epoch chart (x=step, y=epoch).
    """
    del wandb_run  # Presence already gated by the caller; the closure re-imports wandb.
    import wandb  # type: ignore[import-not-found]

    step_metric = f"{model}/{strategy}/epoch"
    wandb.define_metric(step_metric)
    wandb.define_metric(f"{model}/{strategy}/train_mse", step_metric=step_metric)
    wandb.define_metric(f"{model}/{strategy}/val_mse", step_metric=step_metric)

    def _log(epoch: int, train_mse: float, val_mse: float) -> None:
        # Imported lazily so the unused-without-flag path stays free of the dep.
        import wandb  # type: ignore[import-not-found]

        wandb.log(
            {
                f"{model}/{strategy}/train_mse": train_mse,
                f"{model}/{strategy}/val_mse": val_mse,
                step_metric: epoch,
                "epoch": epoch,
            }
        )

    return _log


def _log_results_table(records: list[RunRecord], *, name: str = "results") -> None:
    """Log one results table for the whole pass (one row per model), not N tables."""
    if not records:
        return
    import wandb  # type: ignore[import-not-found]

    columns = list(_result_row(records[0]).keys())
    data = [[_result_row(r)[c] for c in columns] for r in records]
    wandb.log({name: wandb.Table(columns=columns, data=data)})


def _log_regret_to_wandb(curve: list) -> None:
    """Log the regret-vs-budget curve as a line plot + ``regret@1`` summary scalars."""
    import wandb  # type: ignore[import-not-found]

    table = wandb.Table(columns=["budget", "regret", "normalized_regret"])
    for p in curve:
        table.add_data(p.budget, p.regret, p.normalized_regret)
    wandb.log(
        {"regret_vs_budget": wandb.plot.line(table, "budget", "regret", title="Regret vs budget B")}
    )
    if curve and wandb.run is not None:
        wandb.run.summary["regret@1"] = curve[0].regret
        wandb.run.summary["normalized_regret@1"] = curve[0].normalized_regret
