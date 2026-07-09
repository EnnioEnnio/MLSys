"""Summarization task path: LM-head frozen proxy + full fine-tune ground truth.

Parallel to the regression :mod:`mlsys.search.runner` / :mod:`mlsys.finetune`, but for a
:class:`~mlsys.models.backbone.GenerativeBackbone` (seq2seq LM). Reuses :class:`RunRecord`
and the five timing sections:

- ``score_summarization_candidate`` — the **frozen proxy**: train only the LM/generation
  head (``scope="head"``, teacher-forced CE). Direct analog of "FC head on a frozen backbone".
- ``finetune_summarization_candidate`` — the expensive ground truth: unfreeze the whole
  model (``scope="full"``).

``inference_s`` stays ``0.0`` — the forward is fused into the training loop, matching the
encoder *finetune* convention — and generation + ROUGE scoring land in ``eval_s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from mlsys.datasets.registry import REQUIRED_SPLITS
from mlsys.models.registry import ModelSpec, build_backbone
from mlsys.search.metrics import summarization_metrics
from mlsys.search.runner import RunRecord
from mlsys.search.timing import Timer, reset_peak_gpu_memory

if TYPE_CHECKING:
    import torch

    from mlsys.datasets import LoadedDataset, Row
    from mlsys.models.backbone import GenerativeBackbone


@dataclass(frozen=True)
class SummarizeConfig:
    epochs: int = 3
    batch_size: int = 8
    head_lr: float = 1e-3
    full_lr: float = 2e-5
    weight_decay: float = 1e-4
    early_stop_patience: int = 2
    min_delta: float = 1e-3
    # Informational parity with the regression configs; actual decode length is driven
    # by the adapter's `max_target_length` (config/models.yaml `extra`).
    max_target_length: int = 64


@dataclass(frozen=True)
class _Seq2SeqTrainResult:
    train_curve: list[float]
    val_curve: list[float]
    epochs_run: int


def _snapshot(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _restore(params: list[torch.nn.Parameter], snapshot: list[torch.Tensor]) -> None:
    import torch

    with torch.no_grad():
        for p, s in zip(params, snapshot, strict=True):
            p.copy_(s)


def _train_seq2seq(
    backbone: GenerativeBackbone,
    rows: dict[str, list[Row]],
    cfg: SummarizeConfig,
    scope: Literal["head", "full"],
    device: str,
) -> _Seq2SeqTrainResult:
    """Teacher-forced training over the unfrozen ``scope``, early-stopping on val CE loss.

    The backbone is mutated in place and left in its best-val state.
    """
    import torch

    train_rows = rows["train"]
    val_rows = rows["val"]
    if not train_rows or not val_rows:
        raise ValueError(
            "_train_seq2seq requires non-empty train and val splits; got "
            f"{len(train_rows)} train and {len(val_rows)} val rows"
        )

    backbone.set_trainable(scope)
    lr = cfg.head_lr if scope == "head" else cfg.full_lr
    params = list(backbone.trainable_parameters())
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)

    train_src = [r.text for r in train_rows]
    train_tgt = [str(r.target) for r in train_rows]
    val_src = [r.text for r in val_rows]
    val_tgt = [str(r.target) for r in val_rows]

    bs = cfg.batch_size
    best_val = float("inf")
    best_state = _snapshot(params)
    train_curve: list[float] = []
    val_curve: list[float] = []
    patience = 0
    epochs_run = 0

    for epoch in range(cfg.epochs):
        epochs_run = epoch + 1
        backbone.train()
        perm = torch.randperm(len(train_src)).tolist()
        total = 0.0
        seen = 0
        for start in range(0, len(train_src), bs):
            idx = perm[start : start + bs]
            src = [train_src[i] for i in idx]
            tgt = [train_tgt[i] for i in idx]
            optim.zero_grad()
            loss = backbone.teacher_forcing_loss(src, tgt)
            loss.backward()
            optim.step()
            total += float(loss.detach()) * len(src)
            seen += len(src)
        train_curve.append(total / max(seen, 1))

        backbone.eval()
        with torch.no_grad():
            vtot = 0.0
            vseen = 0
            for start in range(0, len(val_src), bs):
                src = val_src[start : start + bs]
                tgt = val_tgt[start : start + bs]
                loss = backbone.teacher_forcing_loss(src, tgt)
                vtot += float(loss) * len(src)
                vseen += len(src)
            val_ce = vtot / max(vseen, 1)
        val_curve.append(val_ce)

        if val_ce + cfg.min_delta < best_val:
            best_val = val_ce
            best_state = _snapshot(params)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    _restore(params, best_state)
    return _Seq2SeqTrainResult(train_curve=train_curve, val_curve=val_curve, epochs_run=epochs_run)


def _run_summarization(
    dataset: LoadedDataset,
    spec: ModelSpec,
    *,
    scope: Literal["head", "full"],
    strategy: str,
    device: str,
    config: SummarizeConfig | None,
) -> RunRecord:
    config = config or SummarizeConfig()
    reset_peak_gpu_memory()
    timer = Timer(label=f"{strategy}:{spec.name}")

    with timer.section("prepare_model_s"):
        backbone = cast("GenerativeBackbone", build_backbone(spec, device=device))

    with timer.section("prepare_data_s"):
        rows = {split: list(dataset.split(split)) for split in REQUIRED_SPLITS}

    # inference_s stays 0.0: the forward is fused into the training loop (encoder-finetune
    # convention). The default TimingBreakdown value already covers this.

    with timer.section("train_head_s"):
        result = _train_seq2seq(backbone, rows, config, scope, device)

    with timer.section("eval_s"):
        backbone.eval()
        test_rows = rows["test"]
        bs = config.batch_size
        preds: list[str] = []
        for start in range(0, len(test_rows), bs):
            batch = test_rows[start : start + bs]
            preds.extend(backbone.generate([r.text for r in batch]))
        refs = [str(r.target) for r in test_rows]
        metrics = summarization_metrics(preds, refs)

    timer.record_peak_gpu_mb()

    return RunRecord(
        dataset=dataset.spec.name,
        model=spec.name,
        metrics=metrics,
        timing=timer.breakdown.to_dict(),
        head_train_curve=result.train_curve,
        head_val_curve=result.val_curve,
        epochs_run=result.epochs_run,
        strategy=strategy,
        extras={"task": "summarization", "head_repeats": 1},
    )


def score_summarization_candidate(
    dataset: LoadedDataset,
    spec: ModelSpec,
    *,
    device: str = "cpu",
    config: SummarizeConfig | None = None,
) -> RunRecord:
    """Frozen proxy: train only the LM/generation head, score generated summaries by ROUGE."""
    return _run_summarization(
        dataset, spec, scope="head", strategy="frozen", device=device, config=config
    )


def finetune_summarization_candidate(
    dataset: LoadedDataset,
    spec: ModelSpec,
    *,
    device: str = "cpu",
    config: SummarizeConfig | None = None,
) -> RunRecord:
    """Ground truth: unfreeze the whole seq2seq model, score generated summaries by ROUGE."""
    return _run_summarization(
        dataset, spec, scope="full", strategy="finetune", device=device, config=config
    )
