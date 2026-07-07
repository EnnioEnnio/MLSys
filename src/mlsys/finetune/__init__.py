"""Joint backbone+head fine-tune loop (the expensive ground-truth signal for regret).

Mirrors :mod:`mlsys.head` (AdamW + MSE + early-stop + train/val curves), but unfreezes
the backbone and trains it jointly with the head. Inference is fused into training here:
each step does ``head(backbone.encode_trainable(texts))`` so there is no separate embedding
pass. The trained head is returned in a :class:`~mlsys.head.HeadTrainResult` (same shape as
the frozen path) and the backbone is mutated in place, restored to its best-val state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mlsys.head import EpochCallback, FCHead, HeadTrainConfig, HeadTrainResult

if TYPE_CHECKING:
    import torch

    from mlsys.datasets import Row
    from mlsys.models.backbone import TrainableBackbone


@dataclass(frozen=True)
class FinetuneConfig:
    epochs: int = 8
    # Fine-tune is memory-heavy (full backbone activations + grads), so a much
    # smaller default batch than the frozen head trainer.
    batch_size: int = 16
    backbone_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 3
    min_delta: float = 1e-3


def _snapshot(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _restore(params: list[torch.nn.Parameter], snapshot: list[torch.Tensor]) -> None:
    import torch

    with torch.no_grad():
        for p, s in zip(params, snapshot, strict=True):
            p.copy_(s)


def train_full_model(
    backbone: TrainableBackbone,
    head: FCHead,
    rows: dict[str, list[Row]],
    head_cfg: HeadTrainConfig,
    finetune_cfg: FinetuneConfig,
    device: str,
    epoch_callback: EpochCallback | None = None,
) -> HeadTrainResult:
    """Train ``backbone`` + ``head`` jointly with AdamW (two LR groups) + MSE, early-stopping
    on val MSE. The backbone is mutated in place and left in its best-val state; the trained
    head comes back inside the returned result.
    """
    import torch
    from torch import nn

    train_rows = rows["train"]
    val_rows = rows["val"]
    if not train_rows or not val_rows:
        raise ValueError(
            "train_full_model requires non-empty train and val splits; got "
            f"{len(train_rows)} train and {len(val_rows)} val rows"
        )

    train_texts = [r.text for r in train_rows]
    y_train = torch.tensor([r.target for r in train_rows], dtype=torch.float32, device=device)
    val_texts = [r.text for r in val_rows]
    y_val = torch.tensor([r.target for r in val_rows], dtype=torch.float32, device=device)

    backbone_params = list(backbone.parameters())
    head_params = list(head.parameters())
    optim = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": finetune_cfg.backbone_lr},
            {"params": head_params, "lr": finetune_cfg.head_lr},
        ],
        weight_decay=finetune_cfg.weight_decay,
    )
    loss_fn = nn.MSELoss()

    bs = finetune_cfg.batch_size
    tracked = backbone_params + head_params
    best_val = float("inf")
    best_state = _snapshot(tracked)
    train_curve: list[float] = []
    val_curve: list[float] = []
    patience = 0
    epochs_run = 0

    for epoch in range(finetune_cfg.epochs):
        epochs_run = epoch + 1
        backbone.train()
        head.train()
        perm = torch.randperm(len(train_texts))
        total = 0.0
        seen = 0
        for start in range(0, len(train_texts), bs):
            idx = perm[start : start + bs]
            batch_texts = [train_texts[i] for i in idx.tolist()]
            yb = y_train[idx]
            optim.zero_grad()
            emb = backbone.encode_trainable(batch_texts)
            pred = head(emb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            total += float(loss.detach()) * len(batch_texts)
            seen += len(batch_texts)
        train_curve.append(total / max(seen, 1))

        backbone.eval()
        head.eval()
        with torch.no_grad():
            preds = [
                head(backbone.encode(val_texts[s : s + bs])) for s in range(0, len(val_texts), bs)
            ]
            val_pred = torch.cat(preds)
            val_mse = float(loss_fn(val_pred, y_val))
        val_curve.append(val_mse)
        if epoch_callback is not None:
            epoch_callback(epoch, train_curve[-1], val_curve[-1])

        if val_mse + finetune_cfg.min_delta < best_val:
            best_val = val_mse
            best_state = _snapshot(tracked)
            patience = 0
        else:
            patience += 1
            if patience >= finetune_cfg.early_stop_patience:
                break

    _restore(tracked, best_state)
    return HeadTrainResult(
        head=head,
        train_curve=train_curve,
        val_curve=val_curve,
        best_val_mse=best_val,
        epochs_run=epochs_run,
    )
