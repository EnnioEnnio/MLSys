"""FC regression head + linear-probe trainer."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn


class FCHead(nn.Module):
    """Linear-probe head. Single `nn.Linear` by default; optional 2-layer MLP."""

    def __init__(self, in_dim: int, hidden: int | None = None, out_dim: int = 1) -> None:
        super().__init__()
        if hidden is None or hidden <= 0:
            self.net: nn.Module = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class HeadTrainConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 3
    min_delta: float = 1e-4
    hidden: int | None = None


@dataclass(frozen=True)
class HeadTrainResult:
    head: FCHead
    train_curve: list[float]
    val_curve: list[float]
    best_val_mse: float
    epochs_run: int


def _iter_minibatches(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    n = x.size(0)
    idx = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for s in range(0, n, batch_size):
        sl = idx[s : s + batch_size]
        yield x[sl], y[sl]


def train_head(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    config: HeadTrainConfig | None = None,
) -> HeadTrainResult:
    """Train an FCHead with AdamW + MSE, early-stop on val-MSE plateau."""
    if config is None:
        config = HeadTrainConfig()
    device = x_train.device
    head = FCHead(in_dim=x_train.size(1), hidden=config.hidden).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    train_curve: list[float] = []
    val_curve: list[float] = []
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    patience = 0
    epochs_run = 0

    for epoch in range(config.epochs):
        epochs_run = epoch + 1
        head.train()
        total = 0.0
        seen = 0
        for xb, yb in _iter_minibatches(x_train, y_train, config.batch_size, shuffle=True):
            optim.zero_grad()
            pred = head(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            total += float(loss.detach()) * xb.size(0)
            seen += xb.size(0)
        train_curve.append(total / max(seen, 1))

        head.eval()
        with torch.inference_mode():
            val_pred = head(x_val)
            val_mse = float(loss_fn(val_pred, y_val))
        val_curve.append(val_mse)

        if val_mse + config.min_delta < best_val:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                break

    head.load_state_dict(best_state)
    return HeadTrainResult(
        head=head,
        train_curve=train_curve,
        val_curve=val_curve,
        best_val_mse=best_val,
        epochs_run=epochs_run,
    )
