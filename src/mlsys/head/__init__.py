"""FC regression head + linear-probe trainer."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn

# Activations selectable for the MLP head's hidden layer (--activation). The linear
# probe has no activation, so this only applies when hidden > 0.
ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


def _build_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; known: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class EpochCallback(Protocol):
    """Invoked after each epoch so callers can stream curves.

    Trainers may attach extra per-epoch scalars as keyword floats (e.g. the finetune
    joint loop's ``grad_norm``/``grad_norm_max``); implementations must accept them.
    """

    def __call__(self, epoch: int, train_mse: float, val_mse: float, **extras: float) -> None: ...


class FCHead(nn.Module):
    """Linear-probe head. Single `nn.Linear` by default; optional 2-layer MLP.

    ``activation`` names the nonlinearity between the MLP's two layers (see
    :data:`ACTIVATIONS`); it is ignored on the linear path, which has none.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int | None = None,
        out_dim: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden is None or hidden <= 0:
            self.net: nn.Module = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                _build_activation(activation),
                nn.Linear(hidden, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class HeadTrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 3
    min_delta: float = 1e-3
    hidden: int | None = None
    # Nonlinearity between the MLP head's two layers; ignored by the linear probe
    # (hidden None/<=0), which has no activation.
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {self.activation!r}; known: {sorted(ACTIVATIONS)}"
            )


@dataclass(frozen=True)
class HeadTrainResult:
    head: FCHead
    train_curve: list[float]
    val_curve: list[float]
    best_val_mse: float
    epochs_run: int
    # Per-epoch pre-clip global grad norm over all trained params (mean / max of the
    # step norms). Populated by the finetune joint loop only; empty for the frozen
    # head trainer, which doesn't clip.
    grad_norm_curve: list[float] = field(default_factory=list)
    grad_norm_max_curve: list[float] = field(default_factory=list)


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
    seed: int | None = None,
    epoch_callback: EpochCallback | None = None,
    head: FCHead | None = None,
) -> HeadTrainResult:
    """Train an FCHead with AdamW + MSE, early-stop on val-MSE plateau.

    Pass ``head`` to train an existing FCHead in place (e.g. the LP-FT warmup phase);
    its ``in_dim``/``hidden``/activation come from the module, so ``config.hidden`` and
    ``config.activation`` are ignored. Otherwise a fresh FCHead is built from ``x_train``
    width + ``config.hidden`` + ``config.activation``.
    """
    if config is None:
        config = HeadTrainConfig()
    if x_train.size(0) == 0 or x_val.size(0) == 0:
        raise ValueError(
            "train_head requires non-empty train and val tensors; got "
            f"{x_train.size(0)} train and {x_val.size(0)} val rows "
            "(check the dataset splits aren't empty after loading/filtering)"
        )
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    device = x_train.device
    if head is None:
        head = FCHead(
            in_dim=x_train.size(1), hidden=config.hidden, activation=config.activation
        ).to(device)
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
        if epoch_callback is not None:
            epoch_callback(epoch, train_curve[-1], val_curve[-1])

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
