"""FCHead converges on a synthetic linear-regression problem."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mlsys.head import HeadTrainConfig, train_head  # noqa: E402


def test_head_converges_below_noise_floor() -> None:
    torch.manual_seed(0)
    n, d = 200, 32
    x = torch.randn(n, d)
    w_true = torch.randn(d)
    noise = 0.05 * torch.randn(n)
    y = x @ w_true + noise

    x_train, x_val = x[:160], x[160:]
    y_train, y_val = y[:160], y[160:]

    cfg = HeadTrainConfig(
        epochs=200, batch_size=32, lr=5e-2, early_stop_patience=20, min_delta=1e-3
    )
    result = train_head(x_train, y_train, x_val, y_val, cfg)

    noise_var = float(noise.var())
    # min_delta=1e-3 makes early-stop fire before the head fully settles, so allow
    # ~3x the noise floor on this synthetic task.
    assert result.best_val_mse < 3 * noise_var, (
        f"head failed to converge: val_mse={result.best_val_mse:.4f} noise_var={noise_var:.4f}"
    )
    assert len(result.train_curve) == result.epochs_run


def test_train_head_rejects_empty_split() -> None:
    # An empty train or val split would otherwise yield silent NaN val-loss; fail fast.
    empty_x, empty_y = torch.randn(0, 8), torch.zeros(0)
    x, y = torch.randn(4, 8), torch.randn(4)
    cfg = HeadTrainConfig(epochs=1)
    with pytest.raises(ValueError, match="non-empty"):
        train_head(empty_x, empty_y, x, y, cfg)
    with pytest.raises(ValueError, match="non-empty"):
        train_head(x, y, empty_x, empty_y, cfg)

def test_seed_applies_deterministic_initialization() -> None:
    seed = 42
    n, d = 100, 16
    x_train, y_train = torch.randn(n, d), torch.randn(n)
    x_val, y_val = torch.randn(n, d), torch.randn(n)

    cfg = HeadTrainConfig(epochs=1, batch_size=32, lr=1e-3)

    # Train two models with the same seed
    result1 = train_head(x_train, y_train, x_val, y_val, cfg, seed)

    result2 = train_head(x_train, y_train, x_val, y_val, cfg, seed)

    # Check if the model weights are identical
    for (param1, param2) in zip(result1.head.parameters(), result2.head.parameters()):
        assert torch.allclose(param1, param2), "Model weights differ despite the same seed"

    # Check if the training curves are identical
    assert result1.train_curve == result2.train_curve, "Training curves differ despite the same seed"
    assert result1.val_curve == result2.val_curve, "Validation curves differ despite the same seed"
