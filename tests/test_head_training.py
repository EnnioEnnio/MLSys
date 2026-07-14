"""FCHead converges on a synthetic linear-regression problem."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mlsys.head import (  # noqa: E402
    ACTIVATIONS,
    FCHead,
    HeadTrainConfig,
    target_stats,
    train_head,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("relu", torch.nn.ReLU),
        ("gelu", torch.nn.GELU),
        ("tanh", torch.nn.Tanh),
        ("silu", torch.nn.SiLU),
    ],
)
def test_mlp_head_uses_requested_activation(name, expected) -> None:
    # The four names the sweep compares; the middle module of the 2-layer MLP is the knob.
    assert set(ACTIVATIONS) == {"relu", "gelu", "tanh", "silu"}
    head = FCHead(in_dim=8, hidden=16, activation=name)
    assert isinstance(head.net, torch.nn.Sequential)
    assert isinstance(head.net[1], expected)


def test_linear_probe_has_no_activation() -> None:
    # hidden None/<=0 -> bare nn.Linear; the activation name is inert, not an error.
    for hidden in (None, 0):
        head = FCHead(in_dim=8, hidden=hidden, activation="tanh")
        assert isinstance(head.net, torch.nn.Linear)


def test_unknown_activation_rejected() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        FCHead(in_dim=8, hidden=16, activation="swish")
    with pytest.raises(ValueError, match="unknown activation"):
        HeadTrainConfig(activation="swish")


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


def test_train_head_trains_prebuilt_head_in_place() -> None:
    # Passing head= trains that exact object (the LP-FT warmup path) instead of a fresh one.
    torch.manual_seed(0)
    x = torch.randn(40, 8)
    y = x[:, 0] * 2.0
    x_train, x_val = x[:32], x[32:]
    y_train, y_val = y[:32], y[32:]

    head = FCHead(in_dim=8, hidden=None)
    before = [p.detach().clone() for p in head.parameters()]

    result = train_head(x_train, y_train, x_val, y_val, HeadTrainConfig(epochs=5), head=head)

    assert result.head is head
    assert any(not torch.equal(b, p) for b, p in zip(before, head.parameters(), strict=True)), (
        "prebuilt head params did not change"
    )


def test_target_stats_matches_train_moments() -> None:
    y = torch.tensor([80.0, 85.0, 90.0, 95.0])
    mean, std = target_stats(y)
    assert mean == pytest.approx(87.5)
    assert std == pytest.approx(float(y.std()))


def test_target_stats_guards_degenerate_targets() -> None:
    # Constant targets (std 0) and single-row splits (std NaN) fall back to std=1.0
    # so z-scoring degrades to a mean-shift instead of dividing by ~0.
    mean, std = target_stats(torch.full((10,), 42.0))
    assert mean == pytest.approx(42.0)
    assert std == 1.0
    mean, std = target_stats(torch.tensor([88.0]))
    assert mean == pytest.approx(88.0)
    assert std == 1.0


def test_seed_produces_identical_initial_weights() -> None:
    d = 16
    x = torch.randn(10, d)
    y = torch.randn(10)
    cfg = HeadTrainConfig(epochs=0)

    result1 = train_head(x, y, x, y, cfg, seed=42)
    result2 = train_head(x, y, x, y, cfg, seed=42)

    for p1, p2 in zip(result1.head.parameters(), result2.head.parameters(), strict=True):
        assert torch.equal(p1, p2)
