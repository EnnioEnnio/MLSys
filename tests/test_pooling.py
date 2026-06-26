"""_pool tensor math for the transformers_encoder adapter (no model download).

Assumes right-padding (real tokens first, pads at the end) — the HF default for
encoder tokenizers and what `_pool`'s last-non-pad indexing (sum(mask) - 1) is
built for.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mlsys.models.adapters.transformers_encoder import _pool  # noqa: E402


def _hidden_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    # B=2, T=3, H=2. Row 0: all 3 tokens real. Row 1: 2 real + 1 right-pad.
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[10.0, 20.0], [30.0, 40.0], [-100.0, -100.0]],  # last token is pad
        ]
    )
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    return hidden, mask


def test_pool_cls_takes_first_token() -> None:
    hidden, mask = _hidden_and_mask()
    out = _pool(hidden, mask, "cls")
    assert torch.equal(out, torch.tensor([[1.0, 2.0], [10.0, 20.0]]))


def test_pool_mean_is_mask_weighted() -> None:
    hidden, mask = _hidden_and_mask()
    out = _pool(hidden, mask, "mean")
    # row 0: mean of all 3; row 1: mean of the 2 real tokens (pad excluded)
    expected = torch.tensor([[3.0, 4.0], [20.0, 30.0]])
    assert torch.allclose(out, expected)


def test_pool_last_skips_right_padding() -> None:
    hidden, mask = _hidden_and_mask()
    out = _pool(hidden, mask, "last")
    # row 0: token idx 2; row 1: last *non-pad* token (idx 1), not the pad at idx 2
    expected = torch.tensor([[5.0, 6.0], [30.0, 40.0]])
    assert torch.equal(out, expected)


def test_pool_last_full_length_row() -> None:
    # No padding: the last token is just the final position.
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])
    mask = torch.tensor([[1, 1, 1]])
    out = _pool(hidden, mask, "last")
    assert torch.equal(out, torch.tensor([[3.0]]))


@pytest.mark.parametrize("how", ["cls", "mean", "last"])
def test_pool_output_does_not_alias_hidden_state(how: str) -> None:
    # Regression: a pooled output that *views* into `hidden` keeps the whole
    # (B, T, H) activation tensor alive. The caller stores pooled outputs for the
    # full split on-device, so an aliasing view pins every batch's activations and
    # OOMs the GPU (hit cls, which used a basic-slice view). Each pooling must
    # return a tensor that owns its storage so `hidden` can be freed.
    hidden, mask = _hidden_and_mask()
    out = _pool(hidden, mask, how)
    assert out.untyped_storage().data_ptr() != hidden.untyped_storage().data_ptr()
