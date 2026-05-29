"""Backbone Protocol — encoder-side surface every adapter implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class Backbone(Protocol):
    """A frozen text encoder. Adapters wrap concrete HF/sentence-transformers/model2vec models."""

    name: str
    embedding_dim: int

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Return embeddings of shape ``[B, embedding_dim]`` on the active device."""
        ...
