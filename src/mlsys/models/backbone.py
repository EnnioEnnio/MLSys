"""Backbone Protocol — encoder-side surface every adapter implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch


@runtime_checkable
class Backbone(Protocol):
    """A frozen text encoder. Adapters wrap concrete HF/sentence-transformers/model2vec models."""

    name: str
    embedding_dim: int

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Return embeddings of shape ``[B, embedding_dim]`` on the active device."""
        ...


@runtime_checkable
class TrainableBackbone(Backbone, Protocol):
    """A backbone that can also be fine-tuned (backbone weights unfrozen).

    The frozen :meth:`Backbone.encode` path is left untouched — it stays the cheap,
    ``inference_mode`` ranking signal. Fine-tuning uses :meth:`encode_trainable`
    (grad-enabled, no ``inference_mode``) plus :meth:`train`/:meth:`eval` mode toggles
    and :meth:`parameters` for the optimiser. Adapters that wrap static / non-trainable
    encoders (e.g. model2vec) set ``can_finetune = False``, return ``[]`` from
    :meth:`parameters`, and raise from :meth:`encode_trainable`; callers fall back to the
    frozen score for them.
    """

    can_finetune: bool

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """The backbone weights the optimiser should update."""
        ...

    def encode_trainable(self, texts: list[str]) -> torch.Tensor:
        """Like :meth:`encode` but grad-enabled (no ``inference_mode``), for fine-tuning."""
        ...

    def train(self) -> None:
        """Put the underlying model in training mode (dropout etc. active)."""
        ...

    def eval(self) -> None:
        """Put the underlying model in eval mode."""
        ...
