"""Backbone Protocol — encoder-side surface every adapter implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

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


@runtime_checkable
class GenerativeBackbone(Protocol):
    """A seq2seq generator (the summarization task path).

    Deliberately **not** a :class:`Backbone` — a generator has no meaningful fixed-width
    ``encode``. The frozen proxy trains only the LM/generation head (``set_trainable("head")``,
    teacher-forced cross-entropy); ``finetune`` unfreezes the whole model
    (``set_trainable("full")``). ``generate`` produces summaries for ROUGE scoring. Adapters
    set ``can_finetune`` and expose the unfrozen weights via :meth:`trainable_parameters`.
    """

    name: str
    can_finetune: bool

    def teacher_forcing_loss(self, sources: list[str], targets: list[str]) -> torch.Tensor:
        """Cross-entropy of teacher-forced ``targets`` given ``sources`` (grads flow to
        whatever is currently unfrozen — no ``inference_mode``)."""
        ...

    def generate(self, sources: list[str]) -> list[str]:
        """Decode a summary string for each source (``inference_mode``)."""
        ...

    def set_trainable(self, scope: Literal["head", "full"]) -> None:
        """Unfreeze only the generation head (``"head"``) or the whole model (``"full"``)."""
        ...

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        """The currently-unfrozen parameters the optimiser should update."""
        ...

    def train(self) -> None:
        """Put the underlying model in training mode (dropout etc. active)."""
        ...

    def eval(self) -> None:
        """Put the underlying model in eval mode."""
        ...
