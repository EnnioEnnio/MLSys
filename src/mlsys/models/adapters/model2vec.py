"""Adapter for model2vec / potion-* static lookup encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlsys.models.registry import ModelSpec, register_adapter

if TYPE_CHECKING:
    import torch


class Model2VecBackbone:
    name: str
    embedding_dim: int
    # Static lookup-table encoder: no trainable weights, so it can't be fine-tuned.
    # The finetune path detects this and falls back to the frozen score.
    can_finetune: bool = False

    def __init__(self, spec: ModelSpec, device: str) -> None:
        from model2vec import StaticModel

        self.name = spec.name
        self.embedding_dim = spec.embedding_dim
        self._device = device
        self._input_prefix = spec.input_prefix
        self._model = StaticModel.from_pretrained(spec.hf_repo)

    def encode(self, texts: list[str]) -> torch.Tensor:
        import torch

        if self._input_prefix:
            texts = [self._input_prefix + t for t in texts]
        vectors = self._model.encode(texts)
        return torch.as_tensor(vectors, dtype=torch.float32, device=self._device)

    def encode_trainable(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.name!r} is a static model2vec encoder and cannot be fine-tuned; "
            "callers should check can_finetune and fall back to the frozen score"
        )

    def parameters(self) -> list[object]:
        return []

    def train(self) -> None:  # no-op: nothing to train
        pass

    def eval(self) -> None:  # no-op
        pass


def _build(spec: ModelSpec, device: str) -> Model2VecBackbone:
    return Model2VecBackbone(spec, device)


register_adapter("model2vec", _build)
