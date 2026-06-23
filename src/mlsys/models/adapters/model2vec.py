"""Adapter for model2vec / potion-* static lookup encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.mlsys.models.registry import ModelSpec, register_adapter

if TYPE_CHECKING:
    import torch


class Model2VecBackbone:
    name: str
    embedding_dim: int

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


def _build(spec: ModelSpec, device: str) -> Model2VecBackbone:
    return Model2VecBackbone(spec, device)


register_adapter("model2vec", _build)
