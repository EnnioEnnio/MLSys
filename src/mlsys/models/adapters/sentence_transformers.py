"""Adapter for sentence-transformers checkpoints (built-in pooling)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlsys.models.registry import ModelSpec, register_adapter

if TYPE_CHECKING:
    import torch


class SentenceTransformersBackbone:
    name: str
    embedding_dim: int

    def __init__(self, spec: ModelSpec, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = spec.name
        self.embedding_dim = spec.embedding_dim
        self._device = device
        self._input_prefix = spec.input_prefix
        kwargs = {"device": device}
        self._model = SentenceTransformer(spec.hf_repo, **kwargs)
        if spec.max_length is not None:
            self._model.max_seq_length = spec.max_length

    def encode(self, texts: list[str]) -> torch.Tensor:
        if self._input_prefix:
            texts = [self._input_prefix + t for t in texts]
        return self._model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self._device,
        )


def _build(spec: ModelSpec, device: str) -> SentenceTransformersBackbone:
    return SentenceTransformersBackbone(spec, device)


register_adapter("sentence_transformers", _build)
