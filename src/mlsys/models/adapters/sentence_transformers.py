"""Adapter for sentence-transformers checkpoints (built-in pooling)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.mlsys.models.registry import ModelSpec, register_adapter

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
        kwargs: dict[str, object] = {"device": device}
        # Opt-in per model (config/models.yaml). Some embedding checkpoints ship a
        # custom architecture as Python in their HF repo (e.g. gte's Alibaba-NLP/new-impl,
        # nomic's nomic_bert); transformers refuses to import that code unless we
        # explicitly consent. Left False, the vast majority of the pool never grants it.
        if spec.trust_remote_code:
            kwargs["trust_remote_code"] = True
        self._model = SentenceTransformer(spec.hf_repo, **kwargs)
        # Force fp32. transformers>=5 loads checkpoints in their saved dtype, so
        # fp16 checkpoints (e.g. sentence-t5-base) arrive in half precision. On
        # NGC images that monkeypatch apex's FusedRMSNorm into T5, that fused
        # kernel rejects half input ("expected scalar type Float but found Half").
        # Pinning every backbone to fp32 keeps inference in one kernel-safe
        # precision and embeddings full-precision (older transformers did this here).
        self._model = self._model.float()
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
