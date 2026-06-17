"""Adapter for raw HF AutoModel encoders with explicit pooling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlsys.models.registry import ModelSpec, register_adapter

if TYPE_CHECKING:
    import torch

VALID_POOLING = {"mean", "cls", "last"}


def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor, how: str) -> torch.Tensor:
    import torch

    if how == "cls":
        # `.clone()` is load-bearing: `hidden[:, 0, :]` is a basic-slice VIEW that
        # shares storage with the full (batch, seq, hidden) activation tensor. The
        # caller keeps pooled outputs on-device and `.to(device, float32)` is a
        # no-op (returns the same view) when already on that device+dtype, so the
        # view would pin every batch's entire activation tensor in memory and OOM
        # on cls-pooled models. mean/last already build fresh tensors; only cls
        # aliases, so only cls needs the copy.
        return hidden[:, 0, :].clone()
    if how == "last":
        # Index of the last non-pad token per row.
        lengths = attention_mask.sum(dim=1) - 1
        idx = lengths.clamp(min=0)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, idx, :]
    # mean
    mask = attention_mask.unsqueeze(-1).type_as(hidden)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TransformersEncoderBackbone:
    name: str
    embedding_dim: int

    def __init__(self, spec: ModelSpec, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if spec.pooling not in VALID_POOLING:
            raise ValueError(
                f"transformers_encoder pooling must be one of {VALID_POOLING}; "
                f"got {spec.pooling!r} for {spec.name}"
            )
        self.name = spec.name
        self.embedding_dim = spec.embedding_dim
        self._device = device
        self._pooling = spec.pooling
        self._max_length = spec.max_length
        self._input_prefix = spec.input_prefix
        # Opt-in per model (config/models.yaml): consent to run a checkpoint's
        # custom architecture code from its HF repo. Off for every model that
        # transformers supports natively, which is all of this loader's pool today.
        trc = spec.trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, trust_remote_code=trc)
        if tokenizer is None:
            raise RuntimeError(f"AutoTokenizer returned None for {spec.hf_repo!r}")
        self._tokenizer = tokenizer
        self._model = AutoModel.from_pretrained(
            spec.hf_repo, use_safetensors=True, trust_remote_code=trc
        ).to(device)
        self._model.eval()
        self._torch = torch

    def encode(self, texts: list[str]) -> torch.Tensor:
        if self._input_prefix:
            texts = [self._input_prefix + t for t in texts]
        tok_kwargs: dict[str, object] = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self._max_length is not None:
            tok_kwargs["max_length"] = self._max_length
        batch = self._tokenizer(texts, **tok_kwargs)
        batch = {k: v.to(self._device) for k, v in batch.items()}
        with self._torch.inference_mode():
            out = self._model(**batch)
        hidden = out.last_hidden_state
        return _pool(hidden, batch["attention_mask"], self._pooling)


def _build(spec: ModelSpec, device: str) -> TransformersEncoderBackbone:
    return TransformersEncoderBackbone(spec, device)


register_adapter("transformers_encoder", _build)
