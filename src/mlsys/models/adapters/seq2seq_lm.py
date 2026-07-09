"""Adapter for HF ``AutoModelForSeq2SeqLM`` generators (the summarization task path).

Mirrors :mod:`mlsys.models.adapters.transformers_encoder` in structure (lazy heavy imports
inside ``__init__``, fp32 load), but exposes the generative surface
(:class:`mlsys.models.backbone.GenerativeBackbone`) instead of ``encode``.

**Tied-weight note.** T5/BART tie the LM head to the shared embedding table, so the
``"head"`` scope (the frozen proxy) trains the shared vocab projection / ``final_logits_bias``
while every transformer block stays frozen — a cheap, faithful "lightweight head on a frozen
body" analog. ``"full"`` unfreezes everything (the expensive fine-tune ground truth).
``embedding_dim`` in the yaml is a **nominal** ``d_model``: no FCHead is built for
summarization, so it is semantically ignored by the runner (only kept > 0 to satisfy
``models/registry.py`` validation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from mlsys.models.registry import ModelSpec, register_adapter

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch

    from mlsys.models.backbone import Backbone


class Seq2SeqLMBackbone:
    name: str
    can_finetune: bool = True

    def __init__(self, spec: ModelSpec, device: str) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = spec.name
        self._device = device
        self._input_prefix = spec.input_prefix
        self._max_length = spec.max_length
        self._max_target_length = int(spec.extra.get("max_target_length", 64))
        trc = spec.trust_remote_code
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, trust_remote_code=trc)
        if tokenizer is None:
            raise RuntimeError(f"AutoTokenizer returned None for {spec.hf_repo!r}")
        self._tokenizer = tokenizer
        # Force fp32 on load (some checkpoints carry fp16 torch_dtype in config); plain
        # AdamW without a GradScaler is numerically fragile in raw fp16.
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            spec.hf_repo, use_safetensors=True, trust_remote_code=trc, torch_dtype=torch.float32
        ).to(device)
        self._model.eval()
        self._torch = torch

    def _tokenize_sources(self, sources: list[str]) -> dict[str, torch.Tensor]:
        if self._input_prefix:
            sources = [self._input_prefix + s for s in sources]
        tok_kwargs: dict[str, object] = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self._max_length is not None:
            tok_kwargs["max_length"] = self._max_length
        batch = self._tokenizer(sources, **tok_kwargs)
        return {k: v.to(self._device) for k, v in batch.items()}

    def teacher_forcing_loss(self, sources: list[str], targets: list[str]) -> torch.Tensor:
        batch = self._tokenize_sources(sources)
        labels = self._tokenizer(
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self._max_target_length,
            return_tensors="pt",
        )["input_ids"].to(self._device)
        # Ignore padding in the loss (HF convention: label id -100 is skipped).
        labels = labels.masked_fill(labels == self._tokenizer.pad_token_id, -100)
        # No inference_mode: grads must flow to whatever set_trainable() left unfrozen.
        return self._model(**batch, labels=labels).loss

    def generate(self, sources: list[str]) -> list[str]:
        batch = self._tokenize_sources(sources)
        with self._torch.inference_mode():
            out = self._model.generate(**batch, num_beams=1, max_new_tokens=self._max_target_length)
        return self._tokenizer.batch_decode(out, skip_special_tokens=True)

    def set_trainable(self, scope: Literal["head", "full"]) -> None:
        if scope == "full":
            for p in self._model.parameters():
                p.requires_grad = True
            return
        # "head": freeze the whole body, then unfreeze only the output (LM) head.
        # T5/BART tie the head to the shared embedding table, so PyTorch de-duplicates
        # `lm_head.weight` under `shared.weight` — matching on parameter *names* finds
        # nothing. `get_output_embeddings()` returns the tied projection module directly,
        # so we train the shared vocab projection while every transformer block stays
        # frozen (a cheap, faithful "lightweight head" analog).
        for p in self._model.parameters():
            p.requires_grad = False
        out = self._model.get_output_embeddings()
        if out is None:
            raise RuntimeError(
                f"{self.name}: model exposes no output embeddings to train as the head"
            )
        for p in out.parameters():
            p.requires_grad = True

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        return (p for p in self._model.parameters() if p.requires_grad)

    def train(self) -> None:
        self._model.train()

    def eval(self) -> None:
        self._model.eval()


def _build(spec: ModelSpec, device: str) -> Backbone:
    # GenerativeBackbone isn't a (regression) Backbone — the registry is typed for the
    # encoder surface, so cast at the boundary. The summarization runner casts back to
    # GenerativeBackbone; regression datasets never resolve to this loader.
    return cast("Backbone", Seq2SeqLMBackbone(spec, device))


register_adapter("seq2seq_lm", _build)
