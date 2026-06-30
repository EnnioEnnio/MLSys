"""Real-model end-to-end smoke — the only test that exercises a real adapter's
encode() path against a downloaded checkpoint.

Skipped by default and in CI (network + heavier deps). Run explicitly with:

    uv run pytest -m integration

Uses potion-base-8M (small static model2vec lookup encoder, CPU-friendly).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("torch")
pytest.importorskip("model2vec")
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")


def test_potion_backbone_encodes() -> None:
    from mlsys.models import load_backbone

    backbone = load_backbone("potion-base-8M", device="cpu")
    emb = backbone.encode(["a red wine with notes of cherry", "crisp white, citrus"])
    assert emb.shape[0] == 2
    assert emb.shape[1] == backbone.embedding_dim


def test_sentence_transformers_encode_trainable_flows_gradients() -> None:
    # The one path that can't be trusted from API docs alone (see PLAN.md open risks):
    # the ST module API must keep the graph attached so grads reach the backbone.
    from typing import cast

    import torch

    from mlsys.models import load_backbone
    from mlsys.models.backbone import TrainableBackbone

    backbone = cast(TrainableBackbone, load_backbone("all-MiniLM-L6-v2", device="cpu"))
    backbone.train()
    emb = backbone.encode_trainable(["a red wine", "a crisp white"])
    assert emb.shape == (2, backbone.embedding_dim)
    assert emb.requires_grad

    emb.sum().backward()
    grads = [p.grad for p in backbone.parameters() if p.grad is not None]
    assert grads, "no backbone parameter received a gradient"
    assert any(torch.any(g != 0) for g in grads)


def _tiny_dataset(name: str = "synthetic"):
    from mlsys.datasets import LoadedDataset, Row
    from mlsys.datasets.registry import DatasetSpec

    class _Split:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

        def __len__(self):
            return len(self._rows)

    spec = DatasetSpec(
        name=name,
        hf_repo="local/fake",
        splits={"train": "train", "val": "val", "test": "test"},
        target_column="y",
        target_type="regression",
        text_template="{text}",
    )
    texts = ["red wine cherry", "white wine citrus", "bold tannins oak", "light fruity rose"]
    rows = [Row(text=texts[i % len(texts)], target=float(i % 3)) for i in range(12)]
    ds = LoadedDataset.__new__(LoadedDataset)
    ds.spec = spec
    ds.splits = {
        "train": _Split(rows[:8]),
        "val": _Split(rows[8:10]),
        "test": _Split(rows[10:]),
    }
    return ds


def test_transformers_encoder_finetune_end_to_end() -> None:
    # Real tiny transformers_encoder fine-tuned end-to-end on a tiny in-memory dataset.
    from mlsys.finetune import FinetuneConfig
    from mlsys.head import HeadTrainConfig
    from mlsys.models.registry import get_spec
    from mlsys.search.metrics import RegressionMetrics
    from mlsys.search.runner import finetune_candidate

    spec = get_spec("distilbert-base-uncased")
    record = finetune_candidate(
        _tiny_dataset(),
        spec,
        device="cpu",
        head_config=HeadTrainConfig(),
        finetune_config=FinetuneConfig(epochs=1, batch_size=4),
    )
    assert record.strategy == "finetune"
    assert isinstance(record.metrics, RegressionMetrics)
    assert record.timing["inference_s"] == 0.0
    assert record.timing["train_head_s"] > 0.0
