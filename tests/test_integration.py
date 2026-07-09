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


def _tiny_summarization_dataset(name: str = "synthetic_sum"):
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
        target_column="summary",
        target_type="summarization",
        text_template="{dialogue}",
    )
    docs = [
        "Tom: are you coming tonight? Amy: yes see you at 8",
        "Ken: the meeting moved to 3pm. Sue: thanks for letting me know",
        "Joe: I bought milk and eggs. Pam: great, we needed those",
        "Ann: the train is delayed again. Bob: ugh, take the bus then",
    ]
    sums = [
        "Amy will meet Tom at 8.",
        "Meeting moved to 3pm.",
        "Joe bought groceries.",
        "Train delayed; take the bus.",
    ]
    rows = [Row(text=docs[i % 4], target=sums[i % 4]) for i in range(12)]
    ds = LoadedDataset.__new__(LoadedDataset)
    ds.spec = spec
    ds.splits = {
        "train": _Split(rows[:8]),
        "val": _Split(rows[8:10]),
        "test": _Split(rows[10:]),
    }
    return ds


def test_seq2seq_generate_and_rouge() -> None:
    # Real t5-small: generate summaries and score them with ROUGE.
    from typing import cast

    from mlsys.models.backbone import GenerativeBackbone
    from mlsys.models.registry import build_backbone, get_spec
    from mlsys.search.metrics import summarization_metrics

    backbone = cast(GenerativeBackbone, build_backbone(get_spec("t5-small"), device="cpu"))
    preds = backbone.generate(
        ["summarize: The quick brown fox jumped over the lazy dog many times all day long."]
    )
    assert len(preds) == 1
    assert isinstance(preds[0], str) and preds[0].strip()
    m = summarization_metrics(preds, ["A fox jumped over a dog."])
    assert 0.0 <= m.rougeL <= 1.0


def test_summarization_frozen_proxy_end_to_end() -> None:
    # 1-epoch LM-head-only teacher-forced proxy on a tiny in-memory dataset.
    from mlsys.models.registry import get_spec
    from mlsys.search.metrics import SummarizationMetrics
    from mlsys.search.summarize import SummarizeConfig, score_summarization_candidate

    record = score_summarization_candidate(
        _tiny_summarization_dataset(),
        get_spec("t5-small"),
        device="cpu",
        config=SummarizeConfig(epochs=1, batch_size=4),
    )
    assert record.strategy == "frozen"
    assert isinstance(record.metrics, SummarizationMetrics)
    assert record.timing["inference_s"] == 0.0
    assert record.timing["eval_s"] > 0.0


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
