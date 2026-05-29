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


def test_potion_backbone_encodes() -> None:
    from mlsys.models import load_backbone

    backbone = load_backbone("potion-base-8M", device="cpu")
    emb = backbone.encode(["a red wine with notes of cherry", "crisp white, citrus"])
    assert emb.shape[0] == 2
    assert emb.shape[1] == backbone.embedding_dim
