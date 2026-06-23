"""Model-search strategies + runner."""

from __future__ import annotations

from src.mlsys.search.full_eval import full_eval
from src.mlsys.search.runner import RunRecord, score_candidate

__all__ = ["RunRecord", "full_eval", "score_candidate"]
