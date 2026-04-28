"""Shared types and Protocols that cross module boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Row:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Label:
    row_id: str
    value: str
    confidence: float = 1.0


@dataclass
class Prediction:
    row_id: str
    value: str
    confidence: float = 1.0
    source: str = ""


@dataclass
class RunRecord:
    row: Row
    label: Label
    prediction: Prediction | None = None


@runtime_checkable
class Dataset(Protocol):
    def __iter__(self) -> Any: ...
    def __len__(self) -> int: ...


@runtime_checkable
class LLMClient(Protocol):
    def classify(self, row: Row) -> Label: ...


@runtime_checkable
class CandidateModel(Protocol):
    name: str

    def predict(self, row: Row) -> Prediction: ...
    def fine_tune(self, records: list[RunRecord]) -> None: ...
