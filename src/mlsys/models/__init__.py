"""Model registry + backbone adapters."""

from __future__ import annotations

from mlsys.models.backbone import Backbone
from mlsys.models.registry import (
    ModelSpec,
    build_backbone,
    get_spec,
    load_specs,
    register_adapter,
)

__all__ = [
    "Backbone",
    "ModelSpec",
    "build_backbone",
    "get_spec",
    "load_backbone",
    "load_specs",
    "register_adapter",
]


def load_backbone(name: str, device: str = "cpu") -> Backbone:
    """Resolve `name` against `config/models.yaml` and return a ready Backbone."""
    return build_backbone(get_spec(name), device=device)
