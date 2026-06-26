"""Parse config/models.yaml into typed ModelSpec records and resolve adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from mlsys.models.backbone import Backbone

REQUIRED_FIELDS = ("name", "hf_repo", "loader", "embedding_dim")
# The built-in loaders, kept for reference/error messages. Loader validation in
# load_specs() is dynamic against the *registered* adapter set, so dropping a new
# self-registering adapter module in adapters/ is enough — no edit here required.
KNOWN_LOADERS = ("sentence_transformers", "transformers_encoder", "model2vec")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_repo: str
    loader: str
    embedding_dim: int
    pooling: str = "builtin"
    max_length: int | None = None
    input_prefix: str = ""
    trust_remote_code: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "models.yaml"


def load_specs(path: Path | None = None) -> dict[str, ModelSpec]:
    cfg_path = path or _config_path()
    raw = yaml.safe_load(cfg_path.read_text()) or []
    # Validate loaders against the registered adapters (not a static list) so a
    # newly dropped adapter file resolves at config-load time without editing here.
    _ensure_adapters_registered()
    specs: dict[str, ModelSpec] = {}
    for entry in raw:
        for field_name in REQUIRED_FIELDS:
            if field_name not in entry:
                raise ValueError(f"models.yaml entry missing {field_name!r}: {entry}")
        if entry["loader"] not in _ADAPTERS:
            raise ValueError(
                f"models.yaml entry {entry['name']!r} has unknown loader "
                f"{entry['loader']!r}; registered: {sorted(_ADAPTERS)}"
            )
        dim = int(entry["embedding_dim"])
        if dim <= 0:
            raise ValueError(
                f"models.yaml entry {entry['name']!r}: embedding_dim must be > 0, got {dim}"
            )
        known = {*REQUIRED_FIELDS, "pooling", "max_length", "input_prefix", "trust_remote_code"}
        spec = ModelSpec(
            name=entry["name"],
            hf_repo=entry["hf_repo"],
            loader=entry["loader"],
            embedding_dim=dim,
            pooling=entry.get("pooling", "builtin"),
            max_length=entry.get("max_length"),
            input_prefix=entry.get("input_prefix", "") or "",
            trust_remote_code=bool(entry.get("trust_remote_code", False)),
            extra={k: v for k, v in entry.items() if k not in known},
        )
        if spec.name in specs:
            raise ValueError(f"duplicate model name in models.yaml: {spec.name}")
        specs[spec.name] = spec
    return specs


def get_spec(name: str, path: Path | None = None) -> ModelSpec:
    specs = load_specs(path)
    if name not in specs:
        raise KeyError(f"unknown model {name!r}; known: {sorted(specs)}")
    return specs[name]


_ADAPTERS: dict[str, Callable[[ModelSpec, str], Backbone]] = {}


def register_adapter(loader: str, builder: Callable[[ModelSpec, str], Backbone]) -> None:
    _ADAPTERS[loader] = builder


_BUILTINS_IMPORTED = False


def _ensure_adapters_registered() -> None:
    # Auto-discover every module in the adapters package; each registers itself
    # via register_adapter() at import time. Flag-guarded (not keyed off whether
    # _ADAPTERS is non-empty) so a peer calling register_adapter() before the
    # first build_backbone() can't suppress the built-ins.
    global _BUILTINS_IMPORTED
    if _BUILTINS_IMPORTED:
        return
    import importlib
    import pkgutil

    from mlsys.models import adapters

    for mod in pkgutil.iter_modules(adapters.__path__):
        importlib.import_module(f"{adapters.__name__}.{mod.name}")
    _BUILTINS_IMPORTED = True


def build_backbone(spec: ModelSpec, device: str = "cpu") -> Backbone:
    _ensure_adapters_registered()
    if spec.loader not in _ADAPTERS:
        raise KeyError(
            f"no adapter registered for loader {spec.loader!r}; known: {sorted(_ADAPTERS)}"
        )
    return _ADAPTERS[spec.loader](spec, device)
