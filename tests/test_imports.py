"""Every mlsys submodule imports cleanly.

Adapters lazy-import torch/transformers/etc. inside methods, so importing the
modules themselves is cheap. Importing `mlsys.__main__` is safe: its
`raise SystemExit(main())` is guarded by `if __name__ == "__main__"`, which is
False under import.
"""

from __future__ import annotations

import importlib
import pkgutil

from src import mlsys


def test_all_submodules_import() -> None:
    failures: list[tuple[str, str]] = []
    for mod in pkgutil.walk_packages(mlsys.__path__, prefix="mlsys."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:
            failures.append((mod.name, repr(exc)))
    assert not failures, f"modules failed to import: {failures}"
