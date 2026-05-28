"""Run-directory layout + JSONL appender for per-candidate result rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_run_dir(output_dir: str | Path) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_path(output_dir: str | Path) -> Path:
    return ensure_run_dir(output_dir) / "results.jsonl"


class JsonlWriter:
    """Append-mode JSONL writer. One row per call; flushes after every write."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
