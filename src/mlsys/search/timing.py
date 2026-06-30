"""Per-substep wall-clock + peak-GPU-memory measurement.

This module is the substrate for the RQ2 bottleneck analysis: every numbered
field name (``prepare_model_s``, ``prepare_data_s``, ``inference_s``,
``train_head_s``, ``eval_s``) shows up verbatim in :mod:`mlsys.search.runner`
and in ``results.jsonl``.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


def _cuda_sync() -> None:
    """Block until queued CUDA kernels finish, so wall-clock deltas and peak-memory
    reads cover completed work. No-op when torch/CUDA is absent (CPU test path)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@dataclass
class TimingBreakdown:
    prepare_model_s: float = 0.0
    prepare_data_s: float = 0.0
    inference_s: float = 0.0
    train_head_s: float = 0.0
    eval_s: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {
            "prepare_model_s": self.prepare_model_s,
            "prepare_data_s": self.prepare_data_s,
            "inference_s": self.inference_s,
            "train_head_s": self.train_head_s,
            "eval_s": self.eval_s,
            "peak_gpu_mem_mb": self.peak_gpu_mem_mb,
        }
        out.update(self.extras)
        return out


class Timer:
    """Context-manager-based timer that accumulates into a `TimingBreakdown`."""

    def __init__(self, label: str | None = None) -> None:
        self.breakdown = TimingBreakdown()
        # When set, each section logs its start/finish at DEBUG so a long run shows
        # which substep is currently executing (e.g. "frozen:all-MiniLM-L6-v2").
        self.label = label

    @contextmanager
    def section(self, name: str):
        _cuda_sync()
        if self.label is not None:
            log.debug("[%s] %s …", self.label, name)
        start = time.perf_counter()
        try:
            yield
        finally:
            _cuda_sync()
            elapsed = time.perf_counter() - start
            if self.label is not None:
                log.debug("[%s] %s done in %.2fs", self.label, name, elapsed)
            if hasattr(self.breakdown, name):
                setattr(self.breakdown, name, getattr(self.breakdown, name) + elapsed)
            else:
                self.breakdown.extras[name] = self.breakdown.extras.get(name, 0.0) + elapsed

    def record_peak_gpu_mb(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.breakdown.peak_gpu_mem_mb = float(
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                )
        except ImportError:
            pass


def reset_peak_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
