"""Timer / TimingBreakdown on the CPU path.

The CUDA synchronise added for P4 is guarded by torch.cuda.is_available(), so
these run with no GPU (and even with no torch) present.
"""

from __future__ import annotations

from src.mlsys.search.timing import Timer, TimingBreakdown


def _busy() -> int:
    x = 0
    for _ in range(200_000):
        x += 1
    return x


def test_section_accumulates_named_field() -> None:
    timer = Timer()
    with timer.section("inference_s"):
        _busy()
    first = timer.breakdown.inference_s
    assert first > 0.0
    with timer.section("inference_s"):
        _busy()
    # Accumulates (about 2x first); it does not overwrite with just the 2nd section.
    assert timer.breakdown.inference_s > first


def test_unknown_section_routes_to_extras() -> None:
    timer = Timer()
    with timer.section("custom_step_s"):
        pass
    assert "custom_step_s" in timer.breakdown.extras
    # Custom names land in extras, not the fixed core schema.
    assert "custom_step_s" not in TimingBreakdown().to_dict()


def test_to_dict_emits_all_core_keys() -> None:
    d = TimingBreakdown().to_dict()
    for key in (
        "prepare_model_s",
        "prepare_data_s",
        "inference_s",
        "train_head_s",
        "eval_s",
        "peak_gpu_mem_mb",
    ):
        assert key in d


def test_record_peak_gpu_mb_is_noop_without_cuda() -> None:
    # No CUDA on the test path → the guarded sync/read is a no-op; peak stays 0.0.
    timer = Timer()
    timer.record_peak_gpu_mb()
    assert timer.breakdown.peak_gpu_mem_mb == 0.0
