"""CPU tests for the data-wait split and the 30% rule."""

import time
from unittest.mock import patch

import pytest

from trainpilot.diagnose import GPU_BUSY, WAITING_FOR_DATA, classify_wait_fraction, diagnose
from trainpilot.measure import GPU_NOT_FOUND, StepMeasurement, measure_batches, measure_steps


def test_wait_above_30_percent_means_waiting_for_data() -> None:
    diagnosis = diagnose(_measured(0.31))

    assert diagnosis.verdict == WAITING_FOR_DATA
    assert "waiting for the next batch" in diagnosis.reason
    assert "short run" in diagnosis.reason


def test_wait_at_or_below_30_percent_means_gpu_busy() -> None:
    assert diagnose(_measured(0.30)).verdict == GPU_BUSY
    assert diagnose(_measured(0.0)).verdict == GPU_BUSY
    assert "busy" in diagnose(_measured(0.10)).reason


def test_median_wait_share_ignores_one_slow_batch() -> None:
    # Three measured steps. Whole steps are 0.10, 0.10, and 0.60 seconds.
    # Waits are 0.01, 0.01, and 0.50. The mean share is about 65%.
    # The median share is 10%, so the GPU stays busy.
    clock = iter(
        [
            0.00, 0.01, 0.01, 0.10,
            1.00, 1.01, 1.01, 1.10,
            2.00, 2.50, 2.50, 2.60,
        ]
    )

    with patch("trainpilot.measure.time.perf_counter", side_effect=lambda: next(clock)):
        result = measure_batches([0, 0, 0], lambda _batch: 8, warmup=0, steps=3)

    assert result.wait_fraction == pytest.approx(0.10)
    assert classify_wait_fraction(result.wait_fraction) == GPU_BUSY


def test_warmup_batch_is_excluded_from_the_wait_share() -> None:
    clock = iter(
        [
            0.0, 10.0, 10.0, 10.01,
            20.0, 20.01, 20.01, 20.11,
        ]
    )

    with patch("trainpilot.measure.time.perf_counter", side_effect=lambda: next(clock)):
        result = measure_batches([0, 0], lambda _batch: 4, warmup=1, steps=1)

    assert result.data_wait_seconds == pytest.approx((0.01,))
    assert result.compute_seconds == pytest.approx((0.10,))
    assert result.wait_fraction == pytest.approx(0.01 / 0.11)
    assert classify_wait_fraction(result.wait_fraction) == GPU_BUSY


def test_slow_batches_and_slow_compute_land_on_opposite_sides() -> None:
    def slow_batches():
        for _ in range(3):
            time.sleep(0.05)
            yield 4

    waiting = measure_batches(slow_batches(), lambda _batch: 4, warmup=0, steps=3)
    busy = measure_batches([4, 4, 4], _slow_step, warmup=0, steps=3)

    assert waiting.wait_fraction is not None and waiting.wait_fraction > 0.30
    assert busy.wait_fraction is not None and busy.wait_fraction < 0.30


def test_cpu_does_not_call_the_gpu_waiting_or_busy() -> None:
    def slow_batches():
        time.sleep(0.02)
        yield 1

    result = measure_batches(slow_batches(), lambda _batch: 1, warmup=0, steps=1)
    diagnosis = diagnose(result)

    assert result.status == GPU_NOT_FOUND
    assert result.wait_fraction is not None and result.wait_fraction > 0.30
    assert diagnosis.verdict == GPU_NOT_FOUND
    assert diagnosis.verdict != WAITING_FOR_DATA


def test_whole_step_timing_is_not_classified_as_a_split() -> None:
    measured = measure_steps(lambda: 1, warmup=0, steps=1)
    diagnosis = diagnose(_measured(None, status="ok"))

    assert measured.wait_fraction is None
    assert diagnosis.verdict == "skipped"
    assert diagnosis.reason.startswith("Skipped because")


def test_loader_must_cover_warmup_and_measured_steps() -> None:
    with pytest.raises(RuntimeError, match="Not enough batches"):
        measure_batches([1], lambda _batch: 1, warmup=0, steps=2)


def test_synchronize_around_data_wait_and_compute() -> None:
    with (
        patch("trainpilot.measure.torch.cuda.is_available", return_value=True),
        patch("trainpilot.measure.torch.cuda.synchronize") as sync,
        patch("trainpilot.measure.torch.cuda.reset_peak_memory_stats"),
        patch("trainpilot.measure.torch.cuda.max_memory_allocated", return_value=2048),
        patch("trainpilot.measure.torch.cuda.get_device_name", return_value="Tesla T4"),
    ):
        result = measure_batches([1, 1], lambda _batch: 2, warmup=1, steps=1)

    # One sync before peak reset, four per batch, one before the peak read.
    assert sync.call_count == 1 + 4 * (1 + 1) + 1
    assert result.status == "ok"
    assert result.gpu_name == "Tesla T4"
    assert result.peak_memory_bytes == 2048


def _slow_step(_batch: int) -> int:
    time.sleep(0.05)
    return 4


def _measured(wait_fraction: float | None, *, status: str = "ok") -> StepMeasurement:
    gpu_found = status == "ok"
    return StepMeasurement(
        step_seconds=(1.0,),
        example_counts=(8,),
        median_step_seconds=1.0,
        examples_per_second=8.0,
        peak_memory_bytes=1 if gpu_found else None,
        gpu_name="Tesla T4" if gpu_found else None,
        warmup_steps=0,
        measured_steps=1,
        status=status,
        median_data_wait_seconds=wait_fraction,
        median_compute_seconds=None if wait_fraction is None else 1.0 - wait_fraction,
        wait_fraction=wait_fraction,
    )
