"""CPU tests for short-run step timing."""

import statistics
import time
from unittest.mock import patch

import pytest
import torch

from trainpilot.measure import GPU_NOT_FOUND, SHORT_RUN_LABEL, _loader_settings, measure_batches, measure_steps


def test_warmup_is_excluded_and_defaults_are_10_then_30() -> None:
    calls = 0

    def step() -> int:
        nonlocal calls
        calls += 1
        return 8

    result = measure_steps(step)

    assert result.warmup_steps == 10
    assert result.measured_steps == 30
    assert len(result.step_seconds) == 30
    assert result.example_counts == (8,) * 30
    assert calls == 40
    assert result.label == SHORT_RUN_LABEL


def test_median_ignores_a_slow_step() -> None:
    # warmup duration 5, then measured durations 1, 1, and 100.
    clock = iter([0.0, 5.0, 10.0, 11.0, 20.0, 21.0, 30.0, 130.0])

    def step() -> int:
        return 32

    with patch("trainpilot.measure.time.perf_counter", side_effect=lambda: next(clock)):
        result = measure_steps(step, warmup=1, steps=3)

    assert result.step_seconds == (1.0, 1.0, 100.0)
    assert result.median_step_seconds == 1.0
    assert result.median_step_seconds != pytest.approx(statistics.mean(result.step_seconds))
    assert result.examples_per_second == 32.0


def test_cpu_reports_gpu_not_found_and_skips_memory() -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("CUDA memory must not be read without a GPU")

    def step() -> int:
        return 4

    with (
        patch("trainpilot.measure.torch.cuda.is_available", return_value=False),
        patch("trainpilot.measure.torch.cuda.synchronize", side_effect=fail_if_called),
        patch("trainpilot.measure.torch.cuda.max_memory_allocated", side_effect=fail_if_called),
    ):
        result = measure_steps(step, warmup=0, steps=2)

    assert result.status == GPU_NOT_FOUND
    assert result.gpu_name is None
    assert result.peak_memory_bytes is None
    assert len(result.step_seconds) == 2


def test_synchronize_runs_before_each_timer_read_and_before_memory() -> None:
    def step() -> int:
        return 2

    with (
        patch("trainpilot.measure.torch.cuda.is_available", return_value=True),
        patch("trainpilot.measure.torch.cuda.synchronize") as sync,
        patch("trainpilot.measure.torch.cuda.reset_peak_memory_stats"),
        patch("trainpilot.measure.torch.cuda.max_memory_allocated", return_value=4096),
        patch("trainpilot.measure.torch.cuda.get_device_name", return_value="Tesla T4"),
    ):
        result = measure_steps(step, warmup=1, steps=1)

    # One sync before resetting peak memory, two per timed call, one before the peak read.
    assert sync.call_count == 1 + 2 * (1 + 1) + 1
    assert result.status == "ok"
    assert result.gpu_name == "Tesla T4"
    assert result.peak_memory_bytes == 4096


def test_measure_batches_reads_dataloader_settings() -> None:
    dataset = torch.utils.data.TensorDataset(torch.zeros(6, 2))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=0, pin_memory=True
    )
    result = measure_batches(loader, lambda _batch: 2, warmup=0, steps=3)

    assert result.loader_num_workers == 0
    assert result.loader_pin_memory is True


def test_measure_batches_ignores_plain_lists() -> None:
    result = measure_batches([1, 2], lambda _batch: 1, warmup=0, steps=2)

    assert result.loader_num_workers is None
    assert result.loader_pin_memory is None


def test_loader_settings_read_workers_without_iterating() -> None:
    dataset = torch.utils.data.TensorDataset(torch.zeros(4, 1))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=2, pin_memory=True
    )

    assert _loader_settings(loader) == (2, True)
    assert _loader_settings([1, 2, 3]) == (None, None)


def test_cpu_median_matches_manual_time_time() -> None:
    def step() -> int:
        value = torch.randn(512, 512)
        product = value
        for _ in range(80):
            product = product @ value
        return 512

    on_gpu = torch.cuda.is_available()
    warmup, steps = (5, 12) if on_gpu else (10, 30)
    # GPU stays at 5%. CPU timer noise (especially on Windows) gets more
    # samples and a wider limit so this check does not flake.
    limit = 0.05 if on_gpu else 0.12
    measured = measure_steps(step, warmup=warmup, steps=steps)
    manual = _manual_median_seconds(step, warmup=warmup, steps=steps)
    assert manual > 0
    assert abs(measured.median_step_seconds - manual) / manual < limit


def test_step_fn_must_return_an_example_count() -> None:
    with pytest.raises(TypeError):
        measure_steps(lambda: None, warmup=0, steps=1)  # type: ignore[arg-type, return-value]
    with pytest.raises(ValueError):
        measure_steps(lambda: -1, warmup=0, steps=1)
    with pytest.raises(ValueError):
        measure_steps(lambda: 1, warmup=-1, steps=1)
    with pytest.raises(ValueError):
        measure_steps(lambda: 1, warmup=0, steps=0)


def _manual_median_seconds(step_fn, warmup: int, steps: int) -> float:
    """Independent timer: ``time.time()`` plus ``synchronize()`` on a GPU."""

    samples: list[float] = []
    for index in range(warmup + steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.time()
        step_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - started
        if index >= warmup:
            samples.append(elapsed)
    return float(statistics.median(samples))
