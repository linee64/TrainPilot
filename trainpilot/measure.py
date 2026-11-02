"""Time training steps on a short run.

Warmup steps are executed and then dropped. The reported step time is the
median of the measured steps, so a single slow step does not move it.
On a GPU, every timer read waits for ``torch.cuda.synchronize()`` first.
Without a GPU the same timing still runs, peak memory is omitted, and the
status is ``GPU not found``.

``measure_batches`` also splits each step into time spent waiting for the
next batch and time spent computing after that batch is in hand.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch

SHORT_RUN_LABEL = "short-run result"
GPU_NOT_FOUND = "GPU not found"

_DEFAULT_WARMUP = 10
_DEFAULT_STEPS = 30


@dataclass(frozen=True)
class StepMeasurement:
    """Numbers from one short run. They are not a full-training measurement."""

    step_seconds: tuple[float, ...]
    example_counts: tuple[int, ...]
    median_step_seconds: float
    examples_per_second: float
    peak_memory_bytes: int | None
    gpu_name: str | None
    warmup_steps: int
    measured_steps: int
    status: str
    label: str = SHORT_RUN_LABEL
    data_wait_seconds: tuple[float, ...] = ()
    compute_seconds: tuple[float, ...] = ()
    median_data_wait_seconds: float | None = None
    median_compute_seconds: float | None = None
    wait_fraction: float | None = None
    loader_num_workers: int | None = None
    loader_pin_memory: bool | None = None


def measure_steps(
    step_fn: Callable[[], int],
    *,
    warmup: int = _DEFAULT_WARMUP,
    steps: int = _DEFAULT_STEPS,
) -> StepMeasurement:
    """Run ``step_fn`` and time it.

    ``step_fn`` performs one training step and returns how many examples
    that step consumed, as a Python ``int``. The first ``warmup`` calls
    (default 10) are excluded. The next ``steps`` calls (default 30) are
    kept. Examples per second uses the median step time, not the mean.
    """

    _check_window(warmup, steps)
    _begin_memory()

    for _ in range(warmup):
        _timed_step(step_fn)

    step_seconds: list[float] = []
    example_counts: list[int] = []
    for _ in range(steps):
        elapsed, examples = _timed_step(step_fn)
        step_seconds.append(elapsed)
        example_counts.append(examples)

    return _finish(
        step_seconds,
        example_counts,
        warmup=warmup,
        steps=steps,
    )


def measure_batches(
    batches: Iterable[Any],
    step_fn: Callable[[Any], int],
    *,
    warmup: int = _DEFAULT_WARMUP,
    steps: int = _DEFAULT_STEPS,
) -> StepMeasurement:
    """Time data wait and compute separately.

    Drawing the next batch is the wait. ``step_fn`` then receives that batch,
    does the training work, and returns how many examples it consumed.
    The wait fraction is the median wait divided by the median whole step.
    The first ``warmup`` batches (default 10) are excluded. The next
    ``steps`` batches (default 30) are kept.
    """

    _check_window(warmup, steps)
    _begin_memory()
    loader_num_workers, loader_pin_memory = _loader_settings(batches)
    upcoming = _batch_source(batches, warmup + steps)

    for _ in range(warmup):
        _timed_batch(upcoming, step_fn)

    step_seconds: list[float] = []
    data_wait_seconds: list[float] = []
    compute_seconds: list[float] = []
    example_counts: list[int] = []
    for _ in range(steps):
        data_wait, compute, examples = _timed_batch(upcoming, step_fn)
        data_wait_seconds.append(data_wait)
        compute_seconds.append(compute)
        step_seconds.append(data_wait + compute)
        example_counts.append(examples)

    return _finish(
        step_seconds,
        example_counts,
        warmup=warmup,
        steps=steps,
        data_wait_seconds=data_wait_seconds,
        compute_seconds=compute_seconds,
        loader_num_workers=loader_num_workers,
        loader_pin_memory=loader_pin_memory,
    )


def _check_window(warmup: int, steps: int) -> None:
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if steps < 1:
        raise ValueError("steps must be >= 1")


def _begin_memory() -> None:
    if torch.cuda.is_available():
        _synchronize()
        torch.cuda.reset_peak_memory_stats()


def _finish(
    step_seconds: list[float],
    example_counts: list[int],
    *,
    warmup: int,
    steps: int,
    data_wait_seconds: list[float] | None = None,
    compute_seconds: list[float] | None = None,
    loader_num_workers: int | None = None,
    loader_pin_memory: bool | None = None,
) -> StepMeasurement:
    median_step_seconds = float(statistics.median(step_seconds))
    median_data_wait_seconds: float | None = None
    median_compute_seconds: float | None = None
    wait_fraction: float | None = None
    if data_wait_seconds is not None and compute_seconds is not None:
        median_data_wait_seconds = float(statistics.median(data_wait_seconds))
        median_compute_seconds = float(statistics.median(compute_seconds))
        wait_fraction = _wait_fraction(median_data_wait_seconds, median_step_seconds)

    peak_memory_bytes: int | None = None
    gpu_name: str | None = None
    status = GPU_NOT_FOUND
    if torch.cuda.is_available():
        _synchronize()
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        gpu_name = str(torch.cuda.get_device_name(0))
        status = "ok"

    return StepMeasurement(
        step_seconds=tuple(step_seconds),
        example_counts=tuple(example_counts),
        median_step_seconds=median_step_seconds,
        examples_per_second=_examples_per_second(example_counts, median_step_seconds),
        peak_memory_bytes=peak_memory_bytes,
        gpu_name=gpu_name,
        warmup_steps=warmup,
        measured_steps=steps,
        status=status,
        data_wait_seconds=tuple(data_wait_seconds or ()),
        compute_seconds=tuple(compute_seconds or ()),
        median_data_wait_seconds=median_data_wait_seconds,
        median_compute_seconds=median_compute_seconds,
        wait_fraction=wait_fraction,
        loader_num_workers=loader_num_workers,
        loader_pin_memory=loader_pin_memory,
    )


def _loader_settings(batches: Any) -> tuple[int | None, bool | None]:
    """Read DataLoader knobs if ``batches`` is a torch DataLoader."""

    if isinstance(batches, torch.utils.data.DataLoader):
        return int(batches.num_workers), bool(batches.pin_memory)
    return None, None


def _wait_fraction(median_data_wait_seconds: float, median_step_seconds: float) -> float:
    if median_step_seconds <= 0:
        return 0.0
    return median_data_wait_seconds / median_step_seconds


def _batch_source(batches: Iterable[Any], needed: int) -> Callable[[], Any]:
    iterator = iter(batches)
    seen = 0

    def next_batch() -> Any:
        nonlocal seen
        try:
            batch = next(iterator)
        except StopIteration:
            raise RuntimeError(f"Not enough batches: needed {needed}, got {seen}.") from None
        seen += 1
        return batch

    return next_batch


def _timed_batch(
    next_batch: Callable[[], Any],
    step_fn: Callable[[Any], int],
) -> tuple[float, float, int]:
    _synchronize()
    started = time.perf_counter()
    batch = next_batch()
    _synchronize()
    data_wait = time.perf_counter() - started

    _synchronize()
    started = time.perf_counter()
    examples = step_fn(batch)
    _synchronize()
    compute = time.perf_counter() - started
    _check_example_count(examples)
    return data_wait, compute, examples


def _timed_step(step_fn: Callable[[], int]) -> tuple[float, int]:
    _synchronize()
    started = time.perf_counter()
    examples = step_fn()
    _synchronize()
    elapsed = time.perf_counter() - started
    _check_example_count(examples)
    return elapsed, examples


def _synchronize() -> None:
    """Finish queued GPU work before a timer is read. No-op without a GPU."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _check_example_count(examples: int) -> None:
    if isinstance(examples, bool) or not isinstance(examples, int):
        raise TypeError("step_fn must return the example count as an int")
    if examples < 0:
        raise ValueError("step_fn returned a negative example count")


def _examples_per_second(example_counts: list[int], median_step_seconds: float) -> float:
    if median_step_seconds <= 0:
        return 0.0
    median_examples = float(statistics.median(example_counts))
    return median_examples / median_step_seconds
