"""Time training steps on a short run.

Warmup steps are executed and then dropped. The reported step time is the
median of the measured steps, so a single slow step does not move it.
On a GPU, every timer read waits for ``torch.cuda.synchronize()`` first.
Without a GPU the same timing still runs, peak memory is omitted, and the
status is ``GPU not found``.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

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

    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    gpu_found = torch.cuda.is_available()
    if gpu_found:
        _synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        _timed_step(step_fn)

    step_seconds: list[float] = []
    example_counts: list[int] = []
    for _ in range(steps):
        elapsed, examples = _timed_step(step_fn)
        step_seconds.append(elapsed)
        example_counts.append(examples)

    median_step_seconds = float(statistics.median(step_seconds))
    examples_per_second = _examples_per_second(example_counts, median_step_seconds)

    peak_memory_bytes: int | None = None
    gpu_name: str | None = None
    status = GPU_NOT_FOUND
    if gpu_found:
        _synchronize()
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        gpu_name = str(torch.cuda.get_device_name(0))
        status = "ok"

    return StepMeasurement(
        step_seconds=tuple(step_seconds),
        example_counts=tuple(example_counts),
        median_step_seconds=median_step_seconds,
        examples_per_second=examples_per_second,
        peak_memory_bytes=peak_memory_bytes,
        gpu_name=gpu_name,
        warmup_steps=warmup,
        measured_steps=steps,
        status=status,
    )


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
