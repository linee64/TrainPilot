"""Decide whether the GPU is waiting for data or busy computing.

The wait share is the median time spent getting the next batch divided by
the median whole step. Above 30% the GPU is waiting for data. Otherwise it
is busy. Without a GPU the verdict is ``GPU not found``.
"""

from __future__ import annotations

from dataclasses import dataclass

from trainpilot.measure import GPU_NOT_FOUND, SHORT_RUN_LABEL, StepMeasurement

WAIT_LIMIT = 0.30
WAITING_FOR_DATA = "waiting for data"
GPU_BUSY = "gpu busy"
SKIPPED = "skipped"


@dataclass(frozen=True)
class Diagnosis:
    """Plain-language result of one short run. Not a full-training claim."""

    verdict: str
    reason: str
    wait_fraction: float | None
    label: str = SHORT_RUN_LABEL


def diagnose(measurement: StepMeasurement) -> Diagnosis:
    """Classify a short run that already split data wait from compute."""

    if measurement.status == GPU_NOT_FOUND:
        return _remember(
            Diagnosis(
                verdict=GPU_NOT_FOUND,
                reason=(
                    "GPU not found. This short run cannot tell whether a GPU would "
                    "be waiting for data or busy computing."
                ),
                wait_fraction=measurement.wait_fraction,
            )
        )
    if measurement.wait_fraction is None:
        return _remember(
            Diagnosis(
                verdict=SKIPPED,
                reason="Skipped because data-wait time was not measured separately.",
                wait_fraction=None,
            )
        )
    verdict = classify_wait_fraction(measurement.wait_fraction)
    return _remember(
        Diagnosis(
            verdict=verdict,
            reason=_reason(verdict, measurement.wait_fraction),
            wait_fraction=measurement.wait_fraction,
        )
    )


def _remember(diagnosis: Diagnosis) -> Diagnosis:
    from trainpilot.report import record

    record(diagnosis=diagnosis)
    return diagnosis


def classify_wait_fraction(wait_fraction: float) -> str:
    """Return the verdict for one wait share. The share must already be a median."""

    if wait_fraction > WAIT_LIMIT:
        return WAITING_FOR_DATA
    return GPU_BUSY


def _reason(verdict: str, wait_fraction: float) -> str:
    percent = wait_fraction * 100
    if verdict == WAITING_FOR_DATA:
        return (
            f"On this short run, {percent:.0f}% of a typical step is spent waiting "
            "for the next batch. The GPU is idle while that happens."
        )
    return (
        f"On this short run, {percent:.0f}% of a typical step is spent waiting "
        "for the next batch. The GPU is busy computing the rest."
    )
