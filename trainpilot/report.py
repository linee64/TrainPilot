"""One-page report and a short snippet the user pastes in themselves.

The first screen is plain language. Numbers are from a short test. A full
training run is only an estimate. Anything that was not checked is listed
as ``Skipped because ...``. The report does not edit the user's script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trainpilot.measure import GPU_NOT_FOUND, SHORT_RUN_LABEL

_FULL_RUN_NOTE = (
    "These numbers are from a short test. "
    "A full training run is only an estimate, not a measurement."
)
_SKIPPED_BATCH_SIZE = "Skipped because this version does not change the batch size."
_SKIPPED_NO_DIAGNOSIS = "Skipped because data-wait time was not measured separately."
_SKIPPED_NO_EXPERIMENT = "Skipped because mixed precision was not tried."
_SKIPPED_NO_PASTE = "Skipped because there is no safe change to paste."
_SKIPPED_NO_GPU = "Skipped because GPU not found."
_NOTEBOOK_WORKER_CAP = 2
_LOADER_ALREADY_SET = (
    "Your DataLoader already uses these settings. "
    "Colab and Kaggle give you only two CPU cores, so more workers will not help. "
    "Make loading cheaper: smaller images, lighter augmentations, "
    "or preprocess the data ahead of time."
)


@dataclass
class _Recorded:
    diagnosis: Any | None = None
    experiment: Any | None = None
    measurement: Any | None = None


_recorded = _Recorded()


def record(
    *,
    diagnosis: Any | None = None,
    experiment: Any | None = None,
    measurement: Any | None = None,
) -> None:
    """Keep the latest results so ``trainpilot run`` can print them."""

    if diagnosis is not None:
        _recorded.diagnosis = diagnosis
    if experiment is not None:
        _recorded.experiment = experiment
    if measurement is not None:
        _recorded.measurement = measurement


def clear_record() -> None:
    """Drop results remembered during an earlier run."""

    _recorded.diagnosis = None
    _recorded.experiment = None
    _recorded.measurement = None


def recorded() -> _Recorded:
    return _recorded


def render_report(
    diagnosis: Any | None = None,
    experiment: Any | None = None,
    measurement: Any | None = None,
) -> str:
    """Return the one-page report as markdown."""

    lines: list[str] = ["# TrainPilot", ""]
    lines.extend(_finding_lines(diagnosis, measurement))
    lines.extend(_tried_lines(diagnosis, experiment))
    lines.extend(_memory_lines(experiment, measurement))
    lines.append(_FULL_RUN_NOTE)
    lines.append("")
    lines.append("## Not checked")
    lines.append("")
    for note in _skipped_lines(diagnosis, experiment):
        lines.append(note)
    lines.append("")
    lines.append("## Code to paste")
    lines.append("")
    lines.append("TrainPilot did not change your script. Paste this yourself, or skip it.")
    lines.append("")
    snippet = code_snippet(diagnosis, experiment)
    if snippet is None:
        lines.append(_SKIPPED_NO_PASTE)
    else:
        lines.append("```python")
        lines.append(snippet)
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def code_snippet(diagnosis: Any | None, experiment: Any | None) -> str | None:
    """Return 3–5 lines to paste, or None when no change is safe."""

    if diagnosis is not None and getattr(diagnosis, "verdict", None) == "waiting for data":
        return _loader_snippet(diagnosis)
    if _amp_is_safe(experiment):
        return "\n".join(
            [
                'scaler = torch.amp.GradScaler("cuda")',
                'with torch.autocast(device_type="cuda", dtype=torch.float16):',
                "    loss = loss_fn(model(inputs), targets)",
                "scaler.scale(loss).backward()",
                "scaler.step(optimizer); scaler.update()",
            ]
        )
    return None


def _loader_already_tuned(diagnosis: Any | None) -> bool:
    """True when the loader already has the notebook-safe worker cap and pin_memory."""

    if diagnosis is None:
        return False
    workers = getattr(diagnosis, "loader_num_workers", None)
    pin_memory = getattr(diagnosis, "loader_pin_memory", None)
    return (
        workers is not None
        and workers >= _NOTEBOOK_WORKER_CAP
        and pin_memory is True
    )


def _loader_snippet(diagnosis: Any) -> str | None:
    """Suggest only DataLoader knobs that are not already set. Never above 2 workers."""

    if _loader_already_tuned(diagnosis):
        return None
    workers = getattr(diagnosis, "loader_num_workers", None)
    pin_memory = getattr(diagnosis, "loader_pin_memory", None)
    workers_ok = workers is not None and workers >= _NOTEBOOK_WORKER_CAP
    pin_ok = pin_memory is True
    lines = ["loader = DataLoader(", "    dataset,"]
    if not workers_ok:
        lines.append(f"    num_workers={_NOTEBOOK_WORKER_CAP},")
    if not pin_ok:
        lines.append("    pin_memory=True,")
    lines.append(")")
    return "\n".join(lines)


def _finding_lines(diagnosis: Any | None, measurement: Any | None) -> list[str]:
    lines: list[str] = []
    if diagnosis is None:
        if measurement is not None and getattr(measurement, "status", None) == GPU_NOT_FOUND:
            lines.append("GPU not found, so this run cannot say whether a GPU is waiting or busy.")
        elif measurement is not None and getattr(measurement, "median_step_seconds", None):
            lines.append(
                "A typical step on this short test took "
                f"{_format_seconds(measurement.median_step_seconds)}."
            )
        else:
            lines.append("GPU not found." if not _cuda_available() else "No short test was recorded.")
        lines.append("")
        return lines

    verdict = getattr(diagnosis, "verdict", "")
    if verdict == "waiting for data":
        lines.append("The GPU is waiting for the next batch, not doing math.")
    elif verdict == "gpu busy":
        lines.append("The GPU is busy computing. It is not sitting idle.")
    elif verdict == GPU_NOT_FOUND:
        lines.append("GPU not found, so this run cannot say whether a GPU is waiting or busy.")
    else:
        lines.append("This short test did not decide whether the GPU is waiting or busy.")
    reason = getattr(diagnosis, "reason", "")
    if reason:
        lines.append("")
        lines.append(reason)
    if verdict == "waiting for data" and _loader_already_tuned(diagnosis):
        lines.append("")
        lines.append(_LOADER_ALREADY_SET)
    lines.append("")
    return lines


def _tried_lines(diagnosis: Any | None, experiment: Any | None) -> list[str]:
    if experiment is None:
        return []
    if getattr(experiment, "skipped_reason", None):
        return []
    baseline = getattr(experiment, "baseline_median_step_seconds", None)
    faster = getattr(experiment, "amp_median_step_seconds", None)
    speedup = getattr(experiment, "median_step_speedup", None)
    waiting = getattr(diagnosis, "verdict", None) == "waiting for data"
    lines = ["## What we tried", ""]
    if baseline is None or faster is None or speedup is None:
        lines.append("Mixed precision did not produce a step time on this short test.")
    elif speedup >= 1.1 and getattr(experiment, "loss_ok", False):
        subject = "the math" if waiting else "a typical step"
        lines.append(
            "Mixed precision (FP16) made "
            f"{subject} faster on this short test: "
            f"{_format_seconds(baseline)} down to {_format_seconds(faster)}. "
            f"That is about {_format_ratio(speedup)} times faster."
        )
        if waiting:
            lines.append(_waiting_followup(diagnosis))
        elif getattr(diagnosis, "verdict", None) == "gpu busy":
            lines.append(
                "A busy GPU can still be busy with slow math. "
                "This short test found a faster step, so that is the change to paste."
            )
    elif speedup >= 1.1:
        lines.append(
            "Mixed precision (FP16) made a typical step faster on this short test: "
            f"{_format_seconds(baseline)} down to {_format_seconds(faster)}. "
            "The loss did not stay healthy, so this is not a change to keep."
        )
        if waiting:
            lines.append(_waiting_followup(diagnosis))
    else:
        lines.append(
            "Mixed precision (FP16) did not make a typical step clearly faster "
            f"on this short test: {_format_seconds(baseline)} and {_format_seconds(faster)}. "
            "Little is left to speed up from this change."
        )
    loss_note = getattr(experiment, "loss_note", "")
    if loss_note:
        lines.append(loss_note)
        if getattr(experiment, "loss_ok", False):
            lines.append("That only means the short test did not blow up. It does not prove the model is unchanged.")
    compile_note = getattr(experiment, "compile_note", None)
    compile_seconds = getattr(experiment, "compile_seconds", None)
    compile_step = getattr(experiment, "compile_median_step_seconds", None)
    amp_step = getattr(experiment, "amp_median_step_seconds", None)
    if compile_seconds is not None and compile_step is not None:
        if amp_step is not None:
            lines.append(
                f"Preparing the compiled model took {_format_seconds(compile_seconds)}. "
                f"After that, a typical step took {_format_seconds(compile_step)}, "
                f"compared with {_format_seconds(amp_step)} for mixed precision without compile. "
                "The prepare time is separate from the step time."
            )
        else:
            lines.append(
                f"Preparing the compiled model took {_format_seconds(compile_seconds)}. "
                f"After that, a typical step took {_format_seconds(compile_step)}. "
                "The prepare time is separate from the step time."
            )
        if compile_note and not str(compile_note).startswith("Skipped because"):
            lines.append(compile_note)
        compile_loss_ok = getattr(experiment, "compile_loss_ok", None)
        if compile_loss_ok:
            lines.append(
                "On the compile run, loss did not diverge and there are no NaN. "
                "That only means the short test did not blow up."
            )
    lines.append("")
    return lines


def _memory_lines(experiment: Any | None, measurement: Any | None) -> list[str]:
    peak = None
    if experiment is not None:
        peak = getattr(experiment, "baseline_peak_memory_bytes", None)
    if peak is None and measurement is not None:
        peak = getattr(measurement, "peak_memory_bytes", None)
    if not peak:
        return []
    return [f"Peak GPU memory on this short test was about {_format_bytes(peak)}.", ""]


def _skipped_lines(diagnosis: Any | None, experiment: Any | None) -> list[str]:
    notes: list[str] = []
    if diagnosis is None:
        notes.append(_SKIPPED_NO_DIAGNOSIS if _cuda_available() else _SKIPPED_NO_GPU)
    elif getattr(diagnosis, "verdict", None) == GPU_NOT_FOUND:
        notes.append(_SKIPPED_NO_GPU)
    elif getattr(diagnosis, "verdict", None) == "skipped":
        reason = getattr(diagnosis, "reason", "")
        notes.append(reason if reason.startswith("Skipped because") else _SKIPPED_NO_DIAGNOSIS)

    if experiment is None:
        notes.append(_SKIPPED_NO_EXPERIMENT if _cuda_available() else _SKIPPED_NO_GPU)
    else:
        skipped = getattr(experiment, "skipped_reason", None)
        if skipped:
            notes.append(skipped if skipped.startswith("Skipped because") else f"Skipped because {skipped}")
        compile_note = getattr(experiment, "compile_note", None)
        if compile_note and compile_note.startswith("Skipped because"):
            notes.append(compile_note)
    notes.append(_SKIPPED_BATCH_SIZE)
    return _unique(notes)


def _waiting_followup(diagnosis: Any | None) -> str:
    if _loader_already_tuned(diagnosis):
        return (
            "Most of the step is still spent waiting for the next batch. "
            + _LOADER_ALREADY_SET
        )
    return (
        "Most of the step is still spent waiting for the next batch, "
        "so the change to paste is the data loader."
    )


def _amp_is_safe(experiment: Any | None) -> bool:
    if experiment is None or getattr(experiment, "skipped_reason", None):
        return False
    speedup = getattr(experiment, "median_step_speedup", None)
    return bool(getattr(experiment, "loss_ok", False) and speedup is not None and speedup >= 1.1)


def _format_ratio(speedup: float) -> str:
    return f"{speedup:.2f}".rstrip("0").rstrip(".")


def _format_seconds(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.1f} s"


def _format_bytes(size: int) -> str:
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    if size >= 1_000:
        return f"{size / 1_000:.1f} KB"
    return f"{size} bytes"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _unique(notes: list[str]) -> list[str]:
    seen: list[str] = []
    for note in notes:
        if note not in seen:
            seen.append(note)
    return seen


def short_run_label() -> str:
    return SHORT_RUN_LABEL
