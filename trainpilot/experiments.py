"""Compare a baseline run with mixed precision, and optionally torch.compile.

Experiments run on copies. The caller's model, optimizer, and script stay
unchanged. Both runs see the same seed and the same batches. Mixed precision
is FP16 with GradScaler. BF16 is never selected, including on a T4.

``torch.compile`` is optional. Its one-time compile cost is timed apart from
the median step time. A skipped check is reported with ``Skipped because``.
"""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from trainpilot.measure import GPU_NOT_FOUND, SHORT_RUN_LABEL, measure_steps

LOSS_DIVERGENCE_FACTOR = 10.0
LOSS_DIVERGENCE_MARGIN = 10.0
LOSS_OK_NOTE = "Loss did not diverge and there are no NaN."
SKIPPED_NO_GPU = "Skipped because GPU not found."
SKIPPED_NOT_CUDA_MODEL = "Skipped because the model is not on a CUDA device."
SKIPPED_COMPILE_NOT_REQUESTED = "Skipped because torch.compile was not requested."

LossFn = Callable[[nn.Module, Any], torch.Tensor]


@dataclass(frozen=True)
class ExperimentResult:
    """Short-run comparison. A faster step is not a claim that quality is unchanged."""

    status: str
    skipped_reason: str | None
    baseline_median_step_seconds: float | None
    amp_median_step_seconds: float | None
    median_step_speedup: float | None
    baseline_losses: tuple[float, ...]
    amp_losses: tuple[float, ...]
    loss_ok: bool | None
    loss_note: str
    amp_dtype: str | None
    label: str = SHORT_RUN_LABEL
    compile_seconds: float | None = None
    compile_median_step_seconds: float | None = None
    compile_note: str | None = None
    seed: int | None = None
    warmup_steps: int = 10
    measured_steps: int = 30
    baseline_examples_per_second: float | None = None
    amp_examples_per_second: float | None = None
    baseline_peak_memory_bytes: int | None = None
    amp_peak_memory_bytes: int | None = None


def compare_mixed_precision(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[Any],
    loss_fn: LossFn,
    *,
    seed: int = 0,
    warmup: int = 10,
    steps: int = 30,
    try_compile: bool = False,
) -> ExperimentResult:
    """Time a baseline copy against an FP16 copy.

    ``loss_fn(model, batch)`` must use the model argument it is given and
    return a scalar loss. ``batches`` is read once. Both runs then replay
    that list. ``seed`` is applied before each run so random ops inside the
    step match. The originals are not stepped.
    """

    if not torch.cuda.is_available():
        return _skipped(SKIPPED_NO_GPU, status=GPU_NOT_FOUND, seed=seed, warmup=warmup, steps=steps)
    if _parameter_device(model) != "cuda":
        return _skipped(SKIPPED_NOT_CUDA_MODEL, seed=seed, warmup=warmup, steps=steps)

    window = _collect_batches(batches, warmup + steps)
    baseline_model, baseline_optimizer = _clone_model_and_optimizer(model, optimizer)
    amp_model, amp_optimizer = _clone_model_and_optimizer(model, optimizer)

    baseline = _execute(
        baseline_model,
        baseline_optimizer,
        window,
        loss_fn,
        seed=seed,
        warmup=warmup,
        steps=steps,
        amp=False,
    )
    del baseline_model, baseline_optimizer
    amp = _execute(
        amp_model,
        amp_optimizer,
        window,
        loss_fn,
        seed=seed,
        warmup=warmup,
        steps=steps,
        amp=True,
    )
    del amp_model, amp_optimizer

    loss_ok, loss_note = assess_losses(baseline.losses, amp.losses)
    compile_seconds = None
    compile_median = None
    compile_note = SKIPPED_COMPILE_NOT_REQUESTED
    if try_compile:
        compile_seconds, compile_median, compile_note = _try_compile(
            model,
            optimizer,
            window,
            loss_fn,
            seed=seed,
            warmup=warmup,
            steps=steps,
        )

    return ExperimentResult(
        status="ok",
        skipped_reason=None,
        baseline_median_step_seconds=baseline.measurement.median_step_seconds,
        amp_median_step_seconds=amp.measurement.median_step_seconds,
        median_step_speedup=_speedup(
            baseline.measurement.median_step_seconds,
            amp.measurement.median_step_seconds,
        ),
        baseline_losses=baseline.losses,
        amp_losses=amp.losses,
        loss_ok=loss_ok,
        loss_note=loss_note,
        amp_dtype="float16",
        compile_seconds=compile_seconds,
        compile_median_step_seconds=compile_median,
        compile_note=compile_note,
        seed=seed,
        warmup_steps=warmup,
        measured_steps=steps,
        baseline_examples_per_second=baseline.measurement.examples_per_second,
        amp_examples_per_second=amp.measurement.examples_per_second,
        baseline_peak_memory_bytes=baseline.measurement.peak_memory_bytes,
        amp_peak_memory_bytes=amp.measurement.peak_memory_bytes,
    )


def assess_losses(
    baseline_losses: Sequence[float],
    amp_losses: Sequence[float],
) -> tuple[bool, str]:
    """Return whether mixed precision broke the short run.

    Broken means a non-finite loss, or a mixed-precision median far above
    the baseline median. This does not claim the model quality is unchanged.
    """

    if not baseline_losses or not amp_losses:
        return False, "No measured losses."
    if any(not math.isfinite(value) for value in baseline_losses):
        return False, "NaN in the baseline loss."
    if any(not math.isfinite(value) for value in amp_losses):
        return False, "NaN in the mixed-precision loss."
    baseline_median = float(statistics.median(baseline_losses))
    amp_median = float(statistics.median(amp_losses))
    limit = max(baseline_median * LOSS_DIVERGENCE_FACTOR, baseline_median + LOSS_DIVERGENCE_MARGIN)
    if amp_median > limit:
        return False, "Mixed-precision loss diverged."
    return True, LOSS_OK_NOTE


def select_amp_dtype(device_name: str) -> torch.dtype:
    """Return the autocast dtype. v0.1 always uses FP16, including on a T4."""

    if "t4" in device_name.lower():
        return torch.float16
    return torch.float16


@dataclass(frozen=True)
class _Run:
    measurement: Any
    losses: tuple[float, ...]


def _execute(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[Any],
    loss_fn: LossFn,
    *,
    seed: int,
    warmup: int,
    steps: int,
    amp: bool,
) -> _Run:
    _set_seed(seed)
    model.train()
    pending = list(batches)
    if len(pending) < warmup + steps:
        raise RuntimeError(f"Not enough batches: needed {warmup + steps}, got {len(pending)}.")

    losses: list[float] = []
    index = 0
    scaler = _grad_scaler() if amp else None
    if amp and select_amp_dtype(_device_name()) is not torch.float16:
        raise RuntimeError("mixed precision must use FP16")

    def step() -> int:
        nonlocal index
        batch = pending[index]
        index += 1
        optimizer.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_fn(model, batch)
            _check_scalar_loss(loss)
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model, batch)
            _check_scalar_loss(loss)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
        return _examples_in_batch(batch)

    measurement = measure_steps(step, warmup=warmup, steps=steps)
    return _Run(measurement=measurement, losses=tuple(losses[warmup:]))


def _try_compile(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[Any],
    loss_fn: LossFn,
    *,
    seed: int,
    warmup: int,
    steps: int,
) -> tuple[float | None, float | None, str | None]:
    compile_model, compile_optimizer = _clone_model_and_optimizer(model, optimizer)
    try:
        compiled = torch.compile(compile_model)
        trigger = _execute(
            compiled,
            compile_optimizer,
            [batches[0]],
            loss_fn,
            seed=seed,
            warmup=0,
            steps=1,
            amp=False,
        )
        steady = _execute(
            compiled,
            compile_optimizer,
            batches,
            loss_fn,
            seed=seed,
            warmup=warmup,
            steps=steps,
            amp=False,
        )
    except Exception as exc:
        return None, None, f"Skipped because torch.compile failed: {exc}"
    return trigger.measurement.median_step_seconds, steady.measurement.median_step_seconds, None


def _clone_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    model_copy = copy.deepcopy(model)
    mapping = _parameter_mapping(model, model_copy)
    groups = []
    for group in optimizer.param_groups:
        cloned = {key: value for key, value in group.items() if key != "params"}
        try:
            cloned["params"] = [mapping[id(param)] for param in group["params"]]
        except KeyError as exc:
            raise RuntimeError(
                "An optimizer parameter is not part of the model, so no copy was made."
            ) from exc
        groups.append(cloned)
    optimizer_copy = optimizer.__class__(groups)
    optimizer_copy.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return model_copy, optimizer_copy


def _parameter_mapping(model: nn.Module, model_copy: nn.Module) -> dict[int, nn.Parameter]:
    original = list(model.parameters())
    copied = list(model_copy.parameters())
    if len(original) != len(copied):
        raise RuntimeError("The copied model does not have the same parameters.")
    return {id(old): new for old, new in zip(original, copied)}


def _collect_batches(batches: Iterable[Any], count: int) -> list[Any]:
    collected: list[Any] = []
    for batch in batches:
        collected.append(batch)
        if len(collected) == count:
            return collected
    raise RuntimeError(f"Not enough batches: needed {count}, got {len(collected)}.")


def _examples_in_batch(batch: Any) -> int:
    if isinstance(batch, bool):
        raise TypeError("batch example count is not a bool")
    if isinstance(batch, int):
        return batch
    tensor = batch if torch.is_tensor(batch) else None
    if tensor is None and isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        tensor = batch[0]
    if tensor is None or tensor.ndim == 0:
        return 1
    return int(tensor.shape[0])


def _check_scalar_loss(loss: torch.Tensor) -> None:
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise TypeError("loss_fn must return a scalar loss tensor")


def _grad_scaler() -> Any:
    return torch.amp.GradScaler("cuda")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameter_device(model: nn.Module) -> str | None:
    try:
        return next(model.parameters()).device.type
    except StopIteration:
        return None


def _device_name() -> str:
    if not torch.cuda.is_available():
        return ""
    return str(torch.cuda.get_device_name(0))


def _speedup(baseline_seconds: float, amp_seconds: float) -> float | None:
    if amp_seconds <= 0:
        return None
    return baseline_seconds / amp_seconds


def _skipped(
    reason: str,
    *,
    status: str = "skipped",
    seed: int,
    warmup: int,
    steps: int,
) -> ExperimentResult:
    return ExperimentResult(
        status=status,
        skipped_reason=reason,
        baseline_median_step_seconds=None,
        amp_median_step_seconds=None,
        median_step_speedup=None,
        baseline_losses=(),
        amp_losses=(),
        loss_ok=None,
        loss_note=reason,
        amp_dtype=None,
        compile_note=reason,
        seed=seed,
        warmup_steps=warmup,
        measured_steps=steps,
    )
