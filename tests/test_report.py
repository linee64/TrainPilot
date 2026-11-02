"""The one-page report stays in plain language."""

from trainpilot.diagnose import Diagnosis
from trainpilot.experiments import ExperimentResult
from trainpilot.report import code_snippet, render_report

_BANNED = ("profiler", "kernel", "bottleneck", "occupancy", "nvtx")


def test_busy_gpu_report_is_plain_and_labels_the_short_test() -> None:
    text = render_report(_busy_diagnosis(), _fast_experiment())
    prose, _, _snippet = text.partition("## Code to paste")

    assert "The GPU is busy computing" in prose
    assert "574.2 ms" in prose
    assert "176.5 ms" in prose
    assert "3.25 times faster" in prose
    assert "does not prove the model is unchanged" in prose
    assert "short test" in prose
    assert "estimate" in prose
    assert "Skipped because torch.compile was not requested." in prose
    assert "Skipped because this version does not change the batch size." in prose
    for word in _BANNED:
        assert word not in prose.lower()
    assert 'dtype=torch.float16' in text
    assert _line_count(code_snippet(_busy_diagnosis(), _fast_experiment())) == 5


def test_waiting_report_offers_a_loader_snippet() -> None:
    diagnosis = Diagnosis(
        verdict="waiting for data",
        reason="On this short run, 89% of a typical step is spent waiting for the next batch. The GPU is idle while that happens.",
        wait_fraction=0.894,
        loader_num_workers=0,
        loader_pin_memory=False,
    )
    snippet = code_snippet(diagnosis, None)
    text = render_report(diagnosis, None)

    assert snippet is not None
    assert 3 <= _line_count(snippet) <= 5
    assert "num_workers=2" in snippet
    assert "pin_memory=True" in snippet
    assert "num_workers=3" not in snippet
    assert "The GPU is waiting for the next batch" in text
    assert "Skipped because" in text
    assert "GPU not found" in text or "mixed precision was not tried" in text


def test_waiting_report_does_not_repeat_loader_settings_already_set() -> None:
    diagnosis = Diagnosis(
        verdict="waiting for data",
        reason="On this short run, 65.0% of a typical step is spent waiting for the next batch. The GPU is idle while that happens.",
        wait_fraction=0.65,
        loader_num_workers=2,
        loader_pin_memory=True,
    )
    snippet = code_snippet(diagnosis, None)
    text = render_report(diagnosis, None)

    assert snippet is None
    assert "num_workers=2" not in text.split("## Code to paste")[1]
    assert "two CPU cores" in text
    assert "smaller images" in text
    assert "lighter augmentations" in text
    assert "preprocess the data" in text
    assert "Skipped because there is no safe change to paste." in text


def test_waiting_loader_advice_is_not_repeated_in_what_we_tried() -> None:
    diagnosis = Diagnosis(
        verdict="waiting for data",
        reason="On this short run, 75.8% of a typical step is spent waiting for the next batch. The GPU is idle while that happens.",
        wait_fraction=0.758,
        loader_num_workers=2,
        loader_pin_memory=True,
    )
    text = render_report(diagnosis, _fast_experiment())
    finding, _, rest = text.partition("## What we tried")
    tried, _, _ = rest.partition("##")

    assert text.count("two CPU cores") == 1
    assert "two CPU cores" in finding
    assert "two CPU cores" not in tried
    assert "574.2 ms" in tried
    assert "176.5 ms" in tried
    assert "Most of the step is still spent waiting for the next batch." in tried
    assert "Your DataLoader already uses these settings." not in tried


def test_waiting_report_suggests_only_the_missing_loader_knob() -> None:
    diagnosis = Diagnosis(
        verdict="waiting for data",
        reason="On this short run, 71.2% of a typical step is spent waiting for the next batch. The GPU is idle while that happens.",
        wait_fraction=0.712,
        loader_num_workers=2,
        loader_pin_memory=False,
    )
    snippet = code_snippet(diagnosis, None)

    assert snippet is not None
    assert "pin_memory=True" in snippet
    assert "num_workers" not in snippet


def test_broken_loss_is_not_offered_as_code() -> None:
    experiment = _fast_experiment()
    broken = ExperimentResult(
        status=experiment.status,
        skipped_reason=None,
        baseline_median_step_seconds=experiment.baseline_median_step_seconds,
        amp_median_step_seconds=experiment.amp_median_step_seconds,
        median_step_speedup=experiment.median_step_speedup,
        baseline_losses=experiment.baseline_losses,
        amp_losses=experiment.amp_losses,
        loss_ok=False,
        loss_note="Mixed-precision loss diverged.",
        amp_dtype="float16",
        compile_note=experiment.compile_note,
    )

    assert code_snippet(_busy_diagnosis(), broken) is None
    assert "not a change to keep" in render_report(_busy_diagnosis(), broken)
    assert "Skipped because there is no safe change to paste." in render_report(_busy_diagnosis(), broken)


def test_busy_gpu_with_a_real_speedup_offers_mixed_precision() -> None:
    text = render_report(_busy_diagnosis(), _fast_experiment())

    assert "A busy GPU can still be busy with slow math." in text
    assert "that is the change to paste" in text
    assert "GradScaler" in text
    assert "Little is left" not in text


def test_waiting_gpu_keeps_the_loader_even_if_math_is_faster() -> None:
    diagnosis = Diagnosis(
        verdict="waiting for data",
        reason="On this short run, 89% of a typical step is spent waiting for the next batch. The GPU is idle while that happens.",
        wait_fraction=0.894,
    )
    text = render_report(diagnosis, _fast_experiment())
    snippet = code_snippet(diagnosis, _fast_experiment())

    assert snippet is not None
    assert "num_workers=2" in snippet
    assert "GradScaler" not in snippet
    assert "made the math faster" in text
    assert "change to paste is the data loader" in text


def test_small_speedup_says_little_is_left() -> None:
    experiment = _fast_experiment()
    small = ExperimentResult(
        status=experiment.status,
        skipped_reason=None,
        baseline_median_step_seconds=0.372,
        amp_median_step_seconds=0.360,
        median_step_speedup=1.03,
        baseline_losses=experiment.baseline_losses,
        amp_losses=experiment.amp_losses,
        loss_ok=True,
        loss_note=experiment.loss_note,
        amp_dtype="float16",
        compile_note=experiment.compile_note,
    )
    text = render_report(_busy_diagnosis(), small)

    assert "Little is left to speed up from this change." in text
    assert code_snippet(_busy_diagnosis(), small) is None


def test_speedup_without_a_diagnosis_does_not_call_the_gpu_busy() -> None:
    text = render_report(None, _fast_experiment())

    assert "3.25 times faster" in text
    assert "A busy GPU" not in text


def test_compile_not_worth_is_plain() -> None:
    experiment = _fast_experiment()
    compiled = ExperimentResult(
        status=experiment.status,
        skipped_reason=None,
        baseline_median_step_seconds=0.1966,
        amp_median_step_seconds=0.0586,
        median_step_speedup=3.13,
        baseline_losses=experiment.baseline_losses,
        amp_losses=experiment.amp_losses,
        loss_ok=True,
        loss_note=experiment.loss_note,
        amp_dtype="float16",
        compile_seconds=19.0,
        compile_median_step_seconds=0.1865,
        compile_note="torch.compile barely speeds this up on this GPU. It is not worth the wait.",
        compile_loss_ok=True,
        compile_speedup=1.05,
        compile_payback_steps=None,
    )
    text = render_report(_busy_diagnosis(), compiled)

    assert "Preparing the compiled model took 19.0 s" in text
    assert "186.5 ms" in text
    assert "58.6 ms" in text
    assert "mixed precision without compile" in text
    assert "not worth the wait" in text
    assert "On the compile run, loss did not diverge" in text
    assert "Skipped because torch.compile was not requested." not in text
    snippet = code_snippet(_busy_diagnosis(), compiled)
    assert snippet is not None
    assert "torch.compile" not in snippet
    assert snippet.splitlines()[0].startswith("scaler = torch.amp.GradScaler")


def test_compile_payback_is_in_the_report() -> None:
    experiment = _fast_experiment()
    compiled = ExperimentResult(
        status=experiment.status,
        skipped_reason=None,
        baseline_median_step_seconds=experiment.baseline_median_step_seconds,
        amp_median_step_seconds=0.0963,
        median_step_speedup=3.22,
        baseline_losses=experiment.baseline_losses,
        amp_losses=experiment.amp_losses,
        loss_ok=True,
        loss_note=experiment.loss_note,
        amp_dtype="float16",
        compile_seconds=19.0,
        compile_median_step_seconds=0.04815,
        compile_note="torch.compile made a typical mixed-precision step faster. The extra prepare time pays for itself after about 395 steps.",
        compile_loss_ok=True,
        compile_speedup=2.0,
        compile_payback_steps=395,
    )
    text = render_report(_busy_diagnosis(), compiled)

    assert "pays for itself after about 395 steps" in text
    assert "48.2 ms" in text or "48.1 ms" in text
    snippet = code_snippet(_busy_diagnosis(), compiled)
    assert snippet is not None
    first = snippet.splitlines()[0]
    assert first.startswith("model = torch.compile(model)")
    assert "pays off after ~395 steps" in first
    assert "19.0 s" in first
    assert "GradScaler" in snippet
    assert snippet.index("torch.compile") < snippet.index("GradScaler")


def test_compile_is_absent_from_snippet_when_not_requested() -> None:
    snippet = code_snippet(_busy_diagnosis(), _fast_experiment())

    assert snippet is not None
    assert "torch.compile" not in snippet
    assert _line_count(snippet) == 5


def test_empty_report_on_cpu_says_gpu_not_found() -> None:
    text = render_report()

    assert "GPU not found" in text
    assert "Skipped because" in text
    assert "estimate" in text


def _line_count(snippet: str | None) -> int:
    assert snippet is not None
    return len([line for line in snippet.splitlines() if line.strip()])


def _busy_diagnosis() -> Diagnosis:
    return Diagnosis(
        verdict="gpu busy",
        reason="On this short run, 2% of a typical step is spent waiting for the next batch. The GPU is busy computing the rest.",
        wait_fraction=0.022,
    )


def _fast_experiment() -> ExperimentResult:
    return ExperimentResult(
        status="ok",
        skipped_reason=None,
        baseline_median_step_seconds=0.5742,
        amp_median_step_seconds=0.1765,
        median_step_speedup=3.25,
        baseline_losses=(2.0,),
        amp_losses=(2.1,),
        loss_ok=True,
        loss_note="Loss did not diverge and there are no NaN.",
        amp_dtype="float16",
        compile_note="Skipped because torch.compile was not requested.",
    )
