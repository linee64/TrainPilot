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
    )
    snippet = code_snippet(diagnosis, None)
    text = render_report(diagnosis, None)

    assert snippet is not None
    assert 3 <= _line_count(snippet) <= 5
    assert "num_workers=2" in snippet
    assert "pin_memory=True" in snippet
    assert "The GPU is waiting for the next batch" in text
    assert "Skipped because" in text
    assert "GPU not found" in text or "mixed precision was not tried" in text


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
