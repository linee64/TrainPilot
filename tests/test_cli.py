"""``trainpilot run`` prints a report and does not edit the script."""

import pytest

from trainpilot.cli import main
from trainpilot.report import clear_record


def test_run_prints_a_report_without_editing_the_script(tmp_path, monkeypatch, capsys) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "\n".join(
            [
                "from trainpilot.diagnose import diagnose",
                "from trainpilot.measure import StepMeasurement",
                "diagnose(StepMeasurement(",
                "    step_seconds=(0.1, 0.1, 0.1),",
                "    example_counts=(8, 8, 8),",
                "    median_step_seconds=0.1,",
                "    examples_per_second=80.0,",
                "    peak_memory_bytes=None,",
                "    gpu_name='Tesla T4',",
                "    warmup_steps=10,",
                "    measured_steps=3,",
                "    status='ok',",
                "    data_wait_seconds=(0.09, 0.09, 0.09),",
                "    compute_seconds=(0.01, 0.01, 0.01),",
                "    median_data_wait_seconds=0.09,",
                "    median_compute_seconds=0.01,",
                "    wait_fraction=0.9,",
                "))",
                "print('script-ran')",
            ]
        ),
        encoding="utf-8",
    )
    before = script.read_bytes()
    monkeypatch.chdir(tmp_path)
    clear_record()

    main(["run", "train.py"])

    assert script.read_bytes() == before
    report = (tmp_path / "trainpilot-report.md").read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "script-ran" in output
    assert "The GPU is waiting for the next batch" in report
    assert "short test" in report
    assert "estimate" in report
    assert "Skipped because" in report
    assert "num_workers=2" in report
    assert output.endswith(report) or report in output


def test_missing_script_exits(tmp_path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["run", str(tmp_path / "missing.py")])
    assert caught.value.code == 2


def test_usage_exits_when_the_command_is_wrong() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["train.py"])
    assert caught.value.code == 2
