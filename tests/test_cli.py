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


def test_cli_does_not_reprint_a_report_the_script_already_printed(
    tmp_path, monkeypatch, capsys
) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "\n".join(
            [
                "from trainpilot.diagnose import diagnose",
                "from trainpilot.measure import StepMeasurement",
                "from trainpilot.report import render_report",
                "diagnosis = diagnose(StepMeasurement(",
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
                "print(render_report(diagnosis))",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    clear_record()

    main(["run", "train.py"])

    output = capsys.readouterr().out
    report = (tmp_path / "trainpilot-report.md").read_text(encoding="utf-8")
    assert "The GPU is waiting for the next batch" in report
    assert output.count("The GPU is waiting for the next batch") == 1
    assert output.count("# TrainPilot") == 1


def test_crash_still_writes_the_report(tmp_path, monkeypatch) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "\n".join(
            [
                "from trainpilot.diagnose import diagnose",
                "from trainpilot.measure import StepMeasurement",
                "diagnose(StepMeasurement(",
                "    step_seconds=(0.1,),",
                "    example_counts=(8,),",
                "    median_step_seconds=0.1,",
                "    examples_per_second=80.0,",
                "    peak_memory_bytes=None,",
                "    gpu_name='Tesla T4',",
                "    warmup_steps=10,",
                "    measured_steps=1,",
                "    status='ok',",
                "    data_wait_seconds=(0.09,),",
                "    compute_seconds=(0.01,),",
                "    median_data_wait_seconds=0.09,",
                "    median_compute_seconds=0.01,",
                "    wait_fraction=0.9,",
                "))",
                "raise RuntimeError('boom')",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    clear_record()

    with pytest.raises(RuntimeError, match="boom"):
        main(["run", "train.py"])

    report = (tmp_path / "trainpilot-report.md").read_text(encoding="utf-8")
    assert "waiting for the next batch" in report


def test_missing_script_exits(tmp_path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["run", str(tmp_path / "missing.py")])
    assert caught.value.code == 2


def test_usage_exits_when_the_command_is_wrong() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["train.py"])
    assert caught.value.code == 2


def test_compile_flag_is_passed_to_try_compile(tmp_path, monkeypatch, capsys) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "from trainpilot.experiments import _cli_try_compile\n"
        "print(f'cli_compile={_cli_try_compile}')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    clear_record()

    main(["run", "train.py", "--compile"])
    with_flag = capsys.readouterr().out
    assert "cli_compile=True" in with_flag

    main(["run", "--compile", "train.py"])
    flagged_before_path = capsys.readouterr().out
    assert "cli_compile=True" in flagged_before_path

    main(["run", "train.py"])
    without_flag = capsys.readouterr().out
    assert "cli_compile=False" in without_flag


def test_help_mentions_compile() -> None:
    import trainpilot.cli as cli

    assert "--compile" in cli._USAGE
