"""The T4 notebook runner must not crash when this machine has no GPU."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(torch.cuda.is_available(), reason="GPU not found check is for a CPU machine")
def test_runner_reports_gpu_not_found() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "notebooks" / "run_t4_checks.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "GPU not found" in completed.stdout


def test_wrong_verdict_is_not_ok() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks" / "run_t4_checks.py"
    spec = importlib.util.spec_from_file_location("run_t4_checks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._verdict_ok("01 CIFAR", "waiting for data")
    assert module._verdict_ok("03 ResNet-50", "gpu busy")
    assert not module._verdict_ok("01 CIFAR", "gpu busy")
