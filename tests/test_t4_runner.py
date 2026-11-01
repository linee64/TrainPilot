"""The T4 notebook runner must not crash when this machine has no GPU."""

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
