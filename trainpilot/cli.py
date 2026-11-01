"""Command-line interface for ``trainpilot run train.py``.

The command runs a script and prints a one-page report. It does not edit
the script. Paste any suggested lines yourself.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from trainpilot.report import clear_record, recorded, render_report

_USAGE = "Usage: trainpilot run train.py"
_REPORT_NAME = "trainpilot-report.md"


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``trainpilot`` console script."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        print(_USAGE)
        return
    if len(args) != 2 or args[0] != "run":
        print(_USAGE, file=sys.stderr)
        raise SystemExit(2)

    script = Path(args[1])
    if not script.is_file():
        print(f"No such script: {script}", file=sys.stderr)
        raise SystemExit(2)

    before = script.read_bytes()
    clear_record()
    try:
        runpy.run_path(str(script.resolve()), run_name="__main__")
    except SystemExit as exc:
        _emit_report()
        if exc.code not in (0, None):
            raise
    else:
        _emit_report()
    if script.read_bytes() != before:
        print("Skipped because the script file changed during the run.", file=sys.stderr)


def _emit_report() -> None:
    state = recorded()
    text = render_report(state.diagnosis, state.experiment, state.measurement)
    print(text)
    try:
        Path(_REPORT_NAME).write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"Skipped because the report file could not be written: {exc}", file=sys.stderr)
