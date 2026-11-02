"""Command-line interface for ``trainpilot run train.py``.

The command runs a script and prints a one-page report. It does not edit
the script. Paste any suggested lines yourself.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any, TextIO

from trainpilot.experiments import set_cli_try_compile
from trainpilot.report import clear_record, recorded, render_report

_USAGE = "Usage: trainpilot run train.py [--compile]"
_REPORT_NAME = "trainpilot-report.md"


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``trainpilot`` console script."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        print(_USAGE)
        return
    try_compile, args = _split_compile_flag(args)
    if len(args) != 2 or args[0] != "run":
        print(_USAGE, file=sys.stderr)
        raise SystemExit(2)

    script = Path(args[1])
    if not script.is_file():
        print(f"No such script: {script}", file=sys.stderr)
        raise SystemExit(2)

    before = script.read_bytes()
    clear_record()
    original_stdout = sys.stdout
    tee = _StdoutTee(original_stdout)
    set_cli_try_compile(try_compile)
    try:
        sys.stdout = tee
        try:
            runpy.run_path(str(script.resolve()), run_name="__main__")
        except SystemExit as exc:
            sys.stdout = original_stdout
            _emit_report(tee.getvalue())
            if exc.code not in (0, None):
                raise
        except Exception:
            sys.stdout = original_stdout
            _emit_report(tee.getvalue())
            raise
        else:
            sys.stdout = original_stdout
            _emit_report(tee.getvalue())
    finally:
        sys.stdout = original_stdout
        set_cli_try_compile(False)
    if script.read_bytes() != before:
        print("Skipped because the script file changed during the run.", file=sys.stderr)


def _split_compile_flag(args: list[str]) -> tuple[bool, list[str]]:
    try_compile = False
    kept: list[str] = []
    for arg in args:
        if arg == "--compile":
            try_compile = True
        else:
            kept.append(arg)
    return try_compile, kept


def _emit_report(already_printed: str = "") -> None:
    state = recorded()
    text = render_report(state.diagnosis, state.experiment, state.measurement)
    try:
        Path(_REPORT_NAME).write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"Skipped because the report file could not be written: {exc}", file=sys.stderr)
    if _report_already_printed(already_printed, text):
        return
    print(text)


def _report_already_printed(captured: str, text: str) -> bool:
    body = text.strip()
    if not body:
        return False
    return body in captured.replace("\r\n", "\n")


class _StdoutTee:
    """Copy script prints so the CLI can skip a report the script already showed."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self.chunks: list[str] = []

    def write(self, data: str) -> int:
        if data:
            self.chunks.append(data)
        return self._stream.write(data)

    def flush(self) -> None:
        self._stream.flush()

    def getvalue(self) -> str:
        return "".join(self.chunks)

    def isatty(self) -> bool:
        getter = getattr(self._stream, "isatty", None)
        return bool(getter()) if callable(getter) else False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
