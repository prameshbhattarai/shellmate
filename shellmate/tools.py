"""Tools: the only actions the agent can take.

Implemented in milestones 1 and 4. Guarantees each tool must keep:
- ``run_command``: runs only after the risk gate, with a timeout, inside the working directory,
  and returns truncated stdout/stderr plus the exit code (never raises on a failing command).
- ``read_file`` / ``list_dir``: read-only, confined to the working directory.
- ``last_error``: the most recent failed command and its stderr, from shellmate's own history.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

TIMEOUT_EXIT_CODE = 124  # same code GNU `timeout` uses


@dataclass(frozen=True)
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool = False


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]", True


def run_command(command: str, *, cwd: str, timeout_s: int, max_output_chars: int) -> CommandResult:
    """Run ``command`` in the user's shell. Never raises for a failing or hanging command."""
    shell = os.environ.get("SHELL") or "/bin/sh"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            executable=shell,
            cwd=cwd,
            stdin=subprocess.DEVNULL,  # nothing may wait for keyboard input (it would hang until the timeout)
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = TIMEOUT_EXIT_CODE
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = f"shellmate: command timed out after {timeout_s}s"

    stdout, out_cut = _truncate(stdout, max_output_chars)
    stderr, err_cut = _truncate(stderr, max_output_chars)
    return CommandResult(command, exit_code, stdout, stderr, truncated=out_cut or err_cut)


def read_file(path: str, *, cwd: str, max_chars: int) -> str:
    raise NotImplementedError("read_file lands in milestone 4")


def list_dir(path: str, *, cwd: str) -> list[str]:
    raise NotImplementedError("list_dir lands in milestone 4")


def last_error() -> CommandResult | None:
    raise NotImplementedError("last_error lands in milestone 4")
