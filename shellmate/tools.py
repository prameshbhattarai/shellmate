"""Tools: the only actions the agent can take.

Implemented in milestones 1 and 4. Guarantees each tool must keep:
- ``run_command``: runs only after the risk gate, with a timeout, inside the working directory,
  and returns truncated stdout/stderr plus the exit code (never raises on a failing command).
- ``read_file`` / ``list_dir``: read-only, confined to the working directory.
- ``last_error``: the most recent failed command and its stderr, from shellmate's own history.
"""

from __future__ import annotations

import os
import signal
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


# After the kill, how long to wait for the pipes to close before giving up on the rest of the output.
DRAIN_AFTER_KILL_S = 2


def _shell() -> str:
    shell = os.environ.get("SHELL") or ""
    return shell if os.path.isfile(shell) and os.access(shell, os.X_OK) else "/bin/sh"


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the command and everything it started (pipeline members, background children)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_command(command: str, *, cwd: str, timeout_s: float, max_output_chars: int) -> CommandResult:
    """Run ``command`` in the user's shell. Never raises for a failing, hanging or binary-printing command.

    The command gets its own process group, so a timeout (or Ctrl-C in shellmate) kills the whole
    pipeline, not just the shell. Output is decoded as UTF-8 with bad bytes replaced.
    """
    proc = subprocess.Popen(
        command,
        shell=True,
        executable=_shell(),
        cwd=cwd,
        stdin=subprocess.DEVNULL,  # nothing may wait for keyboard input (it would hang until the timeout)
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,  # own process group: Ctrl-C no longer reaches it, so we kill it ourselves
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            stdout, _ = proc.communicate(timeout=DRAIN_AFTER_KILL_S)
        except subprocess.TimeoutExpired:  # something escaped the group and still holds the pipe
            stdout = ""
        exit_code = TIMEOUT_EXIT_CODE
        stderr = f"shellmate: command timed out after {timeout_s}s and was stopped"
    except BaseException:  # Ctrl-C while waiting: don't leave the command running
        _kill_group(proc)
        proc.wait()
        raise

    stdout, out_cut = _truncate(stdout or "", max_output_chars)
    stderr, err_cut = _truncate(stderr or "", max_output_chars)
    return CommandResult(command, exit_code, stdout, stderr, truncated=out_cut or err_cut)


def read_file(path: str, *, cwd: str, max_chars: int) -> str:
    raise NotImplementedError("read_file lands in milestone 4")


def list_dir(path: str, *, cwd: str) -> list[str]:
    raise NotImplementedError("list_dir lands in milestone 4")


def last_error() -> CommandResult | None:
    raise NotImplementedError("last_error lands in milestone 4")
