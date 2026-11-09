"""Tools: the only actions the agent can take.

Implemented in milestones 1 and 4. Guarantees each tool must keep:
- ``run_command``: runs only after the risk gate, with a timeout, inside the working directory,
  and returns truncated stdout/stderr plus the exit code (never raises on a failing command).
- ``read_file`` / ``list_dir``: read-only, confined to the working directory.
- ``last_error``: the most recent failed command and its stderr, from shellmate's own history.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool = False


def run_command(command: str, *, cwd: str, timeout_s: int, max_output_chars: int) -> CommandResult:
    raise NotImplementedError("run_command lands in milestone 1")


def read_file(path: str, *, cwd: str, max_chars: int) -> str:
    raise NotImplementedError("read_file lands in milestone 4")


def list_dir(path: str, *, cwd: str) -> list[str]:
    raise NotImplementedError("list_dir lands in milestone 4")


def last_error() -> CommandResult | None:
    raise NotImplementedError("last_error lands in milestone 4")
