"""Agent concept: tools, the only things the agent can do in the world.

The model only *names* a tool (an action in its Decision); graph.py calls the function here and
sends the result back to the model as the next message. That's the whole idea of tool use.

- ``run_command``: runs only after the risk gate, with a timeout, inside the working folder, and
  returns capped stdout/stderr plus the exit code (never raises on a failing command).
- ``read_file`` / ``list_dir``: read-only and confined to the working folder, so they need no
  approval. They raise :class:`ToolError` with a message meant for the model.

(LangChain also has a ``@tool`` decorator and ``model.bind_tools()`` for native tool calling.
shellmate uses structured output instead, because small local models follow a JSON schema more
reliably than tool-call formats. Adding a tool is the same either way: a function plus a way for
the model to ask for it.)
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass

TIMEOUT_EXIT_CODE = 124  # same code GNU `timeout` uses
START_FAILED_EXIT_CODE = 127  # what shells report for a command that could not be started


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
    try:
        proc = _start(command, cwd)
    except OSError as exc:  # the working folder is gone, the shell can't start ...
        return CommandResult(command, START_FAILED_EXIT_CODE, "", f"shellmate: could not start the command: {exc}")
    return _wait(proc, command, timeout_s=timeout_s, max_output_chars=max_output_chars)


def _start(command: str, cwd: str) -> subprocess.Popen:
    return subprocess.Popen(
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


def _wait(proc: subprocess.Popen, command: str, *, timeout_s: float, max_output_chars: int) -> CommandResult:
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


class ToolError(Exception):
    """A tool could not do what was asked; the message goes back to the model."""


LIST_DIR_MAX_ENTRIES = 200
READ_FILE_MAX_BYTES = 1_000_000  # never pull more than this into memory


def _inside(path: str, cwd: str) -> str:
    """Resolve ``path`` against ``cwd`` and refuse anything outside it (symlinks included)."""
    root = os.path.realpath(cwd)
    target = os.path.realpath(os.path.join(root, os.path.expanduser(path or ".")))
    if os.path.commonpath([root, target]) != root:
        raise ToolError(f"{path} is outside the current folder; tools only read inside {root}")
    return target


def read_file(path: str, *, cwd: str, max_chars: int) -> str:
    target = _inside(path, cwd)
    if os.path.isdir(target):
        raise ToolError(f"{path} is a folder; use list_dir")
    try:
        size = os.path.getsize(target)
        with open(target, "rb") as f:
            raw = f.read(READ_FILE_MAX_BYTES)
    except FileNotFoundError:
        raise ToolError(f"{path} does not exist") from None
    except OSError as exc:
        raise ToolError(f"can't read {path}: {exc.strerror}") from None
    if b"\0" in raw[:8000]:
        raise ToolError(f"{path} looks like a binary file")
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= max_chars and size <= len(raw):
        return text
    return text[:max_chars] + f"\n... [file continues: {size} bytes in all, showing the first {max_chars} chars]"


def list_dir(path: str, *, cwd: str) -> list[str]:
    """Entries of a folder, folders first and marked with a trailing /."""
    target = _inside(path, cwd)
    try:
        with os.scandir(target) as it:
            entries = [(not e.is_dir(), e.name + ("/" if e.is_dir() else "")) for e in it]
    except FileNotFoundError:
        raise ToolError(f"{path} does not exist") from None
    except NotADirectoryError:
        raise ToolError(f"{path} is a file; use read_file") from None
    except OSError as exc:
        raise ToolError(f"can't list {path}: {exc.strerror}") from None
    names = [name for _, name in sorted(entries)]
    if len(names) > LIST_DIR_MAX_ENTRIES:
        names = names[:LIST_DIR_MAX_ENTRIES] + [f"... {len(names) - LIST_DIR_MAX_ENTRIES} more"]
    return names
