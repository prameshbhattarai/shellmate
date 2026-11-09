"""What shellmate keeps between sessions, all under ``Settings.home`` (default ~/.shellmate):

- ``history.jsonl``      every command shellmate ran: command, folder, risk, exit code, output tails.
                         Feeds the `history` / `last_error` tools, `shellmate history` and `shellmate undo`.
- ``profile.json``       facts you asked it to remember (`shellmate remember ...`), added to the prompt.
- ``checkpoints.sqlite`` LangGraph checkpoints, so a run paused at an approval survives a restart.
- ``pending.json``       which paused run `shellmate resume` should pick up.

Every reader tolerates a missing or damaged file: memory is a convenience, never a reason to crash.
History holds command output, which can include secrets, so the folder is private to you (0700) and
every file in it is 0600.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

HISTORY_FILE = "history.jsonl"
PROFILE_FILE = "profile.json"
PENDING_FILE = "pending.json"
CHECKPOINT_FILE = "checkpoints.sqlite"

# Kept per history entry, and how many entries are kept in all.
HISTORY_OUTPUT_CHARS = 2000
HISTORY_MAX_ENTRIES = 1000


PRIVATE_DIR = 0o700
PRIVATE_FILE = 0o600


def _ensure(home: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR)
    try:
        os.chmod(home, PRIVATE_DIR)  # also tightens a folder made by an older version
    except OSError:
        pass
    return home


def _private(path: Path) -> None:
    try:
        os.chmod(path, PRIVATE_FILE)
    except OSError:
        pass


def same_folder(a: str | None, b: str | None) -> bool:
    """True when two paths name the same folder, through symlinks (/tmp and /private/tmp ...)."""
    if a is None or b is None:
        return False
    return os.path.realpath(a) == os.path.realpath(b)


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "..." + text[-limit:]


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    _private(tmp)
    os.replace(tmp, path)  # atomic: a crash never leaves half a file


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


# --- user profile ---------------------------------------------------------------------------------

@dataclass
class UserProfile:
    os: str | None = None
    shell: str | None = None
    aliases: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def load_profile(home: Path) -> UserProfile:
    data = _read_json(home / PROFILE_FILE, {})
    if not isinstance(data, dict):
        return UserProfile()
    known = {k: v for k, v in data.items() if k in UserProfile.__dataclass_fields__}
    try:
        return UserProfile(**known)
    except TypeError:
        return UserProfile()


def save_profile(home: Path, profile: UserProfile) -> None:
    _write_json(_ensure(home) / PROFILE_FILE, asdict(profile))


def remember(home: Path, note: str) -> UserProfile:
    profile = load_profile(home)
    note = note.strip()
    if note and note not in profile.notes:
        profile.notes.append(note)
        save_profile(home, profile)
    return profile


def forget(home: Path, number: int | None = None) -> list[str]:
    """Forget note ``number`` (1-based), or every note. Returns what was forgotten."""
    profile = load_profile(home)
    if number is None:
        gone, profile.notes = profile.notes, []
    elif 1 <= number <= len(profile.notes):
        gone = [profile.notes.pop(number - 1)]
    else:
        return []
    save_profile(home, profile)
    return gone


# --- command history ------------------------------------------------------------------------------

class History:
    """Append-only log of the commands shellmate ran, newest last."""

    def __init__(self, home: Path):
        self.home = home
        self.path = home / HISTORY_FILE

    def append(self, entry: dict) -> dict:
        record = {
            "id": uuid.uuid4().hex[:12],
            "time": time.time(),
            **entry,
            "stdout": _tail(entry.get("stdout", ""), HISTORY_OUTPUT_CHARS),
            "stderr": _tail(entry.get("stderr", ""), HISTORY_OUTPUT_CHARS),
        }
        _ensure(self.home)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, PRIVATE_FILE)
        with os.fdopen(fd, "a") as f:
            f.write(json.dumps(record) + "\n")
        _private(self.path)
        self._trim()
        return record

    def _trim(self) -> None:
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return
        if len(lines) > 2 * HISTORY_MAX_ENTRIES:  # rewrite only now and then
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("\n".join(lines[-HISTORY_MAX_ENTRIES:]) + "\n")
            _private(tmp)
            os.replace(tmp, self.path)

    def entries(self) -> list[dict]:
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                item = json.loads(line)
            except ValueError:
                continue  # a damaged line costs that entry, not the whole history
            if isinstance(item, dict) and "command" in item:
                out.append(item)
        return out

    def recent(self, n: int = 10, *, cwd: str | None = None) -> list[dict]:
        if n <= 0:
            return []
        items = [e for e in self.entries() if cwd is None or same_folder(e.get("cwd"), cwd)]
        return items[-n:]

    def last_error(self, *, cwd: str | None = None) -> dict | None:
        return next(
            (e for e in reversed(self.entries()) if e.get("exit_code") not in (0, None)
             and (cwd is None or same_folder(e.get("cwd"), cwd))),
            None,
        )


def format_entry(entry: dict, *, with_output: bool = False) -> str:
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.get("time", 0)))
    line = f"[{when}] exit {entry.get('exit_code')}  {entry['command']}  (in {entry.get('cwd', '?')})"
    if with_output:
        if entry.get("stdout", "").strip():
            line += "\nstdout:\n" + entry["stdout"].rstrip()
        if entry.get("stderr", "").strip():
            line += "\nstderr:\n" + entry["stderr"].rstrip()
    return line


# --- paused runs ----------------------------------------------------------------------------------

def save_pending(home: Path, *, thread_id: str, request: str, cwd: str) -> None:
    _write_json(_ensure(home) / PENDING_FILE, {"thread_id": thread_id, "request": request, "cwd": cwd, "time": time.time()})


def load_pending(home: Path) -> dict | None:
    data = _read_json(home / PENDING_FILE, None)
    return data if isinstance(data, dict) and {"thread_id", "request", "cwd"} <= data.keys() else None


def clear_pending(home: Path, thread_id: str | None = None) -> None:
    """Forget the paused run (only if it is ``thread_id``, when given)."""
    pending = load_pending(home)
    if pending and (thread_id is None or pending["thread_id"] == thread_id):
        try:
            (home / PENDING_FILE).unlink()
        except OSError:
            pass


def checkpointer(home: Path):
    """A SQLite-backed LangGraph checkpointer under ``home``."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = _ensure(home) / CHECKPOINT_FILE
    conn = sqlite3.connect(path, check_same_thread=False)
    _private(path)
    return SqliteSaver(conn)
