"""LangGraph concept: memory with a checkpointer.

A checkpointer saves the graph's state after every step, keyed by a ``thread_id``. That gives you
two things for free:

- **resume**: a run paused by ``interrupt()`` can continue later, from exactly where it stopped,
  even in a new process, because the state is on disk, not in memory.
- **history**: ``graph.get_state(config)`` reads the latest state of a thread at any time.

shellmate keeps both files under ``Settings.home`` (default ~/.shellmate):

- ``checkpoints.sqlite`` the LangGraph checkpoints (SqliteSaver). Finished runs are deleted; only a
                         paused run is kept.
- ``pending.json``       which paused thread `shellmate resume` should pick up, its request and folder.

The folder is private to you (0700) and its files 0600: a saved state holds command output, which
can include secrets.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

PENDING_FILE = "pending.json"
CHECKPOINT_FILE = "checkpoints.sqlite"
# Written by older shellmate versions and no longer used. history.jsonl kept command output tails,
# which can include secrets, so it is deleted rather than left behind unmanaged.
LEGACY_FILES = ("history.jsonl", "history.tmp")
PRIVATE_DIR = 0o700
PRIVATE_FILE = 0o600


def _ensure(home: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR)
    try:
        os.chmod(home, PRIVATE_DIR)  # also tightens a folder made by an older version
    except OSError:
        pass
    for name in LEGACY_FILES:
        try:
            (home / name).unlink()
        except OSError:
            pass  # not there (the usual case)
    return home


def _private(path: Path) -> None:
    try:
        os.chmod(path, PRIVATE_FILE)
    except OSError:
        pass


def checkpointer(home: Path):
    """A SQLite-backed LangGraph checkpointer under ``home``.

    LangGraph: InMemorySaver (graph.py's default) forgets everything when the process exits;
    SqliteSaver writes the same checkpoints to a file. Swapping one for the other is the only change
    needed for a run to survive a restart.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = _ensure(home) / CHECKPOINT_FILE
    conn = sqlite3.connect(path, check_same_thread=False)
    _private(path)
    return SqliteSaver(conn)


# --- the paused run -------------------------------------------------------------------------------

def save_pending(home: Path, *, thread_id: str, request: str, cwd: str) -> None:
    path = _ensure(home) / PENDING_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"thread_id": thread_id, "request": request, "cwd": cwd, "time": time.time()}))
    _private(tmp)
    os.replace(tmp, path)  # atomic: a crash never leaves half a file


def load_pending(home: Path) -> dict | None:
    try:
        data = json.loads((home / PENDING_FILE).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and {"thread_id", "request", "cwd"} <= data.keys() else None


def clear_pending(home: Path, thread_id: str | None = None) -> None:
    """Forget the paused run (only if it is ``thread_id``, when given)."""
    pending = load_pending(home)
    if pending and (thread_id is None or pending["thread_id"] == thread_id):
        try:
            (home / PENDING_FILE).unlink()
        except OSError:
            pass
