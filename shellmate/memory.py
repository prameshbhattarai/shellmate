"""Memory across sessions (milestone 5).

- Session state: a LangGraph SQLite checkpointer under ``Settings.home``, so a paused run
  (waiting for approval) survives a restart.
- User profile: small facts worth remembering (OS, shell, project roots, aliases), stored as JSON
  and added to the system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UserProfile:
    os: str | None = None
    shell: str | None = None
    aliases: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def load_profile(home: Path) -> UserProfile:
    """Stub: returns an empty profile until milestone 5."""
    return UserProfile()


def save_profile(home: Path, profile: UserProfile) -> None:
    raise NotImplementedError("profile storage lands in milestone 5")
