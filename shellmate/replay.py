"""Replay mode, for demos: scripted model answers instead of a live model.

    SHELLMATE_REPLAY=scripts/demo_replay.json shellmate

The file is a JSON list of decisions, used in order across the whole session, e.g.
``[{"action": "run", "command": "ls", "explanation": "..."}, {"action": "answer", "answer": "..."}]``.
Everything except the model is real: the risk rules, the approval prompts, the commands and their
output. So a recording shows exactly how shellmate behaves, with no Ollama call, and the banner says
the session is replayed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .agent import Decision

REPLAY_ENV = "SHELLMATE_REPLAY"
DELAY_ENV = "SHELLMATE_REPLAY_DELAY"  # seconds of "thinking" per answer, so a recording reads naturally


class ReplayDecider:
    def __init__(self, decisions: list[Decision], *, delay_s: float = 0.0):
        self.decisions = list(decisions)
        self.delay_s = delay_s

    @classmethod
    def from_file(cls, path: str | Path, *, delay_s: float = 0.0) -> "ReplayDecider":
        items = json.loads(Path(path).read_text())
        if not isinstance(items, list):
            raise ValueError(f"{path}: expected a JSON list of decisions")
        return cls([Decision(**item) for item in items], delay_s=delay_s)

    def invoke(self, messages: list) -> Decision:
        if not self.decisions:
            raise RuntimeError("the replay script has no more answers")
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.decisions.pop(0)


def replay_path() -> str | None:
    return os.environ.get(REPLAY_ENV) or None


def replay_decider() -> ReplayDecider | None:
    path = replay_path()
    if path is None:
        return None
    return ReplayDecider.from_file(path, delay_s=float(os.environ.get(DELAY_ENV, "0") or 0))
