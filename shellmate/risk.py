"""Risk classification for shell commands.

Design (to be implemented in milestone 2):
1. Rules decide first: regex deny-list (rm, sudo, mv, chmod, redirects, ``curl | sh``,
   ``git push --force`` ...) and an allow-list of read-only commands (ls, du, cat, git status ...).
2. Only commands the rules don't cover are labelled by the model.
3. Anything unsure is DANGER. DANGER always needs human approval, whatever the model says.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Risk(str, Enum):
    SAFE = "safe"          # read-only; may auto-run
    CAUTION = "caution"    # changes state but is easy to undo; ask
    DANGER = "danger"      # destructive, privileged or unknown; always ask


@dataclass(frozen=True)
class RiskVerdict:
    risk: Risk
    reason: str
    decided_by: str  # "rule" or "model"


def classify(command: str) -> RiskVerdict:
    """Classify ``command``. Stub: everything is DANGER until the rules are implemented."""
    return RiskVerdict(Risk.DANGER, "risk rules not implemented yet", "rule")


def needs_approval(verdict: RiskVerdict) -> bool:
    """Only SAFE commands may run without asking."""
    return verdict.risk is not Risk.SAFE
