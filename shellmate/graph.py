"""The agent loop as a LangGraph state graph (milestone 3).

    understand -> propose_command -> classify_risk -> auto_run | ask_human -> run -> observe
               -> next_step (back to propose_command) | answer

``ask_human`` uses LangGraph's ``interrupt()`` so the run pauses for y/n/edit and resumes from the
checkpoint. ``max_steps`` caps the loop.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import BaseMessage

from .risk import RiskVerdict
from .tools import CommandResult

NODES = (
    "understand",
    "propose_command",
    "classify_risk",
    "ask_human",
    "run",
    "observe",
    "answer",
)


class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]
    request: str
    proposed_command: str | None
    verdict: RiskVerdict | None
    approved: bool | None
    last_result: CommandResult | None
    steps: int
    done: bool


def build_graph():
    """Compile and return the agent graph. Stub until milestone 3."""
    raise NotImplementedError("the agent graph lands in milestone 3")
