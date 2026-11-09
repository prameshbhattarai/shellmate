r"""The agent loop as a LangGraph state graph.

    START -> decide --answer / step limit / model error--> END
               |
               run? -> classify_risk --blocked--> refuse --> decide
                            |   \--dry run--> END
                            |--SAFE--> run --> decide
                            \--else--> ask_human --yes--> run
                                           |--edit--> classify_risk  (an edited command is checked again)
                                           \--no----> END

``ask_human`` pauses the run with ``interrupt()``. The caller (the CLI) shows the command, collects
the user's reply and resumes with ``Command(resume={"action": "run" | "edit" | "skip", "command": ...})``.
The graph never prompts anyone itself, so any front end can drive it.

State holds only plain JSON values (no dataclasses), so it checkpoints cleanly.
"""

from __future__ import annotations

import operator
from dataclasses import asdict
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agent import (
    ANSWER_NUDGE,
    REFUSED_NUDGE,
    REPEAT_NUDGE,
    Decider,
    Decision,
    answer_is_a_command,
    build_messages,
    clean_command,
    refused_command,
    repeated_command,
)
from .config import Settings
from .risk import Risk, classify
from .tools import run_command

NODES = ("decide", "classify_risk", "ask_human", "run", "refuse")

Status = Literal["answered", "declined", "dry_run", "step_limit", "model_error"]


class Step(TypedDict, total=False):
    command: str
    outcome: Literal["ran", "blocked"]
    risk: str  # the verdict it ran under; a non-SAFE step may have changed what later commands see
    exit_code: int
    stdout: str
    stderr: str
    note: str


class AgentState(TypedDict, total=False):
    # input
    request: str
    dry_run: bool
    confirm_all: bool
    # working
    steps: Annotated[list[Step], operator.add]  # finished steps, appended by run and refuse
    command: str
    explanation: str
    risk: str
    reason: str
    reply: Literal["run", "edit", "skip"]  # the user's answer in ask_human
    # output
    answer: str
    status: Status
    error: str


def _problem(decision: Decision, steps: list[Step], request: str) -> str | None:
    """What is wrong with a decision the graph must not act on, as a message back to the model."""
    command = clean_command(decision.command)
    if decision.action == "run" and command:
        refused = refused_command(command, steps)
        if refused:
            return REFUSED_NUDGE.format(command=command, reason=refused.get("note", "too dangerous"))
        if repeated_command(command, steps):
            return REPEAT_NUDGE.format(command=command, request=request)
    if decision.action == "answer" and answer_is_a_command(decision.answer, steps):
        return ANSWER_NUDGE.format(request=request)
    return None


def _echo(decision: Decision) -> str:
    return f"run: {decision.command}" if decision.action == "run" else decision.answer


FALLBACK_OUTPUT_CHARS = 800


def _fallback_answer(steps: list[Step], decision: Decision) -> dict:
    """The model kept repeating itself: show the user the last output instead of looping."""
    refused = refused_command(clean_command(decision.command), steps) if decision.action == "run" else None
    if refused:
        reason = refused.get("note", "too dangerous")
        text = f"shellmate won't run `{refused['command']}` ({reason}), and I found no safer way to do it."
        return {"status": "answered", "answer": text, "command": ""}
    last = next((s for s in reversed(steps) if s["outcome"] == "ran"), None)
    if last is None:
        return {"status": "answered", "answer": "I couldn't work out how to answer that.", "command": ""}
    out = (last.get("stdout") or last.get("stderr") or "(no output)").strip()
    if len(out) > FALLBACK_OUTPUT_CHARS:
        out = out[:FALLBACK_OUTPUT_CHARS] + " ..."
    return {"status": "answered", "answer": f"Here is what `{last['command']}` printed:\n{out}", "command": ""}


def build_graph(decider: Decider, *, settings: Settings, cwd: str, checkpointer=None):
    """Compile the agent graph. ``decider`` is the structured model (or a fake in tests)."""

    def decide(state: AgentState) -> dict:
        steps = state.get("steps", [])
        if len(steps) >= settings.max_steps:
            return {
                "status": "step_limit",
                "answer": f"I stopped after {settings.max_steps} steps without a final answer.",
            }
        request = state["request"]
        messages = build_messages(request, steps, cwd=cwd)
        try:
            decision = decider.invoke(messages)
            problem = _problem(decision, steps, request)
            if problem:  # one retry, with the problem spelled out
                decision = decider.invoke(messages + [("ai", _echo(decision)), ("human", problem)])
                if _problem(decision, steps, request):
                    return _fallback_answer(steps, decision)
        except Exception as exc:  # connection refused, model missing, bad JSON ...
            return {"status": "model_error", "error": str(exc)}
        command = clean_command(decision.command)
        if decision.action == "answer" or not command:
            text = decision.answer.strip() or decision.explanation.strip() or "Done."
            return {"status": "answered", "answer": text, "command": ""}
        return {"command": command, "explanation": decision.explanation.strip()}

    def classify_risk(state: AgentState) -> dict:
        verdict = classify(state["command"], cwd=cwd)
        update = {"risk": verdict.risk.value, "reason": verdict.reason}
        if state.get("dry_run"):
            update["status"] = "dry_run"
        return update

    def ask_human(state: AgentState) -> dict:
        reply = interrupt(
            {
                "command": state["command"],
                "risk": state["risk"],
                "reason": state["reason"],
                "explanation": state.get("explanation", ""),
            }
        )
        action = reply.get("action")
        if action == "edit" and reply.get("command", "").strip():
            # the model's explanation was for the old command
            return {"command": reply["command"].strip(), "explanation": "", "reply": "edit"}
        if action == "run":
            return {"reply": "run"}
        return {"reply": "skip", "status": "declined"}

    def run(state: AgentState) -> dict:
        result = run_command(
            state["command"],
            cwd=cwd,
            timeout_s=settings.command_timeout_s,
            max_output_chars=settings.max_output_chars,
        )
        step = {"outcome": "ran", "risk": state.get("risk", "danger"), **asdict(result)}
        step.pop("truncated", None)
        return {"steps": [step]}

    def refuse(state: AgentState) -> dict:
        return {"steps": [{"command": state["command"], "outcome": "blocked", "risk": "blocked", "note": state["reason"]}]}

    def after_decide(state: AgentState) -> str:
        return END if state.get("status") else "classify_risk"

    def after_classify(state: AgentState) -> str:
        if state.get("dry_run"):
            return END
        if state["risk"] == Risk.BLOCKED.value:
            return "refuse"
        if state["risk"] == Risk.SAFE.value and not state.get("confirm_all"):
            return "run"
        return "ask_human"

    def after_ask(state: AgentState) -> str:
        return {"run": "run", "edit": "classify_risk"}.get(state.get("reply"), END)

    graph = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("classify_risk", classify_risk)
    graph.add_node("ask_human", ask_human)
    graph.add_node("run", run)
    graph.add_node("refuse", refuse)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", after_decide, ["classify_risk", END])
    graph.add_conditional_edges("classify_risk", after_classify, ["refuse", "run", "ask_human", END])
    graph.add_conditional_edges("ask_human", after_ask, ["run", "classify_risk", END])
    graph.add_edge("run", "decide")
    graph.add_edge("refuse", "decide")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
