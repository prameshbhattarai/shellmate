r"""The agent loop as a LangGraph state graph.

    START -> decide --answer / step limit / model error--> END
               |---tool (read_file, list_dir, history, last_error)--> use_tool --> decide
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
    FAILED_ANSWER_NUDGE,
    PLACEHOLDER_NUDGE,
    answer_has_placeholders,
    failure_label,
    step_failed,
    REFUSED_NUDGE,
    REPEAT_NUDGE,
    TOOL_ACTIONS,
    Decider,
    Decision,
    PromptContext,
    answer_is_a_command,
    last_failure,
    build_messages,
    clean_command,
    refused_command,
    repeated_command,
    step_key,
)
from .config import Settings
from .memory import History, format_entry
from .risk import Risk, classify
from .tools import ToolError, list_dir, read_file, run_command
from .undo import snapshot_after

NODES = ("decide", "use_tool", "classify_risk", "ask_human", "run", "refuse")

HISTORY_TOOL_ENTRIES = 10

Status = Literal["answered", "declined", "dry_run", "step_limit", "model_error"]


class Step(TypedDict, total=False):
    command: str
    outcome: Literal["ran", "blocked", "tool"]
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
    tool: str  # set when the next step is a read-only tool, not a command
    path: str
    risk: str
    reason: str
    reply: Literal["run", "edit", "skip"]  # the user's answer in ask_human
    # output
    answer: str
    status: Status
    error: str


def _problem(decision: Decision, steps: list[Step], request: str) -> str | None:
    """What is wrong with a decision the graph must not act on, as a message back to the model."""
    key = step_key(decision)
    if decision.action == "run" and key:
        refused = refused_command(key, steps)
        if refused:
            return REFUSED_NUDGE.format(command=key, reason=refused.get("note", "too dangerous"))
    if key and repeated_command(key, steps):
        return REPEAT_NUDGE.format(command=key, request=request)
    if decision.action == "answer" and answer_is_a_command(decision.answer, steps):
        return ANSWER_NUDGE.format(request=request)
    if decision.action == "answer" and answer_has_placeholders(decision.answer):
        return PLACEHOLDER_NUDGE
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
    last = next((s for s in reversed(steps) if s["outcome"] in ("ran", "tool")), None)
    if last is None:
        return {"status": "answered", "answer": "I couldn't work out how to answer that.", "command": ""}
    if step_failed(last):  # never dress an error up as an answer
        error = (last.get("stderr") or last.get("stdout") or "no output").strip().splitlines()[0]
        text = f"I couldn't get an answer: my last attempt, `{last['command']}`, failed ({error})."
        return {"status": "answered", "answer": text, "command": ""}
    out = (last.get("stdout") or last.get("stderr") or "(no output)").strip()
    if len(out) > FALLBACK_OUTPUT_CHARS:
        out = out[:FALLBACK_OUTPUT_CHARS] + " ..."
    return {"status": "answered", "answer": f"Here is what `{last['command']}` printed:\n{out}", "command": ""}


def _run_tool(tool: str, path: str, *, cwd: str, settings: Settings, history: History | None) -> str:
    if tool == "read_file":
        return read_file(path, cwd=cwd, max_chars=settings.max_output_chars)
    if tool == "list_dir":
        return "\n".join(list_dir(path or ".", cwd=cwd)) or "(empty folder)"
    if history is None:
        raise ToolError("history is not available in this session")
    if tool == "history":
        entries = history.recent(HISTORY_TOOL_ENTRIES)
        return "\n".join(format_entry(e) for e in entries) or "shellmate has not run any commands yet"
    if tool == "last_error":
        entry = history.last_error(cwd=cwd) or history.last_error()  # this folder first
        return format_entry(entry, with_output=True) if entry else "no failed command in shellmate's history"
    raise ToolError(f"unknown tool {tool!r}")


def build_graph(
    decider: Decider,
    *,
    settings: Settings,
    cwd: str,
    checkpointer=None,
    history: History | None = None,
    context: PromptContext | None = None,
):
    """Compile the agent graph. ``decider`` is the structured model (or a fake in tests).

    ``history`` logs every command that runs and backs the history tools; ``context`` adds the
    persona, teaching mode and remembered notes to the prompt.
    """

    def decide(state: AgentState) -> dict:
        steps = state.get("steps", [])
        if len(steps) >= settings.max_steps:
            return {
                "status": "step_limit",
                "answer": f"I stopped after {settings.max_steps} steps without a final answer.",
            }
        request = state["request"]
        messages = build_messages(request, steps, cwd=cwd, context=context)
        try:
            decision = decider.invoke(messages)
            problem = _problem(decision, steps, request)
            if problem:  # one retry, with the problem spelled out
                decision = decider.invoke(messages + [("ai", _echo(decision)), ("human", problem)])
                if _problem(decision, steps, request):
                    return _fallback_answer(steps, decision)
            failed = last_failure(steps)
            if decision.action == "answer" and failed:
                # answering straight after a failure: make the model look at the error once more.
                # Whatever it says then is accepted, since "it failed because ..." is a fine answer.
                error = (failed.get("stderr") or failed.get("stdout") or "").strip().splitlines()[:1]
                nudge = FAILED_ANSWER_NUDGE.format(how=failure_label(failed), error=error[0] if error else "(no output)")
                retry = decider.invoke(messages + [("ai", _echo(decision)), ("human", nudge)])
                decision = retry if not _problem(retry, steps, request) else decision
        except Exception as exc:  # connection refused, model missing, bad JSON ...
            return {"status": "model_error", "error": str(exc)}
        if decision.action in TOOL_ACTIONS:
            return {
                "tool": decision.action,
                "path": decision.path.strip(),
                "command": step_key(decision),
                "explanation": decision.explanation.strip(),
            }
        command = clean_command(decision.command)
        if decision.action == "answer" or not command:
            text = decision.answer.strip() or decision.explanation.strip() or "Done."
            return {"status": "answered", "answer": text, "command": "", "tool": ""}
        return {"command": command, "explanation": decision.explanation.strip(), "tool": ""}

    def use_tool(state: AgentState) -> dict:
        step = {"command": state["command"], "outcome": "tool", "risk": Risk.SAFE.value, "stderr": ""}
        try:
            step.update(exit_code=0, stdout=_run_tool(state["tool"], state.get("path", ""), cwd=cwd,
                                                       settings=settings, history=history))
        except ToolError as exc:
            step.update(exit_code=1, stdout="", stderr=str(exc))
        return {"steps": [step]}

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
        if history is not None:
            history.append({"cwd": cwd, **step, **snapshot_after(state["command"], cwd)})
        return {"steps": [step]}

    def refuse(state: AgentState) -> dict:
        return {"steps": [{"command": state["command"], "outcome": "blocked", "risk": "blocked", "note": state["reason"]}]}

    def after_decide(state: AgentState) -> str:
        if state.get("status"):
            return END
        return "use_tool" if state.get("tool") else "classify_risk"

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
    graph.add_node("use_tool", use_tool)
    graph.add_node("classify_risk", classify_risk)
    graph.add_node("ask_human", ask_human)
    graph.add_node("run", run)
    graph.add_node("refuse", refuse)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", after_decide, ["use_tool", "classify_risk", END])
    graph.add_edge("use_tool", "decide")
    graph.add_conditional_edges("classify_risk", after_classify, ["refuse", "run", "ask_human", END])
    graph.add_conditional_edges("ask_human", after_ask, ["run", "classify_risk", END])
    graph.add_edge("run", "decide")
    graph.add_edge("refuse", "decide")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
