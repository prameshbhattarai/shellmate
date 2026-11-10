r"""LangGraph concepts: state, nodes, edges, conditional routing and human-in-the-loop.

This is the agent loop. Read it top to bottom: the state, the nodes (plain functions), the routing
rules, then how they are wired into a graph.

    START -> decide --answer / step limit / model error--> END
               |---read_file / list_dir--> use_tool --> decide
               |
               run? -> classify_risk --blocked--> refuse --> decide
                            |   \--dry run--> END
                            |--SAFE--> run --> decide
                            \--else--> ask_human --yes--> run
                                           |--edit--> classify_risk  (an edited command is checked again)
                                           \--no----> END

Why not ``langchain.agents.create_agent``? It builds this same kind of loop for you, and with
middleware it can even pause for a human (``HumanInTheLoopMiddleware``) or run custom checks.
shellmate builds the graph by hand for two reasons: so every piece (state, nodes, routing,
interrupt, checkpointer) is visible in one file, and because the model picks its next step through
structured output (a ``Decision``) rather than native tool calls, which small local models follow
more reliably. Once this file makes sense, create_agent is the same ideas with less code.
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
    REFUSED_NUDGE,
    REPEAT_NUDGE,
    TOOL_ACTIONS,
    Decider,
    Decision,
    answer_has_placeholders,
    answer_is_a_command,
    build_messages,
    clean_command,
    failure_label,
    last_failure,
    refused_command,
    repeated_command,
    step_failed,
    step_key,
)
from .config import Settings
from .risk import Risk, classify
from .tools import ToolError, list_dir, read_file, run_command

NODES = ("decide", "use_tool", "classify_risk", "ask_human", "run", "refuse")

# --- 1. State -------------------------------------------------------------------------------------
# LangGraph: the state is one dict shared by every node. A node returns only the keys it changes,
# and LangGraph merges them in. Keep it to plain JSON values so a checkpointer can save it.

Status = Literal["answered", "declined", "dry_run", "step_limit", "model_error"]


class Step(TypedDict, total=False):
    """One finished step: a command that ran, a tool call, or a refused command."""

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
    # LangGraph: a "reducer". Annotated[list, operator.add] means a node's {"steps": [new]} is
    # *appended* to the list instead of replacing it. Every other key is simply overwritten.
    steps: Annotated[list[Step], operator.add]
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


# --- 2. Guards used by `decide` (plain Python, no LangGraph) --------------------------------------

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
    """The model kept making the same mistake: end with what really happened instead of looping."""
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


def _ask_model(decider: Decider, messages: list, steps: list[Step], request: str) -> Decision | dict:
    """One model turn, plus at most one retry when a guard spots a problem.

    Returns a Decision to act on, or a finished state update (the fallback answer).
    """
    decision = decider.invoke(messages)
    problem = _problem(decision, steps, request)
    if problem:  # one retry, with the problem spelled out
        decision = decider.invoke(messages + [("ai", _echo(decision)), ("human", problem)])
        if _problem(decision, steps, request):
            return _fallback_answer(steps, decision)
    failed = last_failure(steps)
    if decision.action == "answer" and failed:
        # Answering straight after a failure: make the model look at the error once more. Whatever
        # it says then is accepted, since "it failed because ..." is a fine answer.
        lines = (failed.get("stderr") or failed.get("stdout") or "").strip().splitlines()[:1]
        nudge = FAILED_ANSWER_NUDGE.format(how=failure_label(failed), error=lines[0] if lines else "(no output)")
        retry = decider.invoke(messages + [("ai", _echo(decision)), ("human", nudge)])
        decision = retry if not _problem(retry, steps, request) else decision
    return decision


def _run_tool(tool: str, path: str, *, cwd: str, settings: Settings) -> str:
    if tool == "read_file":
        return read_file(path, cwd=cwd, max_chars=settings.max_output_chars)
    if tool == "list_dir":
        return "\n".join(list_dir(path or ".", cwd=cwd)) or "(empty folder)"
    raise ToolError(f"unknown tool {tool!r}")


def build_graph(decider: Decider, *, settings: Settings, cwd: str, checkpointer=None):
    """Compile the agent graph. ``decider`` is the structured model (or a fake in tests)."""

    # --- 3. Nodes ---------------------------------------------------------------------------------
    # LangGraph: a node is a plain function, state in, partial update out. Nothing else.

    def decide(state: AgentState) -> dict:
        """Ask the model for the next step."""
        steps = state.get("steps", [])
        if len(steps) >= settings.max_steps:  # the loop's hard stop
            return {"status": "step_limit", "answer": f"I stopped after {settings.max_steps} steps without a final answer."}
        try:
            decision = _ask_model(decider, build_messages(state["request"], steps, cwd=cwd), steps, state["request"])
        except Exception as exc:  # connection refused, model missing, bad JSON ...
            return {"status": "model_error", "error": str(exc)}
        if isinstance(decision, dict):  # the fallback answer
            return decision
        if decision.action in TOOL_ACTIONS:
            return {"tool": decision.action, "path": decision.path.strip(), "command": step_key(decision),
                    "explanation": decision.explanation.strip()}
        command = clean_command(decision.command)
        if decision.action == "answer" or not command:
            text = decision.answer.strip() or decision.explanation.strip() or "Done."
            return {"status": "answered", "answer": text, "command": "", "tool": ""}
        return {"command": command, "explanation": decision.explanation.strip(), "tool": ""}

    def use_tool(state: AgentState) -> dict:
        """Read a file or list a folder: read-only, so no risk check and no approval."""
        step = {"command": state["command"], "outcome": "tool", "risk": Risk.SAFE.value, "stderr": ""}
        try:
            step.update(exit_code=0, stdout=_run_tool(state["tool"], state.get("path", ""), cwd=cwd, settings=settings))
        except ToolError as exc:  # the error goes back to the model, which can try something else
            step.update(exit_code=1, stdout="", stderr=str(exc))
        return {"steps": [step]}  # appended to state["steps"] by the reducer

    def classify_risk(state: AgentState) -> dict:
        """Label the command with the rules in risk.py. No model involved."""
        verdict = classify(state["command"], cwd=cwd)
        update = {"risk": verdict.risk.value, "reason": verdict.reason}
        if state.get("dry_run"):
            update["status"] = "dry_run"
        return update

    def ask_human(state: AgentState) -> dict:
        """Pause and wait for the user.

        LangGraph: interrupt() stops the run right here and saves the state with the checkpointer.
        Whoever runs the graph (the CLI) gets the payload, asks the user, and resumes with
        graph.stream(Command(resume=reply), config). The node then re-runs from the top, and this
        time interrupt() returns the reply instead of pausing.
        """
        reply = interrupt({"command": state["command"], "risk": state["risk"], "reason": state["reason"],
                           "explanation": state.get("explanation", "")})
        action = reply.get("action")
        if action == "edit" and reply.get("command", "").strip():
            return {"command": reply["command"].strip(), "explanation": "", "reply": "edit"}
        if action == "run":
            return {"reply": "run"}
        return {"reply": "skip", "status": "declined"}

    def run(state: AgentState) -> dict:
        """Run the approved command and record what happened."""
        result = run_command(state["command"], cwd=cwd, timeout_s=settings.command_timeout_s,
                             max_output_chars=settings.max_output_chars)
        step = {"outcome": "ran", "risk": state.get("risk", "danger"), **asdict(result)}
        step.pop("truncated", None)
        return {"steps": [step]}

    def refuse(state: AgentState) -> dict:
        """Record a BLOCKED command, so the model is told and can choose another way."""
        return {"steps": [{"command": state["command"], "outcome": "blocked", "risk": "blocked", "note": state["reason"]}]}

    # --- 4. Routing -------------------------------------------------------------------------------
    # LangGraph: a conditional edge is a function that reads the state and returns the next node's
    # name (or END).

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

    # --- 5. Wiring --------------------------------------------------------------------------------
    graph = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("use_tool", use_tool)
    graph.add_node("classify_risk", classify_risk)
    graph.add_node("ask_human", ask_human)
    graph.add_node("run", run)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", after_decide, ["use_tool", "classify_risk", END])
    graph.add_conditional_edges("classify_risk", after_classify, ["refuse", "run", "ask_human", END])
    graph.add_conditional_edges("ask_human", after_ask, ["run", "classify_risk", END])
    graph.add_edge("use_tool", "decide")  # these three edges make it a loop
    graph.add_edge("run", "decide")
    graph.add_edge("refuse", "decide")

    # LangGraph: a checkpointer saves the state after every step, keyed by the thread_id in the
    # run's config. interrupt() needs one (to resume), and memory.py swaps in SQLite so a paused run
    # survives a restart.
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
