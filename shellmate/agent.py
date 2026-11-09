"""What the model sees and what it may say back.

Each turn the model returns a :class:`Decision`: either run ONE command, or answer the user. It never
runs anything itself; the graph (graph.py) risk-checks the command and asks the user when needed.
"""

from __future__ import annotations

import os
import platform
from typing import Literal, Protocol

from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are shellmate. You help the user by running shell commands one at a time and
reading their output, until you can answer them.

Environment: {os_name}, shell {shell}. Commands run inside the user's current folder ({cwd}).

Each turn, reply with exactly one of:
- action "run": ONE command to run next, and a short explanation of why.
- action "answer": a short, plain-English answer for the user, based on the outputs you have seen.

How to work:
- Answer as soon as the outputs answer the request. Most requests need one command, some need two
  or three.
- Never run the same command twice. If a command failed, read the error, then try a fix or explain
  the problem in your answer.
- In the answer, give the facts the user asked for (names, numbers, sizes). Summarize; don't paste
  whole outputs back.
- If no command is needed (a greeting, a general question), answer directly.
- If shellmate refused a command, don't retry it; pick a safer way or explain why it can't be done.

Rules for commands:
- One command line. Pipes and && are fine; no scripts, no comments.
- Return the bare command: never wrap it in $( ), backticks or quotes, and no leading "$ ".
- Use relative paths ("." or "*"). Never write out the current folder's absolute path.
- Prefer the simplest standard tool for the job (du for sizes, ls for listings, git for git).
- Print only what was asked for. Narrow output with flags (--format, --stat, --name-only, -n)
  instead of dumping whole files, diffs or logs.
- git: `git diff` shows changes not yet committed. `git show` ALWAYS prints the whole diff unless
  you add -s, so for a commit's message or details use `git log`.
- Prefer read-only commands. Never add sudo unless the user asked for it.
{os_rules}
Examples of good commands:
- "which folders here are eating my disk?" -> du -sh * | sort -rh | head -n 10
- "what branch am I on?" -> git branch --show-current
- "what was my last commit message?" -> git log -1 --format=%B
- "how many python files are in this project?" -> find . -name '*.py' -type f | wc -l
"""

MACOS_RULES = """- This is macOS: BSD tools, not GNU. No `find -printf`, no `sed -i` without '', no `du --max-depth`
  (use `du -d 1`), no `ls --sort`.
"""

# Older outputs are cut harder than the newest ones, so a long session still fits a small context.
RECENT_STEPS_IN_FULL = 2
OLD_OUTPUT_CHARS = 300


class Decision(BaseModel):
    action: Literal["run", "answer"] = Field(description='"run" to run a command next, "answer" to reply to the user.')
    command: str = Field("", description='The single shell command to run when action is "run".')
    explanation: str = Field("", description="One short sentence: why this command.")
    answer: str = Field("", description='The reply to the user when action is "answer".')


class Decider(Protocol):
    """Anything with ``invoke(messages) -> Decision``: the structured chat model, or a fake in tests."""

    def invoke(self, messages: list) -> Decision: ...


def build_decider(model: str | None = None) -> Decider:
    from .config import chat_model

    return chat_model(model).with_structured_output(Decision, method="json_schema")


def _unwrap_subshell(command: str) -> str | None:
    """Return X when ``command`` is exactly ``$(X)``, else None."""
    if not (command.startswith("$(") and command.endswith(")")):
        return None
    depth = 0
    for i, ch in enumerate(command):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(command) - 1:
                return None  # the first $( closes early: "$(a) | $(b)" is not one wrapper
    return command[2:-1].strip()


def clean_command(command: str) -> str:
    """Undo wrappers small models add: code fences, backticks, a "$ " prompt, a whole-line $( ).

    The $( ) case matters most: ``$(git show)`` would run git, then run its *output* as a command.
    """
    cmd = command.strip()
    while True:
        before = cmd
        if cmd.startswith("```") and cmd.endswith("```") and len(cmd) >= 6:
            cmd = cmd[3:-3].strip()
            first, _, rest = cmd.partition("\n")
            if rest and first.strip().isalpha():  # ```bash\n...```
                cmd = rest.strip()
        if len(cmd) >= 2 and cmd[0] == cmd[-1] == "`":
            cmd = cmd[1:-1].strip()
        if cmd.startswith("$ "):
            cmd = cmd[2:].strip()
        inner = _unwrap_subshell(cmd)
        if inner is not None:
            cmd = inner
        if cmd == before:
            return cmd


def system_prompt(cwd: str) -> str:
    os_name = "macOS" if platform.system() == "Darwin" else platform.system()
    shell = os.path.basename(os.environ.get("SHELL") or "/bin/sh")
    return SYSTEM_PROMPT.format(
        os_name=os_name, shell=shell, cwd=cwd, os_rules=MACOS_RULES if os_name == "macOS" else ""
    )


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} more chars]"


def observation(step: dict, *, full: bool) -> str:
    """How one finished step is reported back to the model."""
    if step["outcome"] == "blocked":
        return f"shellmate refused to run that: {step['note']}. Do not retry it."
    limit = 10**9 if full else OLD_OUTPUT_CHARS
    parts = [f"exit code {step['exit_code']}"]
    if step.get("stdout", "").strip():
        parts.append("stdout:\n" + _cut(step["stdout"].rstrip(), limit))
    if step.get("stderr", "").strip():
        parts.append("stderr:\n" + _cut(step["stderr"].rstrip(), limit))
    if len(parts) == 1:
        parts.append("(no output)")
    return "\n".join(parts)


NEXT_STEP_HINT = (
    '\n\nIf this output answers "{request}", reply with action "answer" and tell the user what it '
    "means in a sentence. Otherwise run a DIFFERENT command."
)


def build_messages(request: str, steps: list[dict], *, cwd: str) -> list[tuple[str, str]]:
    messages = [("system", system_prompt(cwd)), ("human", request)]
    for i, step in enumerate(steps):
        latest = i == len(steps) - 1
        text = observation(step, full=i >= len(steps) - RECENT_STEPS_IN_FULL)
        if latest:
            text += NEXT_STEP_HINT.format(request=request)  # small models need the reminder right here
        messages.append(("ai", f"run: {step['command']}"))
        messages.append(("human", text))
    return messages


def repeated_command(command: str, steps: list[dict]) -> bool:
    return any(step["command"] == command for step in steps)


def answer_is_a_command(answer: str, steps: list[dict]) -> bool:
    """Small models sometimes 'answer' with the command they just ran."""
    text = clean_command(answer.strip().strip("."))
    return bool(text) and repeated_command(text, steps)


REPEAT_NUDGE = (
    "You already ran `{command}` and its output is above. Do not run it again. Reply with action "
    '"answer" and tell the user, in a sentence, what that output means for "{request}".'
)
ANSWER_NUDGE = (
    'Your answer was a command, not an answer. Reply with action "answer" and a sentence for the '
    'user that answers "{request}" using the output above.'
)
