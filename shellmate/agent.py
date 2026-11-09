"""What the model sees and what it may say back.

Each turn the model returns a :class:`Decision`: run ONE command, use a read-only tool, or answer the
user. It never runs anything itself; the graph (graph.py) risk-checks commands and asks the user
when needed.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are shellmate. You help the user by running shell commands one at a time and
reading their output, until you can answer them.

Environment: {os_name}, shell {shell}. Commands run inside the user's current folder ({cwd}).

Each turn, reply with exactly one action:
- "run": ONE command in "command", and a short explanation of why.
- "read_file": read a text file; put its path in "path". Prefer this to cat/head for looking
  inside a file.
- "list_dir": list a folder; put its path in "path" ("." for the current folder).
- "history": see the last commands shellmate ran for the user, with their exit codes.
- "last_error": see the most recent command that failed, with its error output. Use it when the
  user asks about "my last error" or "why did that fail".
- "answer": a short, plain-English answer for the user in "answer", based on what you have seen.
The tools only read, and only inside the current folder, so they never need the user's approval.

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
{extras}"""

MACOS_RULES = """- This is macOS: BSD tools, not GNU. No `find -printf`, no `sed -i` without '', no `du --max-depth`
  (use `du -d 1`), no `ls --sort`.
"""

# Older outputs are cut harder than the newest ones, so a long session still fits a small context.
RECENT_STEPS_IN_FULL = 2
OLD_OUTPUT_CHARS = 300


Action = Literal["run", "read_file", "list_dir", "history", "last_error", "answer"]
TOOL_ACTIONS = ("read_file", "list_dir", "history", "last_error")


class Decision(BaseModel):
    action: Action = Field(description="What to do next: run a command, use a read-only tool, or answer.")
    command: str = Field("", description='The single shell command to run when action is "run".')
    path: str = Field("", description='The file or folder for "read_file" and "list_dir".')
    explanation: str = Field("", description="One short sentence: why this step.")
    answer: str = Field("", description='The reply to the user when action is "answer".')


def step_key(decision: Decision) -> str:
    """How a decision is written in the step history, e.g. "ls -la" or "read_file README.md"."""
    if decision.action == "run":
        return clean_command(decision.command)
    if decision.action in TOOL_ACTIONS:
        return f"{decision.action} {decision.path.strip()}".strip()
    return ""


PERSONAS = {
    "default": "",
    "terse": "Keep answers to one short line. No pleasantries.",
    "mentor": "Answer warmly, like a patient senior engineer, and mention one useful tip when it fits.",
    "pirate": "Answer in a light pirate voice (keep commands and facts exact).",
}


@dataclass(frozen=True)
class PromptContext:
    """Per-session additions to the system prompt."""

    persona: str = "default"
    teach: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)  # from `shellmate remember`

    def render(self) -> str:
        parts = []
        if self.notes:
            parts.append("Things the user asked you to remember:\n" + "\n".join(f"- {n}" for n in self.notes))
        if PERSONAS.get(self.persona):
            parts.append("Style: " + PERSONAS[self.persona])
        if self.teach:
            parts.append(
                "Teaching mode: end every answer with one short line that explains the main command you "
                "used and its flags, so the user learns it."
            )
        return ("\n" + "\n\n".join(parts) + "\n") if parts else ""


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


def system_prompt(cwd: str, context: PromptContext | None = None) -> str:
    os_name = "macOS" if platform.system() == "Darwin" else platform.system()
    shell = os.path.basename(os.environ.get("SHELL") or "/bin/sh")
    return SYSTEM_PROMPT.format(
        os_name=os_name,
        shell=shell,
        cwd=cwd,
        os_rules=MACOS_RULES if os_name == "macOS" else "",
        extras=(context or PromptContext()).render(),
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
    "means in a sentence. Otherwise take a DIFFERENT step."
)


def build_messages(
    request: str, steps: list[dict], *, cwd: str, context: PromptContext | None = None
) -> list[tuple[str, str]]:
    messages = [("system", system_prompt(cwd, context)), ("human", request)]
    for i, step in enumerate(steps):
        latest = i == len(steps) - 1
        text = observation(step, full=i >= len(steps) - RECENT_STEPS_IN_FULL)
        if latest:
            text += NEXT_STEP_HINT.format(request=request)  # small models need the reminder right here
        said = step["command"] if step["outcome"] == "tool" else f"run: {step['command']}"
        messages.append(("ai", said))
        messages.append(("human", text))
    return messages


def repeated_command(command: str, steps: list[dict]) -> bool:
    """True when ``command`` already ran and nothing that could change its output ran since.

    Walking back from the newest step: a step that wasn't SAFE (mkdir, git add ...) may have changed
    what the command would print, so running it again after one is a real new step, not a repeat.
    """
    for step in reversed(steps):
        if step["outcome"] not in {"ran", "tool"}:
            continue
        if step["command"] == command:
            return True
        if step.get("risk", "safe") != "safe":
            return False
    return False


def refused_command(command: str, steps: list[dict]) -> dict | None:
    """The earlier refusal of ``command``, if shellmate already refused it."""
    return next((s for s in steps if s["outcome"] == "blocked" and s["command"] == command), None)


def answer_is_a_command(answer: str, steps: list[dict]) -> bool:
    """Small models sometimes 'answer' with the command they just ran."""
    text = clean_command(answer.strip().strip("."))
    return bool(text) and any(step["command"] == text for step in steps)


REPEAT_NUDGE = (
    "You already did `{command}` and its output is above. Do not do it again. Reply with action "
    '"answer" and tell the user, in a sentence, what that output means for "{request}".'
)
REFUSED_NUDGE = (
    "shellmate already refused `{command}` ({reason}) and will never run it. Do not propose it again. "
    'Use a safer command, or reply with action "answer" and explain why it can\'t be done.'
)
ANSWER_NUDGE = (
    'Your answer was a command, not an answer. Reply with action "answer" and a sentence for the '
    'user that answers "{request}" using the output above.'
)
