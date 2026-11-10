"""LangChain concepts: structured output, prompts and message history.

The model never runs anything. Each turn it returns one :class:`Decision` (run a command, read a
file, list a folder, or answer), and graph.py acts on it. This file is everything the model sees
and everything it may say back:

1. ``Decision``          the schema the model must fill in (structured output)
2. ``build_decider``     a chat model that returns a ``Decision`` instead of free text
3. ``SYSTEM_PROMPT``     the instructions, with rules for the OS shellmate runs on
4. ``build_messages``    the conversation so far, as (role, text) messages
5. guards                small checks that catch common small-model mistakes
"""

from __future__ import annotations

import os
import platform
import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field

# --- 1. Structured output: the schema the model fills in ----------------------------------------

# LangChain: with_structured_output() turns a Pydantic model into a JSON schema, tells the model
# to answer in that shape, and parses the reply back into the Pydantic object. No regex parsing of
# free text, and a reply that doesn't fit the schema raises instead of slipping through.
# The read-only tools, listed once: adding a name here is what routes it to graph.use_tool.
TOOL_ACTIONS = ("read_file", "list_dir")
Action = Literal["run", *TOOL_ACTIONS, "answer"]


class Decision(BaseModel):
    action: Action = Field(description="What to do next: run a command, use a read-only tool, or answer.")
    command: str = Field("", description='The single shell command to run when action is "run".')
    path: str = Field("", description='The file or folder for "read_file" and "list_dir".')
    explanation: str = Field("", description="One short sentence: why this step.")
    answer: str = Field("", description='The reply to the user when action is "answer".')


class Decider(Protocol):
    """Anything with ``invoke(messages) -> Decision``: the structured chat model, or a fake in tests.

    LangChain: every model, prompt and chain is a "Runnable" with the same ``invoke()`` method, which
    is why a ten-line fake (tests/fakes.py) can stand in for the real model.
    """

    def invoke(self, messages: list) -> Decision: ...


# --- 2. The model ---------------------------------------------------------------------------------

def build_decider(model: str | None = None) -> Decider:
    """The chat model, wrapped so ``invoke()`` returns a ``Decision``."""
    from .replay import replay_decider

    replay = replay_decider()  # SHELLMATE_REPLAY: scripted answers for demos, no model at all
    if replay is not None:
        return replay
    from .config import chat_model

    # method="json_schema" uses the model's native JSON mode (Ollama's `format`), which small local
    # models follow far more reliably than tool calling.
    return chat_model(model).with_structured_output(Decision, method="json_schema")


# --- 3. The system prompt -------------------------------------------------------------------------

SYSTEM_PROMPT = """You are shellmate. You help the user by running shell commands one at a time and
reading their output, until you can answer them.

Environment: {os_name}, shell {shell}. Commands run inside the user's current folder ({cwd}).

Each turn, reply with exactly one action:
- "run": ONE command in "command", and a short explanation of why.
- "read_file": read a text file; put its path in "path". Prefer this to cat/head for looking
  inside a file.
- "list_dir": list a folder; put its path in "path" ("." for the current folder).
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
- When an example below fits the request, use that command as it is.
{os_rules}
Examples of good commands:
- "which folders here are eating my disk?" -> du -sh * | sort -rh | head -n 10
- "what branch am I on?" -> git branch --show-current
- "what was my last commit message?" -> git log -1 --format=%B
- "how many python files are in this project?" -> find . -name '*.py' -type f | wc -l
"""

# Rules for one operating system. Only the rules for the OS shellmate runs on go into the prompt,
# so a small model never sees (and mixes in) another system's flags.
MACOS_RULES = """- This is macOS: BSD tools, not GNU. No `find -printf`, no `sed -i` without '', no `du --max-depth`
  (use `du -d 1`), no `ls --sort`, no `ps --sort` or `ps -e --format`.
- Processes by memory: `ps -Ao pid,%mem,rss,comm -m | head -n 11`; by CPU: `ps -Ao pid,%cpu,comm -r | head -n 11`.
  `ps -m` / `ps -r` already sort, so don't pipe them into sort. macOS ps has no `cmd` column (use `comm`).
- Ports: `lsof -nP -iTCP:3000 -sTCP:LISTEN`. Free memory: `vm_stat` or `top -l 1 | head -n 10` (no `free`).
- OS version: `sw_vers`. Open a file or folder: `open`. Clipboard: `pbcopy` / `pbpaste`.
- Example: "what is using the most memory?" -> ps -Ao pid,%mem,rss,comm -m | head -n 11
"""

LINUX_RULES = """- This is Linux: GNU coreutils. `sed -i` needs no '' argument, `du --max-depth=1` and `find -printf` work.
- Processes by memory: `ps aux --sort=-%mem | head -n 11`; by CPU: `ps aux --sort=-%cpu | head -n 11`.
- Free memory: `free -h`. Ports: `ss -ltnp` (or `lsof -i :3000`). Disks: `df -h`, `lsblk`.
- OS version: `cat /etc/os-release` (no `sw_vers`). Services: `systemctl status NAME` (reading only).
  Open a file: `xdg-open`. Logs: `journalctl -n 50 --no-pager`.
- Package managers differ by distro (apt, dnf, pacman): check /etc/os-release before suggesting one.
- Example: "what is using the most memory?" -> ps aux --sort=-%mem | head -n 11
"""

WINDOWS_RULES = """- This is Windows, running shellmate from a POSIX shell (Git Bash / MSYS2). Unix tools there are
  GNU versions, but Windows' own tools are not Unix: `ps` is limited, there is no `free`, `lsof` or `sw_vers`.
- Processes by memory: `tasklist //FO TABLE | sort -k5 -rn | head -n 11`, or
  `powershell -NoProfile -Command "Get-Process | Sort-Object WS -Descending | Select-Object -First 10"`.
- Ports: `netstat -ano | grep LISTENING`. OS version: `cmd //c ver`. Open a file: `start`.
- Paths: use forward slashes (C:/Users/...), and quote paths with spaces. Drives appear as /c/, /d/.
- In Git Bash, flags for Windows tools start with // (e.g. `tasklist //FI "IMAGENAME eq node.exe"`).
- Example: "what is using the most memory?" -> tasklist //FO TABLE | sort -k5 -rn | head -n 11
"""

OS_RULES = {"macOS": MACOS_RULES, "Linux": LINUX_RULES, "Windows": WINDOWS_RULES}


def os_name() -> str:
    """The OS shellmate runs on: "macOS", "Linux", "Windows", or whatever platform.system() says."""
    system = platform.system()
    if system == "Darwin":
        return "macOS"
    if system.startswith(("CYGWIN", "MSYS", "MINGW")):
        return "Windows"
    return system or "unknown"


def os_rules(name: str | None = None) -> str:
    """Only the rules for this OS; an OS without rules gets none."""
    return OS_RULES.get(name or os_name(), "")


def system_prompt(cwd: str) -> str:
    name = os_name()
    shell = os.path.basename(os.environ.get("SHELL") or "/bin/sh")
    return SYSTEM_PROMPT.format(os_name=name, shell=shell, cwd=cwd, os_rules=os_rules(name))


# --- 4. Message history ---------------------------------------------------------------------------

# Older outputs are cut harder than the newest ones, so a long session still fits a small context.
RECENT_STEPS_IN_FULL = 2
OLD_OUTPUT_CHARS = 300

NEXT_STEP_HINT = (
    '\n\nIf this output answers "{request}", reply with action "answer" and tell the user what it '
    "means in a sentence. Otherwise take a DIFFERENT step."
)
FAILED_HINT = (
    "\n\nThat step FAILED ({how}): the output above is an error, not an answer. "
    "Do not describe results you did not get. Read the error, then run a corrected command "
    '(for example with different flags), or reply with action "answer" and tell the user it failed and why.'
)


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} more chars]"


def observation(step: dict, *, full: bool) -> str:
    """How one finished step is reported back to the model."""
    if step["outcome"] == "blocked":
        return f"shellmate refused to run that: {step['note']}. Do not retry it."
    limit = 10**9 if full else OLD_OUTPUT_CHARS
    parts = [f"exit code {step['exit_code']}"]
    if step["exit_code"] == 0 and step_failed(step):
        parts[0] += " (but it printed only errors, so it did not work)"
    if step.get("stdout", "").strip():
        parts.append("stdout:\n" + _cut(step["stdout"].rstrip(), limit))
    if step.get("stderr", "").strip():
        parts.append("stderr:\n" + _cut(step["stderr"].rstrip(), limit))
    if len(parts) == 1:
        parts.append("(no output)")
    return "\n".join(parts)


def build_messages(request: str, steps: list[dict], *, cwd: str) -> list[tuple[str, str]]:
    """The whole conversation, rebuilt from the steps each turn.

    LangChain: chat models take a list of messages. ("system", text), ("human", text) and
    ("ai", text) tuples are shorthand for SystemMessage, HumanMessage and AIMessage. Each step adds
    what the model did ("ai") and what came back ("human"), so the model sees its own history.
    """
    messages = [("system", system_prompt(cwd)), ("human", request)]
    for i, step in enumerate(steps):
        text = observation(step, full=i >= len(steps) - RECENT_STEPS_IN_FULL)
        if i == len(steps) - 1:  # small models need the reminder right after the newest output
            failed = last_failure(steps)
            text += FAILED_HINT.format(how=failure_label(failed)) if failed else NEXT_STEP_HINT.format(request=request)
        said = step["command"] if step["outcome"] == "tool" else f"run: {step['command']}"
        messages.append(("ai", said))
        messages.append(("human", text))
    return messages


def step_key(decision: Decision) -> str:
    """How a decision is written in the step history, e.g. "ls -la" or "read_file README.md"."""
    if decision.action == "run":
        return clean_command(decision.command)
    if decision.action in TOOL_ACTIONS:
        return f"{decision.action} {decision.path.strip()}".strip()
    return ""


# --- 5. Guards: catching common small-model mistakes ----------------------------------------------
# Prompts are suggestions; these checks are not. graph.decide() runs them on every Decision and, on
# a problem, asks the model once more with the matching nudge.

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
PLACEHOLDER_NUDGE = (
    "Your answer contains placeholders like [name] instead of real values. Only state values you "
    "actually saw in a command's output. If no command worked yet, run a working one, or tell the "
    "user it didn't work."
)
FAILED_ANSWER_NUDGE = (
    "Your last step failed ({how}) with: {error}\n"
    "Your answer must not claim it worked. Either run a corrected command, or answer by saying the "
    "command failed and why."
)


def step_failed(step: dict) -> bool:
    """A command or tool that failed. A pipeline reports only its last command's exit code, so
    `ps -x | sort -bad | head` exits 0 with nothing but errors: errors and no output count as failed."""
    if step["outcome"] not in {"ran", "tool"}:
        return False
    if step.get("exit_code", 0) != 0:
        return True
    return bool(step.get("stderr", "").strip()) and not step.get("stdout", "").strip()


def failure_label(step: dict) -> str:
    code = step.get("exit_code", 0)
    return f"exit code {code}" if code else "it printed only errors"


def last_failure(steps: list[dict]) -> dict | None:
    """The newest step, when it failed."""
    return steps[-1] if steps and step_failed(steps[-1]) else None


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


PLACEHOLDER = re.compile(r"\[[a-z%_]+(?: [a-z%_]+){0,2}]")


def answer_has_placeholders(answer: str) -> bool:
    """Small models sometimes answer with a template, e.g. "[comm] uses [percent] percent"."""
    return bool(PLACEHOLDER.search(answer))


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
