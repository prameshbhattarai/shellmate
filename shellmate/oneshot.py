"""Milestone 1, the one-shot flow: request -> one proposed command -> confirm -> run -> print.

The model only *proposes*. Running is always the user's decision, made in the CLI.
"""

from __future__ import annotations

import os
import platform
from typing import Protocol

from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You turn a plain-English request into exactly ONE shell command.

Environment: {os_name}, shell {shell}. The command already runs inside the user's current folder
({cwd}).

Rules:
- Return a single command line. Pipes and && are fine; no scripts, no comments.
- Return the bare command: never wrap it in $( ), backticks or quotes, and no leading "$ ".
- Use relative paths ("." or "*"). Never write out the current folder's absolute path.
- Prefer the simplest standard tool for the job (du for sizes, ls for listings, git for git).
- Print only what was asked for, not everything related. Narrow the output with flags
  (--format, --stat, --name-only, -n) instead of dumping whole files, diffs or logs.
- git: `git diff` shows changes not yet committed; `git show` shows a whole commit (message + diff);
  `git log -1 --format=%B` shows just the last commit message.
- Prefer read-only commands. Never add sudo unless the user asked for it.
{os_rules}- If the request is not something a shell command can do, leave "command" empty and say why in
  "explanation".

Examples:
- "which folders here are eating my disk?" -> du -sh * | sort -rh | head -n 10
- "what branch am I on?" -> git branch --show-current
- "how many python files are in this project?" -> find . -name '*.py' -type f | wc -l
"""

MACOS_RULES = """- This is macOS: BSD tools, not GNU. No `find -printf`, no `sed -i` without '', no `du --max-depth`
  (use `du -d 1`), no `ls --sort`.
"""


class Proposal(BaseModel):
    command: str = Field(description="The single shell command to run, or empty if none fits.")
    explanation: str = Field(description="One short sentence: what the command does.")


class Proposer(Protocol):
    """Anything with ``invoke(messages) -> Proposal``: the structured chat model, or a fake in tests."""

    def invoke(self, messages: list) -> Proposal: ...


def build_proposer(model: str | None = None) -> Proposer:
    from .config import chat_model

    return chat_model(model).with_structured_output(Proposal, method="json_schema")


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


def _shell_name() -> str:
    return os.path.basename(os.environ.get("SHELL") or "/bin/sh")


def propose_command(request: str, proposer: Proposer, *, cwd: str) -> Proposal:
    os_name = "macOS" if platform.system() == "Darwin" else platform.system()
    os_rules = MACOS_RULES if os_name == "macOS" else ""
    system = SYSTEM_PROMPT.format(os_name=os_name, shell=_shell_name(), cwd=cwd, os_rules=os_rules)
    proposal = proposer.invoke([("system", system), ("human", request)])
    return Proposal(command=clean_command(proposal.command), explanation=proposal.explanation.strip())
