"""`shellmate undo`: work out the command that reverses the last change shellmate made here.

Only simple, well-understood changes have an inverse. Anything else is reported as not undoable,
never guessed. The inverse is shown, risk-checked and confirmed like any other command, so undoing
a `git commit` (a `git reset --soft`) still needs the full word "yes".

Git inverses (`reset --soft HEAD~1`, `switch -`, `stash pop`) are relative to the repository's
*current* state, and you may have committed or switched by hand since. So right after a git command
runs, shellmate records HEAD, the branch and the top stash, and undo refuses if any of them moved.
"""

from __future__ import annotations

import os
import shlex
import subprocess

from .memory import same_folder
from .risk import SEPARATORS, _tokenize

REDIRECTS = {">", ">>", ">|", "&>", "&>>", "<", "<<", "<<<", ">&"}


def _simple(command: str) -> list[str] | None:
    """The tokens of a single simple command, or None for pipelines, chains, redirections."""
    if any(ch in command for ch in "$`\n"):
        return None
    try:
        tokens = _tokenize(command)
    except ValueError:
        return None
    if not tokens or any(t in SEPARATORS or t in REDIRECTS for t in tokens):
        return None
    return tokens


def _positional(args: list[str]) -> list[str]:
    return [a for a in args if not a.startswith("-")]


def _git_inverse(args: list[str], cwd: str) -> tuple[str, str] | None:
    if not args:
        return None
    sub, rest = args[0], args[1:]
    paths = _positional(rest)
    if sub == "add":
        return shlex.join(["git", "restore", "--staged", "--", *(paths or ["."])]), "unstages what was added"
    if sub == "commit" and "--amend" not in rest:
        return "git reset --soft HEAD~1", "un-commits, keeping the changes staged"
    if sub in {"switch", "checkout"} and paths and not {"-b", "-c", "-B", "-C", "--", "."} & set(rest):
        if any(os.path.lexists(os.path.join(cwd, p)) for p in paths):
            return None  # `git checkout README.md` restored a file: its old changes are gone
        return "git switch -", "switches back to the previous branch"
    if sub == "stash" and (not rest or rest[0] == "push"):
        return "git stash pop", "puts the stashed changes back"
    if sub == "branch" and len(paths) == 1 and not [a for a in rest if a.startswith("-")]:
        return shlex.join(["git", "branch", "-d", paths[0]]), "deletes the branch it created"
    return None


def inverse(command: str, *, cwd: str) -> tuple[str, str] | None:
    """(inverse command, what it does) for ``command`` as it ran in ``cwd``, or None."""
    tokens = _simple(command)
    if tokens is None:
        return None
    prog, args = os.path.basename(tokens[0]), tokens[1:]
    targets = _positional(args)

    if prog == "mkdir" and targets:
        return shlex.join(["rmdir", *reversed(targets)]), "removes the folder(s) it created, if still empty"
    if prog == "git":
        return _git_inverse(args, cwd)
    if prog == "mv" and len(targets) == 2 and len(args) == 2:
        src, dst = targets
        moved_into = os.path.join(dst, os.path.basename(src))
        if os.path.isdir(os.path.join(cwd, dst)) and os.path.lexists(os.path.join(cwd, moved_into)):
            dst = moved_into  # `mv a.txt folder` put it at folder/a.txt
        return shlex.join(["mv", dst, src]), f"moves {dst} back to {src}"
    if prog == "ln" and "-s" in args and len(targets) == 2:
        return shlex.join(["rm", targets[1]]), "removes the link it created"
    return None


def last_undoable(entries: list[dict], *, cwd: str) -> dict | None:
    """The newest successful change made in ``cwd`` that has not been undone yet."""
    undone = {e["undoes"] for e in entries if e.get("undoes")}
    for entry in reversed(entries):
        if not same_folder(entry.get("cwd"), cwd) or entry.get("undoes") or entry.get("id") in undone:
            continue
        if entry.get("exit_code") != 0 or entry.get("risk") in {None, "safe"}:
            continue  # failed, or read-only: nothing to undo
        return entry
    return None


GIT_STATE_QUERIES = {
    "git_head": ["git", "rev-parse", "-q", "--verify", "HEAD"],
    "git_branch": ["git", "branch", "--show-current"],
    "git_stash": ["git", "rev-parse", "-q", "--verify", "refs/stash"],
}


def _is_git(command: str) -> bool:
    tokens = _simple(command)
    return bool(tokens) and os.path.basename(tokens[0]) == "git"


def git_state(cwd: str) -> dict:
    """HEAD, branch and top stash of the repository in ``cwd`` ({} outside a repository)."""
    state = {}
    for key, argv in GIT_STATE_QUERIES.items():
        try:
            out = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            return {}
        state[key] = out.stdout.strip() if out.returncode == 0 else ""
    return state if state.get("git_head") or state.get("git_branch") else {}


def snapshot_after(command: str, cwd: str) -> dict:
    """What to record with a history entry so a later undo can tell whether things moved since."""
    return git_state(cwd) if _is_git(command) else {}


def changed_since(entry: dict, cwd: str) -> str | None:
    """Why undoing ``entry`` is no longer safe, or None when nothing moved since it ran."""
    if not _is_git(entry["command"]):
        return None
    recorded = {k: entry.get(k) for k in GIT_STATE_QUERIES}
    if not any(recorded.values()):
        return "shellmate didn't record the repository state for it, so it can't tell what changed since"
    now = git_state(cwd)
    moved = [k.removeprefix("git_") for k in GIT_STATE_QUERIES if now.get(k) != recorded[k]]
    if moved:
        return f"the repository changed since ({', '.join(moved)} moved), so the undo would hit the wrong thing"
    return None
