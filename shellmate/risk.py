"""Risk classification for shell commands. Rules only: the model never decides what is safe.

Levels, from least to most risky:

- SAFE     read-only (ls, du, cat, git log ...). Still shown before it runs.
- CAUTION  changes something, but it is usually easy to undo (mkdir, git add, >> append ...).
- DANGER   destructive, privileged, outward-facing or unknown (rm, mv, sudo, git push, > overwrite,
           any program not on a list). Runs only after the user types the full word "yes".
- BLOCKED  catastrophic (rm -rf ~, curl | sh, dd to a disk ...). shellmate never runs it,
           whatever the user answers.

How a command is judged: it is split into segments on | && || ; and each segment is judged on its
program and flags. Redirections are judged on their own. The command's verdict is its worst
segment's. Anything the rules cannot understand (command substitution, unbalanced quotes, several
lines) is DANGER: when in doubt, ask harder.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath


class Risk(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    DANGER = "danger"
    BLOCKED = "blocked"

    @property
    def rank(self) -> int:
        return list(Risk).index(self)


@dataclass(frozen=True)
class RiskVerdict:
    risk: Risk
    reason: str
    decided_by: str = "rule"


def needs_approval(verdict: RiskVerdict) -> bool:
    """Only SAFE commands could ever run without asking."""
    return verdict.risk is not Risk.SAFE


def is_blocked(verdict: RiskVerdict) -> bool:
    return verdict.risk is Risk.BLOCKED


def _v(risk: Risk, reason: str) -> RiskVerdict:
    return RiskVerdict(risk, reason)


def _worst(verdicts: list[RiskVerdict]) -> RiskVerdict:
    return max(verdicts, key=lambda v: v.risk.rank)


# --- program tables -------------------------------------------------------------------------------

SAFE_PROGRAMS = frozenset(
    """
    ls pwd echo printf cat head tail wc du df sort uniq cut tr grep egrep fgrep rg which whereis
    type whoami id date cal uname hostname file stat tree basename dirname realpath readlink ps
    uptime jq column diff cmp md5 md5sum shasum sha256sum true false test exit cd export seq
    printenv sleep nl fold expand rev tac comm join paste look strings lsof sw_vers say
    """.split()
)

CAUTION_PROGRAMS = {
    "mkdir": "creates a folder",
    "touch": "creates or updates a file",
    "cp": "copies files (can overwrite the destination)",
    "ln": "creates a link",
    "open": "opens a file or app",
    "code": "opens an editor",
    "tar": "creates or extracts an archive",
    "zip": "creates an archive",
    "unzip": "extracts an archive (can overwrite files)",
    "gzip": "compresses a file in place",
    "gunzip": "decompresses a file in place",
    "pbcopy": "replaces your clipboard",
    "wget": "downloads from the network",
}

DANGER_PROGRAMS = {
    "rm": "deletes files permanently (no trash)",
    "rmdir": "deletes folders",
    "mv": "moves or renames files (can overwrite the destination)",
    "chmod": "changes permissions",
    "chown": "changes file ownership",
    "chgrp": "changes file group",
    "kill": "stops processes",
    "killall": "stops processes",
    "pkill": "stops processes",
    "truncate": "empties files",
    "shred": "destroys files",
    "srm": "destroys files",
    "ssh": "runs commands on another machine",
    "scp": "copies to or from another machine",
    "rsync": "syncs files (can delete at the destination)",
    "sftp": "transfers files to another machine",
    "crontab": "changes scheduled jobs",
    "launchctl": "changes system services",
    "shutdown": "shuts the machine down",
    "reboot": "restarts the machine",
    "halt": "stops the machine",
    "sysctl": "can change kernel settings",
    "networksetup": "changes network settings",
    "systemsetup": "changes system settings",
    "security": "touches the keychain",
    "csrutil": "changes system integrity protection",
    "spctl": "changes app security policy",
    "pmset": "changes power settings",
    "nvram": "changes firmware variables",
    "scutil": "changes system configuration",
}

# Programs that run whatever code they are given.
INTERPRETERS = frozenset(
    "sh bash zsh fish dash ksh csh tcsh python python3 node deno bun ruby perl php osascript "
    "eval exec source .".split()
)

# Prefixes that run the rest of the line as a command.
WRAPPERS = frozenset("env nohup time nice command builtin xargs caffeinate".split())
PRIVILEGE = frozenset("sudo doas su".split())

SEPARATORS = frozenset({"|", "||", "&&", ";", "&", "|&", "(", ")", ";;"})
OVERWRITE_REDIRECTS = frozenset({">", ">|", "&>"})
APPEND_REDIRECTS = frozenset({">>", "&>>"})
INPUT_REDIRECTS = frozenset({"<", "<<", "<<<"})

FORK_BOMB = re.compile(r":\s*\(\s*\)\s*\{")
RAW_DEVICE = re.compile(r"/dev/(r?disk\d|sd[a-z]|nvme\d|hd[a-z])")


# --- paths ----------------------------------------------------------------------------------------

def _is_precious(target: str) -> bool:
    """True for paths whose recursive deletion would wreck the machine or the user's files:
    /, a top-level folder, the home folder or a folder directly in it, the current folder or above.
    """
    t = target.strip()
    if not t:
        return False
    home = os.path.expanduser("~")
    for prefix in ("${HOME}", "$HOME"):
        if t.startswith(prefix):
            t = home + t[len(prefix):]
    t = os.path.expanduser(t)
    while t.endswith("/*") or (t.endswith("/") and t != "/"):
        t = t[:-2] if t.endswith("/*") else t[:-1]
    if t in {"", "/", ".", ".."} or t.startswith("../"):
        return True
    if not t.startswith("/"):
        return False  # a path inside the current folder, e.g. node_modules or ./build
    path = PurePosixPath(os.path.normpath(t))
    home_path = PurePosixPath(home)
    if len(path.parts) <= 2:  # "/" or a top-level folder like /Users, /usr, /System
        return True
    if path == home_path or home_path.is_relative_to(path):
        return True
    return path.is_relative_to(home_path) and len(path.relative_to(home_path).parts) <= 1


def _is_recursive(args: list[str]) -> bool:
    for a in args:
        if a == "--recursive":
            return True
        if a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a):
            return True
    return False


def _targets(args: list[str]) -> list[str]:
    return [a for a in args if not a.startswith("-")]


# --- per-program rules ----------------------------------------------------------------------------

GIT_SAFE = frozenset(
    "status log show diff rev-parse blame shortlog describe ls-files ls-tree reflog grep cat-file "
    "show-ref whatchanged count-objects help version".split()
)
GIT_CAUTION = frozenset("add commit fetch stash switch init mv restore merge cherry-pick tag notes".split())


def _git(args: list[str]) -> RiskVerdict:
    i = 0  # skip global options like -C dir or --no-pager
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in {"-C", "-c"} else 1
    if i >= len(args):
        return _v(Risk.SAFE, "git with no subcommand")
    sub, rest = args[i], args[i + 1:]
    if sub in GIT_SAFE:
        return _v(Risk.SAFE, f"git {sub} only reads")
    if sub == "branch":
        if any(a in {"-d", "-D", "--delete", "-m", "-M", "--move", "-f", "--force"} for a in rest):
            return _v(Risk.DANGER, "deletes, renames or force-moves a branch")
        listing = {"-a", "-r", "-v", "-vv", "--list", "--show-current", "--all", "--merged", "--no-merged"}
        if _targets(rest) and not any(a in listing for a in rest):
            return _v(Risk.CAUTION, "creates a branch")
        return _v(Risk.SAFE, "lists branches")
    if sub == "remote":
        reads = not rest or rest[0] in {"-v", "show", "get-url"}
        return _v(Risk.SAFE, "lists remotes") if reads else _v(Risk.CAUTION, "changes remotes")
    if sub == "config":
        reads = {"--get", "--list", "-l", "--get-all", "--show-origin"}
        if any(a in reads for a in rest):
            return _v(Risk.SAFE, "reads git config")
        return _v(Risk.CAUTION, "changes git config")
    if sub == "stash" and rest[:1] in (["list"], ["show"]):
        return _v(Risk.SAFE, "lists stashes")
    if sub == "stash" and rest[:1] in (["drop"], ["clear"]):
        return _v(Risk.DANGER, "throws away stashed work")
    if sub == "checkout":
        if "--" in rest or "-f" in rest or "--force" in rest or "." in rest:
            return _v(Risk.DANGER, "discards uncommitted changes")
        return _v(Risk.CAUTION, "switches branch")
    if sub == "restore" and "--staged" not in rest:
        return _v(Risk.DANGER, "discards uncommitted changes")
    if sub == "commit" and "--amend" in rest:
        return _v(Risk.DANGER, "rewrites the last commit")
    if sub in GIT_CAUTION:
        return _v(Risk.CAUTION, f"git {sub} changes the repository")
    if sub == "push":
        if any(a.startswith("--force") or a == "-f" or a.startswith("+") for a in rest):
            return _v(Risk.DANGER, "force-pushes, which can overwrite others' work")
        return _v(Risk.DANGER, "publishes commits to a remote")
    if sub == "reset":
        return _v(Risk.DANGER, "resets history" + (" and throws away changes" if "--hard" in rest else ""))
    if sub == "clean":
        return _v(Risk.DANGER, "deletes untracked files")
    if sub == "pull":
        return _v(Risk.CAUTION, "merges remote changes into your branch")
    return _v(Risk.DANGER, f"git {sub} can rewrite or remove history")


READ_SUBCOMMANDS = frozenset("list ls show info outdated search freeze view why doctor".split())
INSTALL_SUBCOMMANDS = frozenset("install add i update upgrade sync lock".split())


def _package_manager(prog: str, args: list[str]) -> RiskVerdict:
    sub = next((a for a in args if not a.startswith("-")), "")
    if not sub or sub in READ_SUBCOMMANDS:
        return _v(Risk.SAFE, f"{prog} only reads")
    if sub in INSTALL_SUBCOMMANDS:
        return _v(Risk.CAUTION, f"{prog} {sub} changes installed packages")
    return _v(Risk.DANGER, f"{prog} {sub} may remove packages or run scripts")


def _program(prog: str, args: list[str]) -> RiskVerdict:
    name = os.path.basename(prog)

    if name in PRIVILEGE:
        inner = _segment(args) if args else _v(Risk.DANGER, "opens a root shell")
        if inner.risk is Risk.BLOCKED:
            return inner
        return _v(Risk.DANGER, f"runs as administrator ({inner.reason})")

    if name in WRAPPERS:
        rest = list(args)
        while rest and (rest[0].startswith("-") or (name == "xargs" and rest[0].isdigit())
                        or re.match(r"^\w+=", rest[0])):
            rest.pop(0)
        return _segment(rest) if rest else _v(Risk.SAFE, f"{name} only prints")

    if name in {"rm", "rmdir", "chmod", "chown", "chgrp"}:
        precious = [t for t in _targets(args) if _is_precious(t)]
        if precious and (name == "rmdir" or _is_recursive(args)):
            return _v(Risk.BLOCKED, f"{name} on {precious[0]} would wreck your system or home folder")
        return _v(Risk.DANGER, DANGER_PROGRAMS[name])

    if name == "dd":
        if any(a.startswith("of=/dev/") for a in args):
            return _v(Risk.BLOCKED, "dd writing to a device can erase a disk")
        return _v(Risk.DANGER, "dd copies raw data")
    if name.startswith(("mkfs", "newfs")) or name in {"fdisk", "gdisk", "sfdisk", "wipefs"}:
        return _v(Risk.BLOCKED, "formats or repartitions a disk")
    if name == "diskutil":
        sub = args[0].lower() if args else ""
        if sub in {"list", "info", "activity"}:
            return _v(Risk.SAFE, "lists disks")
        if sub.startswith(("erase", "partition", "zero", "reformat", "secureerase", "randomdisk")):
            return _v(Risk.BLOCKED, "erases or repartitions a disk")
        return _v(Risk.DANGER, "changes disks or volumes")

    if name == "git":
        return _git(args)
    if name in {"brew", "pip", "pip3", "npm", "pnpm", "yarn", "uv", "cargo", "gem"}:
        return _package_manager(name, args)

    if name == "find":
        if "-delete" in args:
            return _v(Risk.DANGER, "find -delete removes every match")
        if any(a in {"-exec", "-execdir", "-ok", "-okdir"} for a in args):
            return _v(Risk.DANGER, "find -exec runs a command on every match")
        if any(a.startswith("-fprint") or a == "-fls" for a in args):
            return _v(Risk.CAUTION, "find writes to a file")
        return _v(Risk.SAFE, "find only lists")
    if name in {"sed", "gsed"}:
        if any(a.startswith("-i") or a.startswith("--in-place") for a in args):
            return _v(Risk.DANGER, "sed -i edits files in place")
        return _v(Risk.SAFE, "sed only prints")
    if name in {"awk", "gawk"}:
        if any("system" in a or ">" in a or "|" in a for a in args):
            return _v(Risk.DANGER, "awk script writes files or runs commands")
        return _v(Risk.SAFE, "awk only prints")
    if name == "curl":
        sends = {"-d", "-F", "--form", "-T", "--upload-file"}
        method = next((args[i + 1] for i, a in enumerate(args[:-1]) if a in {"-X", "--request"}), "GET").upper()
        if any(a in sends or a.startswith("--data") for a in args) or method not in {"GET", "HEAD"}:
            return _v(Risk.DANGER, "sends data over the network")
        if any(a in {"-o", "-O", "--output", "--remote-name"} for a in args):
            return _v(Risk.CAUTION, "downloads to a file")
        return _v(Risk.CAUTION, "makes a network request")
    if name == "defaults":
        if args[:1] in (["read"], ["domains"]):
            return _v(Risk.SAFE, "reads settings")
        return _v(Risk.DANGER, "changes system settings")
    if name == "docker":
        sub = args[0] if args else ""
        if sub in {"ps", "images", "logs", "inspect", "version", "info", "stats"}:
            return _v(Risk.SAFE, f"docker {sub} only reads")
        return _v(Risk.DANGER, f"docker {sub} changes containers or images")

    if name in INTERPRETERS:
        if args in (["--version"], ["-V"]):
            return _v(Risk.SAFE, "prints a version")
        return _v(Risk.DANGER, f"{name} runs arbitrary code")
    if name in SAFE_PROGRAMS:
        return _v(Risk.SAFE, "read-only")
    if name in CAUTION_PROGRAMS:
        return _v(Risk.CAUTION, CAUTION_PROGRAMS[name])
    if name in DANGER_PROGRAMS:
        return _v(Risk.DANGER, DANGER_PROGRAMS[name])
    return _v(Risk.DANGER, f"{name!r} is not a command shellmate knows, so it can't vouch for it")


def _segment(tokens: list[str]) -> RiskVerdict:
    """Judge one simple command (no | && ; inside): its redirections, then its program."""
    verdicts: list[RiskVerdict] = []
    args: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok.isdigit() and nxt[:1] in {"<", ">"}:
            i += 1  # file-descriptor number, e.g. the 2 in 2>/dev/null
            continue
        if tok in OVERWRITE_REDIRECTS or tok in APPEND_REDIRECTS:
            if RAW_DEVICE.search(nxt):
                verdicts.append(_v(Risk.BLOCKED, "writes straight to a disk device"))
            elif nxt != "/dev/null":
                if tok in APPEND_REDIRECTS:
                    verdicts.append(_v(Risk.CAUTION, f"appends to {nxt}"))
                else:
                    verdicts.append(_v(Risk.DANGER, f"overwrites {nxt}"))
            i += 2
            continue
        if tok == ">&":
            if not (nxt.isdigit() or nxt == "-"):
                verdicts.append(_v(Risk.DANGER, f"overwrites {nxt}"))
            i += 2
            continue
        if tok in INPUT_REDIRECTS:
            i += 2
            continue
        args.append(tok)
        i += 1

    while args and re.match(r"^[A-Za-z_]\w*=", args[0]):
        args.pop(0)  # FOO=bar before the program
    if args:
        verdicts.append(_program(args[0], args[1:]))
    return _worst(verdicts) if verdicts else _v(Risk.SAFE, "nothing to run")


def _effective_program(tokens: list[str]) -> str:
    """The program a segment really runs, looking past sudo, env, xargs, FOO=bar and their flags."""
    rest = list(tokens)
    while rest:
        name = os.path.basename(rest[0])
        if name in PRIVILEGE or name in WRAPPERS or rest[0].startswith("-") or re.match(r"^\w+=", rest[0]):
            rest.pop(0)
            continue
        return name
    return ""


def _tokenize(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def classify(command: str) -> RiskVerdict:
    cmd = command.strip()
    if not cmd:
        return _v(Risk.DANGER, "empty command")
    if FORK_BOMB.search(cmd):
        return _v(Risk.BLOCKED, "fork bomb: it would freeze the machine")
    if "\n" in cmd or "\r" in cmd:
        return _v(Risk.DANGER, "several lines: shellmate checks one command line at a time")
    if "$(" in cmd or "`" in cmd or "<(" in cmd or ">(" in cmd:
        return _v(Risk.DANGER, "command substitution runs a hidden command shellmate can't check")
    try:
        tokens = _tokenize(cmd)
    except ValueError:
        return _v(Risk.DANGER, "could not parse it (unbalanced quotes?)")

    verdicts: list[RiskVerdict] = []
    segments: list[tuple[str, list[str]]] = []  # (operator before it, tokens)
    current: list[str] = []
    op_before = ""
    for tok in tokens:
        if tok in SEPARATORS:
            if current:
                segments.append((op_before, current))
            current, op_before = [], tok
            if tok == "&":
                verdicts.append(_v(Risk.DANGER, "runs in the background, outside the timeout"))
            continue
        current.append(tok)
    if current:
        segments.append((op_before, current))

    for idx, (op, seg) in enumerate(segments):
        verdicts.append(_segment(seg))
        if idx and op in {"|", "|&"} and _effective_program(seg) in INTERPRETERS:
            if _effective_program(segments[idx - 1][1]) in {"curl", "wget"}:
                verdicts.append(_v(Risk.BLOCKED, "pipes a download straight into a shell"))

    return _worst(verdicts) if verdicts else _v(Risk.DANGER, "nothing recognisable to run")
