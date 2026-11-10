"""Risk classification for shell commands. Rules only: the model never decides what is safe.

(Plain Python, no LangChain: optional reading. The idea that matters for agents: put a hard,
testable check between "the model chose an action" and "the action runs". graph.classify_risk is
where this file plugs into the loop.)

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
    "rmdir": "removes empty folders",
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

# Options of those prefixes that take a value, so the value is not mistaken for the program:
# `sudo -u root rm -rf /` runs rm, not a program called "root".
SHORT_OPTIONS_WITH_VALUE = {
    "sudo": set("ugChpDrtTU"),
    "doas": set("uC"),
    "su": set("lsc"),
    "nice": set("n"),
    "env": set("uCS"),
    "xargs": set("IJLnPsEdaR"),
    "caffeinate": set("tw"),
}
LONG_OPTIONS_WITH_VALUE = {
    "sudo": {"--user", "--group", "--host", "--prompt", "--chdir", "--close-from", "--role", "--type",
             "--other-user", "--command-timeout"},
    "env": {"--unset", "--chdir", "--split-string"},
    "nice": {"--adjustment"},
    "xargs": {"--max-args", "--replace", "--max-procs", "--delimiter", "--arg-file", "--eof", "--max-lines",
              "--max-chars"},
}
# Options whose value is itself a command line: `su -c "rm -rf ~"`, `env -S "rm -rf ~"`.
COMMAND_STRING_OPTIONS = {"su": {"-c", "--command"}, "env": {"-S", "--split-string"}}

SEPARATORS = frozenset({"|", "||", "&&", ";", "&", "|&", "(", ")", ";;"})
OVERWRITE_REDIRECTS = frozenset({">", ">|", "&>"})
APPEND_REDIRECTS = frozenset({">>", "&>>"})
INPUT_REDIRECTS = frozenset({"<", "<<", "<<<"})

FORK_BOMB = re.compile(r":\s*\(\s*\)\s*\{")
RAW_DEVICE = re.compile(r"/dev/(r?disk\d|sd[a-z]|nvme\d|hd[a-z])")


# --- paths ----------------------------------------------------------------------------------------

def _resolve(target: str, base: str | None) -> str | None:
    """``target`` as an absolute, normalised path, or None when it can't be known (a variable, an
    unknown working folder). Expands ~ and $HOME; a trailing /* or a bare * means the folder itself."""
    t = target.strip()
    if not t:
        return None
    home = os.path.expanduser("~")
    for prefix in ("${HOME}", "$HOME"):
        if t.startswith(prefix):
            t = home + t[len(prefix):]
    t = os.path.expanduser(t)
    while t.endswith("/*") or (t.endswith("/") and t != "/"):
        t = (t[:-2] or "/") if t.endswith("/*") else t[:-1]  # "/*" is the root, not ""
    if t in {"", "*"}:
        t = "."
    if "$" in t:
        return None
    if not t.startswith("/"):
        if base is None:
            return None
        t = os.path.join(base, t)
    return os.path.normpath(t)


# Regenerable folders that are fine to delete even when they sit directly in the home folder.
DISPOSABLE_FOLDERS = frozenset(
    "node_modules __pycache__ .cache .npm .pytest_cache .mypy_cache .venv venv build dist target tmp .tmp".split()
)


def _is_precious_path(path: str) -> bool:
    """/, a top-level folder, the home folder, an ancestor of it, or a folder directly in it."""
    p = PurePosixPath(path)
    home = PurePosixPath(os.path.expanduser("~"))
    if len(p.parts) <= 2:  # "/" or a top-level folder like /Users, /usr, /System
        return True
    if p == home or home.is_relative_to(p):
        return True
    # a folder directly in home (Documents, Desktop, .ssh ...), unless it is a known throwaway one
    return p.parent == home and p.name not in DISPOSABLE_FOLDERS


def _is_precious(target: str, base: str | None) -> bool:
    """True when wiping ``target`` would wreck the machine or the user's files. ``.`` and ``..``
    always count (deleting the folder you are in, or above it); other relative paths are resolved
    against ``base``, the folder the command runs in, so `rm -rf *` from ~ counts too."""
    t = target.strip().rstrip("/") or target.strip()
    if t in {".", "..", "./*", "../*"} or t.startswith("../"):
        return True
    resolved = _resolve(target, base)
    return resolved is not None and _is_precious_path(resolved)


def _is_precious_tree(target: str, base: str | None) -> bool:
    """Like :func:`_is_precious`, but `.` is just the working folder (for find, mv ...)."""
    resolved = _resolve(target, base)
    return resolved is not None and _is_precious_path(resolved)


def _names_a_path(arg: str, base: str | None) -> bool:
    """True when ``arg`` is an existing file or folder (relative to ``base``), not e.g. a branch name.
    Unknown base: anything that looks like a path counts, to stay on the safe side."""
    if base is None:
        return "/" in arg or "." in arg
    resolved = _resolve(arg, base)
    return resolved is not None and os.path.lexists(resolved)


def _strip_prefix(name: str, args: list[str]) -> list[str]:
    """Drop a prefix's own options (and their values) and return the command it runs."""
    short = SHORT_OPTIONS_WITH_VALUE.get(name, set())
    long = LONG_OPTIONS_WITH_VALUE.get(name, set())
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            i += 1
            break
        if name in {"env", "sudo"} and re.match(r"^[A-Za-z_]\w*=", a):
            i += 1  # env FOO=1 cmd
            continue
        if a.startswith("--"):
            i += 2 if (a.split("=", 1)[0] in long and "=" not in a) else 1
            continue
        if a.startswith("-") and len(a) > 1:
            consumed = 1
            for j, ch in enumerate(a[1:]):
                if ch in short:
                    consumed = 2 if j == len(a) - 2 else 1  # value is the next token, or attached
                    break
            i += consumed
            continue
        break
    return args[i:]


def _command_string(name: str, args: list[str]) -> str | None:
    """The command line passed as a string to `su -c` / `env -S`, if any."""
    opts = COMMAND_STRING_OPTIONS.get(name, set())
    for i, a in enumerate(args):
        key, _, attached = a.partition("=")
        if key in opts:
            return attached or (args[i + 1] if i + 1 < len(args) else "")
        if a.startswith("-") and not a.startswith("--") and len(a) > 2 and a[:2] in opts:
            return a[2:]
    return None


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


def _git(args: list[str], base: str | None = None) -> RiskVerdict:
    i = 0  # skip global options like -C dir or --no-pager
    while i < len(args) and args[i].startswith("-"):
        if args[i] == "-c" or args[i].startswith(("--config-env", "--exec-path")):
            # `git -c core.pager='rm -rf ~' log` runs any command it is given
            return _v(Risk.DANGER, "git -c can make git run arbitrary commands")
        i += 2 if args[i] == "-C" else 1
    if i >= len(args):
        return _v(Risk.SAFE, "git with no subcommand")
    sub, rest = args[i], args[i + 1:]
    if any(a == "--output" or a.startswith("--output=") for a in rest):
        return _v(Risk.DANGER, f"git {sub} --output overwrites a file")
    if sub == "reflog" and rest[:1] in (["expire"], ["delete"]):
        return _v(Risk.DANGER, "throws away the reflog, the usual way to recover lost commits")
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
        files = [t for t in _targets(rest) if _names_a_path(t, base)]
        if files:  # `git checkout README.md` restores the file, throwing away its changes
            return _v(Risk.DANGER, f"discards uncommitted changes to {files[0]}")
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


SED_WRITE_OR_EXEC = re.compile(r"(^|[;}/\s])[wWe](\s|$)")


def _short_flags(args: list[str]) -> str:
    return "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--"))


def _writes_despite_being_read_only(name: str, args: list[str]) -> str | None:
    """Read-only programs that still write files with some flags. Returns why, or None."""
    if name == "sort" and ("o" in _short_flags(args) or any(a.split("=")[0] == "--output" for a in args)):
        return "sort -o overwrites a file"
    if name == "tree" and "o" in _short_flags(args):
        return "tree -o writes a file"
    if name == "say" and ("o" in _short_flags(args) or any(a.split("=")[0] == "--output-file" for a in args)):
        return "say -o writes an audio file"
    if name == "file" and "C" in _short_flags(args):
        return "file -C writes a magic file"
    if name == "uniq":
        positional, skip = [], False
        for a in args:
            if skip:
                skip = False
            elif a in {"-f", "-s"}:
                skip = True  # these take a number
            elif not a.startswith("-") or a == "-":
                positional.append(a)
        if len(positional) >= 2:
            return f"uniq writes its output to {positional[1]}"
    return None


def _program(prog: str, args: list[str], base: str | None) -> RiskVerdict:
    """Judge one program and its arguments. ``base`` is the folder it runs in (None if unknown)."""
    name = os.path.basename(prog)

    if name in PRIVILEGE or name in WRAPPERS:
        if name == "command" and args[:1] in (["-v"], ["-V"]):
            return _v(Risk.SAFE, "prints where a command is")
        script = _command_string(name, args)
        if script is not None:
            inner = classify(script, cwd=base)
        else:
            rest = _strip_prefix(name, args)
            if not rest:
                if name in PRIVILEGE:
                    return _v(Risk.DANGER, "opens a root shell")
                return _v(Risk.SAFE, f"{name} only prints")
            inner = _segment(rest, base)
        if inner.risk is Risk.BLOCKED:
            return inner
        if name in PRIVILEGE:
            return _v(Risk.DANGER, f"runs as administrator ({inner.reason})")
        if script is not None:
            return _worst([inner, _v(Risk.DANGER, f"{name} runs a command given as a string")])
        return inner

    if name in {"rm", "rmdir", "chmod", "chown", "chgrp"}:
        precious = [t for t in _targets(args) if _is_precious(t, base)]
        if precious and (name == "rmdir" or _is_recursive(args)):
            return _v(Risk.BLOCKED, f"{name} on {precious[0]} would wreck your system or home folder")
        if name == "rmdir":
            return _v(Risk.CAUTION, "removes folders, only if they are empty")
        return _v(Risk.DANGER, DANGER_PROGRAMS[name])

    if name == "mv":
        sources = _targets(args)[:-1]
        precious = [t for t in sources if _is_precious(t, base)]
        if precious:
            return _v(Risk.BLOCKED, f"moving {precious[0]} would break your system or home folder")
        return _v(Risk.DANGER, DANGER_PROGRAMS["mv"])

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
        return _git(args, base)
    if name in {"brew", "pip", "pip3", "npm", "pnpm", "yarn", "uv", "cargo", "gem"}:
        return _package_manager(name, args)

    if name == "find":
        destructive = "-delete" in args or any(a in {"-exec", "-execdir", "-ok", "-okdir"} for a in args)
        if destructive:
            starts = []
            for a in args:  # start paths come before the first expression token
                if a.startswith(("-", "(", "!")):
                    break
                starts.append(a)
            precious = [t for t in (starts or ["."]) if _is_precious_tree(t, base)]
            if precious:
                return _v(Risk.BLOCKED, f"find over {precious[0]} could wipe your system or home folder")
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
        if any(SED_WRITE_OR_EXEC.search(a) for a in args if not a.startswith("-")):
            return _v(Risk.DANGER, "sed script writes a file (w) or runs a command (e)")
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

    writer = _writes_despite_being_read_only(name, args)
    if writer:
        return _v(Risk.DANGER, writer)

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


def _segment(tokens: list[str], base: str | None) -> RiskVerdict:
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
        verdicts.append(_program(args[0], args[1:], base))
    return _worst(verdicts) if verdicts else _v(Risk.SAFE, "nothing to run")


def _effective_program(tokens: list[str]) -> str:
    """The program a segment really runs, looking past sudo, env, xargs, FOO=bar and their flags."""
    rest = list(tokens)
    while rest:
        name = os.path.basename(rest[0])
        if name in PRIVILEGE or name in WRAPPERS:
            rest = _strip_prefix(name, rest[1:])
            continue
        if re.match(r"^[A-Za-z_]\w*=", rest[0]):
            rest.pop(0)
            continue
        return name
    return ""


def _cd_target(seg: list[str], base: str | None) -> str | None:
    """The folder a `cd` segment moves to (None when it can't be known)."""
    args = [a for a in seg[1:] if a not in {"-L", "-P", "--"}]
    if not args:
        return os.path.expanduser("~")
    if args[0] == "-":
        return None
    return _resolve(args[0], base)


def _tokenize(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def classify(command: str, cwd: str | None = None) -> RiskVerdict:
    """Label ``command``. ``cwd`` is the folder it will run in (default: the current one); relative
    paths, and any `cd` inside the command, are resolved against it."""
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

    base: str | None = os.path.abspath(cwd or os.getcwd())
    for idx, (op, seg) in enumerate(segments):
        verdicts.append(_segment(seg, base))
        if _effective_program(seg) == "cd" and os.path.basename(seg[0]) == "cd":
            base = _cd_target(seg, base)  # later segments run in the new folder
        if idx and op in {"|", "|&"} and _effective_program(seg) in INTERPRETERS:
            if _effective_program(segments[idx - 1][1]) in {"curl", "wget"}:
                verdicts.append(_v(Risk.BLOCKED, "pipes a download straight into a shell"))

    return _worst(verdicts) if verdicts else _v(Risk.DANGER, "nothing recognisable to run")
