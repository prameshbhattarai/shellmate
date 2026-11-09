# shellmate

[![CI](https://github.com/prameshbhattarai/shellmate/actions/workflows/ci.yml/badge.svg)](https://github.com/prameshbhattarai/shellmate/actions/workflows/ci.yml)

**Talk to your terminal.** Say what you want in plain English. shellmate works out the shell
commands, labels each one by risk, runs it after you approve, reads the output, and keeps going until
it can answer you. It runs on a local model (Qwen via Ollama), so nothing leaves your machine.

<!-- Record with ./scripts/demo.sh, then uncomment:
![shellmate demo](docs/demo.gif)
-->

```text
you> which folders here are eating my disk?
╭─ command ────────────────────────╮
│ du -sh * | sort -rh | head -n 10 │
╰──────────────────────────────────╯
risk  SAFE  read-only
3.0M    node_modules
8.0K    src
mate> node_modules is the biggest at 3.0M; everything else is tiny.

you> delete the node_modules folder
╭─ command ───────────╮
│ rm -rf node_modules │
╰─────────────────────╯
risk  DANGER  deletes files permanently (no trash)
Dangerous. Type yes to run it, e to edit, Enter to skip: yes
done (no output)
mate> Deleted node_modules. `npm install` brings it back.

you> delete everything in my home folder
╭─ command ──╮
│ rm -rf ~/* │
╰────────────╯
risk  BLOCKED  rm on ~/* would wreck your system or home folder
shellmate will not run this. Asking the model for another way.
mate> I won't do that: it would wipe your home folder.
```
*(An example session: the layout is exactly what shellmate prints; the model's wording varies.)*

## What it does

- **An agent loop, not a one-shot.** The model proposes one step at a time, reads the output, and
  decides the next step or answers. It's built on [LangGraph](https://langchain-ai.github.io/langgraph/)
  with a step cap, and repeats are caught in code.
- **Risk is decided by rules, never by the model.** Every command is labelled before it runs, edits
  included:

  | Level | Examples | What happens |
  |---|---|---|
  | SAFE | `ls`, `du`, `git log`, `grep` | runs straight away |
  | CAUTION | `mkdir`, `git add`, `>> file`, `pip install` | asks `y/N` |
  | DANGER | `rm`, `mv`, `git push`, `> file`, `sudo`, unknown programs | needs the full word `yes` |
  | BLOCKED | `rm -rf ~`, `curl … \| sh`, `dd` to a disk, `cd ~ && rm -rf *` | never runs |

  The rules look through `sudo -u`, `xargs`, `env -S` and `su -c`, follow `cd`, and resolve paths
  against the folder you're in.
- **Read-only tools** let the model read a file, list a folder, or look at your command history or
  last error without running anything. They only see the current folder.
- **Human in the loop.** Approvals use LangGraph's `interrupt()`. Press Ctrl-C at a prompt to
  pause, and `shellmate resume` continues later from a SQLite checkpoint, even after a restart.
- **`shellmate explain`**: pipe an error in (`make 2>&1 | shellmate explain`) or just run it after a
  failure, and it investigates and suggests a fix.
- **`shellmate undo`** reverses the last change it made here (`mkdir` → `rmdir`, `git commit` →
  `git reset --soft HEAD~1` …). It only acts when the inverse is known, and asks first.
- **Memory**: `shellmate remember "my projects live in ~/code"` is added to every prompt.
- **Personas and teach mode**: `--persona terse|mentor|pirate`, and `--teach` adds a line
  explaining each command so you learn it.
- **Ctrl-G from your prompt**: `eval "$(shellmate init zsh)"`, then type a request and press
  Ctrl-G instead of Enter.
- **Evals**: 55 cases scored per model, and checked for consistency in CI.

## Quick start

```bash
# 1. A local model (one-off)
brew install ollama && ollama pull qwen2.5:3b     # or qwen2.5:7b: slower, noticeably smarter

# 2. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 3. Run
shellmate                                   # interactive
shellmate ask "what was my last commit message?"
```

More in [Examples](#examples).

## Examples

### Ask a question

Read-only commands run straight away, and you get a plain-English answer at the end:

```bash
shellmate ask "what was my last git commit message?"
```
```text
╭─ command ──────────────╮
│ git log -1 --format=%B │
╰────────────────────────╯
risk  SAFE  git log only reads
agent skeleton.
mate> Your last commit was "agent skeleton."
```

Some more to try:

```bash
shellmate ask "which folders here are eating my disk?"
shellmate ask "how many python files are in this project?"
shellmate ask "what is listening on port 3000?"
shellmate ask "show me what's in README.md"        # uses read_file, no command needed
```

### Make a change, with your approval

Anything that changes something asks first. A CAUTION command takes `y`, and a DANGER one needs
the full word `yes`:

```bash
shellmate ask "create a folder called notes"        # CAUTION: Run it? [y/N/e(dit)]
shellmate ask "delete the node_modules folder"      # DANGER: Type yes to run it ...
```

At the prompt, `e` lets you edit the command first. The edited command is checked again, so you
can't edit your way past the rules. Enter, or `n`, skips it.

### See the plan without running anything

```bash
shellmate --dry-run ask "remove all .pyc files"
```

This shows the first command and its risk level, and stops there.

### Have a conversation

```bash
shellmate
```
```text
you> what branch am I on?
...
mate> You're on main.
you> show my last 5 commits
...
you> exit
```

### Explain an error

```bash
npm run build 2>&1 | shellmate explain              # pipe the error in
shellmate explain "fatal: not a git repository"     # or paste it
shellmate explain                                    # or use the last command shellmate saw fail
```

Before it explains, shellmate may run a few commands to investigate, such as `git status` or `ls`.
They go through the usual risk checks.

### Pause now, decide later

Press **Ctrl-C** at any approval prompt. Nothing runs, and the run is saved:

```text
Dangerous. Type yes to run it, e to edit, Enter to skip: ^C
paused - nothing ran. `shellmate resume` picks it up, even after a restart.
```
```bash
shellmate resume                                     # shows the same question again
```

### Undo the last change

```bash
shellmate ask "create a folder called scratch"       # answer y
shellmate undo                                       # proposes `rmdir scratch`, asks, then runs it
```

Undo only covers changes it understands (`mkdir`, `git add`, `git commit`, `git switch`,
`git stash`, `git branch`, `mv`, `ln -s`). For anything else it tells you it can't, instead of
guessing. For git changes it also checks that you haven't committed, switched branch or stashed by
hand since; if you have, it refuses rather than undo the wrong thing.

### Teach it about you

```bash
shellmate remember "my projects live in ~/Documents/workspace"
shellmate remember "I use pnpm, not npm"
shellmate profile                                    # list the notes
shellmate forget 2                                   # forget note 2 (no number: forget all)
```

Notes are added to every prompt, so a request like "how big are my projects?" knows where to look.

### Change the style, or learn as you go

```bash
shellmate --persona terse ask "how much free disk space do I have?"
shellmate --persona pirate ask "what branch am I on?"
shellmate --teach ask "find files bigger than 100MB here"   # adds a line explaining find -size
```

Set `SHELLMATE_PERSONA` in `.env` to make a persona the default.

### Use a bigger model for a harder request

```bash
shellmate --model qwen2.5:7b ask "why is my git push being rejected?"
```

### Press Ctrl-G from your normal prompt

```bash
echo 'eval "$(shellmate init zsh)"' >> ~/.zshrc && source ~/.zshrc
```

Now type a request at the zsh prompt, for example `what is using port 8080`, and press **Ctrl-G**
instead of Enter.

### Look back

```bash
shellmate history                 # the last 10 commands shellmate ran
shellmate history -n 30 --here    # the last 30, only in this folder
```

## Commands

| Command | Does |
|---|---|
| `shellmate` | interactive session |
| `shellmate ask "..."` | one request, then exit with the outcome's exit code |
| `shellmate explain [TEXT]` | explain an error: an argument, piped input, or your last failed command |
| `shellmate resume` | continue the run you paused with Ctrl-C at a prompt |
| `shellmate undo` | reverse the last change shellmate made in this folder |
| `shellmate history [-n 20] [--here]` | commands shellmate ran |
| `shellmate remember "..."` / `forget [N]` / `profile` | notes added to every prompt |
| `shellmate init zsh` | prints the Ctrl-G key binding |

Global options go before the command: `--model qwen2.5:7b`, `--dry-run` (show the first command, run
nothing), `--confirm-all` (ask even for SAFE), `--verbose` (full output), `--persona`, `--teach`.

## How it works

```
decide ─┬─ tool (read_file, list_dir, history, last_error) ──────────────────────┐
        ├─ run → classify_risk ─┬─ SAFE ──────────────→ run ──────────────────────┤→ decide …
        │                       ├─ CAUTION / DANGER ──→ ask_human (interrupt) → run ┘
        │                       └─ BLOCKED ───────────→ refuse ──────────────────→ decide …
        └─ answer → END
```

**Full walkthrough with flowcharts: [docs/how-it-works.md](docs/how-it-works.md)**: the graph node
by node, what the model sees, the approval pause, the risk rules, memory, undo and evals.

## Evals

```bash
python evals/run_evals.py --check                              # no model: is the eval set consistent?
python evals/run_evals.py --model qwen2.5:3b,qwen2.5:7b        # score models (one call per case)
```

Scoring asks each model for the first step of every case and never runs anything. It checks the
action, matches the command against a regex, and confirms the rules give the expected risk level.

| Model | Pass | Right action | Command matches | Risk label |
|---|---|---|---|---|
| qwen2.5:3b | not run yet | | | |
| qwen2.5:7b | not run yet | | | |

## Configuration

Settings come from `.env` (see [.env.example](.env.example)): the model and Ollama URL, the context
size, the step cap, the command timeout, the output cap, the persona, and `SHELLMATE_HOME` (default
`~/.shellmate`, where history, notes and checkpoints live).

## Project layout

```
shellmate/
  cli.py         # typer commands, REPL, rich rendering, answers the graph's approval pauses
  agent.py       # system prompt, Decision schema, personas, how outputs are fed back, repeat guards
  graph.py       # LangGraph state and nodes: decide, use_tool, classify_risk, ask_human, run, refuse
  risk.py        # rule-based risk classifier: SAFE / CAUTION / DANGER / BLOCKED
  tools.py       # run_command, read_file, list_dir
  memory.py      # history, remembered notes, SQLite checkpoints, the paused run
  undo.py        # inverse commands for `shellmate undo`
  shell_init.py  # zsh Ctrl-G binding
  config.py      # settings from env; builds the chat model lazily
docs/            # how-it-works.md (flowcharts); demo.gif once recorded
evals/           # 55 cases and the scorer
scripts/demo.sh  # records the demo GIF
tests/           # 261 unit tests with a scripted fake model; never call Ollama
```

## Roadmap

- [x] One-shot: request → command → confirm → run → print
- [x] Risk classifier with unit tests
- [x] LangGraph loop with `interrupt()` approvals
- [x] Tools for files, history and "explain my last error"
- [x] Memory across sessions: SQLite checkpoints, resume, notes
- [x] Eval set (55 cases) and scorer; consistency check in CI
- [x] Extras: personas, teach mode, `shellmate undo`, zsh key binding
- [ ] Record the demo GIF (`./scripts/demo.sh`) and fill in the eval table

## License

MIT
