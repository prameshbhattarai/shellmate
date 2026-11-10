# How shellmate works

shellmate turns a plain-English request into shell commands, runs them one at a time and answers
you from their output. Three rules hold everywhere:

- **The model only proposes.** It can't run anything itself; it returns a decision, and the graph
  acts on it.
- **Safety is decided by rules in code, never by the model.** Every command, including one you
  edited, is classified before it runs.
- **You stay in control.** Anything that changes your system waits for your answer.

The diagrams are Mermaid, so they render on GitHub and in most Markdown previews.

---

## 1. The big picture

```mermaid
flowchart LR
    U([You]) -->|request| CLI["CLI<br/>cli.py"]
    CLI -->|streams| G["Agent graph<br/>graph.py"]
    G -->|messages| M["Model: Qwen via Ollama<br/>agent.py builds the prompt"]
    M -->|"Decision: run or answer"| G
    G -->|command| R["Risk rules<br/>risk.py"]
    R -->|"SAFE / CAUTION / DANGER / BLOCKED"| G
    G -->|"pauses: approve?"| CLI
    CLI -->|"y / yes / edit / no"| G
    G -->|approved command| T["run_command<br/>tools.py"]
    T -->|"exit code, stdout, stderr"| G
    G -->|answer| CLI
    CLI -->|"mate> ..."| U
```

| Module | Job |
|---|---|
| `cli.py` | The front end: reads your request, streams the graph, draws each step, asks for approval, prints the answer |
| `graph.py` | The agent loop as a LangGraph state graph: decide, check risk, ask, run, repeat |
| `agent.py` | Everything the model sees and returns: system prompt, `Decision` schema, how outputs are fed back, repeat guards |
| `risk.py` | Pure-Python rules that label a command SAFE, CAUTION, DANGER or BLOCKED |
| `tools.py` | `run_command` (timeout, own process group, no keyboard input, capped output) and the read-only `read_file` / `list_dir`, confined to the working folder |
| `memory.py` | The SQLite checkpointer and the paused-run file, for `shellmate resume` |
| `replay.py` | Scripted model answers for the demo recording (no model) |
| `config.py` | Settings from `.env` and builds the Ollama chat model lazily (only on the first request) |

---

## 2. One request, end to end

```mermaid
sequenceDiagram
    actor You
    participant CLI as CLI (cli.py)
    participant Graph as Graph (graph.py)
    participant Model as Qwen (Ollama)
    participant Risk as Risk rules
    participant Shell as run_command

    You->>CLI: what was my last git commit message?
    CLI->>Graph: stream({request, steps: []})
    Graph->>Model: system prompt + request
    Model-->>Graph: run "git log -1 --format=%B"
    Graph->>Risk: classify(command)
    Risk-->>Graph: SAFE (git log only reads)
    Graph->>Shell: run (SAFE runs without asking)
    Shell-->>Graph: exit 0, "agent skeleton."
    Graph-->>CLI: command panel + output preview
    Graph->>Model: request + "run: git log..." + output + "answer if this is enough"
    Model-->>Graph: answer "Your last commit was 'agent skeleton.'"
    Graph-->>CLI: status answered
    CLI->>You: mate> Your last commit was 'agent skeleton.'
```

A riskier command takes the same path, except the graph **pauses** before running it and the CLI
asks you (section 5).

---

## 3. The agent graph

This is the LangGraph state graph built by `build_graph()` in `graph.py`. Each box is a node, and
the labels on the arrows are the routing rules.

```mermaid
flowchart TD
    START([START]) --> decide

    decide{{"decide<br/>ask the model for the next step"}}
    decide -->|"answer, or no command"| END_A([END: answered])
    decide -->|"step limit reached"| END_L([END: step_limit])
    decide -->|"model unreachable / bad output"| END_E([END: model_error])
    decide -->|"run a command"| classify
    decide -->|"read_file / list_dir"| tool

    tool["use_tool<br/>read-only, inside the working folder"]
    tool --> decide

    classify["classify_risk<br/>rules label the command"]
    classify -->|"--dry-run"| END_D([END: dry_run])
    classify -->|BLOCKED| refuse
    classify -->|"SAFE and not --confirm-all"| run
    classify -->|"CAUTION / DANGER, or --confirm-all"| ask

    ask[/"ask_human<br/>interrupt(): pause for you"/]
    ask -->|"y, or yes for DANGER"| run
    ask -->|"edit"| classify
    ask -->|"no / Enter"| END_N([END: declined])

    run["run<br/>run_command, record the step"]
    run --> decide

    refuse["refuse<br/>record 'shellmate refused'"]
    refuse --> decide
```

### What each node does

| Node | Reads | Writes | Notes |
|---|---|---|---|
| `decide` | `request`, `steps` | `command` and `explanation`, or `answer` and `status` | Calls the model. Also checks the step limit and the repeat guards (section 4) |
| `use_tool` | `tool`, `path` | appends a step: `{command: "read_file README.md", outcome: "tool", ...}` | No risk check or approval: the tools only read, and refuse paths outside the working folder (symlinks included) |
| `classify_risk` | `command` | `risk`, `reason` (and `status: dry_run` in a dry run) | Rules only, no model call |
| `ask_human` | `command`, `risk`, `reason`, `explanation` | `reply` (`run` / `edit` / `skip`), maybe a new `command` | Calls LangGraph's `interrupt()`. The graph pauses here until the CLI resumes it |
| `run` | `command`, `risk` | appends a step: `{command, outcome: "ran", risk, exit_code, stdout, stderr}` | Output is cut at 4000 chars and decoded as UTF-8 (bad bytes replaced). 30 s timeout, after which the whole process group is killed |
| `refuse` | `command`, `reason` | appends a step: `{command, outcome: "blocked", note}` | The model is told it was refused, so it can try something else |

### The loop's safety limits

- **Step cap.** `decide` stops after `SHELLMATE_MAX_STEPS` (default 8) finished steps. Refused
  commands count as steps.
- **Edits are checked again.** An edit routes back to `classify_risk`, never straight to `run`, so
  editing `ls` into `rm -rf ~` gets refused.
- **No means stop.** Declining ends the run, and the model is not asked again.
- **Recursion limit.** The graph is compiled with a LangGraph recursion limit of `10 x max_steps + 10`
  as a backstop.

### State

The state holds only plain JSON values, so any LangGraph checkpointer can save it. The CLI uses
SQLite (`~/.shellmate/checkpoints.sqlite`); the tests use the in-memory one.

```text
request       your words                      dry_run / confirm_all   flags from the CLI
steps[]       finished steps (appended)       command / explanation   the step being worked on
tool / path   set when the next step is a tool
risk / reason the verdict for `command`       reply                   your answer in ask_human
answer        the final reply                 status                  answered | declined | dry_run | step_limit | model_error
error         set when status is model_error
```

---

## 4. Inside `decide`: talking to the model

```mermaid
flowchart TD
    A["steps >= max_steps?"] -->|yes| L([status step_limit])
    A -->|no| B["build_messages()<br/>system prompt, request, then for each step:<br/>ai 'run: cmd' + human 'output'"]
    B --> C["model.invoke() returns a Decision<br/>action: run | answer, command, explanation, answer"]
    C --> P{"problem?"}
    P -->|"repeats a command with nothing changed since,<br/>or one shellmate refused"| N1["retry once with a nudge:<br/>'you already ran X' / 'X was refused'"]
    P -->|"answer is just a command it ran"| N2["retry once with a nudge:<br/>'that was a command, not an answer'"]
    P -->|no| OK
    N1 --> C2["model.invoke() again"]
    N2 --> C2
    C2 --> P2{"still a problem?"}
    P2 -->|yes| FB(["fallback answer:<br/>'Here is what X printed: ...'"])
    P2 -->|no| OK
    OK["clean_command()<br/>strip $( ), backticks, fences, '$ '"] --> D{"action"}
    D -->|"answer, or empty command"| ANS([status answered])
    D -->|run| CMD(["command, explanation -> classify_risk"])
```

What the model is told:

- **The system prompt** (`agent.py`, `SYSTEM_PROMPT`): your OS and shell, and that commands run in
  your current folder. It must reply with either run ONE command or answer. Never repeat a command.
  Use relative paths, the simplest tool, read-only where possible. macOS uses BSD flags. It also
  gets three example request/command pairs.
- **Every finished step** goes back to the model as its exit code, stdout and stderr. The newest
  two outputs are sent in full and older ones are cut to 300 chars, so a long session fits a 4096
  token context.
- **The newest output** ends with a reminder: "if this answers the request, reply with action
  answer; otherwise run a different command." Small models need that reminder right where they
  look.
- **Structured output.** The model must return JSON matching `Decision`, via
  `with_structured_output(..., method="json_schema")`, so there's no free text to parse.

---

## 5. Approval: how the pause works

`ask_human` calls `interrupt()`. LangGraph saves the state, stops the stream, and hands the CLI a
request: `{command, risk, reason, explanation}`. The CLI asks you, then resumes the same thread with
`Command(resume=reply)`.

```mermaid
sequenceDiagram
    participant Graph
    participant CLI
    actor You
    Graph->>CLI: __interrupt__ {command, risk: "danger", reason}
    CLI->>You: Dangerous. Type yes to run it, e to edit, Enter to skip
    You->>CLI: y
    CLI->>You: for a dangerous command, type the whole word: yes
    You->>CLI: yes
    CLI->>Graph: Command(resume={action: "run"})
    Graph->>Graph: ask_human returns reply=run, then run
```

What each answer does, by risk level:

| Risk | Asked? | What runs it | Edit | Enter / n |
|---|---|---|---|---|
| SAFE | only with `--confirm-all` | `y` | re-checked | stop |
| CAUTION | yes | `y` | re-checked | stop |
| DANGER | yes | the full word `yes` (a plain `y` is rejected) | re-checked | stop |
| BLOCKED | never | nothing: it is refused and the model is told | n/a | n/a |

At any prompt, **Ctrl-C pauses** the run: nothing runs, the checkpoint stays in SQLite, and
`shellmate resume` shows the same question again, even after a restart (section 9). **Ctrl-D**, or
no terminal at all (piped input), counts as no.

The graph never prompts anyone itself; it only pauses and resumes. A different front end (a TUI,
a web page, a test) can drive it the same way. The tests do exactly that in `tests/test_graph.py`.

---

## 6. The risk rules

```mermaid
flowchart TD
    C[command] --> F{"fork bomb?"}
    F -->|yes| BL([BLOCKED])
    F -->|no| S{"several lines, $( ), backticks, <( )?"}
    S -->|yes| DG([DANGER: can't check it])
    S -->|no| T["tokenize with shlex<br/>unbalanced quotes -> DANGER"]
    T --> SP["split on | && || ; &<br/>a trailing & -> DANGER (background)"]
    SP --> SEG["for each segment"]
    SEG --> RD["redirections:<br/>overwrite a file -> DANGER, append -> CAUTION,<br/>write to /dev/disk -> BLOCKED, to /dev/null -> ignored"]
    RD --> PR["program rules:<br/>sudo / env / nice / xargs / su -c -> skip their options, judge what they run<br/>rm, chmod, rmdir, mv, find -delete on / ~ ~/X .. -> BLOCKED<br/>relative paths and cd resolved against the working folder<br/>git, find, sed, awk, curl, brew, pip... -> by subcommand and flags<br/>known read-only -> SAFE, known writers -> CAUTION / DANGER<br/>unknown program -> DANGER"]
    PR --> PIPE{"curl / wget piped into a shell?"}
    PIPE -->|yes| BL
    PIPE -->|no| W(["verdict = the worst segment"])
```

Examples (all are in `tests/test_risk.py`):

| Command | Verdict | Why |
|---|---|---|
| `du -sh * \| sort -rh \| head` | SAFE | every segment only reads |
| `git log -1 --format=%B` | SAFE | git log only reads |
| `mkdir build` | CAUTION | creates a folder |
| `echo hi >> notes.txt` | CAUTION | appends |
| `rm -rf node_modules` | DANGER | deletes permanently, but only inside the project |
| `git push --force` | DANGER | can overwrite others' work |
| `ls && rm -rf ~` | BLOCKED | the worst segment wins |
| `curl -fsSL x.sh \| sudo bash` | BLOCKED | runs a download as a script |

Paths are resolved against the folder shellmate runs in, and any `cd` earlier in the same command
is followed. So `rm -rf *` started in your home folder, or `cd ~ && rm -rf *`, is BLOCKED, while the
same `rm -rf *` inside a project is only DANGER. Throwaway folders such as `node_modules`, `.cache`
or `build` stay DANGER even directly in home.

A repeated command is only treated as a repeat when nothing that could change its output (any step
that wasn't SAFE) ran in between, so `ls`, `mkdir x`, `ls` works.

The rules are a guardrail, not a sandbox. Once you type `yes`, the command runs with your full
permissions, and `rm` skips the Trash.

---

## 7. How it ends

| Status | When | You see | `shellmate ask` exit code |
|---|---|---|---|
| `answered` | the model answered, or the fallback after repeats | `mate> ...` | 0 |
| `declined` | you said no | `skipped - stopping here` | 0 |
| `dry_run` | `--dry-run` | the first command and its risk | 0 |
| `step_limit` | 8 steps without an answer | `I stopped after 8 steps...` | 1 |
| `model_error` | Ollama not running, model missing, bad JSON | the error and an `ollama serve` hint | 1 |
| paused | Ctrl-C at an approval prompt | `paused - nothing ran` | 130 |
| interrupted | Ctrl-C while the model thinks or a command runs (the command is killed) | `interrupted` | 130 |

---

## 8. Testing without a model

No test calls Ollama. `tests/fakes.py` has a `ScriptedDecider` that returns a fixed list of
decisions, so a test can script "run X, then run X again, then answer" and check that the graph
nudges, falls back, refuses or pauses as described above. Running commands in tests is real but
harmless: `echo`, `mkdir` and `rm` on temp files.

| Test file | Covers |
|---|---|
| `test_risk.py` | 100+ commands across the four levels, plus paths resolved against a fake home folder |
| `test_graph.py` | the loop: auto-run, approvals, edits re-checked, refusals, repeats, step cap, dry run |
| `test_cli.py` | prompts, `yes` vs `y`, output preview, `--verbose`, exit codes, the REPL |
| `test_agent.py` | prompt building, cutting old outputs, `clean_command` |
| `test_tools.py` | timeout kills the pipeline, binary output, truncation, `read_file` / `list_dir` confinement |
| `test_memory.py` | the paused-run file, private permissions, damaged files |
| `test_replay.py` | replay mode for the demo |
| `test_evals.py` | the eval set is consistent, and the scorer with a scripted model |

Every test gets its own `SHELLMATE_HOME` (see `tests/conftest.py`), so none touches `~/.shellmate`.

---

## 9. Memory: checkpoints and resume

Two files live under `SHELLMATE_HOME` (default `~/.shellmate`), private to you (folder 0700, files 0600):

| File | Holds | Used by |
|---|---|---|
| `checkpoints.sqlite` | LangGraph checkpoints of a paused run (finished runs are deleted) | `shellmate resume` |
| `pending.json` | which thread is paused, its request and folder | `shellmate resume` |

```mermaid
sequenceDiagram
    actor You
    participant CLI
    participant DB as checkpoints.sqlite
    You->>CLI: shellmate ask "delete the build folder"
    CLI->>You: DANGER rm -rf build. Type yes to run it...
    You->>CLI: Ctrl-C
    CLI->>DB: checkpoint stays, pending.json written
    CLI->>You: paused - nothing ran
    Note over You,CLI: later, even after a reboot
    You->>CLI: shellmate resume
    CLI->>DB: load the thread, read its interrupt
    CLI->>You: the same command and question again
    You->>CLI: yes
    CLI->>DB: run, finish, delete the thread
```

The only code change between "forgets on exit" and "survives a restart" is the checkpointer passed
to `graph.compile()`: `InMemorySaver` in tests, `SqliteSaver` in the CLI (`memory.checkpointer`).

---

## 10. Evals

`evals/cases.jsonl` holds 55 requests, each with the action that counts, a regex for the command,
the risk level the rules must give it, and a known-good reference command.

- `python evals/run_evals.py --check` needs no model. It checks that every reference matches its
  regex and gets the expected risk. CI runs it, so a rule change that relabels a case fails the
  build.
- `python evals/run_evals.py --model qwen2.5:3b,qwen2.5:7b` asks each model for the first step of
  every case (nothing runs) and prints a pass-rate table. It calls the model once per case.
