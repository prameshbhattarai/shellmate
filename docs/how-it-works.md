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
    U(["You"]) -->|request| CLI["CLI<br/>cli.py"]
    CLI -->|streams| G["Agent graph<br/>graph.py"]
    G -->|messages| M["Model - Qwen via Ollama<br/>agent.py builds the prompt"]
    M -->|"Decision - run or answer"| G
    G -->|command| R["Risk rules<br/>risk.py"]
    R -->|"risk level"| G
    G -->|"pauses - approve?"| CLI
    CLI -->|"your answer"| G
    G -->|approved command| T["run_command<br/>tools.py"]
    T -->|"exit code and output"| G
    G -->|answer| CLI
    CLI -->|"answer"| U
```

| Module | Job |
|---|---|
| `cli.py` | The front end: reads your request, streams the graph, draws each step, asks for approval, prints the answer |
| `graph.py` | The agent loop as a LangGraph state graph: decide, check risk, ask, run, repeat |
| `agent.py` | Everything the model sees and returns: system prompt, `Decision` schema, how outputs are fed back, repeat guards |
| `risk.py` | Pure-Python rules that label a command SAFE, CAUTION, DANGER or BLOCKED |
| `tools.py` | `run_command`: runs one command in your shell with a timeout, no keyboard input and capped output |
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
    CLI->>Graph: stream the request
    Graph->>Model: system prompt + request
    Model-->>Graph: run git log -1 --format=%B
    Graph->>Risk: classify(command)
    Risk-->>Graph: SAFE - git log only reads
    Graph->>Shell: run it - SAFE needs no approval
    Shell-->>Graph: exit 0 and the output agent skeleton.
    Graph-->>CLI: command panel + output preview
    Graph->>Model: request, the command, its output, and a hint to answer
    Model-->>Graph: answer - Your last commit was agent skeleton.
    Graph-->>CLI: status answered
    CLI->>You: mate - Your last commit was agent skeleton.
```

A riskier command takes the same path, except the graph **pauses** before running it and the CLI
asks you (section 5).

---

## 3. The agent graph

This is the LangGraph state graph built by `build_graph()` in `graph.py`. Each box is a node, and
the labels on the arrows are the routing rules.

```mermaid
flowchart TD
    START(["START"]) --> decide

    decide{{"decide<br/>ask the model for the next step"}}
    decide -->|"answer, or no command"| END_A(["END - answered"])
    decide -->|"step limit reached"| END_L(["END - step_limit"])
    decide -->|"model unreachable or bad output"| END_E(["END - model_error"])
    decide -->|"run a command"| classify

    classify["classify_risk<br/>rules label the command"]
    classify -->|"dry run"| END_D(["END - dry_run"])
    classify -->|BLOCKED| refuse
    classify -->|"SAFE, no confirm-all"| run
    classify -->|"CAUTION or DANGER, or confirm-all"| ask

    ask[/"ask_human<br/>interrupt - pause for you"/]
    ask -->|"y, or yes for DANGER"| run
    ask -->|"edit"| classify
    ask -->|"no / Enter"| END_N(["END - declined"])

    run["run<br/>run_command, record the step"]
    run --> decide

    refuse["refuse<br/>record 'shellmate refused'"]
    refuse --> decide
```

### What each node does

| Node | Reads | Writes | Notes |
|---|---|---|---|
| `decide` | `request`, `steps` | `command` and `explanation`, or `answer` and `status` | Calls the model. Also checks the step limit and the repeat guards (section 4) |
| `classify_risk` | `command` | `risk`, `reason` (and `status: dry_run` in a dry run) | Rules only, no model call |
| `ask_human` | `command`, `risk`, `reason`, `explanation` | `reply` (`run` / `edit` / `skip`), maybe a new `command` | Calls LangGraph's `interrupt()`. The graph pauses here until the CLI resumes it |
| `run` | `command` | appends a step: `{command, outcome: "ran", exit_code, stdout, stderr}` | Output is cut at 4000 chars; 30 s timeout |
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

The state holds only plain JSON values, so any LangGraph checkpointer can save it. Today that is
in-memory; SQLite comes in the memory milestone.

```text
request       your words                      dry_run / confirm_all   flags from the CLI
steps[]       finished steps (appended)       command / explanation   the step being worked on
risk / reason the verdict for `command`       reply                   your answer in ask_human
answer        the final reply                 status                  answered | declined | dry_run | step_limit | model_error
error         set when status is model_error
```

---

## 4. Inside `decide`: talking to the model

```mermaid
flowchart TD
    A{"Step limit reached?"} -->|yes| L(["status step_limit"])
    A -->|no| B["build_messages<br/>system prompt, request, then each step and its output"]
    B --> C["Model returns a Decision<br/>run a command, or answer"]
    C --> P{"Problem?"}
    P -->|"repeats a command it already ran"| N1["Retry once<br/>nudge - you already ran it, answer now"]
    P -->|"answer is just a command it ran"| N2["Retry once<br/>nudge - that was a command, not an answer"]
    P -->|no| OK
    N1 --> C2["Ask the model again"]
    N2 --> C2
    C2 --> P2{"Still a problem?"}
    P2 -->|yes| FB(["Fallback answer<br/>shows the last output"])
    P2 -->|no| OK
    OK["clean_command<br/>strip wrappers like backticks and code fences"] --> D{"Action?"}
    D -->|"answer, or empty command"| ANS(["status answered"])
    D -->|run| CMD(["Command goes to classify_risk"])
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
    Graph->>CLI: interrupt with command, risk danger, reason
    CLI->>You: Dangerous. Type yes to run it, e to edit, Enter to skip
    You->>CLI: y
    CLI->>You: for a dangerous command, type the whole word: yes
    You->>CLI: yes
    CLI->>Graph: resume with action run
    Graph->>Graph: ask_human returns run, then the run node
```

What each answer does, by risk level:

| Risk | Asked? | What runs it | Edit | Enter / n |
|---|---|---|---|---|
| SAFE | only with `--confirm-all` | `y` | re-checked | stop |
| CAUTION | yes | `y` | re-checked | stop |
| DANGER | yes | the full word `yes` (a plain `y` is rejected) | re-checked | stop |
| BLOCKED | never | nothing: it is refused and the model is told | n/a | n/a |

The graph never prompts anyone itself; it only pauses and resumes. A different front end (a TUI,
a web page, a test) can drive it the same way. The tests do exactly that in `tests/test_graph.py`.

---

## 6. The risk rules

First the whole command is checked. Anything shellmate can't read reliably is DANGER.

```mermaid
flowchart TD
    C["command"] --> F{"Fork bomb?"}
    F -->|yes| BL1(["BLOCKED"])
    F -->|no| S{"Several lines or command substitution?"}
    S -->|yes| DG1(["DANGER - cannot be checked"])
    S -->|no| Q{"Quotes balanced?"}
    Q -->|no| DG2(["DANGER - cannot be parsed"])
    Q -->|yes| SP["Split into segments at pipes, and, or, semicolon"]
    SP --> BG{"Ends with a background ampersand?"}
    BG -->|yes| DG3(["DANGER - escapes the timeout"])
    BG -->|no| EACH["Check each segment - see next chart"]
    EACH --> PIPE{"Download piped into a shell?"}
    PIPE -->|yes| BL2(["BLOCKED"])
    PIPE -->|no| W(["Verdict = the worst segment"])
```

Then each segment is checked: first its redirections, then the program it runs.

```mermaid
flowchart TD
    SEG["segment"] --> R{"Redirects output?"}
    R -->|"to a disk device"| B1(["BLOCKED"])
    R -->|"overwrites a file"| D1(["DANGER"])
    R -->|"appends to a file"| C1(["CAUTION"])
    R -->|"no, or to /dev/null"| W{"Starts with sudo, env or xargs?"}
    W -->|"yes - check what it runs"| P
    W -->|no| P{"Which program?"}
    P -->|"rm, rmdir, chmod on root, home or a top folder"| B2(["BLOCKED"])
    P -->|"git, find, sed, curl, pip ..."| SUB["Judged by subcommand and flags"]
    P -->|"read-only: ls, cat, du, grep ..."| S1(["SAFE"])
    P -->|"writers: mkdir, cp, touch ..."| C2(["CAUTION"])
    P -->|"rm, mv, kill, python, ssh ..."| D2(["DANGER"])
    P -->|"unknown program"| D3(["DANGER"])
    SUB --> EX["git log - SAFE<br/>git add - CAUTION<br/>git push - DANGER"]
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

---

## 8. Testing without a model

No test calls Ollama. `tests/fakes.py` has a `ScriptedDecider` that returns a fixed list of
decisions, so a test can script "run X, then run X again, then answer" and check that the graph
nudges, falls back, refuses or pauses as described above. Running commands in tests is real but
harmless: `echo`, `mkdir` and `rm` on temp files.

| Test file | Covers |
|---|---|
| `test_risk.py` | 66 commands across the four levels |
| `test_graph.py` | the loop: auto-run, approvals, edits re-checked, refusals, repeats, step cap, dry run |
| `test_cli.py` | prompts, `yes` vs `y`, output preview, `--verbose`, exit codes, the REPL |
| `test_agent.py` | prompt building, cutting old outputs, `clean_command` |
| `test_tools.py` | timeout, truncation, no keyboard input, runs in the current folder |
