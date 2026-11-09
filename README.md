# shellmate

**Talk to your terminal.** Say what you want in plain English; shellmate works out the shell
commands, labels each one by risk, runs it after you approve, reads the output, and keeps going until
the job is done. It runs on a local model (Qwen via Ollama), so nothing leaves your machine.

> Status: working agent loop with a rule-based risk gate. Memory, extra tools and evals are next. See [Roadmap](#roadmap).

```
$ shellmate
you> which folders here are eating my disk?
[plan]  du -sh ./* | sort -rh | head -5          risk: SAFE (read-only) -> auto-run
[out]   4.7G venv   1.2G node_modules   ...
mate>   venv (4.7G) and node_modules (1.2G). Want me to remove node_modules?
you> yes
[plan]  rm -rf ./node_modules                     risk: DANGER (deletes files)
        Run? [y/n/edit]  y
mate>   Done, freed 1.2G. `npm install` brings it back.

you> why did this fail?   (after a failed `git push`)
mate>   Your branch is behind origin. Run `git pull --rebase` then push again. Do it? [y/n]
```

## How it works

```
decide -> classify_risk -> run (SAFE) | ask_human (CAUTION / DANGER) | refuse (BLOCKED) -> decide ... -> answer
```

- **Agent loop** in [LangGraph](https://langchain-ai.github.io/langgraph/): the model proposes one
  command at a time, reads its output and picks the next step or answers, with a step cap.
  Repeated commands are caught in code, not just in the prompt.
- **The model only proposes.** It returns a structured `Decision` (run a command, or answer); the
  graph runs it, and the only action available is `run_command`: timeout, no keyboard input, output
  capped.
- **Risk is decided by rules, never by the model.** Every command, including one you edited, is
  labelled before it runs: SAFE runs, CAUTION asks `y/N`, DANGER needs the full word `yes`, and
  BLOCKED (`rm -rf ~`, `curl | sh`, `dd` to a disk ...) never runs.
- **Human-in-the-loop with `interrupt()`.** Approval pauses the graph and the CLI resumes it with
  your answer, so any front end can drive the same agent.
- Planned: file and history tools, memory across sessions (SQLite), evals scored in CI.

**Full walkthrough with flowcharts: [docs/how-it-works.md](docs/how-it-works.md)** - the agent
graph node by node, what the model sees, the approval pause, and the risk rules.

## Quick start

```bash
# 1. A local model (one-off)
brew install ollama && ollama pull qwen2.5:3b

# 2. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 3. Run
shellmate            # interactive
shellmate --help
```

## Project layout

```
shellmate/
  cli.py        # typer entrypoint, REPL, rich rendering, answers the graph's approval pauses
  agent.py      # system prompt, Decision schema, how outputs are fed back, repeat guards
  graph.py      # LangGraph state and nodes: decide, classify_risk, ask_human, run, refuse
  risk.py       # rule-based risk classifier: SAFE / CAUTION / DANGER / BLOCKED
  tools.py      # run_command (read_file, list_dir, last_error planned)
  config.py     # settings from env; builds the chat model lazily
  memory.py     # session checkpoints + user profile (planned)
docs/           # how-it-works.md: flowcharts of the agent, graph and risk rules
evals/          # request -> expected command + risk, and the scorer
tests/          # unit tests with a scripted fake model; never call Ollama
```

## Roadmap

- [x] Skeleton: CLI, module layout, smoke tests, CI
- [x] One-shot: request -> one command -> confirm -> run -> print
- [x] Risk classifier with unit tests (pure Python)
- [x] LangGraph loop: observe output, decide next step, step cap, `interrupt()` approval
- [ ] Tools for files, history, "explain last error"
- [ ] Memory across sessions
- [ ] Eval set (50 cases) + CI score badge; Qwen 3b vs 7b comparison table
- [ ] Demo GIF, then fun extras: personas, "teach me" mode, `shellmate undo`, zsh key binding

## License

MIT
