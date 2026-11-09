# shellmate

**Talk to your terminal.** Say what you want in plain English; shellmate works out the shell
commands, labels each one by risk, runs it after you approve, reads the output, and keeps going until
the job is done. It runs on a local model (Qwen via Ollama), so nothing leaves your machine.

> Status: skeleton. The CLI starts, nothing is wired to the model yet. See [Roadmap](#roadmap).

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
understand -> propose_command -> classify_risk -> auto_run | ask_human -> run -> observe -> next_step | answer
```

- **Agent loop** in [LangGraph](https://langchain-ai.github.io/langgraph/), with a step cap and retries.
- **Tools are the only way to act:** `run_command` (timeout, locked to the working directory, output
  truncated), `read_file`, `list_dir`, `last_error`.
- **Risk first, model second.** Rules flag `rm`, `sudo`, `mv`, redirects, `curl | sh`,
  `git push --force` and friends before the model is asked. Anything unsure is treated as DANGER, and
  DANGER always waits for your `y/n/edit`, whatever the model says.
- **Memory:** remembers your OS, project and aliases across sessions (SQLite).
- **Evals:** a set of requests with the expected command and risk label, scored in CI.

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
  cli.py        # typer entrypoint, REPL, rich rendering
  config.py     # settings from env; builds the chat model lazily
  graph.py      # LangGraph state, nodes, human-approval interrupt
  tools.py      # run_command, read_file, list_dir, last_error
  risk.py       # rule-based risk classifier (+ LLM fallback)
  memory.py     # session checkpoints + user profile
evals/          # request -> expected command + risk, and the scorer
tests/          # unit tests; never call the model
```

## Roadmap

- [x] Skeleton: CLI, module layout, smoke tests, CI
- [ ] One-shot: request -> one command -> confirm -> run -> print
- [ ] Risk classifier with unit tests (pure Python)
- [ ] LangGraph loop: observe output, decide next step, step cap, `interrupt()` approval
- [ ] Tools for files, history, "explain last error"
- [ ] Memory across sessions
- [ ] Eval set (50 cases) + CI score badge; Qwen 3b vs 7b comparison table
- [ ] Demo GIF, then fun extras: personas, "teach me" mode, `shellmate undo`, zsh key binding

## License

MIT
