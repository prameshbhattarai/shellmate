"""Command-line entrypoint.

    shellmate                     interactive session
    shellmate ask "..."           one request, then exit
    shellmate explain [TEXT]      explain an error (an argument, piped input, or your last failed command)
    shellmate resume              continue a run you paused at an approval prompt (Ctrl-C there)
    shellmate undo                reverse the last change shellmate made in this folder
    shellmate history             commands shellmate ran
    shellmate remember / forget / profile   facts added to every prompt
    shellmate init zsh            a Ctrl-G key binding for your shell

The CLI is only the front end: it streams the agent graph (graph.py), draws each step, and answers
the graph's approval interrupts by asking the user.
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from typing import Optional

import typer
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .agent import PERSONAS, PromptContext, build_decider
from .config import Settings, get_settings
from .graph import build_graph
from .memory import (
    History,
    checkpointer,
    clear_pending,
    forget as forget_notes,
    format_entry,
    load_pending,
    load_profile,
    remember as remember_note,
    save_pending,
)
from .replay import replay_path
from .risk import Risk, classify
from .shell_init import SCRIPTS
from .tools import run_command
from .undo import changed_since, inverse, last_undoable, snapshot_after

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Talk to your terminal: describe what you want, shellmate plans, risk-checks and runs the commands.",
)
console = Console()

RISK_STYLE = {
    Risk.SAFE.value: "green",
    Risk.CAUTION.value: "yellow",
    Risk.DANGER.value: "bold red",
    Risk.BLOCKED.value: "bold white on red",
}

# Exit codes for `shellmate ask` and friends.
EXIT_OK = 0
EXIT_FAILED = 1  # model unreachable, step limit hit, or a refused / impossible undo
EXIT_INTERRUPTED = 130  # Ctrl-C, as shells report it

# The model reads the full output and answers; the user sees only the start of it unless --verbose.
PREVIEW_LINES = 6
# How much piped input `shellmate explain` passes on.
EXPLAIN_MAX_CHARS = 4000


@dataclass(frozen=True)
class Options:
    model: str | None = None
    dry_run: bool = False
    confirm_all: bool = False
    verbose: bool = False
    persona: str = "default"
    teach: bool = False


def _opts(ctx: typer.Context) -> Options:
    return ctx.obj if isinstance(ctx.obj, Options) else Options()


def _version(value: bool) -> None:
    if value:
        console.print(f"shellmate {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model, e.g. qwen2.5:3b or qwen2.5:7b."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the first command, never run anything."),
    confirm_all: bool = typer.Option(
        False, "--confirm-all", "-c", help="Ask before every command, even read-only ones."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print each command's full output."),
    persona: str = typer.Option(
        "default", "--persona", envvar="SHELLMATE_PERSONA", help=f"Answer style: {', '.join(PERSONAS)}."
    ),
    teach: bool = typer.Option(False, "--teach", "-t", help="End each answer with a line explaining the command."),
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show version."),
) -> None:
    if persona not in PERSONAS:
        raise typer.BadParameter(f"choose one of: {', '.join(PERSONAS)}", param_hint="--persona")
    ctx.obj = Options(model, dry_run, confirm_all, verbose, persona, teach)
    if ctx.invoked_subcommand is None:
        repl(ctx.obj)


# --- drawing -------------------------------------------------------------------------------------

def _show_command(command: str, risk: str, reason: str, explanation: str) -> None:
    style = RISK_STYLE[risk]
    console.print(Panel(Text(command), title="command", border_style=style, expand=False))
    if explanation:
        console.print(Text(explanation, style="dim"))
    console.print(Text.assemble("risk ", Text(f" {risk.upper()} ", style=style), f" {reason}"))


def _preview(text: str, verbose: bool) -> str:
    lines = text.rstrip().splitlines()
    if verbose or len(lines) <= PREVIEW_LINES:
        return "\n".join(lines)
    return "\n".join(lines[:PREVIEW_LINES] + [f"... {len(lines) - PREVIEW_LINES} more lines (--verbose shows all)"])


def _show_step(step: dict, verbose: bool = False) -> None:
    if step["outcome"] == "blocked":
        console.print("[bold red]shellmate will not run this.[/bold red] Asking the model for another way.")
        return
    if step["outcome"] == "tool":
        console.print(Text(f"> {step['command']}", style="cyan"))
    out, err, code = step.get("stdout", ""), step.get("stderr", ""), step["exit_code"]
    if out.strip():
        console.print(Text(_preview(out, verbose), style="dim"))
    if err.strip():
        console.print(Text(_preview(err, verbose), style="red"))
    if code != 0 and step["outcome"] != "tool":
        console.print(f"exit {code}", style="red")
    elif not out.strip() and not err.strip():
        console.print("done (no output)", style="dim")


def _render(node: str, update: dict | None, state: dict, verbose: bool = False) -> None:
    """Draw one graph update. ``state`` holds the latest command and explanation seen so far."""
    if node == "classify_risk" and update:
        _show_command(state["command"], update["risk"], update["reason"], state.get("explanation", ""))
    elif node in {"run", "refuse", "use_tool"} and update:
        for step in update.get("steps", []):
            _show_step(step, verbose)


# --- approval ------------------------------------------------------------------------------------

def _edit_line(command: str) -> str:
    """Read an edited command. In a terminal the line starts out holding ``command``, ready to change.

    Plain ``input()`` on purpose: Ctrl-C stays a KeyboardInterrupt (so it pauses, like at every other
    prompt) and Ctrl-D an EOFError, where typer.prompt would turn both into the same Abort.
    """
    try:
        import readline
    except ImportError:  # no readline (some Windows Pythons): type the whole command
        return input(f"command [{command}]: ")
    readline.set_startup_hook(lambda: readline.insert_text(command))
    try:
        return input("command: ")
    finally:
        readline.set_startup_hook()


def _ask_user(request: dict) -> dict:
    """Answer the graph's ask_human interrupt. SAFE/CAUTION take y/N/e; DANGER only the word "yes".

    Ctrl-C pauses the run (``shellmate resume`` picks it up); Ctrl-D or no terminal skips it.
    """
    danger = request["risk"] == Risk.DANGER.value
    prompt = (
        "[bold red]Dangerous.[/bold red] Type [bold]yes[/bold] to run it, e to edit, Enter to skip: "
        if danger
        else "[bold]Run it?[/bold] [y/N/e(dit)] "
    )
    while True:
        try:
            answer = console.input(prompt).strip().lower()
            if answer in {"e", "edit"}:
                edited = _edit_line(request["command"]).strip()
                # an empty line keeps the command as it was; it is shown and asked about again
                return {"action": "edit", "command": edited or request["command"]}
        except KeyboardInterrupt:
            console.print()
            return {"action": "pause"}
        except EOFError:
            console.print()
            return {"action": "skip"}
        if answer == "yes" or (answer == "y" and not danger):
            return {"action": "run"}
        if answer in {"", "n", "no"}:
            return {"action": "skip"}
        if danger and answer == "y":
            console.print("for a dangerous command, type the whole word: yes")
            continue
        console.print("please answer y, n or e")


# --- the loop ------------------------------------------------------------------------------------

def build_session_graph(opts: Options, settings: Settings, cwd: str | None = None):
    """One compiled graph per session, with history, SQLite checkpoints and the user's profile."""
    profile = load_profile(settings.home)
    context = PromptContext(persona=opts.persona, teach=opts.teach, notes=tuple(profile.notes))
    return build_graph(
        build_decider(opts.model),
        settings=settings,
        cwd=cwd or os.getcwd(),
        checkpointer=checkpointer(settings.home),
        history=History(settings.home),
        context=context,
    )


def _config(thread_id: str, settings: Settings) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 10 * settings.max_steps + 10}


def _pending_interrupt(graph, config: dict) -> dict | None:
    state = graph.get_state(config)
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def _forget_thread(graph, thread_id: str, settings: Settings) -> None:
    """A finished run's checkpoints are no longer needed; only paused runs are kept."""
    saver = getattr(graph, "checkpointer", None)
    if saver is not None and hasattr(saver, "delete_thread"):
        saver.delete_thread(thread_id)
    clear_pending(settings.home, thread_id)


def _drive(graph, *, thread_id: str, payload, settings: Settings, request: str, cwd: str, verbose: bool) -> int:
    """Stream the graph until it ends, answering approval pauses. ``payload=None`` resumes a paused run."""
    config = _config(thread_id, settings)
    seen: dict = {}  # latest value of each state key, merged from the stream (steps aside)
    try:
        while True:
            if payload is None:  # resuming: the pause happened in an earlier process
                pending = _pending_interrupt(graph, config)
                if pending is None:
                    break
                _show_command(pending["command"], pending["risk"], pending["reason"], pending.get("explanation", ""))
            else:
                pending = None
                with console.status("thinking...") as status:
                    for chunk in graph.stream(payload, config, stream_mode="updates"):
                        for node, update in chunk.items():
                            if node == "__interrupt__":
                                pending = update[0].value
                                continue
                            if update:
                                seen.update({k: v for k, v in update.items() if k != "steps"})
                            status.stop()  # don't let the spinner draw over output
                            _render(node, update, seen, verbose)
                            status.start()
                if pending is None:
                    break
            reply = _ask_user(pending)
            if reply["action"] == "pause":
                previous = load_pending(settings.home)
                if previous and previous["thread_id"] != thread_id:
                    _forget_thread(graph, previous["thread_id"], settings)  # only one paused run is kept
                save_pending(settings.home, thread_id=thread_id, request=request, cwd=cwd)
                console.print("[yellow]paused - nothing ran.[/yellow] `shellmate resume` picks it up, even after a restart.")
                return EXIT_INTERRUPTED
            payload = Command(resume=reply)
    except KeyboardInterrupt:  # while the model thinks or a command runs (run_command kills it)
        console.print("\n[yellow]interrupted[/yellow]")
        _forget_thread(graph, thread_id, settings)
        return EXIT_INTERRUPTED

    final = graph.get_state(config).values
    _forget_thread(graph, thread_id, settings)
    status = final.get("status")
    if status == "answered":
        console.print(Text.assemble(Text("mate> ", style="magenta"), final["answer"]))
    elif status == "dry_run":
        console.print("[dim]dry run - not executed[/dim]")
    elif status == "declined":
        console.print("[dim]skipped - stopping here[/dim]")
    elif status == "step_limit":
        console.print(Text(final["answer"], style="yellow"))
        return EXIT_FAILED
    elif status == "model_error":
        console.print(Text.assemble(Text("could not get an answer from the model: ", style="red"), final["error"]))
        console.print(f"is Ollama up at {settings.base_url}? start it with: ollama serve", soft_wrap=True)
        return EXIT_FAILED
    return EXIT_OK


def handle_request(
    request: str,
    *,
    graph,
    settings: Settings,
    dry_run: bool,
    confirm_all: bool = False,
    verbose: bool = False,
    cwd: str | None = None,
) -> int:
    """Run the agent on one request, drawing each step. Returns the exit code for `ask`."""
    payload = {"request": request, "dry_run": dry_run, "confirm_all": confirm_all, "steps": []}
    return _drive(
        graph,
        thread_id=uuid.uuid4().hex,
        payload=payload,
        settings=settings,
        request=request,
        cwd=cwd or os.getcwd(),
        verbose=verbose,
    )


def _run_one(opts: Options, request: str) -> int:
    settings = get_settings()
    return handle_request(
        request,
        graph=build_session_graph(opts, settings),
        settings=settings,
        dry_run=opts.dry_run,
        confirm_all=opts.confirm_all,
        verbose=opts.verbose,
    )


# --- commands ------------------------------------------------------------------------------------

@app.command()
def ask(
    ctx: typer.Context,
    request: str = typer.Argument(..., help="What you want done, in plain English."),
) -> None:
    """Work on REQUEST step by step, asking before anything risky, then answer and exit."""
    raise typer.Exit(_run_one(_opts(ctx), request))


def _reattach_tty() -> None:
    """After reading piped input, read approval answers from the terminal instead."""
    try:
        sys.stdin = open("/dev/tty")  # noqa: SIM115 - lives for the rest of the process
    except OSError:
        pass  # no terminal (CI, cron): prompts see end-of-input and skip


@app.command()
def explain(
    ctx: typer.Context,
    text: Optional[str] = typer.Argument(None, help="The error text. Omit to pipe it in, or to use your last failed command."),
) -> None:
    """Explain an error and how to fix it: `make 2>&1 | shellmate explain`, or just `shellmate explain`."""
    if text is None and not sys.stdin.isatty():
        text = sys.stdin.read()[-EXPLAIN_MAX_CHARS:]
        _reattach_tty()
    if text and text.strip():
        request = f"Explain this error and how to fix it:\n\n{text.strip()}"
    else:
        request = (
            "Explain my last error: use last_error to see the failed command and its output, then explain "
            "what went wrong and how to fix it."
        )
    raise typer.Exit(_run_one(_opts(ctx), request))


@app.command()
def resume(ctx: typer.Context) -> None:
    """Continue the run you paused at an approval prompt."""
    opts, settings = _opts(ctx), get_settings()
    pending = load_pending(settings.home)
    if pending is None:
        console.print("nothing is paused")
        raise typer.Exit(EXIT_OK)
    graph = build_session_graph(opts, settings, cwd=pending["cwd"])
    if _pending_interrupt(graph, _config(pending["thread_id"], settings)) is None:
        clear_pending(settings.home)
        console.print("that paused run is no longer there")
        raise typer.Exit(EXIT_OK)
    console.print(Text.assemble("resuming ", Text(pending["request"], style="bold"), f"  (in {pending['cwd']})"))
    code = _drive(
        graph,
        thread_id=pending["thread_id"],
        payload=None,
        settings=settings,
        request=pending["request"],
        cwd=pending["cwd"],
        verbose=opts.verbose,
    )
    raise typer.Exit(code)


@app.command()
def undo(ctx: typer.Context) -> None:
    """Reverse the last change shellmate made in this folder (shown and confirmed first)."""
    opts, settings = _opts(ctx), get_settings()
    cwd = os.getcwd()
    history = History(settings.home)
    entry = last_undoable(history.entries(), cwd=cwd)
    if entry is None:
        console.print("nothing to undo in this folder")
        raise typer.Exit(EXIT_OK)
    found = inverse(entry["command"], cwd=cwd)
    if found is None:
        console.print(Text.assemble("can't undo ", Text(entry["command"], style="bold"), " automatically"))
        raise typer.Exit(EXIT_FAILED)
    command, what = found
    console.print(Text.assemble("last change: ", Text(entry["command"], style="bold")))
    stale = changed_since(entry, cwd)
    if stale:
        console.print(Text(f"can't undo it safely: {stale}. Undo it by hand.", style="yellow"))
        raise typer.Exit(EXIT_FAILED)
    while True:  # loops only when the user edits: an edited command is checked again
        verdict = classify(command, cwd=cwd)
        _show_command(command, verdict.risk.value, verdict.reason, what)
        if verdict.risk is Risk.BLOCKED:
            console.print("[bold red]shellmate will not run this.[/bold red]")
            raise typer.Exit(EXIT_FAILED)
        if opts.dry_run:
            console.print("[dim]dry run - not executed[/dim]")
            raise typer.Exit(EXIT_OK)
        reply = _ask_user({"command": command, "risk": verdict.risk.value, "reason": verdict.reason})
        if reply["action"] == "edit":
            command, what = reply["command"], ""
            continue
        if reply["action"] != "run":
            console.print("[dim]skipped[/dim]")
            raise typer.Exit(EXIT_OK)
        break
    result = run_command(command, cwd=cwd, timeout_s=settings.command_timeout_s, max_output_chars=settings.max_output_chars)
    step = {"command": command, "outcome": "ran", "risk": verdict.risk.value, "exit_code": result.exit_code,
            "stdout": result.stdout, "stderr": result.stderr}
    _show_step(step, opts.verbose)
    history.append({"cwd": cwd, **step, **snapshot_after(command, cwd), "undoes": entry["id"]})
    raise typer.Exit(EXIT_OK if result.exit_code == 0 else EXIT_FAILED)


@app.command("history")
def history_cmd(
    count: int = typer.Option(10, "--count", "-n", help="How many entries."),
    here: bool = typer.Option(False, "--here", help="Only commands run in this folder."),
) -> None:
    """Show the commands shellmate ran."""
    entries = History(get_settings().home).recent(count, cwd=os.getcwd() if here else None)
    if not entries:
        console.print("no history yet")
    for entry in entries:
        style = "red" if entry.get("exit_code") else None
        console.print(Text(format_entry(entry), style=style))


@app.command()
def remember(note: list[str] = typer.Argument(..., help="A fact for every prompt, e.g. my projects live in ~/code")) -> None:
    """Remember a fact about you or your setup; it is added to every prompt."""
    profile = remember_note(get_settings().home, " ".join(note))
    n = len(profile.notes)
    console.print(f"remembered - {n} note{'' if n == 1 else 's'} saved. `shellmate profile` lists them.")


@app.command()
def forget(number: Optional[int] = typer.Argument(None, help="Note number from `shellmate profile`; omit to forget all.")) -> None:
    """Forget one remembered note, or all of them."""
    gone = forget_notes(get_settings().home, number)
    console.print(f"forgot {len(gone)} note(s)" if gone else "nothing to forget")


@app.command()
def profile() -> None:
    """List what shellmate remembers about you."""
    notes = load_profile(get_settings().home).notes
    if not notes:
        console.print('no notes yet. try: shellmate remember "my projects live in ~/code"')
    for i, note in enumerate(notes, 1):
        console.print(f"{i}. {note}", highlight=False)


@app.command()
def init(shell: str = typer.Argument("zsh", help="Shell to set up. Only zsh for now.")) -> None:
    """Print a key binding script: add `eval "$(shellmate init zsh)"` to ~/.zshrc, then press Ctrl-G."""
    script = SCRIPTS.get(shell)
    if script is None:
        console.print(f"no integration for {shell} yet (supported: {', '.join(SCRIPTS)})")
        raise typer.Exit(EXIT_FAILED)
    typer.echo(script, nl=False)


def repl(opts: Options) -> None:
    settings = get_settings()
    graph = None  # built on the first request, then reused
    console.print(
        f"[bold]shellmate[/bold] {__version__}  "
        f"{'model=replay (scripted answers)' if replay_path() else 'model=' + (opts.model or settings.model)}"
        f"{'  (dry run)' if opts.dry_run else ''}  -  type 'exit' to quit"
    )
    while True:
        try:
            line = console.input("[cyan]you>[/cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if line in {"exit", "quit"}:
            break
        if line:
            graph = graph or build_session_graph(opts, settings)
            handle_request(
                line, graph=graph, settings=settings, dry_run=opts.dry_run, confirm_all=opts.confirm_all,
                verbose=opts.verbose,
            )


if __name__ == "__main__":
    app()
