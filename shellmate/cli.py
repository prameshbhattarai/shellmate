"""LangGraph concepts: running a graph, streaming its updates, and answering interrupts.

    shellmate                interactive session
    shellmate ask "..."      one request, then exit
    shellmate resume         continue a run you paused (Ctrl-C) at an approval prompt

The CLI is only the front end. It never decides anything: it streams the graph (graph.py), draws
each update, and when the graph pauses at ``ask_human`` it asks you and resumes the graph with your
answer. A web page or a test could drive the same graph the same way (tests/test_graph.py does).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Optional

import typer
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .agent import build_decider
from .config import Settings, get_settings
from .graph import build_graph
from .memory import checkpointer, clear_pending, load_pending, save_pending
from .replay import replay_path
from .risk import Risk

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

EXIT_OK = 0
EXIT_FAILED = 1  # model unreachable, or the step limit hit before an answer
EXIT_INTERRUPTED = 130  # Ctrl-C, as shells report it

# The model reads the full output and answers; the user sees only the start of it unless --verbose.
PREVIEW_LINES = 6


@dataclass(frozen=True)
class Options:
    model: str | None = None
    dry_run: bool = False
    confirm_all: bool = False
    verbose: bool = False


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
    confirm_all: bool = typer.Option(False, "--confirm-all", "-c", help="Ask before every command, even read-only ones."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print each command's full output."),
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show version."),
) -> None:
    ctx.obj = Options(model, dry_run, confirm_all, verbose)
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
    prompt) and Ctrl-D an EOFError.
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


# --- running the graph ---------------------------------------------------------------------------

def build_session_graph(opts: Options, settings: Settings, cwd: str | None = None):
    """One compiled graph per session, saving its checkpoints to SQLite (memory.py)."""
    return build_graph(build_decider(opts.model), settings=settings, cwd=cwd or os.getcwd(),
                       checkpointer=checkpointer(settings.home))


def _config(thread_id: str, settings: Settings) -> dict:
    # LangGraph: the config picks the thread. Same thread_id = same saved state, which is how a
    # paused run is found again. recursion_limit caps the number of node runs as a backstop.
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 10 * settings.max_steps + 10}


def _pending_interrupt(graph, config: dict) -> dict | None:
    """The question a paused thread is waiting on, read back from its checkpoint."""
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
                    # LangGraph: stream_mode="updates" yields {node_name: what_it_returned} after each
                    # node, so we can draw as we go. A pause shows up as a "__interrupt__" entry, and
                    # the stream then ends.
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
            # LangGraph: Command(resume=...) continues the paused thread; inside ask_human,
            # interrupt() now returns this reply.
            payload = Command(resume=reply)
    except KeyboardInterrupt:  # while the model thinks or a command runs (run_command kills it)
        console.print("\n[yellow]interrupted[/yellow]")
        _forget_thread(graph, thread_id, settings)
        return EXIT_INTERRUPTED

    final = graph.get_state(config).values  # LangGraph: the thread's latest saved state
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
    """Run the agent on one request, in a new thread, drawing each step. Returns the exit code."""
    payload = {"request": request, "dry_run": dry_run, "confirm_all": confirm_all, "steps": []}
    return _drive(graph, thread_id=uuid.uuid4().hex, payload=payload, settings=settings, request=request,
                  cwd=cwd or os.getcwd(), verbose=verbose)


# --- commands ------------------------------------------------------------------------------------

@app.command()
def ask(
    ctx: typer.Context,
    request: str = typer.Argument(..., help="What you want done, in plain English."),
) -> None:
    """Work on REQUEST step by step, asking before anything risky, then answer and exit."""
    opts, settings = _opts(ctx), get_settings()
    code = handle_request(request, graph=build_session_graph(opts, settings), settings=settings,
                          dry_run=opts.dry_run, confirm_all=opts.confirm_all, verbose=opts.verbose)
    raise typer.Exit(code)


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
    code = _drive(graph, thread_id=pending["thread_id"], payload=None, settings=settings,
                  request=pending["request"], cwd=pending["cwd"], verbose=opts.verbose)
    raise typer.Exit(code)


def repl(opts: Options) -> None:
    settings = get_settings()
    graph = None  # built on the first request, then reused; each request gets its own thread
    model = "replay (scripted answers)" if replay_path() else (opts.model or settings.model)
    console.print(
        f"[bold]shellmate[/bold] {__version__}  model={model}"
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
            handle_request(line, graph=graph, settings=settings, dry_run=opts.dry_run,
                           confirm_all=opts.confirm_all, verbose=opts.verbose)


if __name__ == "__main__":
    app()
