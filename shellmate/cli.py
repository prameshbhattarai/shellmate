"""Command-line entrypoint: ``shellmate`` starts an interactive session, ``shellmate ask "..."`` runs one request.

The CLI is only the front end: it streams the agent graph (graph.py), draws each step, and answers
the graph's approval interrupts by asking the user.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

import typer
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .agent import Decider, build_decider
from .config import Settings, get_settings
from .graph import build_graph
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

# Exit codes for `shellmate ask`.
EXIT_OK = 0
EXIT_FAILED = 1  # model unreachable, or the step limit hit before an answer

# The model reads the full output and answers; the user sees only the start of it unless --verbose.
PREVIEW_LINES = 6


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
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show version."),
) -> None:
    ctx.obj = {"model": model, "dry_run": dry_run, "confirm_all": confirm_all, "verbose": verbose}
    if ctx.invoked_subcommand is None:
        repl(model=model, dry_run=dry_run, confirm_all=confirm_all, verbose=verbose)


@app.command()
def ask(
    ctx: typer.Context,
    request: str = typer.Argument(..., help="What you want done, in plain English."),
) -> None:
    """Work on REQUEST step by step, asking before anything risky, then answer and exit."""
    opts = ctx.obj or {}
    code = handle_request(
        request,
        decider=build_decider(opts.get("model")),
        settings=get_settings(),
        dry_run=opts.get("dry_run", False),
        confirm_all=opts.get("confirm_all", False),
        verbose=opts.get("verbose", False),
    )
    raise typer.Exit(code)


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
    out, err, code = step.get("stdout", ""), step.get("stderr", ""), step["exit_code"]
    if out.strip():
        console.print(Text(_preview(out, verbose), style="dim"))
    if err.strip():
        console.print(Text(_preview(err, verbose), style="red"))
    if code != 0:
        console.print(f"exit {code}", style="red")
    elif not out.strip() and not err.strip():
        console.print("done (no output)", style="dim")


def _render(node: str, update: dict | None, state: dict, verbose: bool = False) -> None:
    """Draw one graph update. ``state`` holds the latest command and explanation seen so far."""
    if node == "classify_risk" and update:
        _show_command(state["command"], update["risk"], update["reason"], state.get("explanation", ""))
    elif node in {"run", "refuse"} and update:
        for step in update.get("steps", []):
            _show_step(step, verbose)


# --- approval ------------------------------------------------------------------------------------

def _ask_user(request: dict) -> dict:
    """Answer the graph's ask_human interrupt. SAFE/CAUTION take y/N/e; DANGER only the word "yes"."""
    danger = request["risk"] == Risk.DANGER.value
    prompt = (
        "[bold red]Dangerous.[/bold red] Type [bold]yes[/bold] to run it, e to edit, Enter to skip: "
        if danger
        else "[bold]Run it?[/bold] [y/N/e(dit)] "
    )
    while True:
        answer = console.input(prompt).strip().lower()
        if answer == "yes" or (answer == "y" and not danger):
            return {"action": "run"}
        if answer in {"", "n", "no"}:
            return {"action": "skip"}
        if answer in {"e", "edit"}:
            edited = typer.prompt("command", default=request["command"]).strip()
            return {"action": "edit", "command": edited} if edited else {"action": "skip"}
        if danger and answer == "y":
            console.print("for a dangerous command, type the whole word: yes")
            continue
        console.print("please answer y, n or e")


# --- the loop ------------------------------------------------------------------------------------

def handle_request(
    request: str,
    *,
    decider: Decider,
    settings: Settings,
    dry_run: bool,
    confirm_all: bool = False,
    verbose: bool = False,
) -> int:
    """Run the agent on one request, drawing each step. Returns the exit code for `ask`."""
    graph = build_graph(decider, settings=settings, cwd=os.getcwd())
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 10 * settings.max_steps + 10}
    payload: dict | Command = {"request": request, "dry_run": dry_run, "confirm_all": confirm_all, "steps": []}

    seen: dict = {}  # latest value of each state key, merged from the stream (steps aside)
    while True:
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
        payload = Command(resume=_ask_user(pending))

    final = graph.get_state(config).values
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


def repl(model: str | None, dry_run: bool, confirm_all: bool = False, verbose: bool = False) -> None:
    settings = get_settings()
    decider: Decider | None = None
    console.print(
        f"[bold]shellmate[/bold] {__version__}  model={model or settings.model}"
        f"{'  (dry run)' if dry_run else ''}  -  type 'exit' to quit"
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
            decider = decider or build_decider(model)
            handle_request(
                line, decider=decider, settings=settings, dry_run=dry_run, confirm_all=confirm_all, verbose=verbose
            )


if __name__ == "__main__":
    app()
