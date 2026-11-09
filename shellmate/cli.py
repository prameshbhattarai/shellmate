"""Command-line entrypoint: ``shellmate`` starts an interactive session, ``shellmate ask "..."`` runs one request."""

from __future__ import annotations

import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .config import Settings, get_settings
from .oneshot import Proposer, build_proposer, propose_command
from .risk import Risk, RiskVerdict, classify
from .tools import CommandResult, run_command

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Talk to your terminal: describe what you want, shellmate plans, risk-checks and runs the commands.",
)
console = Console()


def _version(value: bool) -> None:
    if value:
        console.print(f"shellmate {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model, e.g. qwen2.5:3b or qwen2.5:7b."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the commands, never run them."),
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show version."),
) -> None:
    ctx.obj = {"model": model, "dry_run": dry_run}
    if ctx.invoked_subcommand is None:
        repl(model=model, dry_run=dry_run)


@app.command()
def ask(
    ctx: typer.Context,
    request: str = typer.Argument(..., help="What you want done, in plain English."),
) -> None:
    """Propose one command for REQUEST, confirm it with you, run it and exit with its exit code."""
    opts = ctx.obj or {}
    exit_code = handle_request(
        request,
        proposer=build_proposer(opts.get("model")),
        settings=get_settings(),
        dry_run=opts.get("dry_run", False),
    )
    raise typer.Exit(exit_code)


RISK_STYLE = {Risk.SAFE: "green", Risk.CAUTION: "yellow", Risk.DANGER: "bold red", Risk.BLOCKED: "bold white on red"}

# Exit code when shellmate refuses to run a command.
REFUSED_EXIT_CODE = 1


def _show(command: str, verdict: RiskVerdict, explanation: str = "") -> None:
    style = RISK_STYLE[verdict.risk]
    console.print(Panel(Text(command), title="proposed command", border_style=style, expand=False))
    if explanation:
        console.print(Text(explanation, style="dim"))
    label = Text(f" {verdict.risk.value.upper()} ", style=style)
    console.print(Text.assemble("risk ", label, f" {verdict.reason}"), highlight=False)


def _confirm(command: str, verdict: RiskVerdict) -> tuple[str, str | None]:
    """Ask the user. Returns ("run", command), ("skip", None) or ("edit", new_command).

    SAFE and CAUTION take y/N/e. DANGER takes only the full word "yes", so a reflex "y" can't
    run it. BLOCKED never gets here.
    """
    danger = verdict.risk is Risk.DANGER
    prompt = (
        "[bold red]Dangerous.[/bold red] Type [bold]yes[/bold] to run it, e to edit, Enter to skip: "
        if danger
        else "[bold]Run it?[/bold] [y/N/e(dit)] "
    )
    while True:
        answer = console.input(prompt).strip().lower()
        if answer == "yes" or (answer == "y" and not danger):
            return "run", command
        if answer in {"", "n", "no"}:
            return "skip", None
        if answer in {"e", "edit"}:
            edited = typer.prompt("command", default=command).strip()
            return ("edit", edited) if edited else ("skip", None)
        if danger and answer == "y":
            console.print("for a dangerous command, type the whole word: yes")
            continue
        console.print("please answer y, n or e")


def _print_result(result: CommandResult) -> None:
    """Print the output like the shell would. The exit code is shown only when the command failed."""
    if result.stdout.strip():
        console.print(result.stdout.rstrip(), markup=False, highlight=False)
    if result.stderr.strip():
        console.print(result.stderr.rstrip(), style="red", markup=False, highlight=False)
    if result.exit_code != 0:
        console.print(f"exit {result.exit_code}", style="red")
    elif not result.stdout.strip() and not result.stderr.strip():
        console.print("done (no output)", style="dim")


def handle_request(request: str, *, proposer: Proposer, settings: Settings, dry_run: bool) -> int:
    """The one-shot flow. Returns the exit code to report (0 when nothing ran)."""
    cwd = os.getcwd()
    try:
        with console.status("thinking..."):
            proposal = propose_command(request, proposer, cwd=cwd)
    except Exception as exc:  # connection refused, model missing, bad JSON ...
        console.print(f"[red]could not get a command from the model:[/red] {exc}")
        console.print(f"is Ollama up at {settings.base_url}? start it with: ollama serve", soft_wrap=True)
        return 1

    if not proposal.command:
        console.print(f"[magenta]mate>[/magenta] {proposal.explanation or 'no command fits that request.'}")
        return 0

    command, explanation = proposal.command, proposal.explanation
    while True:  # loops only when the user edits: an edited command is checked again
        verdict = classify(command)
        _show(command, verdict, explanation)
        explanation = ""  # after an edit the model's explanation no longer applies
        if verdict.risk is Risk.BLOCKED:
            console.print("[bold red]shellmate will not run this.[/bold red] If you really mean it, run it yourself.")
            return REFUSED_EXIT_CODE
        if dry_run:
            console.print("[dim]dry run - not executed[/dim]")
            return 0
        action, edited = _confirm(command, verdict)
        if action == "skip":
            console.print("[dim]skipped[/dim]")
            return 0
        if action == "run":
            break
        command = edited

    result = run_command(
        command,
        cwd=cwd,
        timeout_s=settings.command_timeout_s,
        max_output_chars=settings.max_output_chars,
    )
    _print_result(result)
    return result.exit_code


def repl(model: str | None, dry_run: bool) -> None:
    settings = get_settings()
    proposer: Proposer | None = None
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
            proposer = proposer or build_proposer(model)
            handle_request(line, proposer=proposer, settings=settings, dry_run=dry_run)


if __name__ == "__main__":
    app()
