"""Command-line entrypoint: ``shellmate`` starts an interactive session, ``shellmate ask "..."`` runs one request."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .config import get_settings

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
def ask(request: str = typer.Argument(..., help="What you want done, in plain English.")) -> None:
    """Run a single request and exit."""
    console.print(f"[yellow]not implemented yet[/yellow] - would handle: {request!r}")


def repl(model: str | None, dry_run: bool) -> None:
    settings = get_settings()
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
            console.print("[magenta]mate>[/magenta] not implemented yet - the agent lands in milestone 1.")


if __name__ == "__main__":
    app()
