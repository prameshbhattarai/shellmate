"""One-shot flow tests. A fake proposer stands in for the model, so nothing calls Ollama."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from shellmate import cli
from shellmate.oneshot import Proposal, clean_command, propose_command
from shellmate.tools import TIMEOUT_EXIT_CODE, run_command


class FakeProposer:
    def __init__(self, command: str, explanation: str = "test command", error: Exception | None = None):
        self.proposal = Proposal(command=command, explanation=explanation)
        self.error = error
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.proposal


def _run(monkeypatch, proposer, args, user_input=""):
    monkeypatch.setattr(cli, "build_proposer", lambda model=None: proposer)
    return CliRunner().invoke(cli.app, args, input=user_input)


# --- tools.run_command -------------------------------------------------------------------------

def test_run_command_captures_output_and_exit_code(tmp_path):
    result = run_command("echo hi; echo oops >&2; exit 3", cwd=str(tmp_path), timeout_s=5, max_output_chars=100)
    assert (result.exit_code, result.stdout, result.stderr.strip()) == (3, "hi\n", "oops")


def test_run_command_runs_in_cwd(tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    assert "marker.txt" in run_command("ls", cwd=str(tmp_path), timeout_s=5, max_output_chars=100).stdout


def test_run_command_truncates(tmp_path):
    result = run_command("printf 'a%.0s' $(seq 1 50)", cwd=str(tmp_path), timeout_s=5, max_output_chars=10)
    assert result.truncated
    assert result.stdout.startswith("a" * 10) and "truncated 40 chars" in result.stdout


def test_run_command_times_out_without_raising(tmp_path):
    result = run_command("sleep 5", cwd=str(tmp_path), timeout_s=1, max_output_chars=100)
    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert "timed out" in result.stderr


# --- oneshot.propose_command -------------------------------------------------------------------

def test_prompt_carries_environment_and_request(tmp_path):
    fake = FakeProposer("  ls -la  ")
    proposal = propose_command("list files", fake, cwd=str(tmp_path))
    assert proposal.command == "ls -la"
    (_, system), (_, human) = fake.calls[0]
    assert str(tmp_path) in system and human == "list files"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$(git diff $(git rev-parse HEAD))", "git diff $(git rev-parse HEAD)"),
        ("`ls -la`", "ls -la"),
        ("$ du -sh *", "du -sh *"),
        ("```bash\ngit show --stat HEAD\n```", "git show --stat HEAD"),
        ("$($(ls))", "ls"),
        # left alone: substitution is only part of the command
        ("echo $(date)", "echo $(date)"),
        ("$(a) | $(b)", "$(a) | $(b)"),
        ("ls", "ls"),
    ],
)
def test_clean_command(raw, expected):
    assert clean_command(raw) == expected


def test_proposal_is_cleaned(tmp_path):
    assert propose_command("x", FakeProposer("$(git show HEAD)"), cwd=str(tmp_path)).command == "git show HEAD"


# --- cli ask -----------------------------------------------------------------------------------

def test_ask_yes_runs_and_returns_exit_code(monkeypatch):
    result = _run(monkeypatch, FakeProposer("echo from-shellmate; exit 2"), ["ask", "say hi"], "y\n")
    assert "from-shellmate" in result.output
    assert result.exit_code == 2


def test_success_hides_exit_code(monkeypatch):
    result = _run(monkeypatch, FakeProposer("printf 'agent skeleton.\\n\\n'"), ["ask", "x"], "y\n")
    assert result.output.rstrip().endswith("agent skeleton.")
    assert "exit 0" not in result.output


def test_silent_success_says_done(monkeypatch):
    result = _run(monkeypatch, FakeProposer("true"), ["ask", "x"], "y\n")
    assert "done (no output)" in result.output


def test_stderr_with_brackets_is_not_markup(monkeypatch):
    result = _run(monkeypatch, FakeProposer("echo '[red]x[/red]' >&2; exit 1"), ["ask", "x"], "y\n")
    assert "[red]x[/red]" in result.output and "exit 1" in result.output


@pytest.mark.parametrize("answer", ["n\n", "\n"])
def test_ask_no_skips(monkeypatch, answer):
    result = _run(monkeypatch, FakeProposer("echo should-not-run"), ["ask", "x"], answer)
    assert "skipped" in result.output
    assert "should-not-run\n" not in result.output.replace("echo should-not-run", "")
    assert result.exit_code == 0


def test_ask_edit_runs_edited_command(monkeypatch):
    # an edited command is shown and checked again, so it needs its own "y"
    result = _run(monkeypatch, FakeProposer("echo original"), ["ask", "x"], "e\necho edited\ny\n")
    assert result.output.rstrip().endswith("edited")
    assert result.exit_code == 0


def test_dry_run_never_executes(monkeypatch):
    result = _run(monkeypatch, FakeProposer("echo nope > dry.txt"), ["--dry-run", "ask", "x"])
    assert "dry run" in result.output
    assert result.exit_code == 0


def test_empty_command_prints_explanation(monkeypatch):
    result = _run(monkeypatch, FakeProposer("", "that is not a shell task"), ["ask", "write me a poem"])
    assert "not a shell task" in result.output
    assert result.exit_code == 0


def test_model_error_is_friendly(monkeypatch):
    result = _run(monkeypatch, FakeProposer("", error=ConnectionError("refused")), ["ask", "x"])
    assert "ollama serve" in result.output
    assert result.exit_code == 1


def test_repl_uses_the_same_flow(monkeypatch):
    result = _run(monkeypatch, FakeProposer("echo repl-ran"), [], "list\ny\nexit\n")
    assert "repl-ran" in result.output


# --- risk gate in the CLI ----------------------------------------------------------------------

def test_dangerous_command_ignores_plain_y(monkeypatch, tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("x")
    result = _run(monkeypatch, FakeProposer(f"rm {victim}"), ["ask", "x"], "y\n\n")
    assert "DANGER" in result.output and "type the whole word" in result.output
    assert victim.exists()


def test_dangerous_command_runs_after_yes(monkeypatch, tmp_path):
    victim = tmp_path / "gone.txt"
    victim.write_text("x")
    result = _run(monkeypatch, FakeProposer(f"rm {victim}"), ["ask", "x"], "yes\n")
    assert result.exit_code == 0
    assert not victim.exists()


def test_blocked_command_never_prompts_or_runs(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_command", lambda *a, **k: calls.append(a))
    result = _run(monkeypatch, FakeProposer("rm -rf ~"), ["ask", "x"], "yes\n")
    assert "will not run this" in result.output
    assert result.exit_code == cli.REFUSED_EXIT_CODE
    assert calls == []


def test_edit_cannot_sneak_past_the_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_command", lambda *a, **k: calls.append(a))
    result = _run(monkeypatch, FakeProposer("ls"), ["ask", "x"], "e\nrm -rf /\nyes\n")
    assert "will not run this" in result.output
    assert calls == []


def test_dry_run_still_reports_risk(monkeypatch):
    result = _run(monkeypatch, FakeProposer("git push --force"), ["--dry-run", "ask", "x"])
    assert "DANGER" in result.output and "dry run" in result.output


def test_command_text_is_not_rich_markup(monkeypatch):
    result = _run(monkeypatch, FakeProposer("echo '[bold]hi[/bold]'"), ["--dry-run", "ask", "x"])
    assert "[bold]hi[/bold]" in result.output


def test_commands_get_no_keyboard_input(tmp_path):
    result = run_command("cat", cwd=str(tmp_path), timeout_s=5, max_output_chars=100)
    assert result.exit_code == 0  # would hang until the timeout if stdin were the terminal
