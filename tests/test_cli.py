"""The CLI front end, with the scripted model. The tests type the user's answers."""

from __future__ import annotations

import pytest
from fakes import ScriptedDecider, answer, run
from typer.testing import CliRunner

from shellmate import cli


@pytest.fixture(autouse=True)
def small_limits(monkeypatch):
    monkeypatch.setenv("SHELLMATE_MAX_STEPS", "3")
    monkeypatch.setenv("SHELLMATE_COMMAND_TIMEOUT_S", "5")


def invoke(monkeypatch, decider, args, user_input=""):
    monkeypatch.setattr(cli, "build_decider", lambda model=None: decider)
    return CliRunner().invoke(cli.app, args, input=user_input)


def test_safe_runs_and_answers(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(run("echo agent-skeleton"), answer("Your last commit: agent skeleton.")), ["ask", "x"])
    assert "Run it?" not in result.output
    assert "agent-skeleton" in result.output
    assert "mate> Your last commit: agent skeleton." in result.output
    assert result.exit_code == 0


def test_confirm_all_prompts_for_safe(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(run("echo hi"), answer("ok")), ["--confirm-all", "ask", "x"], "y\n")
    assert "Run it?" in result.output and "mate> ok" in result.output


def test_dangerous_ignores_plain_y(monkeypatch, tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("x")
    result = invoke(monkeypatch, ScriptedDecider(run(f"rm {victim}"), answer("done")), ["ask", "x"], "y\n\n")
    assert "DANGER" in result.output and "type the whole word" in result.output
    assert "skipped" in result.output
    assert victim.exists()


def test_dangerous_runs_after_yes(monkeypatch, tmp_path):
    victim = tmp_path / "gone.txt"
    victim.write_text("x")
    result = invoke(monkeypatch, ScriptedDecider(run(f"rm {victim}"), answer("deleted it")), ["ask", "x"], "yes\n")
    assert not victim.exists()
    assert "mate> deleted it" in result.output


def test_blocked_is_refused_then_answered(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(run("rm -rf ~"), answer("I won't do that")), ["ask", "x"], "yes\n")
    assert "BLOCKED" in result.output and "will not run this" in result.output
    assert "mate> I won't do that" in result.output


def test_edit_then_confirm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = invoke(monkeypatch, ScriptedDecider(run("mkdir one"), answer("made two")), ["ask", "x"], "e\nmkdir two\ny\n")
    assert (tmp_path / "two").is_dir() and not (tmp_path / "one").exists()
    assert "mate> made two" in result.output


def test_dry_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = invoke(monkeypatch, ScriptedDecider(run("git push --force")), ["--dry-run", "ask", "x"])
    assert "DANGER" in result.output and "dry run" in result.output
    assert result.exit_code == 0


def test_model_error_is_friendly(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(ConnectionError("refused")), ["ask", "x"])
    assert "ollama serve" in result.output
    assert result.exit_code == cli.EXIT_FAILED


def test_step_limit_exit_code(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(*(run(f"echo step-{i}") for i in range(10))), ["ask", "x"])
    assert "stopped after 3 steps" in result.output
    assert result.exit_code == cli.EXIT_FAILED


def test_text_is_not_rich_markup(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(run("echo '[bold]hi[/bold]'"), answer("[red]x[/red]")), ["ask", "x"])
    assert "[bold]hi[/bold]" in result.output and "[red]x[/red]" in result.output


def test_repl(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(run("echo repl-ran"), answer("ran it")), [], "list\nexit\n")
    assert "repl-ran" in result.output and "mate> ran it" in result.output


LONG = "seq 1 50"


def test_long_output_is_previewed(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(run(LONG), answer("fifty lines")), ["ask", "x"])
    assert "44 more lines" in result.output
    assert "\n50\n" not in result.output
    assert "mate> fifty lines" in result.output


def test_verbose_shows_everything(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(run(LONG), answer("fifty lines")), ["--verbose", "ask", "x"])
    assert "more lines" not in result.output and "\n50\n" in result.output


def test_end_of_input_at_the_prompt_skips(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = invoke(monkeypatch, ScriptedDecider(run("mkdir made"), answer("x")), ["ask", "x"], "")
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "skipped" in result.output and not (tmp_path / "made").exists()
    assert result.exit_code == 0


def test_end_of_input_while_editing_skips(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = invoke(monkeypatch, ScriptedDecider(run("mkdir made"), answer("x")), ["ask", "x"], "e\n")
    assert "skipped" in result.output and not (tmp_path / "made").exists()


def test_ctrl_c_is_reported_not_a_traceback(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(KeyboardInterrupt()), ["ask", "x"])
    assert "interrupted" in result.output
    assert result.exit_code == cli.EXIT_INTERRUPTED


def test_repl_builds_the_graph_once(monkeypatch):
    builds = []
    real = cli.build_graph
    monkeypatch.setattr(cli, "build_graph", lambda *a, **k: builds.append(1) or real(*a, **k))
    decider = ScriptedDecider(run("echo one"), answer("first"), run("echo two"), answer("second"))
    result = invoke(monkeypatch, decider, [], "a\nb\nexit\n")
    assert "mate> first" in result.output and "mate> second" in result.output
    assert len(builds) == 1
