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
    result = invoke(monkeypatch, ScriptedDecider(run("echo '[bold]hi[/bold]'"), answer("[#f00]x[/#f00]")), ["ask", "x"])
    assert "[bold]hi[/bold]" in result.output and "[#f00]x[/#f00]" in result.output


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


# --- explain ---------------------------------------------------------------------------------------

def _request_sent(decider) -> str:
    return decider.calls[0][1][1]


def test_explain_text_argument(monkeypatch):
    decider = ScriptedDecider(answer("A missing module: run pip install requests."))
    result = invoke(monkeypatch, decider, ["explain", "ModuleNotFoundError: No module named 'requests'"])
    assert "No module named 'requests'" in _request_sent(decider)
    assert "mate> A missing module" in result.output


def test_explain_piped_input(monkeypatch):
    monkeypatch.setattr(cli, "_reattach_tty", lambda: None)
    decider = ScriptedDecider(answer("ok"))
    invoke(monkeypatch, decider, ["explain"], "make: *** No rule to make target 'build'.\n")
    assert "No rule to make target" in _request_sent(decider)


def test_explain_without_text_uses_last_error(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    decider = ScriptedDecider(answer("ok"))
    invoke(monkeypatch, decider, ["explain"])
    assert "last_error" in _request_sent(decider)


# --- pause and resume ------------------------------------------------------------------------------

def test_ctrl_c_at_the_prompt_pauses_and_resume_continues(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    decider = ScriptedDecider(run("mkdir made"), answer("made it"))
    real_input = cli.console.input

    def ctrl_c(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.console, "input", ctrl_c)
    first = invoke(monkeypatch, decider, ["ask", "make a folder"])
    assert "paused" in first.output and first.exit_code == cli.EXIT_INTERRUPTED
    assert not (tmp_path / "made").exists()

    monkeypatch.setattr(cli.console, "input", real_input)  # a new process, a new terminal session
    second = invoke(monkeypatch, decider, ["resume"], "y\n")
    assert "resuming" in second.output and "make a folder" in second.output
    assert (tmp_path / "made").is_dir()
    assert "mate> made it" in second.output

    third = invoke(monkeypatch, decider, ["resume"])
    assert "nothing is paused" in third.output


def test_finished_runs_leave_no_checkpoints(monkeypatch, isolated_home):
    import sqlite3

    invoke(monkeypatch, ScriptedDecider(run("echo hi"), answer("ok")), ["ask", "x"])
    conn = sqlite3.connect(isolated_home / "checkpoints.sqlite")
    assert conn.execute("select count(*) from checkpoints").fetchone()[0] == 0


# --- undo, history ---------------------------------------------------------------------------------

def test_undo_reverses_the_last_change(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    invoke(monkeypatch, ScriptedDecider(run("mkdir made"), answer("made it")), ["ask", "x"], "y\n")
    assert (tmp_path / "made").is_dir()
    result = invoke(monkeypatch, ScriptedDecider(answer("unused")), ["undo"], "y\n")
    assert "rmdir made" in result.output and not (tmp_path / "made").exists()
    again = invoke(monkeypatch, ScriptedDecider(answer("unused")), ["undo"])
    assert "nothing to undo" in again.output


def test_undo_says_when_it_cannot(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    invoke(monkeypatch, ScriptedDecider(run("touch f.txt"), answer("ok")), ["ask", "x"], "y\n")
    result = invoke(monkeypatch, ScriptedDecider(answer("unused")), ["undo"])
    assert "can't undo" in result.output and result.exit_code == cli.EXIT_FAILED


def test_history_command(monkeypatch):
    invoke(monkeypatch, ScriptedDecider(run("echo logged"), answer("ok")), ["ask", "x"])
    result = CliRunner().invoke(cli.app, ["history"])
    assert "echo logged" in result.output


# --- profile, persona, init ------------------------------------------------------------------------

def test_remembered_notes_reach_the_prompt(monkeypatch):
    runner = CliRunner()
    assert "remembered" in runner.invoke(cli.app, ["remember", "my", "projects", "live", "in", "~/code"]).output
    assert "1. my projects live in ~/code" in runner.invoke(cli.app, ["profile"]).output
    decider = ScriptedDecider(answer("ok"))
    invoke(monkeypatch, decider, ["ask", "x"])
    assert "my projects live in ~/code" in decider.calls[0][0][1]
    assert "forgot 1" in runner.invoke(cli.app, ["forget"]).output


def test_persona_and_teach_flags(monkeypatch):
    decider = ScriptedDecider(answer("ok"))
    invoke(monkeypatch, decider, ["--persona", "terse", "--teach", "ask", "x"])
    system = decider.calls[0][0][1]
    assert "one short line" in system and "Teaching mode" in system


def test_unknown_persona_is_rejected(monkeypatch):
    result = invoke(monkeypatch, ScriptedDecider(answer("ok")), ["--persona", "yoda", "ask", "x"])
    assert result.exit_code != 0


def test_init_zsh(monkeypatch):
    result = CliRunner().invoke(cli.app, ["init", "zsh"])
    assert "bindkey '^G' _shellmate_widget" in result.output
    assert CliRunner().invoke(cli.app, ["init", "fish"]).exit_code == cli.EXIT_FAILED


def test_ctrl_c_while_editing_pauses(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def ctrl_c(command):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_edit_line", ctrl_c)
    result = invoke(monkeypatch, ScriptedDecider(run("mkdir made"), answer("x")), ["ask", "x"], "e\n")
    assert "paused" in result.output and result.exit_code == cli.EXIT_INTERRUPTED


def test_a_second_pause_drops_the_first_runs_checkpoints(monkeypatch, isolated_home):
    import sqlite3

    def ctrl_c(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.console, "input", ctrl_c)
    invoke(monkeypatch, ScriptedDecider(run("mkdir one"), answer("x")), ["ask", "first"])
    invoke(monkeypatch, ScriptedDecider(run("mkdir two"), answer("x")), ["ask", "second"])
    conn = sqlite3.connect(isolated_home / "checkpoints.sqlite")
    assert conn.execute("select count(distinct thread_id) from checkpoints").fetchone()[0] == 1


def test_undo_of_a_commit_refuses_after_a_manual_commit(monkeypatch, tmp_path):
    import subprocess

    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "test")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "test@example.com")
    monkeypatch.chdir(tmp_path)
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "first")
    invoke(monkeypatch, ScriptedDecider(run("git commit -q --allow-empty -m wip"), answer("ok")), ["ask", "x"], "y\n")
    git("commit", "-q", "--allow-empty", "-m", "by hand")
    result = invoke(monkeypatch, ScriptedDecider(answer("unused")), ["undo"])
    assert "can't undo it safely" in result.output and result.exit_code == cli.EXIT_FAILED


def test_undo_of_a_commit_when_nothing_moved(monkeypatch, tmp_path):
    import subprocess

    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "test")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "test@example.com")
    monkeypatch.chdir(tmp_path)
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True)  # noqa: E731
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "first")
    invoke(monkeypatch, ScriptedDecider(run("git commit -q --allow-empty -m wip"), answer("ok")), ["ask", "x"], "y\n")
    result = invoke(monkeypatch, ScriptedDecider(answer("unused")), ["undo"], "yes\n")
    assert "git reset --soft HEAD~1" in result.output
    assert git("log", "-1", "--format=%s").stdout.strip() == "first"
