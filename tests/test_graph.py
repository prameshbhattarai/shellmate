"""The agent graph, driven directly: the test plays the user by answering interrupts."""

from __future__ import annotations

import pytest
from fakes import ScriptedDecider, answer, run
from langgraph.types import Command

from shellmate.config import Settings
from shellmate.graph import build_graph


@pytest.fixture
def settings():
    return Settings(max_steps=3, command_timeout_s=5, max_output_chars=500)


def drive(decider, settings, cwd, replies=(), **inputs):
    """Run the graph to the end. Returns (final state, the approval requests the user saw)."""
    graph = build_graph(decider, settings=settings, cwd=str(cwd))
    config = {"configurable": {"thread_id": "t"}}
    payload = {"request": "do it", "steps": [], **inputs}
    replies, asked = list(replies), []
    while True:
        out = graph.invoke(payload, config)
        pending = out.get("__interrupt__")
        if not pending:
            return graph.get_state(config).values, asked
        asked.append(pending[0].value)
        payload = Command(resume=replies.pop(0))


def test_safe_command_runs_without_asking_and_output_reaches_the_model(settings, tmp_path):
    decider = ScriptedDecider(run("echo hello-world"), answer("it printed hello-world"))
    state, asked = drive(decider, settings, tmp_path)
    assert asked == []
    assert state["status"] == "answered" and state["answer"] == "it printed hello-world"
    assert "hello-world" in decider.last_message()


def test_multi_step(settings, tmp_path):
    (tmp_path / "a.txt").write_text("first line\n")
    decider = ScriptedDecider(run("ls"), run("head -n 1 a.txt"), answer("a.txt starts with 'first line'"))
    state, _ = drive(decider, settings, tmp_path)
    assert [s["command"] for s in state["steps"]] == ["ls", "head -n 1 a.txt"]
    assert "first line" in decider.last_message()
    assert len(decider.calls) == 3


def test_failure_is_fed_back(settings, tmp_path):
    decider = ScriptedDecider(run("cat missing.txt"), answer("that file does not exist"))
    drive(decider, settings, tmp_path)
    assert "exit code 1" in decider.last_message() and "missing.txt" in decider.last_message()


def test_confirm_all_asks_even_for_safe(settings, tmp_path):
    decider = ScriptedDecider(run("echo hi"), answer("ok"))
    _, asked = drive(decider, settings, tmp_path, replies=[{"action": "run"}], confirm_all=True)
    assert [a["risk"] for a in asked] == ["safe"]


def test_caution_asks_and_runs_on_yes(settings, tmp_path):
    decider = ScriptedDecider(run("mkdir made"), answer("made it"))
    state, asked = drive(decider, settings, tmp_path, replies=[{"action": "run"}])
    assert asked[0]["risk"] == "caution"
    assert (tmp_path / "made").is_dir()
    assert state["status"] == "answered"


def test_decline_stops_without_another_model_call(settings, tmp_path):
    decider = ScriptedDecider(run("mkdir made"), answer("never"))
    state, _ = drive(decider, settings, tmp_path, replies=[{"action": "skip"}])
    assert state["status"] == "declined"
    assert not (tmp_path / "made").exists()
    assert len(decider.calls) == 1


def test_blocked_is_refused_and_model_told(settings, tmp_path):
    decider = ScriptedDecider(run("rm -rf ~"), answer("I won't delete your home folder"))
    state, asked = drive(decider, settings, tmp_path)
    assert asked == []
    assert state["steps"][0]["outcome"] == "blocked"
    assert "refused" in decider.last_message()
    assert state["status"] == "answered"


def test_edit_is_checked_again_blocked(settings, tmp_path):
    decider = ScriptedDecider(run("mkdir made"), answer("ok"))
    state, asked = drive(decider, settings, tmp_path, replies=[{"action": "edit", "command": "rm -rf /"}])
    assert len(asked) == 1
    assert state["steps"][0] == {"command": "rm -rf /", "outcome": "blocked", "note": state["steps"][0]["note"]}
    assert not (tmp_path / "made").exists()


def test_edit_is_checked_again_danger(settings, tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("x")
    decider = ScriptedDecider(run("mkdir made"), answer("ok"))
    _, asked = drive(
        decider, settings, tmp_path,
        replies=[{"action": "edit", "command": f"rm {victim}"}, {"action": "skip"}],
    )
    assert [a["risk"] for a in asked] == ["caution", "danger"]
    assert victim.exists()


def test_step_limit(settings, tmp_path):
    decider = ScriptedDecider(*(run(f"echo step-{i}") for i in range(10)))
    state, _ = drive(decider, settings, tmp_path)
    assert state["status"] == "step_limit"
    assert len(state["steps"]) == settings.max_steps
    assert len(decider.calls) == settings.max_steps


def test_model_error(settings, tmp_path):
    state, _ = drive(ScriptedDecider(ConnectionError("refused")), settings, tmp_path)
    assert state["status"] == "model_error" and "refused" in state["error"]


def test_dry_run_runs_nothing(settings, tmp_path):
    decider = ScriptedDecider(run("mkdir made"))
    state, asked = drive(decider, settings, tmp_path, dry_run=True)
    assert state["status"] == "dry_run" and state["risk"] == "caution"
    assert asked == [] and not (tmp_path / "made").exists()


def test_wrapped_command_is_cleaned(settings, tmp_path):
    decider = ScriptedDecider(run("$(echo cleaned)"), answer("ok"))
    state, _ = drive(decider, settings, tmp_path)
    assert state["steps"][0]["command"] == "echo cleaned"


def test_no_command_means_answer(settings, tmp_path):
    from shellmate.agent import Decision

    decider = ScriptedDecider(Decision(action="run", command="", explanation="hello! ask me about your files"))
    state, _ = drive(decider, settings, tmp_path)
    assert state["status"] == "answered" and "hello" in state["answer"]


def test_repeated_command_gets_a_nudge_then_an_answer(settings, tmp_path):
    decider = ScriptedDecider(run("echo msg"), run("echo msg"), answer("The message is msg."))
    state, _ = drive(decider, settings, tmp_path)
    assert len(state["steps"]) == 1  # the repeat never ran
    assert "already ran `echo msg`" in decider.last_message()
    assert state["answer"] == "The message is msg."


def test_model_that_keeps_repeating_gets_the_output_shown(settings, tmp_path):
    decider = ScriptedDecider(run("echo agent run command."), run("echo agent run command."))
    state, _ = drive(decider, settings, tmp_path)
    assert state["status"] == "answered"
    assert "agent run command." in state["answer"] and len(state["steps"]) == 1


def test_command_given_as_answer_gets_a_nudge(settings, tmp_path):
    decider = ScriptedDecider(run("echo msg"), answer("echo msg"), answer("It says msg."))
    state, _ = drive(decider, settings, tmp_path)
    assert "was a command, not an answer" in decider.last_message()
    assert state["answer"] == "It says msg."


def test_latest_output_carries_the_next_step_hint(settings, tmp_path):
    decider = ScriptedDecider(run("echo one"), answer("ok"))
    drive(decider, settings, tmp_path)
    assert 'reply with action "answer"' in decider.last_message()
