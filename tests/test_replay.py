import json

import pytest
from typer.testing import CliRunner

from shellmate import cli
from shellmate.agent import build_decider
from shellmate.replay import ReplayDecider


def _script(tmp_path, items):
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(items))
    return path


def test_replay_returns_decisions_in_order_then_stops(tmp_path):
    decider = ReplayDecider.from_file(_script(tmp_path, [{"action": "run", "command": "ls"}, {"action": "answer", "answer": "done"}]))
    assert decider.invoke([]).command == "ls"
    assert decider.invoke([]).answer == "done"
    with pytest.raises(RuntimeError, match="no more answers"):
        decider.invoke([])


def test_env_switches_build_decider_to_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELLMATE_REPLAY", str(_script(tmp_path, [{"action": "answer", "answer": "hi"}])))
    assert isinstance(build_decider("qwen2.5:3b"), ReplayDecider)


def test_bad_script_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ReplayDecider.from_file(_script(tmp_path, {"action": "run"}))


def test_replayed_repl_session_is_labelled_and_runs_for_real(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "marker.txt").write_text("x")
    monkeypatch.setenv("SHELLMATE_REPLAY", str(_script(tmp_path, [
        {"action": "run", "command": "ls", "explanation": "list"},
        {"action": "answer", "answer": "there is a marker file"},
    ])))
    result = CliRunner().invoke(cli.app, [], input="what is here?\nexit\n")
    assert "model=replay (scripted answers)" in result.output
    assert "marker.txt" in result.output and "mate> there is a marker file" in result.output
