"""The eval harness, with the scripted model. The real model is only used when you run it by hand."""

import sys
from pathlib import Path

from fakes import ScriptedDecider, answer, run, tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import run_evals  # noqa: E402


def test_eval_set_is_consistent():
    cases = run_evals.load_cases()
    assert len(cases) >= 50
    assert run_evals.check_cases(cases) == []


def test_check_catches_a_wrong_label_and_a_bad_regex():
    bad = [
        {"id": "a", "request": "x", "expect_action": "run", "expect": "^ls", "expect_risk": "danger", "reference": "ls"},
        {"id": "b", "request": "x", "expect_action": "run", "expect": "(", "expect_risk": "safe", "reference": "ls"},
        {"id": "c", "request": "x", "expect_action": "run", "expect": "^du", "expect_risk": "safe", "reference": "ls"},
        {"id": "c", "request": "x"},
    ]
    problems = run_evals.check_cases(bad)
    assert any("a: rules label" in p for p in problems)
    assert any("b: bad regex" in p for p in problems)
    assert any("c: reference" in p for p in problems)
    assert any("missing" in p for p in problems)


CASE = {"id": "disk", "request": "disk?", "expect_action": "run", "expect": r"^du\b", "expect_risk": "safe", "reference": "du -sh *"}


def test_score_good_and_bad_answers():
    assert run_evals.score_case(ScriptedDecider(run("du -sh * | sort -rh")), CASE)["passed"]
    wrong = run_evals.score_case(ScriptedDecider(run("ls -la")), CASE)
    assert not wrong["passed"] and wrong["action_ok"] and not wrong["match_ok"]
    risky = run_evals.score_case(ScriptedDecider(run("du -sh * > sizes.txt")), CASE)
    assert risky["match_ok"] and not risky["risk_ok"]


def test_score_tools_and_answers():
    read = {**CASE, "expect_action": "read_file|run", "expect": r"README\.md"}
    assert run_evals.score_case(ScriptedDecider(tool("read_file", "README.md")), read)["passed"]
    hello = {**CASE, "expect_action": "answer", "expect": ""}
    assert run_evals.score_case(ScriptedDecider(answer("hi!")), hello)["passed"]
    assert not run_evals.score_case(ScriptedDecider(run("echo hi")), hello)["passed"]


def test_model_errors_count_as_failures():
    result = run_evals.score_case(ScriptedDecider(ConnectionError("refused")), CASE)
    assert result["passed"] is False and "refused" in result["error"]


def test_main_scores_and_prints_a_table(capsys, tmp_path):
    out = tmp_path / "results.json"
    code = run_evals.main(
        ["--model", "fake-a,fake-b", "--case", "git-branch", "--case", "greeting", "--out", str(out)],
        decider_factory=lambda model: ScriptedDecider(run("git branch --show-current"), answer("hello!")),
    )
    printed = capsys.readouterr().out
    assert code == 0 and "| fake-a |" in printed and "| fake-b |" in printed
    assert out.exists()


def test_main_without_a_model_only_checks(capsys):
    assert run_evals.main([]) == 0
    assert "problems" in capsys.readouterr().out
