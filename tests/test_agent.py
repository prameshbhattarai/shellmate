import pytest

from shellmate.agent import OLD_OUTPUT_CHARS, build_messages, clean_command


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$(git diff $(git rev-parse HEAD))", "git diff $(git rev-parse HEAD)"),
        ("`ls -la`", "ls -la"),
        ("$ du -sh *", "du -sh *"),
        ("```bash\ngit show --stat HEAD\n```", "git show --stat HEAD"),
        ("$($(ls))", "ls"),
        ("  ls -la  ", "ls -la"),
        # left alone: substitution is only part of the command
        ("echo $(date)", "echo $(date)"),
        ("$(a) | $(b)", "$(a) | $(b)"),
    ],
)
def test_clean_command(raw, expected):
    assert clean_command(raw) == expected


def _ran(command, stdout="", stderr="", exit_code=0):
    return {"command": command, "outcome": "ran", "exit_code": exit_code, "stdout": stdout, "stderr": stderr}


def test_first_turn_is_system_and_request(tmp_path):
    messages = build_messages("list files", [], cwd=str(tmp_path))
    (role0, system), (role1, human) = messages
    assert (role0, role1) == ("system", "human")
    assert str(tmp_path) in system and human == "list files"


def test_each_step_is_reported_back():
    steps = [_ran("ls", stdout="a.txt\n"), _ran("cat nope", stderr="No such file", exit_code=1)]
    messages = build_messages("x", steps, cwd="/w")
    assert messages[2] == ("ai", "run: ls")
    assert "a.txt" in messages[3][1]
    assert "exit code 1" in messages[5][1] and "No such file" in messages[5][1]


def test_old_outputs_are_cut_but_recent_ones_are_not():
    big = "x" * (OLD_OUTPUT_CHARS * 3)
    steps = [_ran("one", stdout=big), _ran("two", stdout=big), _ran("three", stdout=big)]
    messages = build_messages("x", steps, cwd="/w")
    assert "more chars" in messages[3][1]  # oldest step
    assert "more chars" not in messages[5][1] and "more chars" not in messages[7][1]


def test_refusals_are_reported():
    steps = [{"command": "rm -rf ~", "outcome": "blocked", "note": "would wreck your home folder"}]
    assert "refused" in build_messages("x", steps, cwd="/w")[-1][1]


def test_answer_is_a_command():
    from shellmate.agent import answer_is_a_command

    steps = [_ran("git log -1 --format=%B")]
    assert answer_is_a_command("git log -1 --format=%B", steps)
    assert answer_is_a_command("`git log -1 --format=%B`", steps)
    assert not answer_is_a_command("Your last commit was 'agent skeleton.'", steps)


def test_placeholders():
    from shellmate.agent import answer_has_placeholders

    assert answer_has_placeholders("The process is [comm] with [percent] percent")
    assert not answer_has_placeholders("IntelliJ uses 8.1% [1.3 GB]")
    assert not answer_has_placeholders("Your last commit was 'agent skeleton.'")


import pytest  # noqa: E402

from shellmate import agent  # noqa: E402


@pytest.mark.parametrize(
    "system, name, present, absent",
    [
        ("Darwin", "macOS", "ps -Ao pid,%mem,rss,comm -m", "--sort=-%mem"),
        ("Linux", "Linux", "ps aux --sort=-%mem", "vm_stat"),
        ("Windows", "Windows", "tasklist", "--sort=-%mem"),
        ("MINGW64_NT-10.0", "Windows", "Git Bash", "ps -Ao"),
    ],
)
def test_only_this_os_rules_reach_the_prompt(monkeypatch, system, name, present, absent):
    monkeypatch.setattr(agent.platform, "system", lambda: system)
    prompt = agent.system_prompt("/w")
    assert f"Environment: {name}" in prompt
    assert present in prompt and absent not in prompt


def test_unknown_os_gets_no_rules(monkeypatch):
    monkeypatch.setattr(agent.platform, "system", lambda: "FreeBSD")
    prompt = agent.system_prompt("/w")
    assert "Environment: FreeBSD" in prompt
    assert not any(rules.strip() in prompt for rules in agent.OS_RULES.values())


def test_tool_actions_and_the_schema_come_from_one_list():
    from typing import get_args

    from shellmate.agent import TOOL_ACTIONS, Action

    assert set(TOOL_ACTIONS) <= set(get_args(Action))
    assert set(get_args(Action)) == {"run", "answer", *TOOL_ACTIONS}
