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
