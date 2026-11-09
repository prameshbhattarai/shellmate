import pytest

from shellmate.undo import inverse, last_undoable


@pytest.mark.parametrize(
    "command, expected",
    [
        ("mkdir notes", "rmdir notes"),
        ("mkdir -p a b", "rmdir b a"),
        ("git add .", "git restore --staged -- ."),
        ("git add src/app.py", "git restore --staged -- src/app.py"),
        ("git commit -m 'wip'", "git reset --soft HEAD~1"),
        ("git switch main", "git switch -"),
        ("git stash", "git stash pop"),
        ("git branch feature-x", "git branch -d feature-x"),
        ("mv a.txt b.txt", "mv b.txt a.txt"),
        ("ln -s target link", "rm link"),
    ],
)
def test_inverse(tmp_path, command, expected):
    assert inverse(command, cwd=str(tmp_path))[0] == expected


def test_mv_into_a_folder_moves_back_out(tmp_path):
    (tmp_path / "folder").mkdir()
    (tmp_path / "folder" / "a.txt").write_text("x")
    assert inverse("mv a.txt folder", cwd=str(tmp_path))[0] == "mv folder/a.txt a.txt"


@pytest.mark.parametrize(
    "command",
    ["rm notes.txt", "touch x", "git commit --amend", "git push", "mkdir a && cd a", "echo x > f", "cp a b",
     "git checkout -b new", "mv -f a b"],
)
def test_no_inverse_is_guessed(tmp_path, command):
    assert inverse(command, cwd=str(tmp_path)) is None


def _e(id, command, *, cwd="/w", exit_code=0, risk="caution", undoes=None):
    return {"id": id, "command": command, "cwd": cwd, "exit_code": exit_code, "risk": risk, "undoes": undoes}


def test_last_undoable_picks_the_newest_change_here():
    entries = [
        _e("1", "mkdir a"),
        _e("2", "mkdir b", cwd="/elsewhere"),
        _e("3", "ls", risk="safe"),
        _e("4", "mkdir c", exit_code=1),
    ]
    assert last_undoable(entries, cwd="/w")["id"] == "1"


def test_undone_changes_are_skipped():
    entries = [_e("1", "mkdir a"), _e("2", "mkdir b"), _e("3", "rmdir b", undoes="2")]
    assert last_undoable(entries, cwd="/w")["id"] == "1"


def test_checkout_of_a_file_is_not_undone_as_a_branch_switch(tmp_path):
    (tmp_path / "README.md").write_text("x")
    assert inverse("git checkout README.md", cwd=str(tmp_path)) is None


def test_undo_finds_entries_through_symlinks(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    entries = [_e("1", "mkdir a", cwd=str(tmp_path / "link"))]
    assert last_undoable(entries, cwd=str(real))["id"] == "1"


# --- git state guard -------------------------------------------------------------------------------

import subprocess  # noqa: E402

from shellmate.undo import changed_since, git_state  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "test")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "test@example.com")
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "first")
    return tmp_path, git


def test_git_undo_is_safe_when_nothing_moved(repo):
    path, git = repo
    git("commit", "-q", "--allow-empty", "-m", "by shellmate")
    entry = {"command": "git commit -m 'by shellmate'", **git_state(str(path))}
    assert changed_since(entry, str(path)) is None


def test_git_undo_refuses_after_a_manual_commit(repo):
    path, git = repo
    git("commit", "-q", "--allow-empty", "-m", "by shellmate")
    entry = {"command": "git commit -m 'by shellmate'", **git_state(str(path))}
    git("commit", "-q", "--allow-empty", "-m", "by hand")
    assert "head moved" in changed_since(entry, str(path))


def test_git_undo_refuses_without_a_recorded_state(repo):
    path, _ = repo
    assert "didn't record" in changed_since({"command": "git commit -m x"}, str(path))


def test_non_git_changes_need_no_state(tmp_path):
    assert changed_since({"command": "mkdir a"}, str(tmp_path)) is None
