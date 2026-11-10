import stat

from shellmate.memory import checkpointer, clear_pending, load_pending, save_pending


def test_missing_pending_is_none(isolated_home):
    assert load_pending(isolated_home) is None


def test_pending_round_trip(isolated_home):
    save_pending(isolated_home, thread_id="t1", request="delete build", cwd="/w")
    assert load_pending(isolated_home)["thread_id"] == "t1"
    clear_pending(isolated_home, "other")  # only clears the matching run
    assert load_pending(isolated_home) is not None
    clear_pending(isolated_home, "t1")
    assert load_pending(isolated_home) is None


def test_damaged_pending_file_is_ignored(isolated_home):
    isolated_home.mkdir(parents=True)
    (isolated_home / "pending.json").write_text("{not json")
    assert load_pending(isolated_home) is None


def test_files_are_private(isolated_home):
    save_pending(isolated_home, thread_id="t1", request="x", cwd="/w")
    checkpointer(isolated_home)
    mode = lambda p: stat.S_IMODE(p.stat().st_mode)  # noqa: E731
    assert mode(isolated_home) == 0o700
    for name in ("pending.json", "checkpoints.sqlite"):
        assert mode(isolated_home / name) == 0o600, name


def test_old_history_file_is_removed(isolated_home):
    isolated_home.mkdir(parents=True)
    (isolated_home / "history.jsonl").write_text('{"command": "cat .env", "stdout": "API_KEY=secret"}\n')
    (isolated_home / "profile.json").write_text('{"notes": ["keep me"]}')
    checkpointer(isolated_home)
    assert not (isolated_home / "history.jsonl").exists()
    assert (isolated_home / "profile.json").exists()  # harmless notes are left alone
