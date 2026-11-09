from shellmate.memory import (
    HISTORY_MAX_ENTRIES,
    History,
    clear_pending,
    forget,
    load_pending,
    load_profile,
    remember,
    save_pending,
)


def _ran(command, exit_code=0, cwd="/w", **extra):
    return {"command": command, "exit_code": exit_code, "cwd": cwd, "risk": "safe", "stdout": "", "stderr": "", **extra}


def test_history_round_trip(isolated_home):
    h = History(isolated_home)
    first = h.append(_ran("ls"))
    h.append(_ran("cat nope", exit_code=1, stderr="No such file"))
    assert first["id"] and [e["command"] for e in h.recent()] == ["ls", "cat nope"]
    assert h.last_error()["stderr"] == "No such file"


def test_history_filters_by_folder(isolated_home):
    h = History(isolated_home)
    h.append(_ran("ls", cwd="/a"))
    h.append(_ran("pwd", cwd="/b"))
    assert [e["command"] for e in h.recent(cwd="/b")] == ["pwd"]
    assert h.last_error(cwd="/a") is None


def test_history_survives_a_damaged_line(isolated_home):
    h = History(isolated_home)
    h.append(_ran("ls"))
    with h.path.open("a") as f:
        f.write("{not json\n")
    h.append(_ran("pwd"))
    assert [e["command"] for e in h.entries()] == ["ls", "pwd"]


def test_history_keeps_output_tails_and_trims(isolated_home):
    h = History(isolated_home)
    entry = h.append(_ran("big", stdout="x" * 10_000))
    assert len(entry["stdout"]) < 3000
    for i in range(2 * HISTORY_MAX_ENTRIES + 1):
        h.append(_ran(f"echo {i}"))
    assert len(h.entries()) <= HISTORY_MAX_ENTRIES + 1


def test_missing_files_are_empty(isolated_home):
    assert History(isolated_home).entries() == []
    assert load_profile(isolated_home).notes == []
    assert load_pending(isolated_home) is None


def test_remember_and_forget(isolated_home):
    remember(isolated_home, "my projects live in ~/code")
    remember(isolated_home, "my projects live in ~/code")  # no duplicates
    remember(isolated_home, "I use pnpm, not npm")
    assert load_profile(isolated_home).notes == ["my projects live in ~/code", "I use pnpm, not npm"]
    assert forget(isolated_home, 1) == ["my projects live in ~/code"]
    assert forget(isolated_home, 9) == []
    assert forget(isolated_home) == ["I use pnpm, not npm"]


def test_damaged_profile_is_ignored(isolated_home):
    isolated_home.mkdir(parents=True)
    (isolated_home / "profile.json").write_text("[1, 2")
    assert load_profile(isolated_home).notes == []


def test_pending_round_trip(isolated_home):
    save_pending(isolated_home, thread_id="t1", request="delete build", cwd="/w")
    assert load_pending(isolated_home)["thread_id"] == "t1"
    clear_pending(isolated_home, "other")  # only clears the matching run
    assert load_pending(isolated_home) is not None
    clear_pending(isolated_home, "t1")
    assert load_pending(isolated_home) is None


def test_files_are_private(isolated_home):
    import stat

    from shellmate.memory import checkpointer

    History(isolated_home).append(_ran("cat .env", stdout="API_KEY=secret"))
    remember(isolated_home, "note")
    checkpointer(isolated_home)
    mode = lambda p: stat.S_IMODE(p.stat().st_mode)  # noqa: E731
    assert mode(isolated_home) == 0o700
    for name in ("history.jsonl", "profile.json", "checkpoints.sqlite"):
        assert mode(isolated_home / name) == 0o600, name


def test_recent_with_zero_or_negative_count_is_empty(isolated_home):
    h = History(isolated_home)
    h.append(_ran("ls"))
    assert h.recent(0) == [] and h.recent(-3) == []


def test_folders_match_through_symlinks(isolated_home, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    h = History(isolated_home)
    h.append(_ran("cat nope", exit_code=1, cwd=str(tmp_path / "link")))
    assert h.last_error(cwd=str(real)) is not None
    assert len(h.recent(cwd=str(real))) == 1
