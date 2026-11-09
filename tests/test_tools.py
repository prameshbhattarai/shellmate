import time

from shellmate.tools import TIMEOUT_EXIT_CODE, run_command


def test_captures_output_and_exit_code(tmp_path):
    result = run_command("echo hi; echo oops >&2; exit 3", cwd=str(tmp_path), timeout_s=5, max_output_chars=100)
    assert (result.exit_code, result.stdout, result.stderr.strip()) == (3, "hi\n", "oops")


def test_runs_in_cwd(tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    assert "marker.txt" in run_command("ls", cwd=str(tmp_path), timeout_s=5, max_output_chars=100).stdout


def test_truncates(tmp_path):
    result = run_command("printf 'a%.0s' $(seq 1 50)", cwd=str(tmp_path), timeout_s=5, max_output_chars=10)
    assert result.truncated
    assert result.stdout.startswith("a" * 10) and "truncated 40 chars" in result.stdout


def test_times_out_without_raising(tmp_path):
    result = run_command("sleep 5", cwd=str(tmp_path), timeout_s=1, max_output_chars=100)
    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert "timed out" in result.stderr


def test_commands_get_no_keyboard_input(tmp_path):
    result = run_command("cat", cwd=str(tmp_path), timeout_s=5, max_output_chars=100)
    assert result.exit_code == 0  # would hang until the timeout if stdin were the terminal


def test_binary_output_does_not_raise(tmp_path):
    result = run_command("printf '\\377\\376ok'", cwd=str(tmp_path), timeout_s=5, max_output_chars=100)
    assert result.exit_code == 0 and result.stdout.endswith("ok")


def test_timeout_kills_the_whole_pipeline(tmp_path):
    marker = tmp_path / "still-running"
    result = run_command(f"(sleep 1.5; touch {marker}) | cat", cwd=str(tmp_path), timeout_s=0.5, max_output_chars=100)
    assert result.exit_code == TIMEOUT_EXIT_CODE
    time.sleep(1.5)
    assert not marker.exists(), "a pipeline member kept running after the timeout"


def test_missing_shell_falls_back_to_sh(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELL", "/no/such/shell")
    assert run_command("echo ok", cwd=str(tmp_path), timeout_s=5, max_output_chars=100).stdout == "ok\n"


# --- read_file / list_dir ------------------------------------------------------------------------

import os  # noqa: E402

import pytest  # noqa: E402

from shellmate.tools import ToolError, list_dir, read_file  # noqa: E402


def test_read_file_inside(tmp_path):
    (tmp_path / "notes.txt").write_text("hello\n")
    assert read_file("notes.txt", cwd=str(tmp_path), max_chars=100) == "hello\n"


def test_read_file_truncates(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 50)
    text = read_file("big.txt", cwd=str(tmp_path), max_chars=10)
    assert text.startswith("x" * 10) and "50 bytes in all" in text


@pytest.mark.parametrize("path", ["../secret.txt", "/etc/hosts", "~/.zshrc"])
def test_read_file_refuses_outside_the_folder(tmp_path, path):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "secret.txt").write_text("no")
    with pytest.raises(ToolError, match="outside the current folder"):
        read_file(path, cwd=str(work), max_chars=100)


def test_read_file_refuses_symlink_escape(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "secret.txt").write_text("no")
    os.symlink(tmp_path / "secret.txt", work / "link.txt")
    with pytest.raises(ToolError, match="outside"):
        read_file("link.txt", cwd=str(work), max_chars=100)


def test_read_file_refuses_binary_and_folders(tmp_path):
    (tmp_path / "img.bin").write_bytes(b"\x89PNG\x00\x00")
    (tmp_path / "sub").mkdir()
    with pytest.raises(ToolError, match="binary"):
        read_file("img.bin", cwd=str(tmp_path), max_chars=100)
    with pytest.raises(ToolError, match="folder"):
        read_file("sub", cwd=str(tmp_path), max_chars=100)
    with pytest.raises(ToolError, match="does not exist"):
        read_file("nope.txt", cwd=str(tmp_path), max_chars=100)


def test_list_dir_puts_folders_first(tmp_path):
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a-dir").mkdir()
    (tmp_path / "c.txt").write_text("")
    assert list_dir(".", cwd=str(tmp_path)) == ["a-dir/", "b.txt", "c.txt"]
    with pytest.raises(ToolError, match="outside"):
        list_dir("..", cwd=str(tmp_path))


def test_missing_working_folder_does_not_raise(tmp_path):
    result = run_command("ls", cwd=str(tmp_path / "gone"), timeout_s=5, max_output_chars=100)
    assert result.exit_code == 127 and "could not start" in result.stderr
