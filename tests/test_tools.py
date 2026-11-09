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
