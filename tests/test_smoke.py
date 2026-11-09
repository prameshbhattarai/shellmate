"""Smoke tests for the skeleton. These never call the model."""

import importlib

import pytest
from typer.testing import CliRunner

from shellmate import __version__
from shellmate.cli import app
from shellmate.risk import Risk, classify, needs_approval

MODULES = ["shellmate.cli", "shellmate.config", "shellmate.graph", "shellmate.memory", "shellmate.risk", "shellmate.tools"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_help_runs():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Talk to your terminal" in result.output


def test_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_repl_exits_on_exit():
    result = CliRunner().invoke(app, [], input="exit\n")
    assert result.exit_code == 0


def test_unknown_commands_default_to_danger_and_need_approval():
    verdict = classify("some-tool-nobody-knows --do-it")
    assert verdict.risk is Risk.DANGER
    assert needs_approval(verdict)
