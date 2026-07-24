"""Smoke tests for the CLI surface.

These do not test behaviour that does not exist yet; they check that the command
surface is wired up, that `--help` works, and that unimplemented commands fail
loudly with a milestone pointer rather than a stack trace.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from buglocalizer.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()

EXPECTED_COMMANDS = ["mine", "dataset-stats", "samples", "index", "retrieve", "eval", "config-show"]

# Commands still stubbed out. `mine` and `dataset-stats` are deliberately absent:
# they are implemented now, and invoking them here would clone repos and mine
# real history from a unit test.
UNIMPLEMENTED = ["index", "retrieve", "eval"]


def test_help_lists_every_planned_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in EXPECTED_COMMANDS:
        assert command in result.stdout


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_config_show_loads_shipped_config():
    result = runner.invoke(app, ["config-show", "--config", str(REPO_ROOT / "config.yaml")])
    assert result.exit_code == 0
    assert "flask" in result.stdout


@pytest.mark.parametrize("command", UNIMPLEMENTED)
def test_unimplemented_commands_exit_with_milestone_pointer(command):
    result = runner.invoke(app, [command, "--config", str(REPO_ROOT / "config.yaml")])
    assert result.exit_code == 2
    assert "Milestone" in result.stdout
