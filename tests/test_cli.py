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

EXPECTED_COMMANDS = [
    "mine",
    "dataset-stats",
    "samples",
    "index",
    "index-stats",
    "retrieve",
    "eval",
    "config-show",
]

# Every command is implemented as of Milestone 3, so nothing is exercised here.
# Kept as an explicit empty list rather than deleting the test: invoking a real
# command from a unit test would clone repos, load a model, write to Postgres,
# and — for `eval` — append a bogus entry to the held-out peek ledger.
UNIMPLEMENTED: list[str] = []


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
