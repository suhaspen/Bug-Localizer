"""Command line entry point.

Every command a reader might want to run is registered here from Milestone 0,
even the ones not built yet -- an unimplemented command exits with the milestone
that will deliver it, which is friendlier than a missing-command error and keeps
`make help` honest about the shape of the finished system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from buglocalizer import __version__
from buglocalizer.config import Config, load_config
from buglocalizer.logging_setup import configure_logging

app = typer.Typer(
    name="bugloc",
    help="Bug localization: rank the source files most likely responsible for a bug.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="Path to config YAML.", show_default=True)
]


def _load(config_path: Path) -> Config:
    cfg = load_config(config_path)
    configure_logging(cfg.logging.level)
    return cfg


def _not_yet(milestone: str, what: str) -> None:
    console.print(f"[yellow]not implemented yet[/] — {what} lands in {milestone}.")
    raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


@app.command("config-show")
def config_show(config: ConfigOpt = Path("config.yaml")) -> None:
    """Load and pretty-print the resolved configuration.

    Useful for confirming that config.local.yaml / BUGLOC_* overrides landed
    where you think they did before kicking off a long run.
    """
    cfg = _load(config)
    console.print_json(cfg.model_dump_json(indent=2))


@app.command()
def mine(config: ConfigOpt = Path("config.yaml")) -> None:
    """Mine fix commits from the target repos into a labeled examples.jsonl."""
    _load(config)
    _not_yet("Milestone 1", "history mining")


@app.command("dataset-stats")
def dataset_stats(config: ConfigOpt = Path("config.yaml")) -> None:
    """Print dataset statistics: examples per repo, gold-file distribution."""
    _load(config)
    _not_yet("Milestone 1", "dataset statistics")


@app.command()
def index(config: ConfigOpt = Path("config.yaml")) -> None:
    """Build the BM25 and pgvector indexes over each example's parent commit."""
    _load(config)
    _not_yet("Milestone 2", "corpus indexing")


@app.command()
def retrieve(config: ConfigOpt = Path("config.yaml")) -> None:
    """Rank candidate files for a single example (BM25, dense, or hybrid)."""
    _load(config)
    _not_yet("Milestone 2", "retrieval")


@app.command("eval")
def eval_cmd(config: ConfigOpt = Path("config.yaml")) -> None:
    """Run the full evaluation and print the comparison table."""
    _load(config)
    _not_yet("Milestone 3", "the evaluation harness")


if __name__ == "__main__":
    app()
