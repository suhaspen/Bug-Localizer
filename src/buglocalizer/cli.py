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
from buglocalizer.dataset import (
    Example,
    assign_temporal_split,
    examples_path,
    load_funnel,
    read_jsonl,
    save_funnel,
    write_jsonl,
)
from buglocalizer.logging_setup import configure_logging
from buglocalizer.mining import ensure_repo, mine_repo
from buglocalizer.reporting import render_samples, render_stats

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
def mine(
    config: ConfigOpt = Path("config.yaml"),
    repo: Annotated[
        list[str] | None, typer.Option("--repo", "-r", help="Mine only these repos.")
    ] = None,
    no_fetch: Annotated[
        bool, typer.Option("--no-fetch", help="Skip `git fetch`; mine the cached clone as-is.")
    ] = False,
) -> None:
    """Mine fix commits from the target repos into a labeled examples.jsonl."""
    cfg = _load(config)
    targets = [cfg.repo(name) for name in repo] if repo else cfg.repos
    if not targets:
        console.print("[red]no repos configured[/]")
        raise typer.Exit(code=1)

    all_examples: list[Example] = []
    funnel: dict[str, dict] = {}
    for repo_cfg in targets:
        repo_path = ensure_repo(repo_cfg, cfg.paths.cache_dir, fetch=not no_fetch)
        examples, counts = mine_repo(repo_cfg, cfg, repo_path)
        all_examples.extend(examples)
        funnel[repo_cfg.name] = dict(counts)

    all_examples = assign_temporal_split(all_examples, cfg.split.dev_fraction)

    out = examples_path(cfg.paths.data_dir)
    write_jsonl(out, all_examples)
    save_funnel(cfg.paths.data_dir, funnel)

    console.print(
        f"\n[green]wrote {len(all_examples):,} examples[/] → {out}\n"
        f"run [bold]bugloc dataset-stats[/] for the breakdown, "
        f"[bold]bugloc samples[/] to eyeball labels"
    )


@app.command("dataset-stats")
def dataset_stats(config: ConfigOpt = Path("config.yaml")) -> None:
    """Print dataset statistics: examples per repo, split sizes, gold-file distribution."""
    cfg = _load(config)
    examples = read_jsonl(examples_path(cfg.paths.data_dir))
    render_stats(console, examples, load_funnel(cfg.paths.data_dir))


@app.command()
def samples(
    config: ConfigOpt = Path("config.yaml"),
    n: Annotated[int, typer.Option("-n", help="Random examples to print.")] = 6,
    borderline: Annotated[
        int, typer.Option("--borderline", "-b", help="Additional borderline examples.")
    ] = 2,
    repo: Annotated[str | None, typer.Option("--repo", "-r", help="Restrict to one repo.")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Sampling seed.")] = None,
) -> None:
    """Print full examples (query, gold files, SHAs) for eyeballing label quality."""
    cfg = _load(config)
    examples = read_jsonl(examples_path(cfg.paths.data_dir))
    if repo:
        examples = [e for e in examples if e.repo == repo]
        if not examples:
            console.print(f"[red]no examples for repo {repo!r}[/]")
            raise typer.Exit(code=1)
    render_samples(console, examples, n=n, n_borderline=borderline, seed=seed or cfg.seed)


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
