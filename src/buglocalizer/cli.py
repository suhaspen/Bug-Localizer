"""Command line entry point.

Every command a reader might want to run is registered here from Milestone 0,
even the ones not built yet -- an unimplemented command exits with the milestone
that will deliver it, which is friendlier than a missing-command error and keeps
`make help` honest about the shape of the finished system.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from buglocalizer import __version__
from buglocalizer.config import Config, load_config
from buglocalizer.dataset import (
    HELDOUT,
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
from buglocalizer.reporting import render_eval, render_samples, render_stats, run_retrieval_demo

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


def _select(
    cfg: Config,
    split: str | None,
    repo: list[str] | None,
    limit: int | None,
    seed: int,
    newest: bool = False,
):
    """Load examples and apply the standard --split/--repo/--limit selection.

    `newest` takes the most recent N by date instead of a random sample. That
    matters for indexing: the blob cache pays off when consecutive commits are
    close in time and share nearly their whole tree. A random sample spread over
    five years shares almost nothing between neighbours, so it costs far more to
    index than a contiguous slice of the same size.
    """
    examples = read_jsonl(examples_path(cfg.paths.data_dir))
    if split:
        examples = [e for e in examples if e.split == split]
    if repo:
        examples = [e for e in examples if e.repo in set(repo)]
    if limit and limit < len(examples):
        if newest:
            examples = sorted(examples, key=lambda e: (e.authored_at, e.fix_sha))[-limit:]
        else:
            # Sample rather than truncate: examples are stored grouped by repo,
            # so head-truncation would silently drop entire repositories.
            examples = random.Random(seed).sample(examples, limit)
    return examples


@app.command()
def index(
    config: ConfigOpt = Path("config.yaml"),
    split: Annotated[
        str | None, typer.Option("--split", help="Only index 'dev' or 'heldout'.")
    ] = None,
    repo: Annotated[list[str] | None, typer.Option("--repo", "-r")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Index only N examples.")] = None,
    newest: Annotated[
        bool,
        typer.Option(
            "--newest",
            help="With --limit, take the newest N by date instead of a random sample. "
            "Much cheaper: neighbouring commits share blobs, a spread-out sample does not.",
        ),
    ] = False,
    include_tests: Annotated[
        bool,
        typer.Option(
            "--include-tests",
            help="Index the WIDE corpus scope (tests included). This is a superset of "
            "the default scope, so one wide build serves evaluation at both scopes.",
        ),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-index covered commits.")] = False,
) -> None:
    """Embed each example's parent commit into Postgres/pgvector."""
    # Imported here, not at module scope: pulling in torch costs ~3s, and
    # `bugloc mine` must not pay for it.
    from buglocalizer.indexing.indexer import index_examples
    from buglocalizer.indexing.store import init_schema

    cfg = _load(config)
    if include_tests:
        cfg.corpus.include_tests = True
    init_schema(cfg)
    examples = _select(cfg, split, repo, limit, cfg.seed, newest=newest)
    if not examples:
        console.print("[red]no examples matched[/]")
        raise typer.Exit(code=1)

    console.print(f"indexing parent commits for [bold]{len(examples):,}[/] examples")
    stats = index_examples(cfg, examples, force=force)

    console.print(
        f"\n[green]indexed[/] {stats.commits_seen - stats.commits_skipped:,} commits "
        f"({stats.commits_skipped:,} already covered)\n"
        f"  file instances seen : {stats.file_instances:,}\n"
        f"  blobs embedded      : {stats.blobs_new:,}  "
        f"([bold]{stats.dedup_factor:.1f}x[/] fewer than file instances)\n"
        f"  blob cache hits     : {stats.blobs_cached:,}\n"
        f"  chunks embedded     : {stats.chunks_embedded:,}\n"
        f"  wall time           : {stats.seconds / 60:.1f} min"
    )


@app.command("index-stats")
def index_stats_cmd(config: ConfigOpt = Path("config.yaml")) -> None:
    """Show what is currently in the corpus index."""
    from buglocalizer.indexing.store import connect, index_stats

    cfg = _load(config)
    with connect(cfg) as conn:
        rows = index_stats(conn)
    if not rows:
        console.print("index is empty — run [bold]bugloc index[/] first")
        return
    table = Table(title="Corpus index", title_justify="left")
    for col in ("repo", "commits", "blobs", "chunks", "content"):
        table.add_column(col, justify="right" if col != "repo" else "left")
    for repo_name, blobs, nbytes, chunks, commits in rows:
        table.add_row(
            repo_name, f"{commits:,}", f"{blobs:,}", f"{chunks:,}", f"{nbytes / 1e6:.0f} MB"
        )
    console.print(table)


@app.command()
def retrieve(
    config: ConfigOpt = Path("config.yaml"),
    example_id: Annotated[
        str | None, typer.Option("--example", "-e", help="Example id; random if omitted.")
    ] = None,
    method: Annotated[str, typer.Option("--method", "-m", help="bm25 | dense | both")] = "both",
    k: Annotated[int, typer.Option("-k", help="How many files to show.")] = 10,
    repo: Annotated[str | None, typer.Option("--repo", "-r", help="Restrict to one repo.")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
) -> None:
    """Rank candidate files for one example, and show whether the gold file was found."""
    from buglocalizer.indexing.store import connect

    cfg = _load(config)
    examples = read_jsonl(examples_path(cfg.paths.data_dir))

    if example_id:
        matches = [e for e in examples if e.example_id == example_id]
        if not matches:
            console.print(f"[red]no example with id {example_id!r}[/]")
            raise typer.Exit(code=1)
        example = matches[0]
    else:
        with connect(cfg) as conn:
            covered = {
                (r[0], r[1])
                for r in conn.execute("SELECT repo, commit_sha FROM indexed_commit").fetchall()
            }
        pool = [e for e in examples if (e.repo, e.parent_sha) in covered]
        if repo:
            pool = [e for e in pool if e.repo == repo]
        if not pool:
            console.print("[red]no indexed examples — run `bugloc index` first[/]")
            raise typer.Exit(code=1)
        example = random.Random(seed if seed is not None else cfg.seed).choice(pool)

    run_retrieval_demo(console, cfg, example, method=method, k=k)


@app.command("eval")
def eval_cmd(
    config: ConfigOpt = Path("config.yaml"),
    split: Annotated[str, typer.Option("--split", help="'heldout' or 'dev'.")] = "heldout",
    repo: Annotated[list[str] | None, typer.Option("--repo", "-r")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    scopes: Annotated[
        str,
        typer.Option(
            "--scopes",
            help="'both' (default), 'narrow' (tests excluded), or 'wide' (tests included).",
        ),
    ] = "both",
    label: Annotated[str, typer.Option("--label", help="Note recorded with the run.")] = "",
) -> None:
    """Evaluate BM25 vs dense vs hybrid and write results/<timestamp>.json."""
    from buglocalizer.eval.harness import evaluate
    from buglocalizer.eval.results import (
        peek_count,
        record_heldout_peek,
        render_markdown,
        run_to_dict,
        save_results,
    )
    from buglocalizer.indexing.embedder import get_embedder
    from buglocalizer.indexing.store import connect

    cfg = _load(config)
    examples = _select(cfg, split, repo, limit, cfg.seed)
    if not examples:
        console.print(f"[red]no examples in split {split!r}[/]")
        raise typer.Exit(code=1)

    wanted = {"both": [False, True], "narrow": [False], "wide": [True]}.get(scopes)
    if wanted is None:
        console.print("[red]--scopes must be one of: both, narrow, wide[/]")
        raise typer.Exit(code=1)

    if split == HELDOUT:
        console.print(
            f"[yellow]held-out evaluation[/] — this will be peek "
            f"#{peek_count(cfg) + 1} in {cfg.paths.results_dir}/heldout_log.jsonl"
        )

    from buglocalizer.indexing.store import covered_commits

    embedder = get_embedder(cfg)
    payloads: list[dict] = []
    with connect(cfg) as conn:
        # Comparing corpus scopes is only meaningful if both scopes score the
        # SAME examples. The wide scope is more expensive to index, so it always
        # covers fewer commits; without this intersection the two columns would
        # be computed over different example sets and the delta between them
        # would conflate scope with sample.
        if len(wanted) > 1:
            common = set.intersection(*(covered_commits(conn, s) for s in wanted))
            before = len(examples)
            examples = [e for e in examples if (e.repo, e.parent_sha) in common]
            if len(examples) < before:
                console.print(
                    f"[yellow]restricted to {len(examples):,} of {before:,} examples[/] "
                    f"covered at every requested scope, so the scopes are comparable"
                )
            if not examples:
                console.print("[red]no examples are indexed at all requested scopes[/]")
                raise typer.Exit(code=1)

        for include_tests in wanted:
            cfg.corpus.include_tests = include_tests
            scope_name = "tests included" if include_tests else "tests excluded"
            console.print(f"\n[bold]evaluating {len(examples):,} examples — {scope_name}[/]")
            run = evaluate(cfg, examples, conn, embedder)
            if run.n_examples == 0:
                console.print(
                    f"[red]nothing evaluated at scope '{scope_name}' — "
                    f"{run.skipped_unindexed:,} examples had no index coverage. "
                    f"Run `bugloc index"
                    f"{' --include-tests' if include_tests else ''}` first.[/]"
                )
                raise typer.Exit(code=1)
            payloads.append(run_to_dict(cfg, run, split, label))
            render_eval(console, payloads[-1])

    for payload in payloads:
        path = save_results(cfg, payload)
        console.print(f"[green]wrote[/] {path}")
        if split == HELDOUT:
            n = record_heldout_peek(cfg, payload, path)
            console.print(f"  held-out peek count is now [bold]{n}[/]")

    md = cfg.paths.results_dir / "latest.md"
    md.write_text(render_markdown(payloads))
    console.print(f"[green]wrote[/] {md}")


if __name__ == "__main__":
    app()
