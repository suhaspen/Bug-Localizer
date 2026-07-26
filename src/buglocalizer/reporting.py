"""Rendering for the dataset stats table and the example sampler.

Kept out of the CLI so the numbers can be computed and tested independently of
how they are printed.
"""

from __future__ import annotations

import random
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from buglocalizer.dataset import (
    GOLD_BUCKETS,
    HELDOUT,
    Example,
    RepoStats,
    borderline_counter,
    bucket_label,
    compute_repo_stats,
)
from buglocalizer.mining.filters import FUNNEL_ORDER

# A repo contributing fewer than this many held-out examples cannot support a
# per-repo accuracy claim, and blending it into the aggregate hides that.
MIN_USEFUL_HELDOUT = 30
MIN_HELDOUT_SHARE = 0.05


def _fmt_date(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "—"


def render_stats(console: Console, examples: list[Example], funnel: dict | None) -> None:
    repos = sorted({e.repo for e in examples})
    stats = [compute_repo_stats(examples, r) for r in repos]

    _render_funnel(console, funnel)
    _render_per_repo(console, stats, examples)
    _render_gold_distribution(console, stats)
    _render_borderline(console, examples)
    _render_warnings(console, stats)


def _render_funnel(console: Console, funnel: dict | None) -> None:
    if not funnel:
        return
    table = Table(title="Mining funnel — commits removed by each filter", title_justify="left")
    table.add_column("filter")
    for repo in funnel:
        table.add_column(repo, justify="right")

    scanned = {r: f.get("scanned", 0) for r, f in funnel.items()}
    table.add_row("commits scanned", *[f"{scanned[r]:,}" for r in funnel], style="dim")
    for reason in FUNNEL_ORDER:
        cells = []
        for repo in funnel:
            count = funnel[repo].get(reason, 0)
            pct = 100 * count / scanned[repo] if scanned[repo] else 0
            cells.append(f"{count:,} ({pct:.1f}%)" if count else "—")
        table.add_row(f"  − {reason}", *cells)
    table.add_row(
        "= examples kept",
        *[f"{funnel[r].get('kept', 0):,}" for r in funnel],
        style="bold green",
    )
    console.print(table)
    console.print()


def _render_per_repo(console: Console, stats: list[RepoStats], examples: list[Example]) -> None:
    table = Table(title="Dataset by repo", title_justify="left")
    table.add_column("repo")
    table.add_column("total", justify="right")
    table.add_column("dev", justify="right")
    table.add_column("held-out", justify="right")
    table.add_column("held-out %", justify="right")
    table.add_column("avg gold", justify="right")
    table.add_column("med", justify="right")
    table.add_column("max", justify="right")
    table.add_column("borderline", justify="right")
    table.add_column("date range", justify="center")
    table.add_column("held-out range", justify="center")

    total_heldout = sum(s.n_heldout for s in stats)

    for s in stats:
        share = s.n_heldout / total_heldout if total_heldout else 0
        low = s.n_heldout < MIN_USEFUL_HELDOUT or share < MIN_HELDOUT_SHARE
        table.add_row(
            s.repo,
            f"{s.total:,}",
            f"{s.n_dev:,}",
            f"{s.n_heldout:,}",
            f"{100 * share:.1f}%",
            f"{s.avg_gold_files:.2f}",
            f"{s.median_gold_files:.0f}",
            str(s.max_gold_files),
            f"{s.n_borderline:,}",
            f"{_fmt_date(s.earliest)} → {_fmt_date(s.latest)}",
            f"{_fmt_date(s.heldout_earliest)} → {_fmt_date(s.heldout_latest)}",
            style="yellow" if low else None,
        )

    all_gold = [len(e.gold_files) for e in examples]
    table.add_section()
    table.add_row(
        "ALL",
        f"{len(examples):,}",
        f"{sum(s.n_dev for s in stats):,}",
        f"{total_heldout:,}",
        "100.0%",
        f"{sum(all_gold) / len(all_gold):.2f}" if all_gold else "0",
        "",
        str(max(all_gold, default=0)),
        f"{sum(s.n_borderline for s in stats):,}",
        "",
        "",
        style="bold",
    )
    console.print(table)
    console.print()


def _render_gold_distribution(console: Console, stats: list[RepoStats]) -> None:
    table = Table(title="Gold files per example", title_justify="left")
    table.add_column("repo")
    for lo, hi in GOLD_BUCKETS:
        table.add_column(bucket_label(lo, hi), justify="right")

    for s in stats:
        total = max(s.total, 1)
        cells = []
        for lo, hi in GOLD_BUCKETS:
            count = s.gold_distribution[bucket_label(lo, hi)]
            cells.append(f"{count:,}\n{100 * count / total:.0f}%" if count else "—")
        table.add_row(s.repo, *cells)
    console.print(table)
    console.print()


def _render_borderline(console: Console, examples: list[Example]) -> None:
    counter = borderline_counter(examples)
    if not counter:
        return
    table = Table(
        title="Borderline markers — examples that nearly got filtered", title_justify="left"
    )
    table.add_column("marker")
    table.add_column("count", justify="right")
    table.add_column("% of dataset", justify="right")
    for marker, count in counter.most_common():
        table.add_row(marker, f"{count:,}", f"{100 * count / len(examples):.1f}%")
    console.print(table)
    console.print()


def _render_warnings(console: Console, stats: list[RepoStats]) -> None:
    total_heldout = sum(s.n_heldout for s in stats)
    problems = []
    for s in stats:
        share = s.n_heldout / total_heldout if total_heldout else 0
        if s.n_heldout < MIN_USEFUL_HELDOUT:
            problems.append(
                f"[bold]{s.repo}[/] contributes only [bold]{s.n_heldout}[/] held-out examples "
                f"({100 * share:.1f}% of the eval set). That is too few to support a per-repo "
                f"accuracy claim — a single example moves top-1 by "
                f"{100 / max(s.n_heldout, 1):.1f} points — and blending it into the aggregate "
                f"hides how little it contributed."
            )
        elif share < MIN_HELDOUT_SHARE:
            problems.append(
                f"[bold]{s.repo}[/] is [bold]{100 * share:.1f}%[/] of the held-out set "
                f"({s.n_heldout} examples). Aggregate numbers will be dominated by the other repos."
            )
    if problems:
        console.print(
            Panel(
                "\n\n".join(problems),
                title="[yellow]per-repo coverage warnings[/]",
                border_style="yellow",
            )
        )
        console.print()


def render_samples(
    console: Console,
    examples: list[Example],
    n: int,
    n_borderline: int,
    seed: int,
) -> None:
    """Print full examples for eyeballing label quality.

    Deliberately samples borderline cases separately from the random draw. A
    random sample of a dataset that is 85% clean shows you 85% clean examples;
    the whole point of reviewing labels by hand is to see the edges.
    """
    rng = random.Random(seed)
    borderline = [e for e in examples if e.borderline]
    ordinary = [e for e in examples if not e.borderline]

    picks: list[tuple[Example, str]] = []
    for ex in rng.sample(ordinary, min(n, len(ordinary))):
        picks.append((ex, "random"))
    for ex in rng.sample(borderline, min(n_borderline, len(borderline))):
        picks.append((ex, "borderline"))

    for ex, kind in picks:
        _render_one(console, ex, kind)


def _render_one(console: Console, ex: Example, kind: str) -> None:
    header = f"[bold]{ex.example_id}[/]  ({ex.split})"
    if kind == "borderline":
        header += "  [yellow]— BORDERLINE[/]"

    body = [
        f"[dim]fix   [/] {ex.fix_sha}",
        f"[dim]parent[/] {ex.parent_sha}  [dim](this is the state we index)[/]",
        f"[dim]date  [/] {ex.authored_at:%Y-%m-%d}"
        f"   [dim]files changed:[/] {ex.n_files_changed}"
        f"   [dim]issues:[/] {', '.join('#' + str(i) for i in ex.issue_refs) or '—'}",
    ]
    if ex.borderline:
        body.append(f"[yellow]flags [/] {', '.join(ex.borderline)}")
    if ex.query_scrubbed:
        body.append(
            "[yellow]note  [/] a gold path appeared verbatim in the message and was scrubbed"
        )

    body.append("")
    body.append("[bold cyan]QUERY[/]")
    query = ex.query_text if len(ex.query_text) <= 1200 else ex.query_text[:1200] + "\n[dim]…[/]"
    body.append(query)
    body.append("")
    body.append(f"[bold green]GOLD FILES[/] ({len(ex.gold_files)})")
    body.extend(f"  • {p}" for p in ex.gold_files)

    console.print(
        Panel(
            "\n".join(body),
            title=header,
            title_align="left",
            border_style="yellow" if kind == "borderline" else "blue",
        )
    )


def heldout_only(examples: list[Example]) -> list[Example]:
    return [e for e in examples if e.split == HELDOUT]


def _metrics_table(title: str, scores: dict, subtitle: str = "") -> Table:
    table = Table(title=title + (f"\n[dim]{subtitle}[/]" if subtitle else ""), title_justify="left")
    table.add_column("method")
    for col in ("top-1", "top-5", "top-10", "MRR", "MAP"):
        table.add_column(col, justify="right")

    keys = ["top1", "top5", "top10", "mrr", "map"]
    best = {k: max((s.get(k, 0.0) for s in scores.values()), default=0.0) for k in keys}
    for method in ("bm25", "dense", "hybrid"):
        if method not in scores:
            continue
        s = scores[method]
        cells = []
        for k in keys:
            v = s.get(k, 0.0)
            cells.append(f"[bold green]{v:.3f}[/]" if v == best[k] and v > 0 else f"{v:.3f}")
        table.add_row(method, *cells)
    return table


def render_eval(console: Console, payload: dict) -> None:
    """Print one scope's results: composition first, then aggregate, then per repo."""
    scope = "tests INCLUDED (harder)" if payload["config"]["include_tests"] else "tests excluded"

    comp = Table(title=f"Eval set composition — corpus scope: {scope}", title_justify="left")
    comp.add_column("repo")
    comp.add_column("examples", justify="right")
    comp.add_column("share of aggregate", justify="right")
    for c in payload["composition"]:
        comp.add_row(
            c["repo"],
            f"{c['n']:,}",
            f"{100 * c['share']:.1f}%",
            style="bold yellow" if c["share"] >= 0.5 else None,
        )
    console.print(comp)

    dominant = max(payload["composition"], key=lambda c: c["share"])
    if dominant["share"] >= 0.5:
        console.print(
            Panel(
                f"The aggregate below is [bold]{100 * dominant['share']:.0f}% "
                f"{dominant['repo']}[/]. Read it as a {dominant['repo']} number that "
                f"other repos nudge, not as a cross-repo result. The per-repo tables "
                f"are the ones that support per-repo claims.",
                border_style="yellow",
                title="[yellow]aggregate composition warning[/]",
            )
        )

    console.print(
        _metrics_table(
            "Aggregate",
            payload["overall"],
            f"{payload['n_examples']:,} examples · mean "
            f"{payload['mean_candidates']:.0f} candidate files/query",
        )
    )
    for repo, scores in payload["per_repo"].items():
        n = next(iter(scores.values()))["n"]
        console.print(_metrics_table(f"{repo}", scores, f"n={n}"))
    console.print()


def run_retrieval_demo(console: Console, cfg, example: Example, method: str, k: int) -> None:
    """Run retrieval for one example and show the ranking against ground truth.

    This is the human-readable counterpart to the eval harness: it shows not
    just whether the gold file was found but *where* it ranked, which is the
    thing top-k accuracy compresses away.
    """
    from buglocalizer.corpus import list_corpus, read_blobs, repo_path
    from buglocalizer.indexing.embedder import get_embedder
    from buglocalizer.indexing.store import connect
    from buglocalizer.retrieval import bm25_search, dense_search

    repo_dir = repo_path(cfg, example.repo)
    files = list_corpus(repo_dir, example.parent_sha, cfg)
    gold = set(example.gold_files)

    console.print(
        Panel(
            f"[bold cyan]QUERY[/]\n{example.query_text[:600]}\n\n"
            f"[bold green]GOLD[/] {', '.join(example.gold_files)}\n"
            f"[dim]corpus at {example.parent_sha[:12]}: {len(files):,} candidate files"
            f"  ·  include_tests={cfg.corpus.include_tests}[/]",
            title=f"[bold]{example.example_id}[/]  ({example.split})",
            title_align="left",
            border_style="blue",
        )
    )

    results = []
    if method in ("bm25", "both"):
        contents = read_blobs(repo_dir, sorted({f.blob_sha for f in files}))
        results.append(bm25_search(cfg, example.example_id, example.query_text, files, contents))
    if method in ("dense", "both"):
        embedder = get_embedder(cfg)
        with connect(cfg) as conn:
            results.append(
                dense_search(
                    cfg,
                    conn,
                    example.repo,
                    example.example_id,
                    example.query_text,
                    files,
                    embedder,
                )
            )

    for result in results:
        table = Table(
            title=f"{result.method.upper()}  —  {result.n_candidates:,} candidates, "
            f"{result.seconds * 1000:.0f} ms",
            title_justify="left",
        )
        table.add_column("#", justify="right")
        table.add_column("score", justify="right")
        table.add_column("file")
        for i, sf in enumerate(result.ranked[:k], 1):
            hit = sf.path in gold
            table.add_row(
                str(i),
                f"{sf.score:.4f}",
                f"{'✓ ' if hit else '  '}{sf.path}",
                style="bold green" if hit else None,
            )
        console.print(table)

        ranks = [i for i, sf in enumerate(result.ranked, 1) if sf.path in gold]
        if ranks:
            console.print(
                f"  [green]gold found at rank {', '.join(map(str, ranks))}[/]"
                f"  (of {result.n_candidates:,})\n"
            )
        else:
            console.print("  [red]gold not ranked at all[/]\n")
