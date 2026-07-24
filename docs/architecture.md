# Architecture

*A map of the codebase: what each file is responsible for, and how data flows
through the system. This is the doc to read before answering "walk me through
your code."*

**Status: Milestone 1.** Mining and the dataset exist; indexing, retrieval and
evaluation do not. Modules planned for later milestones are listed with the
milestone that delivers them, so the intended shape of the system is visible from
the start.

---

## Layout

```
Bug Localizer/
├── config.yaml              # every knob that can change a reported number
├── docker-compose.yml       # Postgres 17 + pgvector on :5433
├── Makefile                 # the single entry point for every command
├── pyproject.toml           # deps; heavy ML libs behind the [ml] extra
├── src/buglocalizer/
│   ├── __init__.py
│   ├── config.py            # typed config models + loader          [M0]
│   ├── logging_setup.py     # one place that configures logging     [M0]
│   ├── cli.py               # the `bugloc` command surface          [M0]
│   ├── dataset.py           # Example schema, JSONL io, split, stats [M1]
│   ├── reporting.py         # stats tables and the sample printer   [M1]
│   ├── mining/
│   │   ├── repos.py         # clone/fetch into .cache/              [M1]
│   │   ├── filters.py       # pure commit-filter logic              [M1]
│   │   └── miner.py         # git log → Example objects             [M1]
│   ├── indexing/            # source files → BM25 + pgvector        [M2]
│   ├── retrieval/           # query → ranked files                  [M2]
│   └── eval/                # ranked files → metrics                [M3]
├── tests/
│   ├── test_config.py       # config loading, overrides, validation
│   ├── test_cli.py          # command surface smoke tests
│   ├── test_filters.py      # every filter + borderline rule
│   └── test_miner.py        # end-to-end mining on a real fixture repo
├── data/                    # examples.jsonl, splits (gitignored)
├── results/                 # eval runs (committed — this is the evidence)
└── docs/                    # the knowledge base
```

The `src/` layout is deliberate: it means tests import the *installed* package
rather than accidentally picking up the source directory from the working
directory, so a passing test suite is evidence the package is actually
installable.

---

## Data flow

Nothing below runs yet beyond configuration loading. This is the target shape.

```
config.yaml
    │
    ▼
[cli.py]  loads Config, configures logging, dispatches
    │
    ├──▶ bugloc mine                                                    [M1]
    │      clone/fetch repos into .cache/
    │      one `git log --name-only` per repo → CommitInfo list
    │      classify() funnel: merge, excluded message, not-a-fix,
    │        mega-commit, no-source-files
    │      batch-verify gold files exist at parent_sha
    │      build query_text (strip trailers, scrub gold paths)
    │      assign temporal split, per repo
    │      → data/examples.jsonl        {query_text, gold_files,
    │                                    repo, fix_sha, parent_sha,
    │                                    authored_at, borderline, split}
    │      → data/mining_funnel.json    per-filter rejection counts
    │
    ├──▶ bugloc dataset-stats                                           [M1]
    │      read examples.jsonl + funnel → per-repo tables,
    │      gold-file distribution, coverage warnings
    │
    ├──▶ bugloc samples                                                 [M1]
    │      print full examples for hand-review of label quality,
    │      over-sampling borderline cases
    │
    ├──▶ bugloc index                                                   [M2]
    │      for each example: checkout parent_sha
    │      extract .py files → chunks → embeddings
    │      → Postgres (pgvector) + BM25 index
    │      cached by git blob hash, so identical file
    │      versions are embedded exactly once
    │
    ├──▶ bugloc retrieve                                                [M2]
    │      query_text → BM25 ranking
    │                 → dense ranking (cosine over pgvector)
    │                 → hybrid (RRF fusion of the two)
    │      → ranked list of file paths
    │
    └──▶ bugloc eval                                                    [M3]
           for each held-out example: retrieve, compare to gold_files
           → top-1/5/10 accuracy, MRR, MAP
           → results/<timestamp>.json + a markdown table
```

---

## Module responsibilities (current)

### `config.py`

Defines the whole configuration as nested pydantic models and loads it from
YAML. Three layers, applied in order: `config.yaml` → `config.local.yaml` (a
gitignored personal override) → `BUGLOC_*` environment variables.

Two design points worth being able to defend:

**Unknown keys are rejected.** Every model sets `extra="forbid"`. A typo'd key
that pydantic silently ignores would mean publishing results attributed to
settings that were never actually applied.

**Environment overrides are deliberately narrow** — only the database DSN and log
level. Anything that can change a metric must come from the version-controlled
file, so that a result is always reproducible from (git SHA + `config.yaml`)
without needing to know what was in someone's shell.

`Config.repo(name)` looks up a repo by name and raises with the list of known
names, so a typo in a repo argument produces a useful error rather than a
`StopIteration` or `None`.

### `logging_setup.py`

A single idempotent `configure_logging()` called from the CLI, using rich's
handler. Mining and indexing are long loops whose per-filter decisions we need to
observe — the counts logged during mining are the raw material for the dataset
stats table. GitPython's own logging is pinned to WARNING because it is extremely
chatty at DEBUG.

### `cli.py`

The `bugloc` Typer application. Every planned command is registered from
Milestone 0; unimplemented ones print the milestone that will deliver them and
exit with code 2. `config-show` is genuinely useful and works today: it prints the
fully resolved config, which is how you confirm an override landed before
starting a long run.

Each command takes `--config/-c`, so a run can be pinned to a specific settings
file — needed for sweeping hyperparameters on the dev set without editing the
committed config in place.

### `mining/filters.py`

The commit-filter rules, written as **pure functions over a `CommitInfo`
dataclass**. Nothing in this module touches git, the filesystem, or the network,
which is the point: the rules that decide what the dataset contains can be tested
against fabricated commits, exhaustively and in milliseconds.

`classify()` runs the funnel and returns a `Decision` carrying either a rejection
reason or the gold files plus any *borderline* markers. Rejection order is
meaningful — cheap structural checks (merge, parent count) run before message
regexes — and each commit is attributed to the first rule that rejected it, so
the logged counts sum to the total scanned.

`path_matches()` implements glob matching with `**` crossing directory
boundaries. Neither `fnmatch` (its `*` matches `/`, which would make `*.py` and
`**/test_*.py` equivalent) nor `PurePath.full_match` (3.13+) was usable.

`CommitInfo.__post_init__` strips the message. That looks fussy until you learn
several pandas commits are written `" DOC: ..."` with a leading space, which
makes every `^`-anchored exclusion silently miss.

### `mining/miner.py`

Reads history and builds `Example` objects. Three things worth knowing:

**One subprocess per repo.** A single `git log --name-only` with a custom format
using `\x01`/`\x1f`/`\x1e` separators — control characters that cannot appear in
a commit message or path, so multi-line messages parse unambiguously. GitPython's
`commit.stats` would be one subprocess per commit: ~38,000 on pandas.

**Dates via `%at`, not `%aI`.** flask has a commit with a `+518:00` timezone
offset that `datetime.fromisoformat` refuses. Epoch seconds always parse.

**Gold reachability is batch-verified.** `paths_present_at()` feeds every
`(parent_sha, path)` pair to one `git cat-file --batch-check` process and maps
results back positionally. This enforces that a gold file exists in the corpus
we index.

### `dataset.py`

The `Example` pydantic model (also `extra="forbid"`), JSONL read/write, the
temporal split, and the statistics functions. The split sorts on
`(authored_at, fix_sha)` so it is byte-identical across runs even when two
commits share a timestamp.

### `reporting.py`

Rich tables for `dataset-stats` and the panel renderer for `samples`. Separate
from the CLI so the numbers can be computed and tested independently of how they
are printed. It owns the per-repo coverage warning thresholds — a repo under 30
held-out examples or 5% of the eval set gets flagged explicitly rather than being
blended into an aggregate.

`render_samples()` draws borderline examples as a *separate quota* from the
random sample, because a random draw from a mostly-clean dataset shows you
mostly-clean examples.

---

## Testing strategy

Tests target the logic where a bug would corrupt a *reported number*, rather than
chasing coverage:

- **Config** — override precedence, rejection of unknown keys, validation of
  impossible values, and a test that the committed `config.yaml` actually parses.
  That last one is small but load-bearing: it means a broken config can never be
  committed silently.
- **CLI** — that the command surface exists and `--help` works. Note that `mine`
  and `dataset-stats` are deliberately *excluded* from the "unimplemented
  command" test: invoking them from a unit test would clone repositories and mine
  real history.
- **Filters** — every rejection reason and every borderline marker, against
  fabricated `CommitInfo` objects. Highest-value tests in the project so far: a
  wrong filter doesn't crash, it silently changes what the dataset contains, and
  every downstream number inherits the error with no visible symptom.
- **Mining end-to-end** — `tests/test_miner.py` builds a real git repository in a
  temp directory containing one commit of each shape (root, feature, clean fix,
  docs-only, `DOC:`-prefixed, mega-commit, a fix that creates its own gold file,
  and a fix whose message names the file it changed), then mines it and asserts
  the exact funnel counts. This covers the parts no filter test can: the `git log`
  parser, the parent-existence check, and query construction.

  The most important test in the file is `test_parent_sha_really_is_the_buggy_state`,
  which reads the gold file at both SHAs and asserts the bug is present at the
  parent and the fix is not. That is the project's central correctness property,
  and it is checked rather than assumed.

Coming: metric functions tested against hand-computed examples (M3) — a wrong
metric silently changes every number we report.
