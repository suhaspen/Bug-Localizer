# Architecture

*A map of the codebase: what each file is responsible for, and how data flows
through the system. This is the doc to read before answering "walk me through
your code."*

**Status: Milestone 0.** Only the skeleton exists. Modules planned for later
milestones are listed with the milestone that delivers them, so the intended
shape of the system is visible from the start.

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
│   ├── mining/              # git history → labeled examples        [M1]
│   ├── indexing/            # source files → BM25 + pgvector        [M2]
│   ├── retrieval/           # query → ranked files                  [M2]
│   └── eval/                # ranked files → metrics                [M3]
├── tests/
│   ├── test_config.py       # config loading, overrides, validation
│   └── test_cli.py          # command surface smoke tests
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
    │      clone/pull repos into .cache/
    │      walk history → identify fix commits
    │      apply filters (merge, docs/test-only, mega-commit)
    │      → data/examples.jsonl        {query_text, gold_files,
    │                                    repo, fix_sha, parent_sha, ts}
    │
    ├──▶ bugloc dataset-stats                                           [M1]
    │      read examples.jsonl → counts, gold-file distribution
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

---

## Testing strategy

Tests target the logic where a bug would corrupt a *reported number*, rather than
chasing coverage:

- **Config** — override precedence, rejection of unknown keys, validation of
  impossible values, and a test that the committed `config.yaml` actually parses.
  That last one is small but load-bearing: it means a broken config can never be
  committed silently.
- **CLI** — that the command surface exists and `--help` works.

Coming: filter logic tested against fabricated commits (M1), metric functions
tested against hand-computed examples (M3). Those two are the highest-value tests
in the project — a wrong filter silently changes the dataset, and a wrong metric
silently changes every number we report.
