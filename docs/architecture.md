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
│   ├── reporting.py         # stats tables, sample + retrieval demo [M1/M2]
│   ├── corpus.py            # commit → searchable files, blob reads [M2]
│   ├── mining/
│   │   ├── repos.py         # clone/fetch into .cache/              [M1]
│   │   ├── filters.py       # pure commit-filter logic              [M1]
│   │   └── miner.py         # git log → Example objects             [M1]
│   ├── indexing/
│   │   ├── chunking.py      # file → overlapping windows            [M2]
│   │   ├── embedder.py      # sentence-transformers wrapper         [M2]
│   │   ├── store.py         # Postgres/pgvector schema + io         [M2]
│   │   └── indexer.py       # the blob-cached index build loop      [M2]
│   ├── retrieval/
│   │   ├── base.py          # ScoredFile, RetrievalResult, tokenizer [M2]
│   │   ├── sparse.py        # BM25 + tokenised-blob LRU cache       [M2]
│   │   ├── dense.py         # pgvector cosine search                [M2]
│   │   ├── hybrid.py        # RRF fusion of ranked lists            [M3]
│   │   └── rerank.py        # cross-encoder over a shortlist        [M4]
│   └── eval/
│       ├── metrics.py       # top-k, MRR, MAP — pure functions      [M3]
│       ├── harness.py       # run all methods, accumulate scores    [M3]
│       └── results.py       # JSON/markdown output + peek ledger    [M3]
├── tests/
│   ├── test_config.py       # config loading, overrides, validation
│   ├── test_cli.py          # command surface smoke tests
│   ├── test_filters.py      # every filter + borderline rule
│   ├── test_miner.py        # end-to-end mining on a real fixture repo
│   ├── test_chunking.py     # chunk coverage/overlap + tokenizer
│   ├── test_corpus.py       # corpus scope + BM25 on a fixture repo
│   ├── test_metrics.py      # hand-computed metric + RRF + McNemar values
│   ├── test_eval.py         # score accumulation, composition, ledger
│   └── test_rerank.py       # shortlist boundary, max-over-chunks, tail
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
    │      for each example's parent_sha:
    │        git ls-tree      → candidate files (label-space scope)
    │        DB lookup        → which blobs are already embedded
    │        git cat-file     → contents of the new ones only
    │        chunk (700 ch)   → embed → COPY into pgvector
    │      → blob(repo, blob_sha) + chunk(…, embedding)
    │      Nothing is keyed by commit: identical file versions
    │      across commits collapse onto one blob row.
    │
    ├──▶ bugloc retrieve                                                [M2]
    │      git ls-tree at parent_sha → candidate files
    │      BM25  : read blobs, tokenize (LRU), score in memory
    │      dense : embed query, cosine over that commit's chunks,
    │              file score = max over its chunks
    │      → ranked list of file paths + where gold landed
    │
    └──▶ bugloc eval                                                    [M3]
           restrict examples to commits covered at EVERY requested scope
           for each scope (tests excluded, tests included):
             for each example, in commit order:
               list_corpus → BM25 ranking
                           → dense ranking
                           → hybrid = RRF(bm25, dense)
               score all three against gold_files
           → per-repo + aggregate: top-1/5/10, MRR, MAP
           → results/<timestamp>.json, results/latest.md
           → append to results/heldout_log.jsonl (the peek ledger)
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

### `corpus.py`

Turns a commit into a searchable candidate set. `list_corpus()` runs
`git ls-tree` and applies the same path exclusions the miner uses, so corpus and
label space agree by construction rather than by two lists being kept in sync by
hand. `read_blobs()` reads many file contents through a single
`git cat-file --batch` process — one process per tree rather than per file, which
is 79 ms for a pandas tree.

The batch reader is worth a glance: it walks the output stream by byte offset
using each header's declared size. A missing sha produces a short `<sha> missing`
line with no payload, and mishandling that would desynchronise the stream and
silently mis-assign contents to *every subsequent file*. There is a test for it.

### `indexing/`

`chunking.py` returns `(start, end)` offsets rather than strings, so file content
is stored once and a chunk is a view into it. `embedder.py` wraps
sentence-transformers, resolves the device, and refuses to start if the model's
dimensionality disagrees with the configured `embedding_dim` — the pgvector
column is fixed-width, so a mismatch would otherwise fail deep inside a COPY.

`indexer.py` is the build loop, and its shape is entirely dictated by the blob
cache: list the corpus, ask the DB which blobs it already has, read and embed
only the remainder. Chunks are batched *across* blobs before encoding, because a
small file yields one chunk and encoding one chunk at a time wastes most of the
GPU's throughput. Examples are processed oldest-first so consecutive commits
share nearly their whole tree.

`store.py` holds the schema and all SQL. Chunks are written with binary `COPY`
rather than INSERT, which matters when a single commit contributes ~10,000 rows.

### `retrieval/`

`base.py` owns the code-aware tokenizer (identifiers emitted whole *and* split on
snake/camel boundaries) and the result types. `sparse.py` is BM25 plus the
bounded LRU of tokenised blobs that makes it affordable. `dense.py` is a single
SQL statement: cosine distance restricted to the commit's blobs, `MIN` per blob,
which is max-similarity pooling expressed as min-distance.

### `eval/`

`metrics.py` is the most carefully isolated module in the project: pure
functions over `(ranked_paths, gold_paths)` with no git, no database and no
model. That isolation is deliberate, because a metric bug does not crash — it
shifts every published number by a plausible amount — so every metric must be
checkable against values computed by hand in a test.

`harness.py` runs each example once per first-stage method and fuses hybrid from
those same rankings, so the three methods provably see identical candidate sets.
It skips examples whose parent commit is not indexed at the current scope:
scoring them would silently record dense misses and depress the dense column for
a reason that has nothing to do with retrieval. Examples are processed in commit
order to keep the tokenised-blob cache warm.

`results.py` serialises a run with its git SHA and full retrieval config — a
number is only reproducible as (code + config) — and owns the held-out peek
ledger.

`retrieval/hybrid.py` implements RRF. Worth reading for the docstring alone: it
explains why score-level fusion is not an option here (BM25 scores are unbounded
and corpus-dependent, cosine lives in [-1, 1]) and what the `k` constant actually
controls.

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

- **Chunking** — that chunks *cover every character* of the file, that overlap is
  exactly as configured, and that no chunk exceeds the limit. Coverage is the
  load-bearing one: a gap means a region of code is permanently unretrievable and
  every dense number is silently capped with no error anywhere.
- **Tokenizer** — that identifiers are emitted both whole and split, and that a
  single-word identifier is *not* emitted twice (which would double-weight it).
- **Corpus + BM25** — against a real fixture repo: that the default corpus equals
  the label space, that `include_tests` widens it, that `read_blobs` survives a
  missing sha without desynchronising the stream, and that ranking is
  deterministic on ties.

Deliberately not tested: the embedding model's output values, and Postgres round
trips. Both are third-party behaviour, and asserting on specific float outputs
would produce a test that fails whenever the model version changes without
indicating anything is actually wrong.

Coming: metric functions tested against hand-computed examples (M3) — a wrong
metric silently changes every number we report.
