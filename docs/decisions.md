# Decision log

One entry per non-obvious choice: what was decided, what the alternatives were,
and why. Newest milestone last. The purpose is that every design decision in this
project has a defensible answer already written down.

---

## Milestone 0

### D1 — Target repos: flask, requests, pandas

**Decision.** Mine three Python repositories: `pallets/flask`, `psf/requests`,
`pandas-dev/pandas`.

**Alternatives.** flask + requests only; substituting `django` for `pandas`;
four mid-size repos (scrapy, sympy, …).

**Why.** The choice trades iteration speed against dataset size, and we wanted
both, so we took a small pair plus one large repo. flask and requests clone and
index in seconds, which makes them the right place to develop and debug the
pipeline. Neither is big enough to matter statistically, though: after filtering
they likely yield a few hundred examples between them, which is too few to
distinguish a 3-point accuracy difference from noise. pandas supplies the volume,
and it does so with unusually clean labels — the project enforces a `BUG: ...`
commit-title convention and links issues consistently, which is precisely the
signal our miner keys on.

Django was the close runner-up; its `Fixed #12345 -- ...` convention is arguably
the cleanest fix signal in Python open source. It lost on issue text: Django uses
Trac rather than GitHub Issues, so fetching the bug report body is extra
machinery for no methodological gain. Revisit if we need a fourth repo.

Three repos also gives a cross-repo generalization story: we can report per-repo
numbers and see whether a method that wins on pandas also wins on flask.

### D2 — Temporal split, per repo, 70/30

**Decision.** Sort each repo's examples by commit date; the oldest 70% is the dev
set, the newest 30% is held out.

**Alternatives.** Random split with a fixed seed; a single global cutoff date
across all repos.

**Why.** Nothing in this project is trained, so the "dev set" exists purely to
give hyperparameter tuning somewhere legitimate to happen. Splitting by time
makes tuning mean the right thing: we tune on past bugs and are judged on future
bugs, which is the actual deployment setting. A random split would let us tune on
a 2024 bug and evaluate on a 2019 bug touching the same file — optimistic in a
way that would not survive contact with reality (see **temporal leakage** in the
glossary).

Per-repo rather than one global cutoff date, because the three repos have very
different activity periods; a single calendar cutoff would let pandas dominate
the eval set and leave flask barely represented, making per-repo comparison
impossible.

Cost, stated honestly: the held-out set is the *newest* bugs, which are also the
ones whose fixes touch the most modern code. If a repo changed character over
time, the held-out numbers are measuring a slightly different distribution than
the dev numbers. We accept that — it's the same asymmetry a deployed system faces.

**Discipline.** Every run against the held-out set is written to `results/` with
a timestamp. That count is a deliberate artifact: "I evaluated on held-out N
times" is an honest and answerable statement about how much hand-tuning could
have leaked in.

### D3 — Postgres + pgvector via Docker Compose, on port 5433

**Decision.** `docker compose up` starts `pgvector/pgvector:pg17`.

**Alternatives.** Local Homebrew Postgres with pgvector installed by hand.

**Why.** "Reproducible from a single command" is a stated goal of the project,
and a local Postgres makes the setup instructions machine-dependent — the reader
has to install the extension, match versions, and hope. The Docker image ships
with pgvector already present. Port 5433 rather than the default 5432 so it can
never collide with a Postgres the reader is already running, which is a common
and annoying failure for anyone trying the repo.

### D4 — Heavy ML dependencies live in an optional extra

**Decision.** `sentence-transformers`, `psycopg`, and `pgvector` are in a `[ml]`
extra, not core dependencies. `make setup` installs core; `make setup-ml`
installs the rest.

**Alternatives.** One dependency list containing everything.

**Why.** `sentence-transformers` pulls in PyTorch — well over a gigabyte, and
minutes of install time. Making that a prerequisite for `make test` means the
test suite is slow to bootstrap and the project is annoying to try. The split
keeps the fast, offline-safe core (mining, filtering, metrics, BM25) independent
of the ML stack, which also enforces a useful architectural boundary: dataset and
evaluation logic must not import torch.

### D5 — Config rejects unknown keys

**Decision.** All pydantic config models set `extra="forbid"`, so an unrecognized
key in `config.yaml` is a hard error.

**Alternatives.** Pydantic's default of ignoring unknown keys.

**Why.** This is a measurement project, and the config file is the record of what
produced a number. If `max_files_per_comit: 20` is silently ignored, we publish
results claiming a threshold of 20 that were actually produced with the default
of 10 — a wrong number with a plausible explanation attached, which is worse than
a crash. Failing loudly at load time costs nothing and closes that hole.

### D6 — Every planned CLI command exists from Milestone 0

**Decision.** `mine`, `index`, `retrieve`, `eval`, and `dataset-stats` are all
registered now; the unbuilt ones exit code 2 with the milestone that delivers them.

**Alternatives.** Add each command when it is implemented.

**Why.** It makes `make help` an accurate description of the finished system
rather than a snapshot of progress, and it means the interface is designed once,
up front, instead of accreting. It also gives the test suite something real to
assert about the command surface from commit 1.

### D7 — `rank_bm25` for the sparse baseline (provisional)

**Decision.** Plan to use the `rank_bm25` library rather than Postgres full-text
search. Marked provisional; confirmed or reversed in Milestone 2.

**Alternatives.** Postgres `tsvector`/`ts_rank_cd` full-text search, which would
keep sparse and dense retrieval in the same database.

**Why (provisional).** `rank_bm25` is a transparent, ~200-line implementation of
textbook BM25 with directly exposed `k1` and `b` parameters, which matters
because we want to *explain* the baseline, not just invoke it. Postgres FTS is
not actually BM25 — `ts_rank_cd` uses a different weighting scheme — so calling
it "our BM25 baseline" would be inaccurate, and its English-language stemmer is
wrong for code tokens like `send_file` anyway. The cost is holding an in-memory
index per commit, which we will re-examine if it becomes a bottleneck.

### D8 — Milestones stop for review

**Decision.** Each milestone ends with tests run, numbers shown, docs written,
and a commit — then work stops.

**Why.** Recorded here because it is a real constraint on how the code is built,
not just a workflow preference: it means each milestone must stand alone as
something demonstrable and defensible, rather than being a half-finished slice of
a larger design.

---

## Milestone 1

### D9 — Query text is the commit message only; issue bodies deferred

**Decision.** `query_text` is built from the commit message. Linked issue numbers
are extracted and stored (72.6% of examples have one), but issue titles and
bodies are not fetched.

**Alternatives.** Fetch issue text from the GitHub API during mining.

**Why, and this is a deliberate cut rather than an omission.** Fetching issue
bodies needs the GitHub API: 60 requests/hour unauthenticated, 5,000/hour with a
token. For 7,466 examples that is either impossible or an hour-long networked
run, and it breaks the project's offline guarantee — mining, indexing and eval
are all supposed to run with no network and no API key.

The cost is real and is documented rather than hidden: commit messages are
written by someone who already knows the answer, so they are subtly easier
queries than real bug reports, and at a median of 69 characters they are much
shorter. Absolute accuracy will therefore be optimistic relative to production.
Method *comparison* — what this project is actually for — is unaffected, since
every method faces the same query.

Issue numbers are stored specifically so this can be added later as a separate
hydration step without re-mining.

### D10 — Test files are never gold labels

**Decision.** Test files are excluded from `gold_files` even though a fix commit
usually edits its test alongside the source.

**Alternatives.** Count every changed file, tests included.

**Why.** The test is *evidence* of the bug, not its location. Including tests
would make the task easier for the wrong reason: test names and assertions
restate the bug report almost verbatim, so a retriever could rank
`tests/test_send.py` first by pure lexical overlap and be scored correct without
ever finding the defect. We would be measuring "can you find the test that
describes this bug," which is a different and much easier question.

### D11 — Gold files must exist at the parent commit

**Decision.** Every gold file is checked for existence at `parent_sha`; absent
ones are removed, and examples with nothing left are dropped.

**Alternatives.** Trust the commit's file list.

**Why.** A fix that *creates* a file leaves a label pointing at something that
does not exist in the corpus we index, so no retriever could ever return it —
a silent, permanent ceiling on every accuracy number. Only 2 examples were
dropped entirely and 24 partially trimmed, so the effect is small, but an
unreachable label is a correctness bug regardless of frequency, and "how do you
know your labels are even retrievable?" is a fair interview question with a much
better answer than "I assumed so."

Implemented as one batched `git cat-file --batch-check` per repo rather than a
lookup per file, which keeps it effectively free.

### D12 — Mine with `git log` subprocess, not GitPython's diff API

**Decision.** History is read via a single `git log --name-only` subprocess per
repo with a custom control-character format. GitPython is used only for cloning.

**Alternatives.** GitPython's `commit.stats.files`, which is the obvious API.

**Why.** `commit.stats` shells out once per commit. On pandas that is ~38,000
subprocesses and takes tens of minutes; one `git log` does the same work in about
8 seconds. The whole mining run is 13 seconds, which matters more than it sounds
— a fast miner is one you re-run freely after changing a filter, and this
milestone was re-mined three times while fixing label defects found by sampling.

Fields are separated with `\x1f`/`\x1e`/`\x01` because those cannot occur in a
commit message or path, so multi-line messages parse unambiguously.

### D13 — Commit dates read as `%at` (epoch), not `%aI` (ISO-8601)

**Decision.** Parse author dates as unix timestamps in UTC.

**Why.** Found by a crash on real data: flask contains a commit whose timezone
offset is written `+518:00`, which `datetime.fromisoformat` rejects outright.
Git will happily store a malformed offset that git itself wrote decades ago. The
epoch form is always valid, and an absolute instant is what temporal ordering
actually needs — the local offset is irrelevant to "which bug came first."

### D14 — Build, benchmark and tooling paths are excluded from gold

**Decision.** `setup.py`, `versioneer.py`, `asv_bench/`, `scripts/`, `ci/`, `web/`
and `**/benchmarks/**` cannot be gold files.

**Alternatives.** Accept any `.py` file outside tests and docs.

**Why.** Found by hand-reviewing sampled examples: `setup.py` alone was the sole
label for 108 examples, and 295 gold entries in total were build or dev-tooling
code. A packaging fix is not a localizable bug in the product, and asking a
retriever to rank `setup.py` for a bug report about DataFrame indexing is asking
it to learn noise. Removing these cost 240 examples and made the rest
trustworthy.

Recorded here mainly because of *how* it was found — reviewing labels by hand is
the only thing that would have caught it, since nothing crashes and every metric
still computes.

### D15 — Borderline markers instead of stricter filters

**Decision.** Examples that *nearly* got filtered are tagged (`short_query`,
`weak_fix_signal`, `at_mega_commit_threshold`, …) and kept, rather than dropped.
`bugloc samples` deliberately over-samples them.

**Alternatives.** Tighten the filters until only unambiguous examples remain.

**Why.** Two reasons. First, review quality: a random sample of a mostly-clean
dataset shows you mostly-clean examples, which tells you nothing about where the
labels break down. Sampling the edges is the only way to see what the filter
boundaries actually admit. Second, honesty about difficulty: dropping every
short or ambiguous query would inflate accuracy by quietly removing the hard
cases, which is exactly the kind of silent dataset curation that makes published
numbers untrustworthy. The markers are stored per example, so results can later
be broken down by borderline status instead of the dataset being pre-filtered to
look good.

### D16 — Normalize commit messages inside `CommitInfo`

**Decision.** `CommitInfo.__post_init__` strips the message.

**Why.** A bug, not a preference. Several pandas commits are written with a
leading space (`" DOC: ..."`), which makes every `^`-anchored exclusion pattern
miss, so docstring commits were labeled as bug fixes. Normalizing at the
construction boundary rather than at each call site means no future code path —
including tests — can reintroduce it.
