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
