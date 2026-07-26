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

---

## Milestone 2

### D17 — D7 resolved: `rank_bm25`, confirmed with timings

**Decision.** Confirm the provisional Milestone 0 choice. The sparse baseline is
`rank_bm25`'s `BM25Okapi`, built in memory per query.

**Alternatives.** Postgres full-text search (`tsvector` + `ts_rank_cd`).

**Why.** The open question was whether a per-commit in-memory index bites at
pandas scale. Measured, the answer splits:

*Memory: no.* A BM25 index over a pandas corpus is ~5 MB of heap; process peak
RSS ~346 MB. There was never a memory problem.

*Time: yes if naive, and the fix is a cache.* Tokenisation dominates the cost —
and it is the one stage that caches perfectly, because the same file version
recurs across hundreds of consecutive commits with an identical token list. A
bounded LRU keyed on blob hash turns the dominant cost into a dictionary lookup.

Postgres FTS lost on both criteria. It was *slower* (~4.0 s per commit to COPY,
build tsvectors and a GIN index, versus the entire `rank_bm25` path), and more
importantly `ts_rank_cd` **is not BM25** — different weighting, no `k1`
saturation, no `b` length normalisation. Labelling that column "BM25" in a
results table would have been a false claim, which is the kind of thing this
project exists to avoid. Its English stemmer is also wrong for code: it would
mangle `send_file` rather than split it.

### D18 — Corpus = label space (non-test source) by default

**Decision.** The searchable candidate set is exactly the set of files that could
be a gold label. `corpus.include_tests: true` restores whole-repo search.

**Alternatives.** Search every `.py` file in the tree.

**Why.** Coherence. The miner strips test files from labels, so a gold file can
never be a test. If tests remained searchable, a method ranking
`pandas/tests/test_frame.py` first would be scored a *miss* even though that test
is genuinely about the bug — penalising sensible behaviour for a reason that is
an artifact of our labelling convention. Matching the candidate set to the label
space removes the incoherence, and it follows standard practice in the
bug-localization literature, where test directories are excluded from the corpus.

**Cost, stated because it changes every number we report.** 80% of pandas' `.py`
files are tests: the corpus drops from 1,405 files / 18.0 MB to 282 files /
6.5 MB per commit. Ranking within 282 candidates is materially easier than within
1,405, so accuracy will read higher than a whole-repo search would give. Anyone
comparing our numbers to another system must check that system's corpus scope
first. The flag exists so the claim is checkable rather than merely asserted.

### D19 — 700-character fixed windows, 10% overlap

**Decision.** Chunk files into 700-character sliding windows overlapping by 70,
stored as offsets into the file content.

**Alternatives.** Whole file; 2000-character windows (the Milestone 0 default);
AST function/class units.

**Why.** The size is not a free parameter — it is dictated by the model. MiniLM
caps at 256 word-piece tokens, and code measures ~2.8 characters per token
(versus ~4 for prose), so the model can see about 700 characters of Python.
Beyond that it truncates **silently**: no error, the tail simply never gets
embedded. The original 2000-char default would have discarded ~65% of every
chunk. Whole-file embedding would have captured the first 8% of the median
corpus file.

*Why not AST chunking*, which is the appealing option: it does not reduce the
unit count (14.6 vs 20.7 per file — the binding constraint is the token budget,
not the split strategy), it does not avoid truncation (the median Python function
measured **536 tokens**, more than twice the limit), and it adds a failure mode
(`ast.parse` fails on older syntax across 17 years of history, needing a fallback
anyway). It is the better idea *with a long-context model*, and Milestone 5 needs
function units for a different reason — localizing *to* functions — which is
where the AST parser earns its place.

Overlap costs 10% more units and prevents a hard cut landing mid-function and
leaving neither half matchable. Offsets rather than stored text avoid keeping a
second, 10%-larger copy of the whole corpus.

### D20 — all-MiniLM-L6-v2, chosen on measured throughput

**Decision.** Keep `all-MiniLM-L6-v2` (384-d).

**Alternatives.** `bge-small-en-v1.5` (512 tokens);
`jina-embeddings-v2-base-code` (8192 tokens, code-trained).

**Why.** `bge-small` halves the chunk count exactly as intended — and loses by
3.5x anyway, because it is **6.8x slower per unit**. Doubling context costs more
than double the compute, and the unit saving does not pay for it. Worth recording
because the units/file column alone predicts the wrong winner; only the product
of units and throughput decides it.

The code-native jina model would have been the genuinely interesting option — an
8192-token context fits most whole files in one unit, eliminating truncation and
chunking together. It could not be loaded: its remote code calls
`find_pruneable_heads_and_indices`, removed from current `transformers`. Pinning
an older `transformers` was judged not worth the dependency risk for a first
pass. **This is the single most promising upgrade available to the project** and
is recorded as such rather than quietly dropped.

### D21 — Blob-hash content addressing, and no paths in the store

**Decision.** Storage is keyed on `(repo, blob_sha)`. Paths are not stored; they
come from `git ls-tree` at query time.

**Why.** Each example searches its own commit, which naively means one corpus per
example: **5,398,769** (commit, path) file-instances across pandas. Git already
content-addresses file versions, and untouched files share a blob exactly, so
keying on blob hash collapses that to **93,848 distinct blobs — 57x**. Without
this, dense retrieval on this dataset is arithmetically impossible on a laptop.

Paths are excluded because a blob is content, and the same content can live at
different paths in different commits — storing paths would reintroduce exactly
the per-commit duplication the blob key removes.

`repo` is in every primary key even though blob hashes are globally unique. That
denormalisation is deliberate: the project reports per-repo results, so index,
drop, count and vector search must all be cheaply scopeable to one repository.

### D22 — Exact vector search, no HNSW index

**Decision.** Dense retrieval scans the candidate blobs exactly; no approximate
nearest-neighbour index.

**Alternatives.** pgvector's HNSW index.

**Why.** HNSW earns its keep when scanning millions of vectors. Here every query
is restricted to one commit's corpus — a few hundred blobs, order 10,000 chunks —
and an exact scan is already fast. An approximate index would inject recall error
into a measurement whose entire purpose is measuring recall. Revisit only if
query time becomes the bottleneck, and if so, report the ANN recall separately.

### D23 — Deterministic tie-breaking on `(-score, path)`

**Decision.** Both retrievers sort by score descending, then path ascending.

**Why.** Most files score exactly zero on a given BM25 query. Without a
tie-break their relative order depends on dictionary iteration order, so top-k
membership could differ between runs on identical inputs. A metric that moves
when nothing changed is indefensible, and this is a one-line guarantee against it.

### D24 — Postgres 17 via Homebrew, not the Docker Compose file (environment only)

**Decision.** The committed `docker-compose.yml` remains the documented, portable
setup. On this machine the Docker daemon would not start, so development uses a
Homebrew PostgreSQL 17 + pgvector 0.8.5 on the same port 5433 with the same DSN.

**Why.** The DSN is the only coupling, so both paths are interchangeable and
nothing in the code knows the difference. Recorded so the discrepancy between the
docs and this machine is not mistaken for a reversal of D3 — the reproducible
story for anyone cloning the repo is still `make db-up`.

Port 5433 also kept an existing local PostgreSQL 16 untouched, which is exactly
the collision D3 chose that port to avoid.

### D25 — `flask/testsuite/**` excluded; `testing.py` modules kept

**Decision.** Add `**/testsuite/**` to the path exclusions. Do *not* exclude
paths merely containing "test".

**Why.** Found while reviewing a retrieval result: old flask kept its suite
inside the package at `flask/testsuite/`, which `**/tests/**` does not match, so
it was gold for 44 examples. But the fix has to be narrow — `flask/testing.py`
and `pandas/util/testing.py` are shipped testing *utilities* and part of the
public API (`pandas.util.testing.assert_frame_equal`), so bugs in them are real
bugs. A broad match on "test" would have silently deleted 80 legitimate examples
while fixing 44 bad ones.

---

## Milestone 3

### D26 — Report top-1/5/10, MRR and MAP together

**Decision.** All five numbers, every run.

**Alternatives.** Just top-k accuracy, which is what the task description implies.

**Why.** They disagree in informative ways. Top-k accuracy is binary per example
and maps directly onto how the tool is used (scan a short list, find one lead),
so it is the headline. But it cannot distinguish "gold at rank 2" from "gold at
rank 10", which is a large practical difference — that is what MRR adds. And MRR
stops at the *first* gold file, so a method that finds one of four gold files at
rank 1 scores a perfect 1.0; MAP is the only one of the three that notices. With
77% of examples single-gold, MAP should track slightly below MRR, and a large
divergence would itself be a signal something is wrong.

### D27 — One retrieval pass per example, hybrid fused from it

**Decision.** Each example is retrieved once per first-stage method; hybrid is
computed by fusing those same two rankings rather than re-running anything.

**Why.** Not just performance, though running the eval three times would triple
the cost. It also guarantees the three methods see *identical* candidate sets and
identical corpus state, so any difference in the table is attributable to ranking
rather than to two runs having drifted apart.

### D28 — Both corpus scopes, evaluated over the same examples

**Decision.** Every evaluation runs at both scopes and reports them side by side.
When more than one scope is requested, the example set is restricted to commits
covered at *every* scope.

**Alternatives.** Report only the default (tests-excluded) scope.

**Why.** Excluding tests makes the task materially easier — the pandas candidate
set drops from ~1,400 files to ~290 — so a single number would be
defensible-but-flattering. Reporting the harder number next to it costs one extra
run and pre-empts the obvious objection.

The intersection restriction is a correctness requirement, not tidiness. The wide
scope is far more expensive to index, so it always covers fewer commits. Without
the restriction, the two columns would be computed over different example sets
and the delta between them would conflate *scope* with *sample* — which is
exactly the kind of quiet confound that makes a comparison worthless.

### D29 — The held-out peek ledger

**Decision.** Every held-out evaluation appends a line to
`results/heldout_log.jsonl` (timestamp, git SHA, scope, n). The next peek number
is printed before the run starts.

**Alternatives.** Just save results files and count them if anyone asks.

**Why.** A held-out set stops being held out the moment decisions are made from
it, and those decisions are rarely explicit — you see dense underperform, decide
700 was "arbitrary anyway", and re-run. Nothing dishonest happened, but judgement
is now baked into the number. The count of looks is the cheapest honest upper
bound on how much, so it is recorded rather than remembered. Printing the next
peek number *before* the run also adds a small useful moment of friction.

The first ledger entry is a deliberate example: a flask-only smoke test to
validate the harness. It is recorded rather than excluded, because a ledger you
edit is not evidence.

### D30 — Aggregate numbers always print their composition first

**Decision.** The composition table precedes the aggregate, and a warning fires
when any repo exceeds 50% of the eval set.

**Why.** pandas is 91% of the full held-out set. An aggregate over that is a
pandas number with a rounding error of flask attached, and calling it "cross-repo
accuracy" would be misleading. Rather than relying on the reader to check, the
tool refuses to show an aggregate without the composition beside it. The per-repo
tables are the ones that support per-repo claims.

### D31 — Eval set capped at pandas' newest 120 held-out examples

**Decision.** The Milestone 3 eval covers flask (85) + requests (120) + pandas
(120 newest held-out) = 325 examples, rather than the full 2,235-example held-out
set.

**Alternatives.** Full held-out; the 505-example set originally planned.

**Why.** Cost, measured rather than assumed — and the estimate was wrong in an
instructive way. The plan projected ~1.6 h to index pandas' newest 300 at the
wide scope. Actual throughput was ~1 commit/minute, projecting to **~5 hours**,
because the wide scope embeds ~5x the files per commit and the marginal blob rate
(~52/commit) was ~3.5x what the sampling-based estimate suggested. The estimate
was built from *distinct blobs across the whole window*, which understates the
per-commit churn that the incremental build actually pays.

Capping at 120 keeps the wide index around 2 hours and, importantly, makes the
pandas slice identical at both scopes so the scope comparison is clean.

**What it costs.** 325 examples gives a standard error of roughly 2.8 points on
an accuracy near 0.5, so differences of 6+ points are trustworthy and 2-point
differences are not. The pandas eval also covers only its newest held-out
examples (2026), not the full 2021+ window.

**Why this is recoverable.** Indexing is incremental and blob-cached, so
expanding to the full held-out set later pays only for commits not yet covered —
no restart. The narrow scope already covers 121 pandas commits; the expensive
part is the wide scope, and it can run unattended.

### D32 — `indexed_commit` records the corpus scope it was built at

**Decision.** A boolean `include_tests` column; a commit counts as covered only
if indexed at a scope at least as wide as the one requested.

**Why.** A bug found while planning this milestone. The wide scope embeds a
superset of the default scope's blobs, which is what lets one build serve both
evaluations — but without recording *which* scope produced the coverage, a later
wide-scope run would see the commit marked "indexed", skip it, and leave every
test file unembedded. Dense retrieval would then score those files at -1 and rank
them last, producing a wide-scope number that looked plausible and was silently
measuring an incomplete index.
