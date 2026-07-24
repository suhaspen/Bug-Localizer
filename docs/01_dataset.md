# The Dataset — how git history labels itself

*Self-contained: you can read this without the code, and without having read the
other docs.*

---

## The problem this solves

To measure whether a bug-localization system works, you need examples of the
form *"here is a bug report; here are the files that were actually at fault."*
Producing those by hand requires someone who knows the codebase to read a bug
report and identify the culprit files. At a few minutes per example, a thousand
examples is weeks of expert time — per repository. This is why the public
datasets in this area are small and old.

We sidestep it entirely. The observation is simple enough to state in one
sentence:

> When a maintainer fixes a bug, they write a commit, and the files that commit
> touches are — by definition — the files that were at fault.

Nobody has to label anything. The label was produced as a side effect of the
work, by the person best qualified to produce it, and it is sitting in git.
Every fix commit in every repository is a free labeled example. This technique
has a name in machine learning: **distant supervision**, meaning labels derived
from a signal that already exists rather than from human annotation.

Running this over three repositories produced **7,466 labeled examples in about
13 seconds.**

---

## What one example looks like

Take pandas commit `5eb9988fcdf0`, from September 2018. Its message reads:

> `BUG: fix failing DataFrame.loc when indexing with an IntervalIndex (#22576)`

It changed three files: `pandas/core/indexing.py`, a test file, and a changelog
entry. From this we build:

| Field | Value | Where it came from |
| --- | --- | --- |
| `query_text` | "BUG: fix failing DataFrame.loc when indexing with an IntervalIndex (#22576)" | The commit message |
| `gold_files` | `["pandas/core/indexing.py"]` | The changed files, minus tests and changelogs |
| `parent_sha` | `9b2e6db5a4ec…` | The commit *before* the fix — the code we will search |
| `fix_sha` | `5eb9988fcdf0…` | The answer key. Never indexed. |
| `authored_at` | 2018-09-08 | Used for the temporal split |
| `issue_refs` | `[22576]` | Parsed from the message |

At evaluation time a retriever sees only `query_text`, searches the repository as
it existed at `parent_sha`, and returns a ranked list of file paths. It scores a
hit if `pandas/core/indexing.py` is near the top. That's the whole task.

---

## The rule everything depends on: index the parent, not the fix

This is the one thing to get right, and the first thing a sharp interviewer will
probe.

Suppose we indexed the repository *after* the fix landed. The fixed code would
be in the searchable corpus. A bug report saying "raises AttributeError on Arrow
date types" would match the very lines the fix added — the retriever would score
a bullseye, and the number would be meaningless, because it was shown the answer.

Indexing at the **parent commit** reconstructs the developer's actual situation:
the buggy code, the bug description, and no knowledge of the repair. This is what
"avoiding **leakage**" means here.

The project enforces this structurally rather than by convention. Every example
stores `parent_sha` as the state to index, and `fix_sha` exists only as the
answer key. There is a test that checks the property directly on a fixture
repository — it asserts that the buggy code is present at `parent_sha` and the
corrected code is present only at `fix_sha`. If someone ever wires up indexing
against the wrong SHA, that test fails.

### Leakage through the query, and an honest caveat

There is a second, subtler path. The commit message is written *at fix time*, so
it can mention the fix. Two cases:

**Case one, handled.** Some messages name the file outright: "fix mimetype in
`src/flask/helpers.py`". That hands the retriever its own answer. Any literal
gold path appearing in the message is removed before the query is stored. This
fired on **47 examples** — small, but they would have been free wins.

The scrub removes only *full paths*, never bare module names. Removing `groupby`
from "BUG: groupby.apply raises on empty frames" because the gold file happens to
be `pandas/core/groupby/groupby.py` would delete the actual bug description and
make the example harder than reality. There's a test pinning that behaviour.

**Case two, acknowledged, not solved.** A commit message is still written by
someone who already knows the answer, so its vocabulary is subtly better-informed
than a real bug report's. A user writes "my dataframe prints wrong"; the
maintainer writes "BUG: incorrect repr for MultiIndex with NaT". The second is a
noticeably easier query.

This means **our absolute accuracy numbers are optimistic relative to real
incoming bug reports.** They are still perfectly valid for *comparing* retrieval
methods, which is what the project is for — every method faces the same
advantage. The fix is to use the linked issue's title and body instead, which
72.6% of examples have an issue number for. That requires the GitHub API
(authenticated, rate-limited), which breaks the offline guarantee, so it is
deliberately deferred rather than skipped. The issue numbers are already stored
so it can be added later without re-mining.

---

## The filters, and what each one is protecting against

Most commits that look like fixes are useless as examples. Each filter below
exists because letting those commits through would **inflate our accuracy for a
reason unrelated to retrieval quality**. The order matters: each rejected commit
is attributed to the *first* rule that caught it, so the counts sum to the total.

| # | Filter | flask | requests | pandas | What it protects against |
| --- | --- | ---: | ---: | ---: | --- |
| | commits scanned | 5,539 | 6,486 | 38,465 | |
| 1 | − no parent | 1 | 1 | 1 | The root commit has no buggy state to index |
| 2 | − merge commit | 1,727 | 1,612 | 3,172 | A merge's file list is an artifact of merging |
| 3 | − excluded message | 478 | 330 | 20,867 | Docs/style/refactor commits that aren't bug fixes |
| 4 | − not a fix | 2,671 | 3,927 | 5,997 | Features and chores |
| 5 | − mega-commit | 11 | 6 | 255 | Refactors, where "the bug" would be 200 files |
| 6 | − no source files | 348 | 211 | 1,400 | Docs-only or test-only: nothing to localize to |
| 7 | − gold missing at parent | 0 | 1 | 1 | Files the fix *created*, which are unretrievable |
| | **= kept** | **303** | **398** | **6,765** | |

**Merge commits (2).** A merge commit's diff lists every file changed on the
merged branch. That is a fact about branch topology, not about a bug. The real
fix is an individual commit inside the branch, which we mine separately — so
including the merge would double-count it with a worse label.

**Excluded messages (3).** Checked *before* the fix patterns, because otherwise
"DOC: fix wording in the merge docstring" reads as a bug fix — it contains "fix",
it touches a `.py` file — and labels a source file that was never at fault. This
is the single largest filter for pandas (54%), because pandas prefixes nearly
every commit with its type (`DOC:`, `TST:`, `ENH:`, `CLN:`), which makes the
noise unusually easy to remove. Note that the pattern deliberately requires the
prefix to be followed by a colon or space, so a combined `DOC/BUG:` prefix — a
real bug fix that also touched docs — survives.

**Mega-commits (5).** A commit touching more than 10 files is a refactor, a
release, or a formatting sweep. If we accepted one and called all 200 of its
files "correct," almost any retriever would land a hit in its top 10 by accident.
Top-10 accuracy would climb and mean nothing. The threshold is configurable and
logged, so its effect is visible rather than buried.

**No source files (6).** A commit touching only `docs/` and `tests/` has no
source file to point at. Note that test files are excluded from `gold_files`
*even when a fix edits them*, which it usually does. The reasoning: the test is
*evidence* of the bug, not its location. If `tests/test_send.py` counted as a
correct answer, a retriever could score well by matching the bug report's
vocabulary against test names — test names restate the bug report almost
verbatim — while never finding the actual defect.

**Gold missing at parent (7).** Rare but important for correctness. If a fix
*creates* a file, that file doesn't exist in the corpus we index, so no retriever
could ever return it. Leaving those labels in would put a silent ceiling on every
accuracy number in the project. All gold files are checked for existence at the
parent commit in one batched `git cat-file` call; examples where nothing survives
are dropped, and the 24 examples where only *some* gold files were absent are
kept with the missing ones removed and a borderline marker attached.

---

## What the dataset looks like

**7,466 examples: 5,225 dev / 2,241 held-out.**

| repo | total | dev | held-out | held-out share | avg gold files | date range |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| flask | 303 | 212 | 91 | 4.1% | 1.38 | 2010-04 → 2026-02 |
| requests | 398 | 278 | 120 | 5.4% | 1.23 | 2011-02 → 2026-06 |
| pandas | 6,765 | 4,735 | 2,030 | 90.6% | 1.39 | 2009-08 → 2026-07 |

**77% of examples have exactly one gold file**, and the average is 1.38. That is
good news for the task: it means most bugs really do live in one place, and
top-1 accuracy is a meaningful thing to ask for. It also means MAP and MRR will
track each other closely, since with one gold file MAP reduces to MRR.

### The composition problem, stated plainly

**pandas is 90.6% of the held-out set.** Any aggregate number this project
reports is, to a first approximation, a pandas number. flask contributes 91
held-out examples and requests 120 — and in the recent window it is starker:

| repo | held-out | 2023 onward | 2024 onward | 2025 onward |
| --- | ---: | ---: | ---: | ---: |
| flask | 91 | 20 | 16 | 3 |
| requests | 120 | 16 | 12 | 8 |
| pandas | 2,030 | 1,210 | 722 | 506 |

With 91 held-out examples, a single example moves flask's top-1 accuracy by 1.1
points, so a 3-point difference between two retrieval methods on flask is noise.
The small repos cannot support per-repo accuracy claims; they can only show
whether a method's ranking *direction* is consistent across codebases.

This is a real limitation and the tooling surfaces it rather than hiding it:
`dataset-stats` breaks every count down per repo and prints an explicit warning
when a repo falls below 30 held-out examples or 5% of the eval set.

Note also that the temporal split is per repo, so "held-out" means different
calendar windows in different repos: pandas held-out starts in 2021, requests in
2015. requests simply hasn't had 120 bug fixes since 2015 — it is a mature,
low-velocity library. A single global cutoff date would have been worse, leaving
the small repos with almost no held-out examples at all.

---

## Label quality: what hand-review found

The dataset ships with a `samples` command that prints full examples — query,
gold files, SHAs — for manual review. It deliberately samples **borderline**
examples separately from the random draw. A random sample of a dataset that is
mostly clean shows you mostly clean examples; the point of reviewing by hand is
to see the edges, so examples are tagged during mining when they *nearly* got
filtered:

| marker | count | share | meaning |
| --- | ---: | ---: | --- |
| `short_query` | 2,497 | 33.4% | Message under 60 characters |
| `lone_source_file_among_many` | 1,052 | 14.1% | One source file among 4+ changed files |
| `weak_fix_signal` | 425 | 5.7% | "fix" appears only in the body, not the subject |
| `many_gold_files` | 119 | 1.6% | 5+ gold files |
| `at_mega_commit_threshold` | 48 | 0.6% | Exactly at the 10-file limit |
| `some_gold_files_absent_at_parent` | 24 | 0.3% | Part of the label was unretrievable |

Reviewing the first mining run's samples by hand found **four real defects**,
each of which was corrupting labels. They are worth recording because they are
the kind of thing that never shows up as a crash:

1. **`setup.py` was gold for 108 examples.** It's a `.py` file that no path
   filter excluded, so packaging fixes became "bug fixes" pointing at the build
   script. Together with `asv_bench/`, `scripts/` and `ci/`, 295 gold entries
   were build or tooling code, and 144 examples had *nothing but* such files.
   Fixed by excluding build/benchmark/tooling paths.
2. **Lowercase commit prefixes bypassed the message filter.** pandas has drifted
   toward conventional-commit style (`docs:`, `style:`), which the uppercase-only
   pattern let through. Fixed by matching plural and lowercase forms.
3. **Squash trailers polluted 357 queries.** Old pandas PRs append the author
   line and every squashed sub-commit subject with its SHA. That block describes
   the *review history*, not the bug, dragging a dozen hex SHAs and unrelated
   vocabulary into the query. Now stripped.
4. **A leading space defeated every anchored pattern.** Several pandas commits
   are written `" DOC: ..."` with a leading space, so `^DOC` never matched and
   five docstring commits were labeled as bug fixes. Fixed by normalizing the
   message inside the `CommitInfo` type itself, so no construction path can skip
   it — with a regression test.

Those fixes removed 240 examples (7,706 → 7,466). The dataset got smaller and
better, which is the correct trade.

---

## Honest limitations

Things a careful reader should know, and that an interviewer might reasonably
push on:

**The labels are a proxy, not truth.** A fix commit sometimes touches a file for
tidiness — updating a call site, adjusting an import — rather than because that
file was at fault. We count every non-test source file as gold, which slightly
over-credits retrievers that find *any* touched file. Measuring how often this
happens would require the manual labeling the project is designed to avoid, so
it stays an acknowledged unknown rather than a quantified one.

**Queries are short.** Median query length is 69 characters; the 10th percentile
is 40. A one-line commit subject is much less to work with than a real bug report
with a traceback and a reproduction. Two consequences: absolute accuracy will
look worse than a system fed real issue text, and methods that need lexical
overlap (BM25) may be handicapped relative to how they'd perform in production.
This is the strongest argument for hydrating issue bodies later.

**8.1% of queries still contain PR bullet lists.** Modern GitHub squash merges
append the sub-commit subjects as `* fix pyarrow interchange`, `* mypy`,
`* reduce diff`. It's noise, but stripping every bullet list would also delete
legitimate bug descriptions written in bullets, so it is left in and documented.

**One commit can fix several unrelated things.** Squashed commits sometimes bundle
independent changes under one message, producing a query that describes three
things and a gold set covering all three. Detecting this automatically is hard;
the mega-commit filter catches the extreme cases only.

**Selection bias toward well-described bugs.** We can only mine fixes whose commit
message announces itself as a fix. Bugs fixed silently, or with a message like
"oops", are invisible to us. The dataset therefore over-represents bugs that
maintainers considered worth describing — which likely correlates with them being
easier to describe, and so easier to localize.

**pandas dominance is a generalization risk.** A method tuned on this dataset is
substantially tuned on one codebase's conventions. Per-repo reporting is the
mitigation, but the small repos are too small to fully validate transfer.

---

## Reproducing it

```bash
make mine    # clone/fetch the repos, mine, filter, split, write data/examples.jsonl
make stats   # the tables above
```

The whole run takes about 13 seconds once the repos are cloned (the pandas clone
is ~470 MB and takes a few minutes the first time). Mining reads the entire
history in a single `git log --name-only` subprocess rather than one diff per
commit — the latter is roughly 38,000 subprocesses on pandas and takes tens of
minutes.

Everything that affects what ends up in the dataset — the filter thresholds, the
message patterns, the path exclusions, the split fraction — lives in
`config.yaml`, so a dataset is reproducible from (git SHA + config file). The
per-filter counts are written to `data/mining_funnel.json` alongside the
examples, so the funnel table can be re-rendered without re-mining.
