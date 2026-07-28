# Glossary

Every domain term used in this project, defined plainly and in one place. Terms
are grouped by topic rather than alphabetized, because they're easier to learn in
clusters. Entries marked *(not yet used)* are defined ahead of the milestone that
introduces them, so this file stays useful as a standalone primer.

---

## The task

**Bug localization** — Given a description of a bug, identify the source files
(or functions) responsible for it. Not fixing the bug; just finding where it
lives. Framed here as a ranking problem: output an ordered list, and hope the
culprit is near the top.

**Retrieval** — Finding the most relevant items in a fixed collection, given a
query. Web search is retrieval. Here the "collection" is the source files of a
repository and the "query" is a bug report.

**Corpus** — The collection being searched. In this project, one corpus per
example: all source files of a repository at one specific commit.

**Query** — The text we search *with*. Here: a bug report's title and body, or a
commit message describing the problem.

**Gold files / ground truth / labels** — The correct answer for an example. Here,
the source files that the real fix commit modified.

**Fix commit** — A commit that repairs a bug. Identified by its message
(`BUG: ...`, `fixes #1234`).

**Parent commit** — The commit immediately before another one. For a fix commit,
the parent is the last state of the code that *still had the bug* — which is
exactly the state we must search. Also called the **buggy state**.

**Leakage** — When information the model should not have access to sneaks into
its input, producing scores that look great and mean nothing. The canonical
example here: indexing the repository *after* the fix, so the corrected code is
sitting in the corpus waiting to be matched. Guarded against by always indexing
the parent commit.

**Temporal leakage** — A specific kind of leakage where information from the
future influences a prediction about the past. Avoided here by splitting the data
by time rather than randomly.

**Held-out set** — Examples deliberately not used while developing or tuning,
reserved to produce the final reported numbers. The moment you tune against it,
it stops being held out and your numbers become optimistic.

---

## Retrieval methods

**Sparse retrieval** — Matching on the literal words. Documents are represented
as very long, mostly-zero vectors (one dimension per vocabulary word — hence
"sparse"). Excellent when the query and document share exact tokens, which is
common in code: a bug report often names `send_file` or `TypeError` verbatim.

**Dense retrieval** — Matching on meaning. Documents and queries are mapped to
short vectors of a few hundred real numbers (every dimension carries signal —
hence "dense") such that similar meanings land near each other. Catches
paraphrase, where sparse fails: "upload breaks for large files" can match code
about `chunk_size` even with no shared words.

**TF-IDF** *(term frequency–inverse document frequency)* — The classic sparse
weighting scheme. A word matters more if it appears often *in this document*
(TF), and less if it appears in *many documents* (IDF). It's why matching on
`the` tells you nothing and matching on `send_file` tells you a lot.

**BM25** — The refined, modern successor to TF-IDF, and the standard sparse
baseline in information retrieval. It adds two corrections: (1) *saturation* — a
word appearing 50 times isn't 50× more relevant than once, so the contribution
levels off; (2) *length normalization* — long documents match more words by
chance, so their scores are damped. Tunable via `k1` (saturation rate) and `b`
(length-normalization strength). Despite being from the 1990s, it is a genuinely
hard baseline to beat on code.

**Embedding** — A vector of numbers representing a piece of text, produced by a
neural network trained so that semantically similar texts get nearby vectors.
"The model's opinion about meaning, written as coordinates."

**Cosine similarity** — How similarity between two embeddings is measured: the
cosine of the angle between them. Ranges from -1 to 1; 1 means pointing the same
direction. It cares about *direction*, not magnitude, so a long document and a
short one can still be judged similar.

**pgvector** — A PostgreSQL extension that adds a `vector` column type plus
similarity operators and indexes. It lets us store embeddings in the same
database as everything else and run nearest-neighbour queries in SQL, instead of
running a separate vector database.

**Chunking** — Splitting a long document into smaller pieces before embedding.
Necessary because embedding models have a token limit, and useful because one
vector for a 2,000-line file is too blurry to be discriminative. The tradeoff:
smaller chunks are sharper but lose surrounding context.

**Token / word-piece token** — The unit a neural model actually reads. Text is
split into sub-word pieces, so `send_file` might become `send`, `_`, `file`.
Models have a hard limit on how many they accept — 256 for the model used here —
and anything past it is **silently truncated**, with no error. Code is far denser
than prose: measured at ~2.8 characters per token versus ~4 for English, because
identifiers and punctuation fragment heavily. This is why the chunk size is 700
characters and not a round number someone liked.

**Context length / context window** — The maximum number of tokens a model can
process at once. The single most important spec when choosing an embedding model
for code, because it determines how much chunking you are forced to do.

**Corpus scope** — The decision about which files a retriever is allowed to
return. Here the corpus defaults to exactly the set of files that could be a gold
label (non-test source), so that a "wrong" answer is genuinely wrong rather than
an artifact of the labelling convention. It matters enormously: 80% of pandas'
Python files are tests, so this choice changes the candidate set from 1,405 files
to 282 and therefore changes every accuracy number.

**Content addressing** — Identifying data by a hash of its contents rather than
by a name or location. Git does this for every file version (see **blob hash**),
which is what lets this project store one row per distinct file *version* instead
of one per (commit, path) pair — a 57x reduction on this dataset.

**Max-pooling over chunks** — Scoring a file by the best-scoring chunk it
contains, rather than the average. A file is relevant if *any* part of it is;
averaging would punish a large file containing one highly relevant function.

**Exact vs approximate nearest neighbour (ANN)** — Exact search compares the
query to every candidate vector; ANN (e.g. pgvector's **HNSW** index) trades a
little recall for speed on very large collections. This project uses exact
search, because each query is restricted to one commit's few hundred files and
approximation would inject error into a measurement of recall.

**Hybrid retrieval** — Combining sparse and dense results, on the reasoning that
they fail in different ways, so their errors partly cancel.

**RRF (Reciprocal Rank Fusion)** — A way to merge two ranked lists using only
*positions*, not scores. Each list contributes `1 / (k + rank)` to an item's total
(with `k` a constant, conventionally 60, that damps how much the very top ranks
dominate). Its virtue is needing no score calibration: BM25 scores are unbounded
positive numbers and cosine similarities live in [-1, 1], so adding them directly
is meaningless, whereas "ranked 3rd" means the same thing in both lists.

**Cross-encoder** — A model that takes the query and one document **together** as
a single input and outputs a relevance score. Contrast with the **bi-encoder**
used for dense retrieval, which embeds query and document *separately* and
compares vectors. A cross-encoder can attend to interactions between the two
texts — that this exact identifier appears in both — which a bi-encoder
structurally cannot, since each side is compressed before it sees the other. The
price is that nothing can be precomputed: every (query, document) pair is a
forward pass at query time. Measured here at 180 pairs/second.

**Reranking** — The two-stage pattern that follows: a cheap retriever narrows
hundreds of files to a shortlist, then an expensive model reorders just those.

**Shortlist ceiling** — Because reranking only reorders the shortlist, the
first-stage retriever's accuracy *at the shortlist depth* is a hard upper bound
on what reranking can reach. If hybrid's top-25 is 0.93, no reranker can push
top-10 past 0.93 from a top-25 shortlist. A rerank gain is only interpretable
against the headroom it actually had, so this project reports the ceiling next to
the result.

**McNemar's test** — A paired significance test for two methods evaluated on the
same examples. Examples both methods get right, or both get wrong, carry no
information about which is better; only the disagreements do. The accuracy
difference is `(a_only − b_only) / n` with standard error `sqrt(a_only + b_only)
/ n`, which is typically much tighter than treating the two accuracies as
independent samples. `z` is how many standard errors the difference sits from
zero; |z| > 1.96 is the conventional 5% threshold.

**MS MARCO** — A large dataset of web search queries paired with relevant
passages, and the training data behind most off-the-shelf cross-encoders. Worth
knowing because a model trained on natural-language web passages is being asked
here to judge relevance between a bug report and Python source, which is a
substantial domain shift.

**Domain shift** — When a model is applied to data drawn from a different
distribution than it was trained on, and its learned notion of the task no longer
transfers. The concrete case in this project: an MS MARCO cross-encoder has
learned what makes a *web passage* relevant to a *search query*, and is being
asked to judge what makes *Python source* relevant to a *bug report*. Different
vocabulary, different structure, and a different meaning of "relevant" — an
identifier match should outweigh a prose overlap, which nothing in its training
taught it. Measured here as a 7-point drop in top-10 accuracy relative to the
ranking it was given. The important discipline is scoping the conclusion to the
model and domain tested, rather than to reranking in general.

**Negative result** — A finding that an approach does *not* work. Useful when it
is measured carefully enough to be trusted and diagnosed well enough to be
actionable — here, that a general-domain reranker degrades code retrieval, which
points at a code-trained reranker as the fix rather than at abandoning the
two-stage pattern.

---

## Metrics

**Top-k accuracy** — The fraction of examples where *at least one* gold file
appears in the top k results. If top-5 accuracy is 0.60, then for 60% of bug
reports a developer looking at the top five suggestions would find a genuinely
faulty file. This is the headline metric because it maps directly onto how the
tool would be used.

**MRR (Mean Reciprocal Rank)** — For each example, take the rank of the *first*
correct result and compute `1 / rank`; then average across examples. Rank 1 scores
1.0, rank 2 scores 0.5, rank 10 scores 0.1. It rewards putting a correct answer
high rather than merely somewhere in the list — the distinction top-k accuracy
throws away.

**MAP (Mean Average Precision)** — Like MRR, but it credits *all* correct results
rather than just the first. For each example, compute precision at every position
where a gold file appears, average those, then average across examples. Matters
here because many bugs have multiple gold files, and a ranking that surfaces three
of four is better than one that surfaces one of four.

**Precision / Recall** — Precision is the share of returned results that are
correct; recall is the share of correct results that were returned. Ranking
metrics are built from these evaluated at each cutoff.

**Standard error** — Roughly, how much a measured accuracy would wobble if you
re-ran it on a different sample of the same size. For an accuracy near 0.5 it is
about `0.5 / sqrt(n)`, so 325 examples gives ~2.8 points. It is what tells you
whether a gap between two methods is a real difference or noise: on 325 examples
a 6-point gap is meaningful and a 2-point gap is not.

**Peek / peek ledger** — A record of every evaluation run against the held-out
set. Each look at held-out data lets a little judgement leak into decisions that
follow, so the number of looks is logged as an honest upper bound on how much.

**Aggregate composition** — The per-repo makeup of an evaluation set. Reported
alongside every aggregate number here, because an "aggregate" over a set that is
91% one repository is that repository's number wearing a cross-repo label.

---

## Tooling & method

**Self-labeling / distant supervision** — Deriving labels from a signal that
already exists in the data rather than from human annotation. Here, git history
provides labels for free. The tradeoff is label *noise*: the signal is a proxy
for truth, not truth itself (a fix commit may touch a file for tidiness rather
than because it was at fault).

**Mega-commit** — A commit touching a large number of files (>10 by default here),
typically a refactor, merge, or release. Excluded, because treating all of its
files as "the bug" would inflate accuracy for free — a wide enough net catches
anything.

**Mining funnel** — The per-filter tally of how many commits each rule removed,
in the order the rules ran. Each rejected commit is attributed to the *first*
rule that caught it, so the counts sum to the total scanned. Publishing the
funnel is what makes a self-labeled dataset auditable: it shows exactly how
7,466 examples were selected out of 50,490 commits.

**Borderline example** — An example that only just survived the filters (a very
short message, a fix keyword appearing only in the body, exactly at the
mega-commit threshold). Marked rather than dropped, so that manual label review
can deliberately over-sample the edges — a random sample of a mostly-clean
dataset shows you mostly-clean examples and teaches you nothing about where the
labels fail.

**Query scrubbing** — Removing text from the query that would give away the
answer. Here: deleting literal gold file paths from a commit message that names
the file it changed. Only full paths are removed, never bare module names, since
stripping `groupby` from "groupby.apply raises" would delete the bug description
itself.

**Gold reachability** — The requirement that every gold file actually exists in
the corpus being searched. A fix that *creates* a file produces a label no
retriever could ever return, which puts a silent ceiling on accuracy. Checked
here against the parent commit.

**Blob** — Git's term for the stored contents of a single file version. See
**blob hash**.

**Blob hash** — Git's content-addressed identifier for a file's contents. Two
files with identical content have the same blob hash regardless of path or
commit, which is what lets us embed each distinct file version exactly once
across thousands of near-identical checkouts.

**Spectrum-based fault localization (SBFL)** *(not implemented — explicitly a
non-goal)* — A different family of techniques that *runs* the test suite and
records which lines each passing and failing test executes. Lines executed mostly
by failing tests are ranked suspicious. **Ochiai** and **Tarantula** are the two
standard formulas for that suspiciousness score. Powerful, but it requires
actually executing every historical version of the test suite, which is
enormously expensive. Named here so the contrast is clear: our approach is purely
textual and never runs the code.

---

*Add terms as they come up. Every term used in another doc should have an entry
here.*
