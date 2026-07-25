# Retrieval — sparse, dense, and the corpus they search

*Self-contained: you can read this without the code and without the other docs.*

Milestone 1 produced 7,445 labeled examples: a bug description, and the files
that actually had to change. This document is about the other half — given that
description, how do we search a repository and produce a ranked list of files?

There are two fundamentally different ways to do it, they fail in different
ways, and most of the interesting engineering is in the corpus they both search.

---

## 1. The two kinds of search

### Sparse retrieval: match the words

The oldest idea in information retrieval is to score a document by how many of
the query's words it contains. Refined slightly, it becomes surprisingly hard to
beat.

The refinement everyone starts with is **TF-IDF**. *Term frequency* says a word
appearing often in this file is evidence the file is about that word. *Inverse
document frequency* says a word appearing in almost every file tells you nothing.
Matching on `the` is worthless; matching on `send_file` is enormously
informative, precisely because `send_file` appears in two files out of three
hundred.

**BM25** is the standard modern version, and it adds two corrections that both
matter for code:

- **Saturation.** A file mentioning `groupby` fifty times is not fifty times more
  relevant than one mentioning it twice. BM25 lets each additional occurrence
  count for less, approaching a ceiling. The parameter `k1` (we use 1.5) controls
  how quickly it saturates.
- **Length normalisation.** A 400 KB file contains more words than a 2 KB file
  and would win every query by sheer size. BM25 penalises length, with parameter
  `b` (we use 0.75) controlling how aggressively. Pandas has files spanning 200
  bytes to 400 KB, so this is not a subtlety here — without it, `pandas/core/frame.py`
  would rank first for everything.

Why sparse retrieval is genuinely strong for this task: bug reports are full of
identifiers. A report saying "`DataFrame.loc` raises on an `IntervalIndex`"
shares literal tokens with the file that defines `loc`. That is the ideal case
for keyword matching, and it is why BM25 is a real baseline here rather than a
strawman.

Its blind spot is vocabulary. If a user writes "uploading big files times out"
and the responsible code talks about `chunk_size` and `deadline`, there is no
word overlap, and BM25 scores zero. It cannot match on meaning, only on spelling.

#### Tokenisation, which matters more than you'd think

Off-the-shelf English tokenisers are wrong for code. `send_file` should match a
query saying "send file", and `DataFrame` should match "data frame". So the
tokeniser emits identifiers **both whole and split** on `snake_case` and
`camelCase` boundaries:

```
"DataFrame.to_csv"  ->  dataframe, data, frame, to_csv, to, csv
```

The whole form is kept because an exact identifier match is the strongest signal
available, and the parts are added because a user describing a bug rarely types
the identifier exactly. This is also one concrete reason Postgres full-text
search was not used — its English stemmer would mangle `send_file` rather than
split it.

### Dense retrieval: match the meaning

The other approach maps text to **embeddings**: lists of numbers (here, 384 of
them) produced by a neural network trained so that things with similar meanings
end up as nearby vectors. Both the query and every piece of code get embedded,
and we rank by **cosine similarity** — the cosine of the angle between two
vectors, which is 1 when they point the same way and 0 when unrelated. Angle
rather than distance, so a long document and a short one can still be judged
similar.

This is exactly what BM25 cannot do: "uploading big files times out" can land
near code about `chunk_size` because the model learned during training that those
concepts are related, with no shared words at all.

Its blind spot is the mirror image. Embeddings are lossy summaries. A 384-number
vector cannot encode that *this specific file* defines `send_file` as opposed to
some conceptually similar function. Dense retrieval is good at topic and bad at
precision, which is why hybrid fusion (Milestone 3) usually beats either alone.

**pgvector** is the piece that makes this practical: a PostgreSQL extension that
adds a `vector` column type and similarity operators, so nearest-neighbour search
happens in SQL, in the same database as everything else, rather than requiring a
separate vector database.

---

## 2. The corpus: what is actually searchable

Before either method runs, something has to decide *what set of files it may
return*. That turns out to be the most consequential decision in the milestone.

### A corpus entry is content, not a file

Every example searches its own commit — the parent of a fix, the buggy state.
Naively that means one corpus per example, and the arithmetic is grim: across
pandas' examples there are **5,398,769** (commit, path) file-instances. Embedding
that many things is not happening on a laptop.

But git already content-addresses every file version with a **blob hash**, and
two commits that didn't touch a file share its blob exactly. So we key storage on
blob hash rather than (commit, path). Measured across every Python file in
pandas' parent trees, 5,398,769 file-instances collapse to **93,848 distinct
blobs — a 57x reduction**. (That measurement predates the corpus-scope decision
below, so it covers all `.py`; restricting to the label-space corpus reduces both
sides of the ratio and leaves the order of magnitude unchanged.) This single
property is what makes dense retrieval feasible here at all.

A useful consequence: paths are not stored at all. A blob is content, and the
same content can appear at different paths in different commits, so paths come
from `git ls-tree` at query time.

### Corpus = label space

The second decision is which files belong in the candidate set. By default the
corpus is **exactly the set of files that could be a gold label** — non-test
Python source, using the same exclusions the miner applies.

The reasoning is coherence. Our labels can never be a test file, because the
miner strips them. If test files were still searchable, a method that ranked
`pandas/tests/test_frame.py` first would be scored as a *miss* — even though that
test is genuinely about the bug. We would be penalising sensible behaviour for a
reason that is an artifact of our labelling convention, not a real error. Making
the candidate set equal the label space removes that incoherence, and it matches
standard practice in the bug-localization literature, where the corpus is the
main source tree and test directories are excluded.

It is also, unavoidably, a large simplification, and it must be stated plainly
because it changes every number this project will report:

| pandas, one commit | files | bytes |
| --- | ---: | ---: |
| all `.py` | 1,405 | 18.0 MB |
| excluded (tests, docs, benchmarks, tooling) | 1,123 (**80%**) | 11.5 MB |
| **corpus** | **282** | **6.5 MB** |

Eighty percent of pandas' Python files are tests. Ranking within 282 candidates
is a substantially easier task than ranking within 1,405, and accuracy numbers
will be correspondingly higher than a whole-repo search would give. Anyone
comparing these numbers to another system must check that system's corpus scope
first. `corpus.include_tests: true` restores the harder setting, and the flag
exists precisely so the claim can be checked rather than trusted.

A related label-quality fix fell out of this. Old flask kept its test suite
inside the package at `flask/testsuite/`, which the `**/tests/**` glob does not
match — it was gold for 44 examples. Adding `**/testsuite/**` removed them. Note
the exclusion is deliberately *not* a broad match on "test": `flask/testing.py`
and `pandas/util/testing.py` are shipped testing *utilities*, part of the public
API, so bugs in them are real bugs and they stay eligible. Over-broad matching
would have silently deleted 80 legitimate examples.

---

## 3. Chunking: a decision forced by the model

**Chunking** means splitting a file into smaller pieces before embedding. It is
not optional, and its size is not a free parameter.

The embedding model, `all-MiniLM-L6-v2`, has a hard limit of **256 word-piece
tokens**. Anything longer is silently truncated — no error, no warning, the tail
of the file simply never gets embedded. And code tokenizes densely: measured on
pandas, roughly **2.8 characters per token**, versus about 4 for English prose,
because identifiers and punctuation fragment. So the model can see about **700
characters of Python at a time.**

The median pandas corpus file is 8,780 characters. One embedding per file would
capture the first 8% of it.

### What was measured

Four strategies, on one pandas commit tree:

| strategy | units/file | median unit | verdict |
| --- | ---: | ---: | --- |
| **A. whole file** | 1.0 | 5,231 ch | Rejected — 92% of the median file silently truncated, and a 400 KB file reduced to its first paragraph |
| **B. fixed window, 2000 ch** | 7.6 | 2,000 ch | Rejected — still ~65% truncated per chunk. Cheap, but cheap and wrong |
| **C. AST function/class** | 14.6 | 1,070 ch | Rejected — see below |
| **D. fixed window, 700 ch** | 20.7 | 700 ch | **Chosen** — median 235 tokens, fits the budget |

**Why not AST-based chunking**, which is the intuitively appealing option? It
splits at function and class boundaries, so each unit is semantically coherent
rather than arbitrarily cut. Three reasons it lost:

1. It doesn't reduce the unit count. Measured 14.6 units/file versus 20.7 — the
   same order, because the binding constraint is the token budget, not the
   splitting strategy. A 3,000-character function still has to be split.
2. It doesn't avoid truncation. The median Python function measured **536
   tokens** — more than twice the model's limit. "One function per chunk" sounds
   like it respects semantics but silently truncates the majority of functions.
3. It adds a failure mode. `ast.parse` fails on any file that isn't valid modern
   Python, which across 17 years of history is a real population, requiring a
   fallback path anyway.

The honest summary: AST chunking is the better idea in principle, and it becomes
the right one with a long-context model. With a 256-token budget its semantic
advantage is destroyed by truncation before it can help. Milestone 5 needs
function-level units for a different reason — localizing *to* functions — and
that is where the AST parser earns its place.

**Overlap.** Windows overlap by 70 characters (10%). A hard cut at 700 characters
will sometimes land in the middle of the one function that answers the query,
leaving neither half with enough context to match. Overlap costs 10% more units
and guarantees every region appears intact in at least one window.

**Storage.** Chunks are stored as `(start_char, end_char)` offsets into the file
content, not as copies of the text. The content is stored once; a chunk is a view
into it. With 10% overlap, storing chunk text would have meant a second, larger
copy of the entire corpus.

**Aggregation.** A file's dense score is the **maximum** over its chunks, not the
average. A file is relevant if *any* part of it is relevant; averaging would
punish a large file containing one highly relevant function, which is precisely
the case we care about.

### Why this model

The obvious alternative was a longer-context model that would need fewer chunks.
Both candidates were measured:

| model | context | units/file | throughput | full corpus |
| --- | ---: | ---: | ---: | ---: |
| **all-MiniLM-L6-v2** | 256 tok | 36.9 | **211 units/s** | **58 min** |
| bge-small-en-v1.5 | 512 tok | 18.7 | 31 units/s | 201 min |
| jina-embeddings-v2-base-code | 8192 tok | ~1.5 | — | incompatible |

`bge-small` halves the chunk count, exactly as intended — and still loses by 3.5x,
because it is **6.8x slower per unit**. Twice the context costs more than twice
the compute per unit, and the unit saving doesn't pay for it. This is a good
illustration of why the decision had to be measured: the units/file column alone
predicts the wrong winner.

The code-native model would have been the interesting one — an 8192-token context
fits most whole files in a single unit, eliminating both truncation and chunking.
It could not be loaded: its remote model code calls
`find_pruneable_heads_and_indices`, removed from current `transformers`. Pinning
an older `transformers` to accommodate it was judged not worth the dependency
risk for a first pass. This is the most promising single upgrade available to the
project, and it is recorded as such rather than quietly dropped.

---

## 4. The BM25 implementation question, settled

Milestone 0 recorded a *provisional* decision (D7) to use the `rank_bm25`
library rather than Postgres full-text search, flagged for confirmation with real
timings. The concern was concrete: a fresh sparse index must be built for every
example, because each one searches a different commit and BM25's IDF statistics
depend on the corpus. At 2,241 held-out examples, a two-second build is 75
minutes of pure overhead.

Both options were measured on real pandas commits.

**Does the in-memory index bite at pandas scale?** Memory: no, not remotely. A
BM25 index over a pandas corpus occupies about **5 MB** of heap, with process
peak RSS around 346 MB. There is no memory problem.

Time: yes, if done naively — and the fix is a cache. Measured over six pandas
held-out commits (corpus ≈ 290 files, ≈ 986,000 tokens):

| stage | cost | cacheable? |
| --- | ---: | --- |
| `git ls-tree` | 323 ms | no (per commit) |
| `git cat-file`, whole tree | 79 ms | no |
| **tokenisation** | **853 ms** | **yes — by blob hash** |
| BM25 index build | 171 ms | no (IDF is per corpus) |
| query scoring | 1 ms | no |
| **total per example** | **1,427 ms** | → **53 min** for 2,241 examples |
| **with token cache** | **574 ms** | → **21 min** |

Tokenisation dominates at 60% of the cost, and it is the one stage that caches
perfectly: the same file version appears in hundreds of consecutive commits, and
its token list is identical every time. A bounded LRU over tokenised blobs turns
the dominant cost into a lookup. Examples are processed in commit order so the
working set stays resident.

Note what becomes the bottleneck afterwards: `git ls-tree` at 323 ms, now larger
than the BM25 build itself. If sparse retrieval ever needs to get faster, that is
the next thing to attack, not the index build that prompted this investigation in
the first place.

**Postgres full-text search lost on both counts.** Building the equivalent index
(COPY + `to_tsvector` + GIN) measured ~4.0 s per commit — slower than the whole
`rank_bm25` path, because generating tsvectors server-side over megabytes of
source is not cheap. And more importantly it would have been the wrong thing:
`ts_rank_cd` is **not BM25**. It is a different weighting scheme with no `k1`
saturation and no `b` length normalisation. Shipping it and labelling the column
"BM25" in a results table would have been a straightforwardly false claim.

**D7 is confirmed: `rank_bm25`.** `BM25Okapi` is a direct implementation of the
textbook Okapi BM25 formula with `k1` and `b` exposed as parameters, which also
means the baseline can be explained rather than merely invoked.

---

## 5. How a query is answered

```
example (query_text, parent_sha, repo)
         │
         ├─ git ls-tree parent_sha ──► candidate files (path, blob_sha)
         │
         ├─ BM25 ───────────────────────────────────────────────┐
         │    git cat-file → contents                           │
         │    tokenize (LRU by blob_sha)                        │
         │    BM25Okapi(corpus).get_scores(tokenize(query))     ├─► ranked
         │                                                      │   file list
         └─ DENSE ──────────────────────────────────────────────┘
              embed(query_text) → 384-d unit vector
              SELECT blob_sha, MIN(embedding <=> query)
                FROM chunk WHERE repo=… AND blob_sha = ANY(…)
                GROUP BY blob_sha
              file score = 1 − min distance  (max similarity over chunks)
```

Two implementation notes worth defending:

**Exact vector search, not an approximate index.** pgvector offers HNSW for
approximate nearest neighbours, which matters when scanning millions of vectors.
Here every query is restricted to one commit's corpus — a few hundred blobs, on
the order of 10,000 chunks — and an exact scan over that is already fast. An
approximate index would add recall error to a measurement whose entire purpose is
measuring recall. That is a bad trade at this scale, and it is why there is no
HNSW index in the schema.

**Deterministic tie-breaking.** Both retrievers sort by `(-score, path)`. Most
files score zero on a given BM25 query, and without a tie-break their order would
depend on dictionary iteration, making top-k jitter between runs. Metrics that
move when nothing changed are indefensible.

### The schema

```sql
blob   (repo, blob_sha, n_bytes, content)              -- content, stored once
chunk  (repo, blob_sha, chunk_idx, start_char,
        end_char, embedding vector(384))               -- a window + its vector
indexed_commit (repo, commit_sha, n_files, indexed_at) -- coverage bookkeeping
```

`repo` is part of every primary key even though a blob hash is already globally
unique. That is deliberate denormalisation: this project reports per-repo
results, so index, drop, count, and vector search all need to be cheaply
scopeable to one repository. `indexed_commit` makes indexing resumable and
incremental — re-running `bugloc index` after adding examples costs only the
blobs that are genuinely new.

---

### What it costs in practice

The index built for Milestone 2 covers flask and requests completely, plus the
120 newest pandas held-out examples:

| repo | commits | blobs embedded | chunks | content |
| --- | ---: | ---: | ---: | ---: |
| flask | 299 | 1,520 | 48,862 | 30 MB |
| requests | 386 | 1,931 | 36,295 | 22 MB |
| pandas (slice) | 121 | 844 | 99,145 | 62 MB |
| **total** | **806** | **4,295** | **184,302** | **434 MB** |

The blob cache is doing the work, and its payoff varies in an instructive way:

| run | commits | file instances | blobs embedded | saving | cache hit | time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flask + requests | 685 | 31,855 | 3,451 | 9.2x | 89% | 25 min |
| pandas, newest 120 | 120 | 35,400 | **593** | **59.7x** | **98%** | 21 min |

**One non-obvious property, learned the hard way.** The first attempt at the
pandas slice took a *random* sample of 150 held-out examples, and it was still on
its second commit after several minutes. pandas' held-out window spans 2021–2026,
so a random sample lands neighbours years apart, and two commits five years apart
share almost no blobs — the cache never warms up. The same-sized *contiguous*
slice ran at a 98% hit rate, because adjacent commits differ by a handful of
files. Hence the `--newest` flag. **The cache's payoff is a function of how
clustered in time the commits are, not just how many there are** — which is worth
knowing before Milestone 3 asks for a full held-out index.

Note also that pandas embeds 99,145 chunks from just 844 blobs — about 117 chunks
each. That is not an error: excluding tests removes the small files and leaves the
large core modules, so the average corpus file is ~73 KB.

On disk the index is **434 MB**, dominated by the chunk table — a 384-dimensional
float vector is 1,536 bytes, so vectors outweigh the source text they describe by
a wide margin.

Query-side latency, measured on flask with the model warm: **13 ms** to embed
the query, and tens of milliseconds for the vector scan over a 27-file corpus.
The first query of a process is much slower (~1.2 s) because Apple's MPS backend
compiles kernels on first use — a one-off cost that amortises immediately and
must not be mistaken for per-query cost when reading timings.

---

## 6. What is not here yet

**Hybrid fusion (RRF)** and the actual accuracy numbers are Milestone 3. Nothing
in this document says whether dense retrieval beats BM25 on this dataset — that
question is deliberately not answered by eyeballing a few examples, because
picking illustrative cases after seeing results is how people fool themselves.
The eval harness answers it over 2,241 held-out examples at once.

**Cross-encoder reranking** is Milestone 4.

---

## Reproducing

```bash
make db-up                        # Postgres 17 + pgvector
make index ARGS="--repo flask"    # build the corpus index
make index-stats                  # what is stored
make retrieve ARGS="--repo flask" # rank files for one example, BM25 vs dense
```

Indexing is incremental and cached by blob hash, so re-running after adding
examples only pays for genuinely new file versions.
