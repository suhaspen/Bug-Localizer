# Evaluation — what the numbers mean and how they were produced

*Self-contained: you can read this without the code and without the other docs.*

Milestone 1 produced 7,445 labeled examples — a bug description plus the files
that actually had to change. Milestone 2 built two ways to search a repository.
This document is about the only question that matters: **do they work, and which
one works better?**

---

## 1. The metrics, in plain terms

All three metrics answer "did we put a correct file near the top?", but they
disagree about what counts, and the disagreement is exactly why we report all
three. Throughout, take this one example:

> **Gold files:** `core/indexing.py` and `core/frame.py`
> **Our ranking:** 1. `core/series.py`  2. `core/indexing.py`  3. `io/csv.py`
> 4. `core/frame.py`  5. `core/base.py`

### Top-k accuracy

**Definition.** The fraction of examples where *at least one* gold file appears
in the first k results.

For our example: top-1 is a **miss** (rank 1 is `series.py`, not gold). Top-5 is
a **hit** (`indexing.py` is at rank 2). Binary — no partial credit.

**What "top-5 accuracy of 0.60" actually means.** Take 100 bug reports. For 60 of
them, a developer who opened the five suggested files would find a genuinely
faulty file among them. For the other 40, all five suggestions are wrong. It says
nothing about *where* in those five the right answer sat, and nothing about
whether the other gold files were found.

This is the headline metric because it maps directly onto how the tool would be
used: a developer scans a short list and needs one real lead to start pulling the
thread.

**Why "any gold file" rather than "all".** 77% of our examples have exactly one
gold file, so the distinction rarely bites; and when a fix touches four files, a
developer who finds one of them has what they need to find the rest. Requiring
all four would measure something stricter than the task.

### MRR — Mean Reciprocal Rank

**Definition.** For each example take the rank of the *first* gold file and
compute `1 / rank`; average across examples.

For our example: the first gold file is at rank 2, so the reciprocal rank is
`1/2 = 0.5`.

The scale is deliberately steep — rank 1 scores 1.0, rank 2 scores 0.5, rank 3
scores 0.33, rank 10 scores 0.1. That encodes something top-k accuracy throws
away: **being second is much better than being tenth**, even though top-10
accuracy scores them identically. If two methods have the same top-10 but one has
a much higher MRR, that one is putting answers higher in the list and is the
better tool in practice.

MRR is bounded above by top-1 accuracy plus a fraction, and a useful sanity
check is that MRR always sits between top-1 and top-10 accuracy.

### MAP — Mean Average Precision

**Definition.** For each example, look at every position where a gold file
appears, compute precision up to that point, average those, then average across
examples.

For our example, with two gold files:

| position | file | gold? | precision so far |
| ---: | --- | :---: | ---: |
| 1 | `core/series.py` | | |
| 2 | `core/indexing.py` | ✓ | 1/2 = 0.500 |
| 3 | `io/csv.py` | | |
| 4 | `core/frame.py` | ✓ | 2/4 = 0.500 |

Average precision = (0.500 + 0.500) / 2 = **0.500**.

The divisor is the number of **gold files** (2), not the number found. So failing
to rank a gold file at all is penalised rather than quietly ignored.

**What MAP adds over MRR.** MRR stops at the first correct answer; MAP credits
all of them. Consider an example with four gold files where a method ranks one at
position 1 and misses the rest: MRR gives it a perfect 1.0, MAP gives it 0.25.
MAP is the only one of the three that notices a method is finding one file and
declaring victory.

**A property to expect in our results.** When an example has exactly one gold
file, MAP reduces exactly to reciprocal rank. Since 77% of our dataset is
single-gold, the MAP and MRR columns should track each other closely, and MAP
should sit slightly *below* MRR (the multi-gold examples drag it down). If they
ever diverged wildly, something would be wrong. There is a unit test pinning this
equivalence.

---

## 2. The evaluation setup

### The split: temporal, per repo, 70/30

Each repo's examples are sorted by commit date; the oldest 70% is the **dev set**
and the newest 30% is **held out**.

Nothing here is trained — BM25 and the embedding model are used off the shelf. So
"dev set" really means *the set on which hyperparameters may be tuned*: chunk
size, the RRF constant, BM25's `k1` and `b`, rerank depth later. Those are fitted
to data by hand, and fitting them on the same examples you report on makes the
report optimistic.

**Why temporal rather than random.** Because it matches how the tool would be
used: you tune on past bugs and are judged on future ones. A random split would
let you tune on a 2024 bug and evaluate on a 2019 bug touching the same file.

That is **temporal leakage**, and it is worth being concrete about why it matters
here rather than treating it as ritual. Codebases have structure that persists:
`pandas/core/indexing.py` has been the answer to indexing bugs for a decade. If
you tune your chunk size while looking at 2024 indexing bugs and then evaluate on
2019 indexing bugs, you have — indirectly, through your own choices — used
knowledge of the future to configure a prediction about the past. The effect is
small for a system with five hyperparameters, but it is real, it is free to
avoid, and "I used a random split" is a bad answer to an obvious question.

**Per repo, not one global cutoff date.** The three repos have very different
activity periods; a single calendar cutoff would let pandas dominate the eval set
and leave flask barely represented, making per-repo comparison impossible.

### The held-out peek ledger

Every held-out evaluation appends a line to `results/heldout_log.jsonl` with a
timestamp, git SHA, corpus scope and example count. The running count is printed
before each run, so the cost of another look is visible before you take it.

This is not bureaucracy. A held-out set stops being held out the moment you start
making decisions based on it — and the decisions are rarely explicit. You run the
eval, see dense underperform, change the chunk size "because 700 was arbitrary
anyway", and run again. Nothing dishonest happened, but the held-out number now
has some of your judgement baked in. The number of looks is the only cheap upper
bound on how much, so it is recorded rather than remembered.

Tuning belongs on the dev split: `make eval-dev`.

### Two corpus scopes, reported side by side

The default corpus contains only files that could be a gold label — non-test
source. This is coherent (a method ranking a test file first would otherwise be
scored wrong for an artifact of our labelling) and standard in the literature,
but it makes the task **substantially easier**: 80% of pandas' Python files are
tests, so the candidate set shrinks from ~1,400 files to ~290.

So every evaluation runs twice, and both numbers are reported:

- **tests excluded** (default) — the coherent setting, and the one comparable to
  published bug-localization results.
- **tests included** — the harder, more realistic whole-repo search, where the
  retriever must also learn to avoid the test files that discuss the bug in
  almost the same words as the report.

The second number is the honest one to quote when someone asks "how well would
this actually work?". Reporting only the first would be defensible-but-flattering;
reporting both costs one extra run and removes the question.

### Aggregate numbers, and why the composition table comes first

The eval prints a per-repo table for each repo *and* an aggregate — but the
aggregate is always preceded by a composition table, and a warning fires when any
repo exceeds 50% of the eval set.

The reason is that our dataset is extremely lopsided: pandas is 91% of the full
held-out set. An aggregate over that is a pandas number with a rounding error of
flask attached. Calling it "cross-repo accuracy" would be misleading, so the tool
refuses to print it without the composition beside it.

---

## 3. What is being compared

| method | one-line description |
| --- | --- |
| **BM25** | Sparse keyword ranking. Scores a file by how many *rare* query terms it contains, with saturation (`k1=1.5`) and length normalisation (`b=0.75`). |
| **dense** | Embed query and code into 384-d vectors; rank by cosine similarity. A file scores as the maximum over its chunks. |
| **hybrid** | Reciprocal Rank Fusion of the two lists above. |

### Why RRF rather than adding the scores

BM25 scores are unbounded positive numbers whose scale depends on corpus
statistics — around 35 on one repo, 6 on another. Cosine similarities live in
[-1, 1]. Adding or averaging them is meaningless, and normalising first (min-max,
z-score) requires assuming a distribution shape that neither actually has, and is
badly distorted by a single outlier at the top of a list.

**RRF throws the scores away and uses only rank position.** Each list contributes
`1 / (k + rank)` to a file's total, with `k = 60` by convention. "Ranked 3rd"
means the same thing in both lists, so no calibration is needed.

A structural property worth knowing, because it is the mechanism by which hybrid
helps: **a file ranked 2nd in both lists always beats a file ranked 1st in only
one**, for every `k ≥ 0` — since `2/(k+2) > 1/(k+1)` always. Agreement between
two independent retrievers outweighs one retriever's confident top pick. That is
pinned by a unit test so it cannot be tuned away by accident.

What `k` does control is how steeply rank matters further down the list. With
`k = 60`, ranks 1 and 2 contribute 1/61 and 1/62 — nearly equal, so fine
distinctions near the top are deliberately flattened. With `k = 1` they are 1/2
and 1/3, so a top rank dominates.

---

## 4. Results

**Eval set: 337 held-out examples** — flask 85 (25.2%), pandas 132 (39.2%),
requests 120 (35.6%). Note this is *not* the full 2,235-example held-out set:
pandas is capped at its newest 132 held-out examples for indexing cost, which
has the side effect of making the aggregate genuinely three-repo rather than 91%
pandas. Both corpus scopes were evaluated over exactly these same examples.

### Tests excluded (default scope) — mean 141 candidate files per query

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.406 | 0.760 | 0.843 | 0.557 | 0.546 |
| dense | 0.341 | 0.685 | 0.834 | 0.497 | 0.482 |
| **hybrid (RRF)** | **0.415** | **0.768** | **0.872** | **0.575** | **0.564** |

### Tests included (whole-repo search) — mean 633 candidate files per query

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.255 | 0.582 | 0.751 | 0.416 | 0.409 |
| dense | 0.285 | 0.582 | 0.721 | 0.426 | 0.411 |
| **hybrid (RRF)** | **0.318** | **0.682** | **0.804** | **0.474** | **0.462** |

### Per repo, tests excluded

| repo | method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| flask (85) | BM25 | 0.459 | 0.765 | 0.824 | 0.582 | 0.575 |
| | dense | 0.459 | 0.718 | 0.882 | 0.584 | 0.568 |
| | hybrid | **0.482** | 0.729 | **0.906** | **0.594** | **0.585** |
| pandas (132) | BM25 | 0.424 | 0.773 | 0.864 | 0.580 | 0.563 |
| | dense | 0.318 | 0.682 | 0.849 | 0.483 | 0.467 |
| | hybrid | **0.447** | **0.826** | **0.894** | **0.617** | **0.600** |
| requests (120) | BM25 | **0.350** | **0.742** | **0.833** | **0.515** | 0.507 |
| | dense | 0.283 | 0.667 | 0.783 | 0.451 | 0.437 |
| | hybrid | 0.333 | 0.733 | 0.825 | **0.515** | **0.509** |

---

## 5. Reading the table

### Hybrid wins, and it wins for the reason RRF predicts

Hybrid is best on 9 of 10 aggregate cells across both scopes. In the default
scope it reaches **top-10 of 0.872** — for 87% of bug reports, a developer
handed ten files would find a genuinely faulty one — and **top-1 of 0.415**,
meaning the single best guess is right about 42% of the time.

This is what RRF is supposed to do: BM25 and dense fail on different examples, so
a file both retrievers rank highly beats one that only a single retriever loved.
The gain over BM25 alone is modest in the default scope (+2.9 points top-10, +1.8
MRR), and it is worth being careful here — with 337 examples the standard error
on a single accuracy is about 2.7 points, so the *aggregate* gap alone is at the
edge of what this sample can resolve. What makes it credible is that the gain is
consistent: hybrid leads on 4 of 5 metrics for flask, 5 of 5 for pandas, and on
every aggregate metric in both scopes. Consistency across independent slices is
stronger evidence than one gap clearing a threshold.

### BM25 beats dense — but only while the corpus is easy

In the default scope BM25 is clearly ahead of dense on the aggregate: **0.406 vs
0.341 top-1**, 0.557 vs 0.497 MRR. That is the un-glamorous result and it is the
honest one. Bug reports are dense with identifiers — `DataFrame.loc`,
`IntervalIndex`, `TypeError` — and a 1990s keyword-matching algorithm exploits
that better than a 384-dimensional embedding of a 700-character window.

Then the corpus gets harder and **the ordering reverses**:

| scope | BM25 top-1 | dense top-1 |
| --- | ---: | ---: |
| tests excluded | **0.406** | 0.341 |
| tests included | 0.255 | **0.285** |

BM25 loses **15.1 points** of top-1 when test files enter the corpus; dense loses
**5.6**. The effect is sharpest on pandas, which has by far the most tests:

| pandas, top-1 | tests excluded | tests included | change |
| --- | ---: | ---: | ---: |
| BM25 | 0.424 | 0.136 | **−0.288** |
| dense | 0.318 | 0.235 | −0.083 |
| hybrid | 0.447 | 0.280 | −0.167 |

BM25's pandas top-1 collapses by 68% relative. The explanation is
straightforward once you look at what a test file contains: a test for a
`send_file` mimetype bug is called `test_send_file_mimetype`, and its body
restates the bug report almost verbatim. To a purely lexical matcher, the test
file is *the single best keyword match in the repository* — and it is never a
correct answer. Dense retrieval is less easily fooled, because a test file is
semantically test-shaped in a way an embedding partly captures.

This is the most interesting finding in the milestone, and it is one that would
have been invisible reporting only the default scope. It also reframes the
sparse-vs-dense question: dense retrieval's value here is not that it finds
things BM25 misses, but that it is **more robust to distractors**.

### Where it doesn't work: requests

requests is the one slice where hybrid does **not** beat BM25 — 0.333 vs 0.350
top-1, 0.825 vs 0.833 top-10, with MRR tied at 0.515. Dense is distinctly weaker
here (0.283 top-1), so fusing it in costs about as much as it adds.

A plausible reason is that requests' held-out window stretches back to 2015 and
much of its corpus is vendored third-party code (`requests/packages/urllib3/...`)
— topically similar to everything, which is exactly the situation where
embedding-based similarity is least discriminative. This is a hypothesis, not a
measured result, and it is the kind of thing an error analysis in a later
milestone should settle rather than something to assert now.

Reporting it matters more than explaining it: "hybrid always wins" would have
been a cleaner story and a false one.

### The sanity checks all hold

- **MRR sits between top-1 and top-10** in every cell, as it must.
- **MAP tracks just below MRR** everywhere (e.g. 0.564 vs 0.575 for hybrid) —
  exactly what is expected when 77% of examples have a single gold file, since
  MAP reduces to reciprocal rank in that case and only the multi-gold examples
  pull it down.
- **Making the corpus 4.5x larger (141 → 633 candidates) hurt every method**, in
  the right direction and by a plausible amount.

If any of these had come out otherwise, the metric implementation would be the
first suspect.

### What the numbers do *not* say

The absolute values are optimistic relative to a real deployment, for reasons
documented elsewhere and worth repeating here:

- **Queries are commit messages**, written by someone who already knew the
  answer, and averaging 69 characters. A real bug report is longer, vaguer, and
  written by someone who does not know which file is at fault.
- **The default scope excludes tests**, which the second table quantifies as
  worth roughly 9 points of top-10 and 15 of top-1.
- **The pandas slice is its newest 132 held-out examples**, not the full 2021+
  window, so pandas is represented by its most recent code.

The comparison *between* methods is unaffected by all three, because every method
sees the same queries and the same corpus. That comparison is what this project
is for.

---

## 6. Reproducing

```bash
make db-up
make index          # embed the eval set's parent commits
make eval           # both corpus scopes, writes results/<timestamp>.json
make peeks          # how many times held-out has been evaluated
```

`make eval-dev` runs the same thing on the dev split, which is where tuning
belongs. Every run records its git SHA and the full retrieval config, so a number
is reproducible from (SHA + config) without needing to know what was in anyone's
shell.
