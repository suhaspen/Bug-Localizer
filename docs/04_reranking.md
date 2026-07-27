# Reranking — a second, expensive opinion on a shortlist

*Self-contained: you can read this without the code and without the other docs.*

Milestone 3 established the baselines: BM25, dense embeddings, and their RRF
fusion. This document is about adding a fourth, much more expensive stage on top
— and about reporting honestly what it bought.

---

## 1. Bi-encoders and cross-encoders

Everything in the previous milestone used a **bi-encoder**. The query goes
through the model on its own and becomes a vector; every chunk of code goes
through the model on its own and becomes a vector; relevance is the cosine
between them.

The reason that's fast is that the document side is **precomputable**. We embed
the corpus once, store the vectors in pgvector, and a query costs one forward
pass plus a similarity scan. It's also the reason it's limited: at the moment the
two vectors meet, each has already been compressed to 384 numbers *without any
knowledge of the other*. The query's `IntervalIndex` and the code's
`IntervalIndex` were encoded independently, and whatever survived that
compression is all the comparison has to work with.

A **cross-encoder** removes that constraint. The query and one document are
concatenated into a single input and passed through the model *together*, and the
model emits a relevance score directly:

```
bi-encoder    embed("...IntervalIndex...")  ─┐
                                             ├─► cosine ─► score
              embed("def _get_loc(...)")    ─┘

cross-encoder  model("...IntervalIndex..." ++ "def _get_loc(...)") ─► score
```

Because the two texts attend to each other inside the network, the model can
condition on their interaction: it can notice that this specific identifier
appears in both, that the code branches on exactly the type the report mentions,
that the error the report quotes is raised on line four. A bi-encoder cannot
represent "these two texts share a specific rare token" in a fixed-size vector
that was computed before it saw the other text.

**The price is that nothing can be precomputed.** Scoring N candidates costs N
forward passes at query time. Measured on this machine with
`cross-encoder/ms-marco-MiniLM-L-6-v2`:

| device | throughput |
| --- | ---: |
| MPS (Apple GPU) | 180 pairs/s |
| CPU | 74 pairs/s |

At 180 pairs/s, scoring one query against a 633-file corpus would take 3.5
seconds — and that is *per query*, with no caching possible. Against the ~1 ms
BM25 scoring and ~13 ms query embedding of the first-stage retrievers, that is
three orders of magnitude more expensive.

---

## 2. The two-stage pattern, and the ceiling it creates

The standard resolution is **retrieve-then-rerank**:

```
   corpus (hundreds of files)
        │
        │  cheap: BM25 + dense + RRF        ~150 ms
        ▼
   shortlist (top 25)
        │
        │  expensive: cross-encoder          ~550 ms
        ▼
   reordered top 25, tail appended unchanged
```

You pay the expensive model only on a shortlist small enough to afford, and you
get most of its quality for a fraction of its cost.

The consequence that must be stated before looking at any results: **reranking
can only reorder what the shortlist contains.** If hybrid's top-25 accuracy is
0.93, then no reranker on earth can push top-10 past 0.93 using a top-25
shortlist. That number is the *ceiling*, and a rerank gain is only interpretable
against the headroom it actually had. The eval computes and reports it.

This also frames the shortlist depth as a real tradeoff rather than a free
parameter:

- **Deeper shortlist** → higher ceiling, more chances to rescue a gold file
  buried at rank 20 — and linearly more cost.
- **Shallower shortlist** → cheaper, but the ceiling drops toward hybrid's own
  top-10, at which point reranking can only rearrange files that were already in
  the window and cannot improve top-10 at all.

We use **top-25**, which sits meaningfully above the top-10 we report while
keeping the cost at 100 pairs per query.

---

## 3. What text does the reranker actually read?

A cross-encoder scores a (query, document) pair, but our unit of retrieval is a
*file*, and files here run to 400 KB — far beyond the model's 512-token input.
So each candidate file has to be represented by some passage, and the choice
matters.

The options, and why we chose as we did:

**The file's head.** Cheap and unbiased, and for code the head is genuinely
informative — imports, module docstring, class definitions. But the buggy
function may be 3,000 lines down, in which case the reranker is scoring the wrong
part of the file.

**Every chunk, take the max.** Most faithful, and far too expensive: a pandas
core file is ~117 chunks, so 25 candidates would be ~2,900 pairs per query, or 16
seconds.

**The file's most query-relevant chunks (chosen).** For each candidate we take up
to **4 chunks, selected by cosine similarity to the query** using the embeddings
already in pgvector, and score the file as the **maximum** over those. This
targets the reranker at the windows most likely to matter — for a 400 KB file,
the difference between reading the imports and reading the function that raises.

One objection worth pre-empting: does selecting chunks by cosine make the
reranker dependent on dense retrieval, and so bias the comparison? Not in a way
that favours dense. The selection happens *within* an already-chosen candidate
file — it decides which part of that file to read, not which files compete. A
file that only BM25 surfaced still gets its own best-matching windows chosen the
same way. What it does mean is that if the embedding model is poor at locating
the relevant region inside a file, the reranker inherits that weakness, and
that's an honest limitation to name.

Files whose chunks are somehow unavailable fall back to their first 700
characters rather than being dropped, and files that cannot be scored at all sink
to the bottom of the shortlist instead of vanishing — the eval needs a complete
ranking, and silently shortening one method's list would make its top-10
incomparable with the others.

---

## 4. Results

*Filled in below once the run against the expanded held-out set completes.*

---

## 5. Reproducing

```bash
make eval ARGS="--rerank"
```

Adds a fourth `rerank` row to every table, at both corpus scopes, and reports the
shortlist ceiling alongside. As with every held-out evaluation, the run is
appended to `results/heldout_log.jsonl`.
