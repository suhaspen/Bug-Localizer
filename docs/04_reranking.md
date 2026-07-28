# Reranking — a second, expensive opinion that made things worse

*Self-contained: you can read this without the code and without the other docs.*

Milestone 3 established three baselines — BM25, dense embeddings, and their RRF
fusion — on 337 held-out examples. This milestone does two things: it expands the
evaluation to **1,308 held-out examples**, and it adds a fourth stage, a
cross-encoder reranker, on top of the best of them.

The expansion settled a question left open in Milestone 3. The reranker produced
the most useful negative result in the project.

---

## Headline findings

### 1. Hybrid beats both baselines — confirmed, and it consolidated rather than washed out

At n=337 I described the hybrid advantage as "at the edge of what this sample can
resolve." At **n=1,308** it is unambiguous:

| comparison | k | hybrid | other | delta | hybrid wins | other wins | z | sig. 5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| hybrid vs BM25 | 1 | 0.437 | 0.394 | **+0.044** | 167 | 110 | +3.42 | yes |
| hybrid vs BM25 | 5 | 0.794 | 0.723 | **+0.071** | **142** | **49** | +6.73 | yes |
| hybrid vs BM25 | 10 | 0.876 | 0.828 | **+0.048** | 98 | 35 | +5.46 | yes |
| hybrid vs dense | 1 | 0.437 | 0.325 | **+0.112** | 208 | 61 | +8.96 | yes |
| hybrid vs dense | 5 | 0.794 | 0.689 | **+0.105** | 173 | 35 | +9.57 | yes |
| hybrid vs dense | 10 | 0.876 | 0.819 | **+0.057** | 97 | 22 | +6.88 | yes |

The win counts are the legible form. At top-5, on the 191 examples where hybrid
and BM25 disagreed, **hybrid found a faulty file on 142 and BM25 on 49** — close
to three to one. That is not a marginal effect.

**Part of the resolution was a bigger sample; part of it was fixing my
statistics.** In Milestone 3 I compared the two accuracies as though they were
independent samples, using `sqrt(p(1-p)/n)` — which gave a standard error of
about 2.7 points and made a 2.9-point gap look like noise. But they are not
independent samples: **every method scores the same bug reports.** Examples that
are easy for both methods, or hard for both, tell you nothing about which is
better. All the information is in the disagreements.

McNemar's test uses only those, with standard error `sqrt(a_only + b_only) / n`.
At top-10 that is **0.0088** rather than 0.027 — a third the width. The gap was
always real; I had been measuring it with the wrong instrument. Switching to a
paired test is what resolved the hedge, and the larger sample made it
comfortable rather than merely defensible.

### 2. Cross-encoder reranking significantly *hurts* — by a lot

This is the most valuable finding in the project, and it is a negative one.

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.394 | 0.723 | 0.828 | 0.539 | 0.509 |
| dense | 0.325 | 0.689 | 0.819 | 0.487 | 0.455 |
| **hybrid** | **0.437** | **0.794** | **0.876** | **0.593** | **0.557** |
| rerank | 0.297 | 0.664 | 0.806 | 0.463 | 0.437 |

| comparison | k | delta | rerank wins | hybrid wins | z | sig. 5% |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| rerank vs hybrid | 1 | **−0.140** | 115 | **298** | −9.00 | yes |
| rerank vs hybrid | 5 | **−0.130** | 68 | **238** | −9.72 | yes |
| rerank vs hybrid | 10 | **−0.070** | 42 | **134** | −6.93 | yes |

At top-1 the reranker moved a correct answer *out* of first place on 298
examples and *into* it on 115. Nearly three to one, in the wrong direction.

**Put that against the headroom it had.** Hybrid's top-25 accuracy — the
shortlist it was given, and therefore the hard ceiling on anything it could
achieve — is **0.947**. From a top-10 of 0.876 there were **7.1 points available**.
The reranker gave back 7.0 instead. It converted a near-perfect opportunity into
a loss of almost exactly the same size.

**Mechanistically, it destroys precisely what fusion added.** Compare it to BM25
alone:

| comparison | k | delta | z | sig. 5% |
| --- | ---: | ---: | ---: | :---: |
| rerank vs BM25 | 1 | −0.096 | −6.10 | yes |
| rerank vs BM25 | 5 | −0.059 | −4.16 | yes |
| rerank vs BM25 | 10 | −0.022 | −1.95 | **no** |

At top-10 reranking is statistically indistinguishable from plain BM25 — it lands
back at roughly the level of the single cheapest component. Fusion's whole
contribution is the RRF signal that a file was ranked highly by *two independent
retrievers*. The cross-encoder discards that consensus and re-sorts on its own
opinion, and its opinion is worse than the consensus it overwrote.

**The harder corpus scope makes this sharper, not softer.** With test files
searchable, reranking becomes indistinguishable from BM25 at *every* cutoff
(z between −1.75 and +0.50), while hybrid beats BM25 there by +0.109 top-10
(z = +9.62) — so the reranker erases the entire fusion advantage, exactly. And the
penalty grows rather than shrinking: **−0.102 top-10 versus −0.070**. That is the
opposite of the natural hypothesis, which is that a semantic model should help
most where lexical matching is brittle. Two-scope analysis below.

**Diagnosis: domain shift, and the claim should be scoped narrowly.** The model
is `cross-encoder/ms-marco-MiniLM-L-6-v2`, trained on **MS MARCO** — Bing search
queries paired with natural-language web passages. It is being asked here to
judge relevance between a terse bug report and a window of Python source. Those
are different distributions in vocabulary, structure, and what "relevant" means.
The model has never been shown that `def _get_loc` is a definition, that an
identifier match is worth more than a prose overlap, or that a docstring is
weaker evidence than a branch condition.

So the honest claim is narrow: **this reranker, trained on web-search passages,
degrades ranking on this domain, Python source.** It is *not* "reranking doesn't
work" — the two-stage pattern is sound and well established. A code-trained
cross-encoder is an untested and entirely plausible fix. The finding is about a
specific model-domain mismatch, and stating it more broadly than that would be
overclaiming from one experiment.

### 3. The per-repo pattern is internally coherent — and that is what makes it mechanism, not coincidence

Reranking helps on exactly one repository:

| repo | n | BM25 top-10 | hybrid top-10 | rerank top-10 | rerank vs hybrid |
| --- | ---: | ---: | ---: | ---: | ---: |
| flask | 85 | 0.824 | **0.906** | 0.871 | −0.035 |
| pandas | 1,103 | 0.828 | **0.879** | 0.797 | −0.082 |
| requests | 120 | 0.833 | 0.825 | **0.842** | **+0.017** |

**requests is also the one repository where hybrid failed to beat BM25** — in
Milestone 3 and again here, hybrid's 0.825 sits *below* BM25's 0.833, the only
slice where fusion has no edge.

Those two facts are the same fact. On flask and pandas, RRF found real consensus
between two retrievers that disagree usefully, and the reranker destroyed it. On
requests, dense retrieval is weak enough (0.783 top-10, well below BM25's 0.833)
that fusing it in already cost more than it added — so there was no consensus
signal to destroy, and the reranker could only improve on a ranking that fusion
had already degraded.

The reranker isn't good at requests. It's that requests is the only place where
the thing it overwrites was not worth keeping. A finding that holds together
across two independent slices of the data this way is describing a mechanism, not
a coincidence — and it is the detail that made me confident the top-line result
isn't a bug in my rerank plumbing.

The pattern's *direction* survives the harder corpus scope — requests is again
the only repo where fusion has no edge (0.800 vs BM25's 0.800, a tie) and again
where reranking is competitive (better at top-1 and MRR, worse at top-10). Its
*magnitude* does not: on a 120-example slice a 0.025 top-10 difference is three
examples. The structural claim holds; the precise per-repo numbers should not be
leaned on.

---

## Results in full

**Eval set: 1,308 held-out examples**, expanded from 337 in Milestone 3.

| repo | examples | share |
| --- | ---: | ---: |
| flask | 85 | 6.5% |
| pandas | 1,103 | **84.3%** |
| requests | 120 | 9.2% |

The aggregate is 84% pandas and the tooling prints a warning saying so. Read it
as a pandas number that the other repos nudge; the per-repo tables are what
support per-repo claims.

### Tests excluded (default scope) — mean 246 candidate files per query

Shortlist ceiling (hybrid top-25): **0.947**

| repo | method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **all (1,308)** | BM25 | 0.394 | 0.723 | 0.828 | 0.539 | 0.509 |
| | dense | 0.325 | 0.689 | 0.819 | 0.487 | 0.455 |
| | **hybrid** | **0.437** | **0.794** | **0.876** | **0.593** | **0.557** |
| | rerank | 0.297 | 0.664 | 0.806 | 0.463 | 0.437 |
| flask (85) | BM25 | 0.459 | **0.765** | 0.824 | 0.582 | 0.575 |
| | dense | 0.459 | 0.718 | 0.882 | 0.584 | 0.568 |
| | **hybrid** | **0.482** | 0.729 | **0.906** | **0.594** | **0.585** |
| | rerank | 0.388 | 0.753 | 0.871 | 0.567 | 0.553 |
| pandas (1,103) | BM25 | 0.394 | 0.718 | 0.828 | 0.539 | 0.504 |
| | dense | 0.319 | 0.689 | 0.818 | 0.484 | 0.449 |
| | **hybrid** | **0.445** | **0.806** | **0.879** | **0.601** | **0.560** |
| | rerank | 0.286 | 0.650 | 0.797 | 0.450 | 0.421 |
| requests (120) | BM25 | **0.350** | **0.742** | 0.833 | **0.515** | 0.507 |
| | dense | 0.283 | 0.667 | 0.783 | 0.451 | 0.437 |
| | hybrid | 0.333 | 0.733 | 0.825 | **0.515** | **0.509** |
| | **rerank** | 0.342 | 0.733 | **0.842** | 0.509 | 0.501 |

### Tests included (whole-repo search) — mean 1,204 candidate files per query

Shortlist ceiling (hybrid top-25): **0.890**

| repo | method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **all (1,308)** | BM25 | 0.209 | 0.534 | 0.670 | 0.363 | 0.341 |
| | dense | 0.238 | 0.558 | 0.691 | 0.385 | 0.355 |
| | **hybrid** | **0.306** | **0.658** | **0.779** | **0.461** | **0.428** |
| | rerank | 0.193 | 0.508 | 0.677 | 0.345 | 0.323 |
| flask (85) | BM25 | 0.365 | 0.671 | 0.765 | 0.504 | 0.501 |
| | dense | 0.400 | 0.647 | 0.753 | 0.525 | 0.508 |
| | **hybrid** | **0.423** | **0.682** | **0.812** | **0.547** | **0.530** |
| | rerank | 0.341 | 0.671 | 0.788 | 0.515 | 0.499 |
| pandas (1,103) | BM25 | 0.187 | 0.511 | 0.649 | 0.341 | 0.316 |
| | dense | 0.223 | 0.548 | 0.681 | 0.372 | 0.339 |
| | **hybrid** | **0.299** | **0.651** | **0.774** | **0.454** | **0.417** |
| | rerank | 0.169 | 0.478 | 0.658 | 0.319 | 0.295 |
| requests (120) | BM25 | 0.308 | 0.642 | **0.800** | 0.465 | 0.456 |
| | dense | 0.258 | 0.592 | 0.742 | 0.405 | 0.392 |
| | hybrid | 0.283 | **0.708** | **0.800** | 0.461 | 0.453 |
| | **rerank** | **0.308** | 0.667 | 0.775 | **0.466** | **0.457** |

McNemar, tests included:

| comparison | k | delta | A wins | B wins | z | sig. 5% |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| hybrid vs BM25 | 1 | +0.096 | 187 | 61 | +8.00 | yes |
| hybrid vs BM25 | 5 | +0.125 | 210 | 47 | +10.17 | yes |
| hybrid vs BM25 | 10 | +0.109 | 180 | 38 | +9.62 | yes |
| hybrid vs dense | 10 | +0.088 | 160 | 45 | +8.03 | yes |
| rerank vs hybrid | 1 | −0.113 | 100 | 248 | −7.93 | yes |
| rerank vs hybrid | 5 | −0.151 | 72 | 269 | −10.67 | yes |
| rerank vs hybrid | 10 | −0.102 | 64 | 197 | −8.23 | yes |
| rerank vs BM25 | 1 | −0.017 | 138 | 160 | −1.27 | **no** |
| rerank vs BM25 | 5 | −0.026 | 172 | 206 | −1.75 | **no** |
| rerank vs BM25 | 10 | +0.007 | 169 | 160 | +0.50 | **no** |

---

## What the two scopes together show

Running both scopes over the *same* 1,308 examples turns three separate
observations into one coherent picture.

### Reranking hurts more where the corpus is harder

| | ceiling | hybrid top-10 | headroom | rerank top-10 | delta | z |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tests excluded | 0.947 | 0.876 | +0.071 | 0.806 | **−0.070** | −6.93 |
| tests included | 0.890 | 0.779 | +0.111 | 0.677 | **−0.102** | −8.23 |

This was worth testing because the natural hypothesis pointed the other way: if
the hard scope is hard because *lexical* matching gets fooled by test files, a
semantic model should help most exactly there. It does the opposite — the penalty
grows from 7.0 to 10.2 points.

The symmetry is the striking part. In **both** scopes the reranker gave back
almost precisely the headroom it was handed: 7.0 of 7.1 available, then 10.2 of
11.1. It is not making partial progress and falling short; it is reliably
converting the shortlist's advantage into a loss of the same magnitude.

The likely reason it is *worse* in the hard scope is the same reason BM25 is worse
there. A test file restates the bug report in near-identical natural language —
which is exactly the surface signal an MS MARCO model is trained to reward. The
added distractors are adversarial to a web-passage reranker for much the same
reason they are adversarial to keyword matching.

### Reranking lands on BM25, and in the hard scope it lands there exactly

| rerank vs BM25 | tests excluded | tests included |
| --- | --- | --- |
| top-1 | −0.096 (z = −6.10, sig) | −0.017 (z = −1.27, **n.s.**) |
| top-5 | −0.059 (z = −4.16, sig) | −0.026 (z = −1.75, **n.s.**) |
| top-10 | −0.022 (z = −1.95, n.s.) | +0.007 (z = **+0.50**, **n.s.**) |

In the tests-included scope, reranking is **statistically indistinguishable from
plain BM25 at every cutoff** — z between −1.75 and +0.50. That is the cleanest
statement of the mechanism available. Hybrid beats BM25 there by a wide margin
(+0.109 top-10, z = +9.62), and applying the cross-encoder erases the entire
difference, landing back on the cheapest baseline.

Whatever the reranker is doing, it is not adding information. It is discarding
the two-retriever consensus that fusion contributes and re-sorting on an opinion
worth about as much as BM25's alone.

### Fusion matters more as the task gets harder — the mirror image

| hybrid vs BM25 | tests excluded | tests included |
| --- | ---: | ---: |
| top-10 delta | +0.048 (z = +5.46) | **+0.109** (z = +9.62) |
| win counts | 98 vs 35 | **180 vs 38** |

The harder the corpus, the more fusion is worth. This also confirms Milestone 3's
scope-flip finding at 4x the sample size: **dense overtakes BM25 once tests are
searchable** (0.691 vs 0.670 top-10; 0.238 vs 0.209 top-1), reversing their order
in the easy scope. Dense retrieval's contribution here is robustness to
distractors, and fusion is what converts that into a ranking better than either.

### The requests exception, re-examined

The per-repo coherence described above holds in both scopes, with a nuance the
second scope adds.

| | requests: hybrid vs BM25 | requests: rerank vs hybrid |
| --- | --- | --- |
| tests excluded | 0.825 vs **0.833** (fusion loses) | **0.842** vs 0.825 (rerank wins) |
| tests included | 0.800 vs 0.800 (tied) | 0.775 vs **0.800** top-10, but **0.308** vs 0.283 top-1 and **0.466** vs 0.461 MRR |

requests remains the only repository where fusion has no edge — it loses in one
scope and ties in the other — and it remains the only place reranking is
competitive. In the harder scope the picture is genuinely mixed rather than a
clean win: reranking is better at top-1 and on MRR/MAP, worse at top-10.

The honest reading is that the *direction* is consistent across both scopes and
the *magnitude* is not. requests is a 120-example slice, so a top-10 difference of
0.025 is two or three examples. What survives is the structural point: the one
repository where fusion adds nothing is the one where the reranker costs nothing,
because there is no consensus signal there to destroy.

---

## How reranking works, and why it costs what it costs

### Bi-encoders and cross-encoders

Everything in Milestone 2 used a **bi-encoder**. The query goes through the model
alone and becomes a vector; every chunk of code goes through alone and becomes a
vector; relevance is the cosine between them.

That is fast because the document side is **precomputable** — embed the corpus
once, store it in pgvector, and a query costs one forward pass plus a similarity
scan. It is also the limitation: when the two vectors finally meet, each has
already been compressed to 384 numbers *without any knowledge of the other*.

A **cross-encoder** removes that constraint. Query and document are concatenated
into one input and passed through the model together:

```
bi-encoder     embed("...IntervalIndex...")   ─┐
                                               ├─► cosine ─► score
               embed("def _get_loc(...)")     ─┘

cross-encoder  model("...IntervalIndex..." ++ "def _get_loc(...)") ─► score
```

Because the texts attend to each other inside the network, the model can
condition on their interaction — that this identifier appears in both, that the
code branches on exactly the type the report names. A bi-encoder cannot represent
"these two share a specific rare token" in a vector computed before it saw the
other text.

**The price is that nothing can be precomputed.** Measured on this machine with
`ms-marco-MiniLM-L-6-v2`: **180 pairs/s on MPS**, 74 on CPU. Scoring one query
against a 633-file corpus would be 3.5 seconds, per query, uncacheable — against
~1 ms for BM25 scoring and ~13 ms for a query embedding.

### The two-stage pattern and its ceiling

```
   corpus (hundreds of files)
        │  cheap: BM25 + dense + RRF        ~150 ms
        ▼
   shortlist (top 25)
        │  expensive: cross-encoder          ~550 ms
        ▼
   reordered top 25, tail appended unchanged
```

Reranking can only reorder what the shortlist contains, so the first-stage
accuracy *at the shortlist depth* is a hard ceiling. We use **top-25** (ceiling
0.947), which sits meaningfully above the top-10 we report while keeping cost at
100 pairs per query. Deeper raises the ceiling linearly in cost; shallower
collapses toward hybrid's own top-10, at which point reranking cannot improve
top-10 at all.

Reporting the ceiling is not decoration. A −7.0 point result against 7.1 points
of available headroom is a much sharper statement than the delta alone.

### What text the reranker actually reads

The model takes 512 tokens; our files run to 400 KB. Each candidate must be
represented by some passage, and we take up to **4 chunks per file, selected by
cosine similarity to the query**, scoring the file as the **maximum** over them.

Scoring every chunk would be faithful and unaffordable — a pandas core file is
~117 chunks, so 25 candidates would be ~2,900 pairs and 16 seconds per query.
Taking the file's head is cheap but may sit 3,000 lines from the buggy function.

Does cosine-selecting the chunks make reranking dependent on dense retrieval, and
so bias the comparison? Not in a way that favours dense: the selection happens
*within* an already-chosen candidate and decides which part of that file to read,
not which files compete. A BM25-only candidate gets its own best windows the same
way. The genuine cost is that if the embedding model is poor at locating the
relevant region inside a file, the reranker inherits that weakness — and given
the result above, that is a live alternative explanation worth naming, though it
does not fit the requests pattern as cleanly as domain shift does.

---

## Honest limitations

**One reranker, one configuration.** Depth 25, 4 chunks per file, one model. I did
not sweep depth or chunk count, and I did not try a code-trained cross-encoder.
The finding is that *this* configuration hurts, tested once and clearly.

**The chunk-selection confound.** Because candidate windows are chosen by cosine,
a poor embedding model could be feeding the reranker the wrong part of each file.
That would produce a similar top-line result. The per-repo pattern argues against
it being the whole story, but it is not ruled out, and a head-representation
ablation would settle it cheaply.

**No error analysis.** I know the reranker demotes correct files; I have not
sampled the demotions to characterise *which* files it prefers instead. That is
the obvious next step and would likely confirm or kill the domain-shift reading
in an afternoon.

**Cost was never the deciding factor, but it is worth stating.** Reranking adds
~550 ms per query against ~150 ms for the whole first stage — roughly 4x the
latency — to deliver a significantly worse ranking. Even at parity it would have
been a poor trade.

---

## Reproducing

```bash
make eval-rerank
```

Adds a fourth `rerank` row at both corpus scopes, reports the shortlist ceiling,
and appends the run to `results/heldout_log.jsonl`.

---

## Provenance

Both scopes cover the same 1,308 held-out examples. They were produced by two
separate runs of the same code: the tests-excluded scope on 2026-07-27 (tables
preserved in `results/logs/m4_eval_run.log` — that run was killed during its
second scope before it could write JSON, which is why per-scope persistence was
added), and the tests-included scope on 2026-07-28
(`results/20260728T052731Z.json`).

A note on partial results, because this milestone produced a good cautionary
example. At 425 examples the tests-excluded run showed dense ahead of BM25; the
completed run reversed it. Every partial figure quoted during the run was flagged
as provisional, and the final tables here are drawn only from completed runs. A
fraction of a run is not a result.
