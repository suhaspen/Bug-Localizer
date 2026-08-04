# Bug Localizer

Given a bug report, rank the source files most likely responsible for it.

The interesting part is the dataset: it labels itself. When a maintainer fixes a
bug, the commit that fixes it *is* the answer key — the files that commit touched
are, by definition, the files responsible. Mining git history therefore produces
thousands of labeled bug → file examples with no manual annotation. We index the
repository at the commit *before* the fix (the buggy state), so the model is
asked the same question a developer faced on the day the bug was filed.

We then evaluate retrieval methods — BM25, dense embeddings, hybrid fusion, and
cross-encoder reranking — on top-1/5/10 file accuracy, MRR, and MAP.

**Status: Milestone 4 complete.** **7,466 labeled examples** mined from 50,490
commits across flask, requests and pandas in ~13 seconds; four retrieval methods
evaluated on a temporal held-out split of 1,308 examples.

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.394 | 0.723 | 0.828 | 0.539 | 0.509 |
| dense | 0.325 | 0.689 | 0.819 | 0.487 | 0.455 |
| **hybrid (RRF)** | **0.437** | **0.794** | **0.876** | **0.593** | **0.557** |
| + cross-encoder rerank | 0.297 | 0.664 | 0.806 | 0.463 | 0.437 |

**Hybrid fusion wins** — it puts the responsible file in the top 5 for 79% of
bug reports, and beats both parents by a paired margin that McNemar's test
resolves comfortably. **Cross-encoder reranking loses**, decisively and
reproducibly; `docs/04_reranking.md` covers why a passage reranker is the wrong
instrument for this task rather than burying the result.

## Quickstart

```bash
make setup      # create the venv, install core dependencies
make setup-ml   # add sentence-transformers + psycopg (needed from here on)
make test       # run the test suite
make mine       # clone the repos and build data/examples.jsonl
make stats      # per-repo dataset breakdown
make db-up      # Postgres 17 + pgvector
make index      # embed each example's parent commit
make retrieve   # rank files for one example, BM25 vs dense
```

## Documentation

The `docs/` folder is written to be read on its own, without the code:

| Doc | What it covers |
| --- | --- |
| [`docs/00_overview.md`](docs/00_overview.md) | The project end to end, and the headline results |
| [`docs/01_dataset.md`](docs/01_dataset.md) | How git history self-labels the dataset, every filter, and the honest limitations |
| [`docs/02_retrieval.md`](docs/02_retrieval.md) | Sparse vs dense retrieval in plain terms, the corpus design, and the chunking decision |
| [`docs/03_evaluation.md`](docs/03_evaluation.md) | Every metric with a worked example, the split rationale, and the results interpreted |
| [`docs/architecture.md`](docs/architecture.md) | Map of the codebase and how data flows through it |
| [`docs/decisions.md`](docs/decisions.md) | Every non-obvious choice, with the alternatives rejected |
| [`docs/04_reranking.md`](docs/04_reranking.md) | Cross-encoder reranking, why it underperforms, and what that rules out |
| [`docs/glossary.md`](docs/glossary.md) | Plain-language definitions of every domain term |

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker (for Postgres + pgvector, from Milestone 2 onward)
