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

**Status: Milestone 1.** The dataset exists — **7,466 labeled examples** mined
from 50,490 commits across flask, requests and pandas in ~13 seconds. No
retrieval results yet.

## Quickstart

```bash
make setup   # create the venv, install dependencies
make test    # run the test suite
make mine    # clone the repos and build data/examples.jsonl
make stats   # per-repo dataset breakdown
make samples # print labeled examples for hand-review
```

## Documentation

The `docs/` folder is written to be read on its own, without the code:

| Doc | What it covers |
| --- | --- |
| [`docs/00_overview.md`](docs/00_overview.md) | The project end to end, and the headline results |
| [`docs/01_dataset.md`](docs/01_dataset.md) | How git history self-labels the dataset, every filter, and the honest limitations |
| [`docs/architecture.md`](docs/architecture.md) | Map of the codebase and how data flows through it |
| [`docs/decisions.md`](docs/decisions.md) | Every non-obvious choice, with the alternatives rejected |
| [`docs/glossary.md`](docs/glossary.md) | Plain-language definitions of every domain term |
| [`docs/interview_qa.md`](docs/interview_qa.md) | The hard questions about this project, with answers |

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker (for Postgres + pgvector, from Milestone 2 onward)
