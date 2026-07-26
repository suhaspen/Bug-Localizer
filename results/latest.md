# Evaluation results

- generated: `2026-07-26T01:15:45.646650+00:00`
- git: `0d1cc12` (dirty tree)
- split: `heldout`
- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- chunk: 700 chars / 70 overlap · RRF k=60

## Corpus scope: tests excluded

337 examples · mean 141 candidate files per query · 1.5 min

**Composition of the aggregate** — an aggregate is only as cross-repo as this table says it is:

| repo | examples | share |
| --- | ---: | ---: |
| flask | 85 | 25.2% |
| pandas | 132 | 39.2% |
| requests | 120 | 35.6% |

### Aggregate

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.406 | 0.760 | 0.843 | 0.557 | 0.546 |
| dense | 0.341 | 0.685 | 0.834 | 0.497 | 0.482 |
| hybrid | **0.415** | **0.768** | **0.872** | **0.575** | **0.564** |

### flask (n=85)

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.459 | **0.765** | 0.824 | 0.582 | 0.575 |
| dense | 0.459 | 0.718 | 0.882 | 0.584 | 0.568 |
| hybrid | **0.482** | 0.729 | **0.906** | **0.594** | **0.585** |

### pandas (n=132)

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.424 | 0.773 | 0.864 | 0.580 | 0.563 |
| dense | 0.318 | 0.682 | 0.849 | 0.483 | 0.467 |
| hybrid | **0.447** | **0.826** | **0.894** | **0.617** | **0.600** |

### requests (n=120)

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | **0.350** | **0.742** | **0.833** | **0.515** | 0.507 |
| dense | 0.283 | 0.667 | 0.783 | 0.451 | 0.437 |
| hybrid | 0.333 | 0.733 | 0.825 | **0.515** | **0.509** |

## Corpus scope: tests INCLUDED (harder)

337 examples · mean 633 candidate files per query · 3.5 min

**Composition of the aggregate** — an aggregate is only as cross-repo as this table says it is:

| repo | examples | share |
| --- | ---: | ---: |
| flask | 85 | 25.2% |
| pandas | 132 | 39.2% |
| requests | 120 | 35.6% |

### Aggregate

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.255 | 0.582 | 0.751 | 0.416 | 0.409 |
| dense | 0.285 | 0.582 | 0.721 | 0.426 | 0.411 |
| hybrid | **0.318** | **0.682** | **0.804** | **0.474** | **0.462** |

### flask (n=85)

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.365 | 0.671 | 0.765 | 0.504 | 0.501 |
| dense | 0.400 | 0.647 | 0.753 | 0.525 | 0.508 |
| hybrid | **0.423** | **0.682** | **0.812** | **0.547** | **0.530** |

### pandas (n=132)

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.136 | 0.470 | 0.697 | 0.314 | 0.307 |
| dense | 0.235 | 0.530 | 0.682 | 0.381 | 0.367 |
| hybrid | **0.280** | **0.659** | **0.803** | **0.440** | **0.427** |

### requests (n=120)

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | **0.308** | 0.642 | **0.800** | **0.465** | **0.456** |
| dense | 0.258 | 0.592 | 0.742 | 0.405 | 0.392 |
| hybrid | 0.283 | **0.708** | **0.800** | 0.461 | 0.453 |

## Scope comparison — top-10, aggregate

| method | tests excluded | tests included | delta |
| --- | ---: | ---: | ---: |
| bm25 | 0.843 | 0.751 | -0.092 |
| dense | 0.834 | 0.721 | -0.113 |
| hybrid | 0.872 | 0.804 | -0.068 |
