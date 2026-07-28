# Evaluation results

- generated: `2026-07-28T05:27:31.047355+00:00`
- git: `1a7d439` (dirty tree)
- split: `heldout`
- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- chunk: 700 chars / 70 overlap · RRF k=60

## Corpus scope: tests INCLUDED (harder)

1308 examples · mean 1204 candidate files per query · 91.8 min

**Composition of the aggregate** — an aggregate is only as cross-repo as this table says it is:

| repo | examples | share |
| --- | ---: | ---: |
| flask | 85 | 6.5% |
| pandas | 1,103 | 84.3% |
| requests | 120 | 9.2% |

Rerank shortlist: top-25 of hybrid. Hybrid's top-25 accuracy is **0.890** — a hard ceiling, since reranking can only reorder what the shortlist contains.

### Aggregate

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.209 | 0.534 | 0.670 | 0.363 | 0.341 |
| dense | 0.238 | 0.558 | 0.691 | 0.385 | 0.355 |
| hybrid | **0.306** | **0.658** | **0.779** | **0.461** | **0.428** |
| rerank | 0.193 | 0.508 | 0.677 | 0.345 | 0.323 |


#### Paired comparisons (McNemar)

Both methods scored the same examples, so only the disagreements carry information. `A only` counts examples A found and B missed.

| comparison | k | A | B | delta | A only | B only | paired SE | z | sig. 5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| hybrid vs bm25 | 1 | 0.306 | 0.209 | +0.096 | 187 | 61 | 0.0120 | +8.00 | yes |
| hybrid vs bm25 | 5 | 0.658 | 0.534 | +0.125 | 210 | 47 | 0.0123 | +10.17 | yes |
| hybrid vs bm25 | 10 | 0.779 | 0.670 | +0.109 | 180 | 38 | 0.0113 | +9.62 | yes |
| hybrid vs dense | 1 | 0.306 | 0.238 | +0.068 | 166 | 77 | 0.0119 | +5.71 | yes |
| hybrid vs dense | 5 | 0.658 | 0.558 | +0.100 | 197 | 66 | 0.0124 | +8.08 | yes |
| hybrid vs dense | 10 | 0.779 | 0.691 | +0.088 | 160 | 45 | 0.0109 | +8.03 | yes |
| rerank vs hybrid | 1 | 0.193 | 0.306 | -0.113 | 100 | 248 | 0.0143 | -7.93 | yes |
| rerank vs hybrid | 5 | 0.508 | 0.658 | -0.151 | 72 | 269 | 0.0141 | -10.67 | yes |
| rerank vs hybrid | 10 | 0.677 | 0.779 | -0.102 | 64 | 197 | 0.0124 | -8.23 | yes |
| rerank vs bm25 | 1 | 0.193 | 0.209 | -0.017 | 138 | 160 | 0.0132 | -1.27 | no |
| rerank vs bm25 | 5 | 0.508 | 0.534 | -0.026 | 172 | 206 | 0.0149 | -1.75 | no |
| rerank vs bm25 | 10 | 0.677 | 0.670 | +0.007 | 169 | 160 | 0.0139 | +0.50 | no |

### flask (n=85)

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.365 | 0.671 | 0.765 | 0.504 | 0.501 |
| dense | 0.400 | 0.647 | 0.753 | 0.525 | 0.508 |
| hybrid | **0.423** | **0.682** | **0.812** | **0.547** | **0.530** |
| rerank | 0.341 | 0.671 | 0.788 | 0.515 | 0.499 |

### pandas (n=1103)

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.187 | 0.511 | 0.649 | 0.341 | 0.316 |
| dense | 0.223 | 0.548 | 0.681 | 0.372 | 0.339 |
| hybrid | **0.299** | **0.651** | **0.774** | **0.454** | **0.417** |
| rerank | 0.169 | 0.478 | 0.658 | 0.319 | 0.295 |

### requests (n=120)

| method | top-1 | top-5 | top-10 | MRR | MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | **0.308** | 0.642 | **0.800** | 0.465 | 0.456 |
| dense | 0.258 | 0.592 | 0.742 | 0.405 | 0.392 |
| hybrid | 0.283 | **0.708** | **0.800** | 0.461 | 0.453 |
| rerank | **0.308** | 0.667 | 0.775 | **0.466** | **0.457** |
