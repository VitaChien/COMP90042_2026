# M1 BM25 baseline results (top-K sweep)

| top-K | F | A (random) | HM |
|---|---|---|---|
| 3 | 0.1089 | 0.2468 | 0.1511 |
| 4 | 0.1072 | 0.2468 | 0.1495 |
| 5 | 0.1099 | 0.2468 | 0.1520 |
| 6 | 0.1059 | 0.2468 | 0.1482 |

**Best top-K for M1:** `k=5` achieves the highest harmonic mean (HM=0.1520) and
should be the default for Phase 4 tuning.

## Observations

- A (random) is identical across all K values (0.2468) because the random label
  is seeded with `cfg.seed=42`; this confirms the seed is applied correctly.
- F (evidence recall/precision) is non-monotonic: it peaks at k=5 then falls at
  k=6, suggesting that adding a sixth evidence document introduces enough
  low-relevance noise to hurt precision more than the marginal recall gain helps.
