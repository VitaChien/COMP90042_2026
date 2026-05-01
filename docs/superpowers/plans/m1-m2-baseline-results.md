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

---

# M2 BM25 -> cross-encoder rerank (dev, K=4)

| Variant                          | F      | A (random) | HM     |
|----------------------------------|--------|------------|--------|
| BM25-only k=4 (M1 baseline)      | 0.1072 | 0.2468     | 0.1495 |
| BM25 top-200 -> CE rerank top-4  | 0.0821 | 0.2468     | 0.1232 |

**Cross-encoder training summary (`bert-base-uncased`, BCE, 2 epochs):**
- 4122 positives + 16488 hard negatives (4:1) from BM25 top-50 \\ gold
- Epoch 1 mean_loss = 0.4101
- Epoch 2 mean_loss = **0.2658** (Plan B target < 0.3 met)
- Trained on Apple MPS, ~30 min BM25 search + ~55 min training

**Diagnostic findings (regression analysis):**
- Pairwise gold > random: **30/30** train, **30/30** dev (model has learned
  basic relevance).
- Pairwise gold > hard-negative (BM25 top-50 \\ gold): **48/50** train (96%),
  **45/50** dev (90%).
- BM25 top-50 recall on a 30-dev-claim subset: 40/104 gold (38%). Of those 40
  golds present in top-50, the CE places 16 in its top-4 (40% conditional
  recall) -> overall recall@4 = 16/104 = 15.4%, vs BM25-only top-4 = 11/104
  = 10.6%.
- MPS vs CPU inference: identical gold ranks -> not a numerical-precision bug.

**Why the *full* retriever-only F still drops** (0.107 -> 0.082):
- At inference the candidate pool is BM25 top-**200** (not top-50), so the
  cross-encoder reorders ~150 unseen candidates whose distribution is
  *out-of-train* for it; pairwise 90% accuracy compounds across many
  comparisons and mis-ranks pull non-gold candidates above the gold ones
  that BM25 had originally placed in its top-4.
- The per-claim F penalty for replacing one BM25-correct gold with a
  CE-confident wrong evidence is asymmetrical (precision drops while
  recall is preserved only at best), so the average F slips below
  the BM25 baseline.

**Plan B prediction vs reality:**
- Plan B Task 2.4 step 3 expected F to climb to ~0.18+; observed F=0.0821.
- The architecture is intact; under-training and an inference / training
  candidate-pool mismatch are the two leading hypotheses. Phase 3
  decisions will record which mitigation we adopt (longer training,
  blended BM25+CE score, or top-50 candidate pool at inference).

**BM25 backend switch (post-M2):**
- Replaced `rank_bm25` with `bm25s` (sparse-matrix BM25, multi-threaded).
- BM25 train top-50 retrieval: ~28 min -> 3.7 s wall clock (~466x faster).
- BM25 index build: ~5-15 min -> 19 s.
- Dev retriever-only F unchanged: 0.0821 (different top-200 ordering between
  the two BM25s, but the cross-encoder converges on the same top-4
  candidates regardless). bm25s switch is a pure speed win.
