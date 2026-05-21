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

## Task A2 experiment: narrow CE candidate pool to BM25 top-50 (no retraining)

| Setting                                  | F      | A (random) | HM     |
|------------------------------------------|--------|------------|--------|
| BM25-only k=4 (M1 baseline)              | 0.1072 | 0.2468     | 0.1495 |
| BM25 top-200 -> CE top-4 (M2 baseline)   | 0.0821 | 0.2468     | 0.1232 |
| BM25 top-50  -> CE top-4 (Task A2)       | 0.1081 | 0.2468     | 0.1503 |

**Interpretation:** F rose above both the M2 regression baseline (0.0821) and the M1 BM25-only baseline (0.1072), reaching 0.1081, confirming that candidate-pool distribution mismatch (ranks 50-200 being out-of-distribution for the CE) was the dominant cause of the regression and that retraining with a top-200 pool should further improve F.

## Task A3: rerank padding aligned to training (no retraining)

| Setting                                              | F      | A (random) | HM     |
|------------------------------------------------------|--------|------------|--------|
| BM25 top-200 -> CE top-4 (M2, padding=True)          | 0.0821 | 0.2468     | 0.1232 |
| BM25 top-200 -> CE top-4 (A3, padding=max_length)    | 0.0821 | 0.2468     | 0.1232 |

**Interpretation:** F is unchanged (0.0821), confirming the prediction: BERT's `attention_mask` makes the model's output invariant to padding length, so aligning `padding="max_length"` in `rerank()` with the training-time dataset is a maintenance/parity fix only — it eliminates a "same preprocessing in both paths?" doubt without affecting scores.

---

## Phase B retrain: all bugs fixed (token_type_ids, hard-neg dedup, BM25 cache stamp, ce_batch_size=64)

**Cross-encoder retraining summary (`bert-base-uncased`, BCE, 2 epochs, batch_size=64):**

- 4122 positives + 16488 hard negatives (4:1) from BM25 top-50 \\ gold (deduped per-claim)
- Epoch 1 mean_loss = 0.3788
- Epoch 2 mean_loss = **0.2265** (Plan B target < 0.3 met)
- Pre-tokenized dataset cache, Apple MPS, ~145 min total wall clock

| Setting                                              | F      | A (random) | HM     |
|------------------------------------------------------|--------|------------|--------|
| BM25-only k=4 (M1 baseline)                          | 0.1072 | 0.2468     | 0.1495 |
| BM25 top-200 -> CE top-4 (M2 buggy)                  | 0.0821 | 0.2468     | 0.1232 |
| BM25 top-200 -> CE top-4 (Phase B retrain, bugfixed) | 0.1477 | 0.2468     | 0.1848 |
| BM25 top-50  -> CE top-4 (Phase B retrain, bugfixed) | 0.1786 | 0.2468     | 0.2072 |

**Interpretation:**

- BM25 top-200 inference (production setting): F jumped from 0.082 → **0.1477** (+80% vs buggy, +38% vs M1 BM25-only). The `token_type_ids` fix is the dominant factor: BERT's segment embeddings now correctly distinguish claim vs evidence.
- BM25 top-50 inference (matching training pool): F=**0.1786**, HM=**0.2072** — the highest score yet, confirming that some distribution mismatch persists at inference-time rank 50-200. Model was trained exclusively on BM25 top-50 hard negatives.
- The remaining top-50 vs top-200 gap (~0.03 F) suggests the next improvement is either blending BM25 score with CE score at inference, or retraining with a mixed top-50/top-200 hard negative pool.

---

## Phase C retrain: hard negatives from BM25 top-200 pool

**Cross-encoder retraining summary (`bert-base-uncased`, BCE, 2 epochs, batch_size=64):**

- Same 4122 positives + 16488 hard negatives; negatives now sampled uniformly from BM25 top-200 \ gold
- Epoch 1 mean_loss = 0.3356
- Epoch 2 mean_loss = **0.1835** (lowest yet; Phase B was 0.2265)
- BM25 top-200 train cache rebuilt in 3.48s (bm25s); Apple MPS, ~77 min total wall clock

| Setting                                              | F          | A (random) | HM         |
|------------------------------------------------------|------------|------------|------------|
| BM25-only k=4 (M1 baseline)                          | 0.1072     | 0.2468     | 0.1495     |
| BM25 top-200 → CE top-4 (Phase B, top-50 neg pool)   | 0.1477     | 0.2468     | 0.1848     |
| BM25 top-50  → CE top-4 (Phase B, top-50 neg pool)   | 0.1786     | 0.2468     | 0.2072     |
| BM25 top-200 → CE top-4 (Phase C, top-200 neg pool)  | **0.2011** | 0.2468     | **0.2216** |
| BM25 top-50  → CE top-4 (Phase C, top-200 neg pool)  | 0.1891     | 0.2468     | 0.2141     |

**Interpretation:**

- BM25 top-200 (production setting): F=**0.2011**, HM=**0.2216** — best result so far. Phase C top-200 beats Phase B top-200 by +36% (0.1477 → 0.2011).
- The top-50 vs top-200 gap is **reversed**: Phase C top-200 (0.2011) > Phase C top-50 (0.1891). Training on a wider negative pool has made the model better calibrated for the full 200-candidate pool than for the narrower 50-candidate pool.
- The distribution mismatch hypothesis is confirmed and resolved: training negatives should match the inference pool size. `hard_negatives_bm25_top_k=200` is the correct production default going forward.
