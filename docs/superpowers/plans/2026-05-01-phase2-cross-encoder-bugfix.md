# Phase 2 Cross-Encoder Bug-Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the bugs that caused Phase 2's cross-encoder to regress dev F from 0.107 (BM25 baseline) down to 0.082 (BM25 -> CE rerank), and bring F above the BM25 baseline as the original Plan B predicted.

**Architecture:** Three-phase fix. **Phase A** runs cheap experiments with the *existing* checkpoint to isolate which bugs matter (no retraining = ~5 min total). **Phase B** bundles all training-affecting fixes (token_type_ids, pre-tokenize cache, hard-neg de-duplication, BM25 cache rebuild, batch_size 32->64) into a SINGLE retraining run because we cannot afford to iterate per-fix; with the perf wins, fine-tuning drops from ~55 min to ~25-30 min on Apple MPS. **Phase C** cleans up correctness landmines that don't affect M2 metrics but will bite Phase 4 (oracle / mixed inference modes). **Wall-clock budget:** ~35-45 min total because B4 training runs in the background while Phase C1 proceeds in parallel.

**Tech Stack:** Python 3.11, PyTorch, HuggingFace transformers, bm25s, pytest. Apple MPS for local training.

---

## Context: which bugs cause which symptoms

| ID | Bug | File | Causes train/inference mismatch? | Fixed by retraining? |
|---|---|---|---|---|
| #1 | `token_type_ids` dropped from `CrossEncoderHead.forward` — BERT's segment embeddings get all-zeros instead of (0=claim, 1=evidence) | `src/retriever_cross_enc.py:48-51` | No (consistent train+infer), but caps achievable F | YES |
| #2 | `rerank` uses `padding=True`, training uses `padding="max_length"` — formal parity gap; in practice `attention_mask` makes padding length output-invariant, so this is *probably* cosmetic | `src/retriever_cross_enc.py:80-87` vs `:181-189` | Theoretical | No — pure inference fix |
| #3 | Hard-neg miner can emit the same `(claim, neg)` triple multiple times when a claim has multiple gold | `src/hard_negatives.py:45-65` | No, biases gradient | YES |
| #4 | BM25 cache `bm25_train_top50.json` was built under `rank_bm25` but now used with `bm25s` candidates at inference | `scripts/train_cross_encoder.py:49-72` | YES | YES |
| #5 | Inference uses BM25 top-200, training uses top-50 — CE reranks ranks 50-200 it has never seen | `scripts/run_inference.py:85` (`cfg.bm25_top_k=200`) | YES | No — config-only fix |
| #6 | `evaluator.py` uses `len(set(predictions))` for precision; official `eval.py` uses `len(list)` | `src/evaluator.py:40-47` | No — landmine for Phase 4 | No |

The `m1-m2-baseline-results.md` paradox (CE wins on a 30-claim subset's recall@4 but loses on full-dev mean F) is most likely explained by **#5 alone** (out-of-train ranks 50-200 in the inference candidate pool), with **#1** capping the achievable F headroom in either case. Phase A's main signal-bearing experiment is therefore A2 (narrow the pool to 50). A1 and A3 are correctness/parity hardening — likely no F movement, but they make future drift detectable.

## File Structure

| File | Role | Touched by |
|---|---|---|
| `src/retriever_cross_enc.py` | Cross-encoder model + dataset + train + rerank | A1, A3, B1 |
| `src/hard_negatives.py` | Hard-negative sampling | B2 |
| `src/config.py` | Hyperparameters | A2 (temp), B4 (`ce_batch_size`), B4 step 8 (temp) |
| `src/evaluator.py` | F / accuracy / HM metric helper | C1 |
| `scripts/train_cross_encoder.py` | Training entry point; uses cached BM25 top-50 | B3 |
| `scripts/run_inference.py` | Inference entry point; calls rerank | A1 |
| `tests/test_retriever_cross_enc.py` | Cross-encoder unit tests | A1, B1 |
| `tests/test_hard_negatives.py` | Hard-neg unit tests | B2 |
| `tests/test_evaluator.py` | Evaluator unit tests | C1 |
| `docs/superpowers/plans/m1-m2-baseline-results.md` | M1+M2 results log (running record) | A2, A3, B4 |

## Commit Message Rules (MUST FOLLOW)

Same as Plan B: imperative present, scope prefix, one-line subject, no AI tags.

```
test(retriever): Train/rerank logit parity guard
fix(retriever): Match rerank padding mode to training
fix(retriever): Pass token_type_ids to BERT cross-encoder
perf(retriever): Pre-tokenize cross-encoder training pairs
fix(retriever): Deduplicate hard negatives per claim
fix(retriever): Stamp BM25 backend in cache file
chore(config): Bump cross-encoder batch_size to 64
fix(evaluator): Match official eval.py precision denominator
docs: Record Phase 2 bug-fix dev F results
docs: Record Task A2 narrow-pool experiment
```

---

# Phase A: Cheap experiments (no retraining)

Goal: with the existing `cross_encoder.pt`, isolate how much of the F regression is caused by the inference-only bugs (#2, #5). This costs ~5 min total and tells us before we spend ~25-30 min retraining whether retraining is even necessary.

### Task A1: Train/rerank logit-parity test (catches future drift)

**Files:**
- Modify: `tests/test_retriever_cross_enc.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_retriever_cross_enc.py`:

```python
def test_train_and_rerank_produce_identical_logit(tokenizer_and_model):
    """The same (claim, evidence) input must produce the SAME score whether
    fed through `CrossEncoderDataset.__getitem__` -> model.forward (training
    path) or through `rerank` (inference path).

    Any divergence here means train and inference disagree on what the
    model "sees" - tokenization parity, padding mode, segment IDs, etc.
    """
    tok, model = tokenizer_and_model
    model.eval()

    claim = "the planet is warming"
    evid = "global mean temperature has risen by 1.1 C since 1880"

    # Training path
    ds = CrossEncoderDataset(
        [{"claim_text": claim, "evidence_text": evid, "label": 1}],
        tok,
        max_len=64,
    )
    item = ds[0]
    batch = {k: v.unsqueeze(0) for k, v in item.items() if k != "labels"}
    with torch.no_grad():
        train_logit = model(**batch).item()
    train_score = torch.sigmoid(torch.tensor(train_logit)).item()

    # Inference path
    out = rerank(
        model,
        tok,
        claim_text=claim,
        candidates=[("e-1", 0.0)],
        evidence_lookup={"e-1": evid},
        top_k=1,
        device="cpu",
        max_len=64,
    )
    rerank_score = out[0][1]

    assert abs(train_score - rerank_score) < 1e-5, (
        f"train vs rerank logit drift: train={train_score:.6f} rerank={rerank_score:.6f}"
    )
```

- [ ] **Step 2: Run; expected outcome is PROBABLY PASS at HEAD (record actual result either way)**

```bash
pytest tests/test_retriever_cross_enc.py::test_train_and_rerank_produce_identical_logit -v
```

**Expected:** PASS at HEAD. Reasoning: although training pads to `max_length` and rerank pads to longest-in-batch, BERT's `attention_mask` masks padded positions before softmax, so the [CLS] hidden state should be padding-length-invariant. Both paths *also* currently drop `token_type_ids` via `**_kwargs`, so segment IDs are effectively zeros in both — another way they end up identical.

**If the test FAILS at HEAD,** that is a real surprise and a stronger bug than #2 — note the divergence value and continue (Task A3 will likely fix it; if it doesn't, escalate before retraining).

**If the test PASSES at HEAD,** that's the expected outcome; the test is now a regression guard for any future change to either path.

- [ ] **Step 3: Commit the test regardless of pass/fail**

```bash
git add tests/test_retriever_cross_enc.py
git commit -m "test(retriever): Train/rerank logit parity guard"
```

---

### Task A2: Quick experiment — narrow inference candidate pool from top-200 to top-50

Goal: test hypothesis #5 (out-of-distribution rerank candidates). Lowering `cfg.bm25_top_k` to 50 means the CE only reranks BM25 top-50 — exactly the distribution it was trained on. No retraining needed.

**Files:**
- (Read-only) `scripts/run_inference.py` accepts default top-k via Config; we'll override via env or temporary edit.

- [ ] **Step 1: Run inference with `bm25_top_k=50` (one-shot edit, do NOT commit)**

Edit `src/config.py:23` temporarily:
```python
    bm25_top_k: int = 50  # WAS 200 — temporary for Task A2
```

Run dev inference:
```bash
python -m scripts.run_inference --split dev --mode retriever-only --top-k 4
```

Note the printed `F=`, `A=`, `HM=` line.

- [ ] **Step 2: Revert the config edit (do not commit)**

```bash
git checkout -- src/config.py
```

- [ ] **Step 3: Record outcome in `docs/superpowers/plans/m1-m2-baseline-results.md`**

Append to the M2 section a sub-table:

```markdown
## Task A2 experiment: narrow CE candidate pool to BM25 top-50 (no retraining)

| Setting                                  | F      | A (random) | HM     |
|------------------------------------------|--------|------------|--------|
| BM25-only k=4 (M1 baseline)              | 0.1072 | 0.2468     | 0.1495 |
| BM25 top-200 -> CE top-4 (M2 baseline)   | 0.0821 | 0.2468     | 0.1232 |
| BM25 top-50  -> CE top-4 (Task A2)       | <fill> | 0.2468     | <fill> |

**Interpretation:** `<fill>` (one sentence: did F rise above 0.0821? above 0.1072?).
```

- [ ] **Step 4: Commit the docs update**

```bash
git add docs/superpowers/plans/m1-m2-baseline-results.md
git commit -m "docs: Record Task A2 narrow-pool experiment"
```

**Decision gate:** if F under top-50 already exceeds the BM25 baseline (>0.107), we have evidence #5 was the dominant cause; we still proceed to Phase B for #1 and #3 because the segment-id fix is independently correct and improves headroom.

---

### Task A3: Align rerank padding with training (`padding="max_length"`)

Goal: align rerank with training as a parity-discipline change. Likely produces no F movement (attention_mask makes padding length output-invariant), but eliminates a class of "is the model seeing the same thing in both paths?" doubts and makes inference memory usage deterministic.

**Files:**
- Modify: `src/retriever_cross_enc.py:181-189`

- [ ] **Step 1: Edit `rerank` to use `padding="max_length"`**

Find this block in `src/retriever_cross_enc.py`:

```python
        enc = tokenizer(
            [claim_text] * len(batch_texts),
            batch_texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(device)
```

Change `padding=True` to `padding="max_length"`:

```python
        enc = tokenizer(
            [claim_text] * len(batch_texts),
            batch_texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
```

- [ ] **Step 2: Run the parity test from Task A1; expect PASS**

```bash
pytest tests/test_retriever_cross_enc.py::test_train_and_rerank_produce_identical_logit -v
```
Expected: PASS (divergence < 1e-5). If A1's test was already passing at HEAD, it remains passing here; if it was failing, this should fix it.

- [ ] **Step 3: Run the full cross-encoder test file**

```bash
pytest tests/test_retriever_cross_enc.py -v
```
Expected: all tests pass.

- [ ] **Step 4: Re-run dev inference (with `bm25_top_k=200`, the committed default)**

```bash
python -m scripts.run_inference --split dev --mode retriever-only --top-k 4
```

Note the F / A / HM line.

- [ ] **Step 5: Append result to `m1-m2-baseline-results.md`**

```markdown
## Task A3: rerank padding aligned to training (no retraining)

| Setting                                              | F      | A (random) | HM     |
|------------------------------------------------------|--------|------------|--------|
| BM25 top-200 -> CE top-4 (M2, padding=True)          | 0.0821 | 0.2468     | 0.1232 |
| BM25 top-200 -> CE top-4 (A3, padding=max_length)    | <fill> | 0.2468     | <fill> |
```

- [ ] **Step 6: Commit**

```bash
git add src/retriever_cross_enc.py docs/superpowers/plans/m1-m2-baseline-results.md
git commit -m "fix(retriever): Match rerank padding mode to training"
```

---

# Phase B: Training-affecting fixes (single retraining run)

Goal: bundle all bugs that require regenerating training data or weights into ONE retraining run. Order: B1 (#1 token_type_ids + pre-tokenize cache for ~10-20% epoch speedup) -> B2 (#3 dedupe negs) -> B3 (#4 BM25 cache rebuild) -> B4 (batch_size 32->64 + run training in background). Tests guard each fix in isolation; the retraining at the end is the integration test.

**Execution flow for time-saving (~80 min -> ~35-45 min wall clock):**

1. Complete B1, B2, B3 sequentially (~15 min total — small commits with tests).
2. **Launch B4 training in the background** (`nohup ... &` or `tmux`). It runs ~25-30 min unattended thanks to batch_size=64 + pre-tokenize cache.
3. **While training runs, complete Phase C1 in parallel** (evaluator fix is fully independent — no shared files). C1 is ~5 min.
4. When training completes, return to B4 step 4+ to run inference and record results.

This pattern means the 25-30 min training cost is *amortized* against productive work instead of being pure waiting.

### Task B1: Pass `token_type_ids` end-to-end

**Files:**
- Modify: `src/retriever_cross_enc.py` (model forward, dataset, rerank)
- Modify: `tests/test_retriever_cross_enc.py`

- [ ] **Step 1: Add a failing test that asserts segment IDs reach the encoder**

Append to `tests/test_retriever_cross_enc.py`:

```python
def test_dataset_yields_token_type_ids(tokenizer_and_model):
    """Dataset MUST emit token_type_ids so BERT can use segment embeddings.

    With segment IDs absent, BertModel falls back to all-zeros and the
    pretrained NSP-trained sentence-pair signal is wasted.
    """
    tok, _ = tokenizer_and_model
    pairs = [{"claim_text": "claim", "evidence_text": "ev", "label": 1}]
    ds = CrossEncoderDataset(pairs, tok, max_len=32)
    item = ds[0]
    assert "token_type_ids" in item
    # claim half should be 0, evidence half should be 1, padding (after [SEP]) is 0
    tt = item["token_type_ids"]
    assert (tt == 1).any(), "evidence half must have token_type_id=1"


def test_forward_uses_token_type_ids(tokenizer_and_model):
    """Different segment assignments must produce different logits (proves
    the encoder actually consumes token_type_ids)."""
    tok, model = tokenizer_and_model
    model.eval()
    enc = tok("claim text", "evidence text", return_tensors="pt",
              padding="max_length", max_length=32, truncation=True)
    with torch.no_grad():
        a = model(input_ids=enc["input_ids"],
                  attention_mask=enc["attention_mask"],
                  token_type_ids=enc["token_type_ids"]).item()
        b = model(input_ids=enc["input_ids"],
                  attention_mask=enc["attention_mask"],
                  token_type_ids=torch.zeros_like(enc["token_type_ids"])).item()
    assert abs(a - b) > 1e-5, "model is ignoring token_type_ids"
```

- [ ] **Step 2: Run; expect FAIL on both**

```bash
pytest tests/test_retriever_cross_enc.py::test_dataset_yields_token_type_ids tests/test_retriever_cross_enc.py::test_forward_uses_token_type_ids -v
```
Expected: FAIL (test_dataset asserts a key the dataset doesn't return; test_forward signature accepts token_type_ids but the encoder is never actually called with it because of `**_kwargs`).

- [ ] **Step 3: Fix the model forward in `src/retriever_cross_enc.py:48-51`**

Replace:
```python
    def forward(self, input_ids, attention_mask, **_kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls)).squeeze(-1)
```

With:
```python
    def forward(self, input_ids, attention_mask, token_type_ids=None, **_kwargs):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls)).squeeze(-1)
```

- [ ] **Step 4: Fix the dataset to return token_type_ids**

In `src/retriever_cross_enc.py:88-92`, replace the return dict:

```python
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "token_type_ids": enc["token_type_ids"][0],
            "labels": torch.tensor(float(p["label"]), dtype=torch.float32),
        }
```

- [ ] **Step 5: Fix the training loop to pass token_type_ids**

In `train_cross_encoder` at `src/retriever_cross_enc.py:126-129`, change:
```python
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
```
to:
```python
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )
```

(The `loss_fn(logits, batch["labels"])` line still works because `labels` is a separate batch key — but verify the `batch = {k: v.to(device) for k, v in batch.items()}` line still moves all tensors to device. It does, because token_type_ids is now in `batch`.)

- [ ] **Step 6: rerank already calls `model(**enc)` so it auto-forwards token_type_ids — verify**

Read `src/retriever_cross_enc.py:190` — `logits = model(**enc)` already unpacks the tokenizer output, which includes `token_type_ids` for sentence-pair encoding. With Step 3's signature change, this Just Works. No edit needed at the rerank call site.

- [ ] **Step 7: Run all CE tests, expect PASS**

```bash
pytest tests/test_retriever_cross_enc.py -v
```
Expected: ALL tests pass, including the parity test from A1 (token_type_ids flows through both paths consistently).

- [ ] **Step 8: Commit token_type_ids fix**

```bash
git add src/retriever_cross_enc.py tests/test_retriever_cross_enc.py
git commit -m "fix(retriever): Pass token_type_ids to BERT cross-encoder"
```

- [ ] **Step 9: Add pre-tokenize cache to `CrossEncoderDataset` (training-time speedup)**

Currently `__getitem__` calls `self.tok(...)` on every access, so each of the 20k pairs is re-tokenized once per epoch. Pre-tokenizing once at `__init__` and indexing into a Python list cuts ~10-20% off epoch wall time and makes `num_workers=0` viable (avoids macOS multiprocessing-with-MPS quirks).

Replace `CrossEncoderDataset.__init__` and `__getitem__` in `src/retriever_cross_enc.py`. Find:

```python
    def __init__(
        self,
        pairs: Sequence[dict],
        tokenizer,
        max_len: int = 256,
        evidence_lookup: dict[str, str] | None = None,
    ) -> None:
        self.pairs = list(pairs)
        self.tok = tokenizer
        self.max_len = max_len
        self.lookup = evidence_lookup or {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        ev = p.get("evidence_text") or self.lookup[p["evidence_id"]]
        enc = self.tok(
            p["claim_text"],
            ev,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "token_type_ids": enc["token_type_ids"][0],
            "labels": torch.tensor(float(p["label"]), dtype=torch.float32),
        }
```

Replace with:

```python
    def __init__(
        self,
        pairs: Sequence[dict],
        tokenizer,
        max_len: int = 256,
        evidence_lookup: dict[str, str] | None = None,
    ) -> None:
        lookup = evidence_lookup or {}
        self.max_len = max_len
        # Pre-tokenize once. Avoids re-running the (slow Python-side) tokenizer
        # on every __getitem__ across epochs and worker processes.
        claim_texts = [p["claim_text"] for p in pairs]
        evidence_texts = [
            p.get("evidence_text") or lookup[p["evidence_id"]] for p in pairs
        ]
        enc = tokenizer(
            claim_texts,
            evidence_texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc["token_type_ids"]
        self.labels = torch.tensor([float(p["label"]) for p in pairs], dtype=torch.float32)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
        }
```

- [ ] **Step 10: Run all CE tests; expect PASS (existing tests are agnostic to internal storage)**

```bash
pytest tests/test_retriever_cross_enc.py -v
```
Expected: ALL pass. The existing tests (`test_dataset_yields_tensors`, `test_dataset_resolves_evidence_via_lookup`, `test_dataset_yields_token_type_ids`) only assert on `__getitem__` output shape/keys — they don't care whether tokenization was eager or lazy.

- [ ] **Step 11: Commit pre-tokenize cache**

```bash
git add src/retriever_cross_enc.py
git commit -m "perf(retriever): Pre-tokenize cross-encoder training pairs"
```

---

### Task B2: De-duplicate hard negatives per claim

**Files:**
- Modify: `src/hard_negatives.py`
- Modify: `tests/test_hard_negatives.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hard_negatives.py`:

```python
def test_no_duplicate_negatives_per_claim():
    """For a claim with N gold and only M < N*n_neg unique candidates, the
    miner must NEVER emit the same (claim, neg_evidence) pair twice. Each
    duplicate pair would silently inflate that negative's gradient weight."""
    claims = {
        "c": {"claim_text": "Q", "evidences": ["g1", "g2", "g3"]},
    }
    # Only 4 unique non-gold candidates available
    bm25_results = {
        "c": [
            ("g1", 9.0), ("g2", 8.0), ("g3", 7.0),
            ("n1", 6.0), ("n2", 5.0), ("n3", 4.0), ("n4", 3.0),
        ],
    }
    # 3 golds * n_neg=4 = 12 negative SLOTS; only 4 unique candidates exist.
    # The miner must cap at 4 unique negatives, not loop and produce 12.
    pairs = build_training_pairs(claims, bm25_results, n_neg=4, seed=0)
    neg_pairs = [(p["claim_id"], p["evidence_id"]) for p in pairs if p["label"] == 0]
    assert len(neg_pairs) == len(set(neg_pairs)), \
        f"duplicate negatives emitted: {neg_pairs}"
```

- [ ] **Step 2: Run; expect FAIL**

```bash
pytest tests/test_hard_negatives.py::test_no_duplicate_negatives_per_claim -v
```
Expected: FAIL — current implementation re-samples per gold and produces duplicates.

- [ ] **Step 3: Rewrite the miner loop to dedupe per claim**

Replace `src/hard_negatives.py:38-66` (the body of `build_training_pairs` from `pairs: list[dict] = []` onwards) with:

```python
    pairs: list[dict] = []
    for cid, claim in claims.items():
        ctext = _claim_field(claim, "claim_text")
        gold_list = _claim_field(claim, "evidences")
        gold_set = set(gold_list)
        bm_hits = bm25_results.get(cid, [])
        candidates = [eid for eid, _ in bm_hits if eid not in gold_set]

        # Sample ALL negatives for this claim in one shot so each candidate
        # appears at most once. Cap at n_neg * len(gold_list) total slots.
        target_n_neg = n_neg * len(gold_list)
        sampled_negs = (
            rng.sample(candidates, k=min(target_n_neg, len(candidates)))
            if candidates else []
        )

        for ge in gold_list:
            pairs.append(
                {"claim_id": cid, "claim_text": ctext,
                 "evidence_id": ge, "label": 1}
            )
        for ne in sampled_negs:
            pairs.append(
                {"claim_id": cid, "claim_text": ctext,
                 "evidence_id": ne, "label": 0}
            )
    return pairs
```

- [ ] **Step 4: Run all hard-neg tests; expect PASS**

```bash
pytest tests/test_hard_negatives.py -v
```
Expected: ALL pass.

**Note:** `test_build_training_pairs_has_pos_and_negs` expects exactly 4 negatives (2 golds × 2 negs each). After this fix, with `n_neg=2`, target_n_neg = 2*2 = 4, and 4 candidates (`{e-9, e-8, e-7, e-6}`) are available, so 4 unique negatives are sampled — same count, same distinct set. Test stays green.

`test_build_training_pairs_is_deterministic` uses 1 gold × n_neg=4, also unaffected.

`test_no_pairs_when_bm25_only_returns_gold` has 0 candidates -> 0 negs, unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/hard_negatives.py tests/test_hard_negatives.py
git commit -m "fix(retriever): Deduplicate hard negatives per claim"
```

---

### Task B3: Stamp BM25 backend in cache + rebuild under bm25s

**Files:**
- Modify: `scripts/train_cross_encoder.py`

- [ ] **Step 1: Add backend stamp to cache write/read**

In `scripts/train_cross_encoder.py:38-73`, replace the BM25 cache section. Find this block:

```python
    bm25_cache = cfg.cache_dir / "bm25_train_top50.json"
    if bm25_cache.exists():
        log.info("Loading cached BM25 train top-50 from %s", bm25_cache)
        bm25_results = load_json(bm25_cache)
    else:
        # Single-threaded numpy inside rank_bm25.get_scores; ~1228 calls over a
        # 1.2M-doc corpus benchmarks at ~1s/call so we surface progress + cache
        # incrementally to avoid losing work if interrupted.
        bm25_results = {}
        partial_cache = cfg.cache_dir / "bm25_train_top50.partial.json"
        if partial_cache.exists():
            bm25_results = load_json(partial_cache)
            log.info(
                "Resuming BM25 search; %d/%d already cached", len(bm25_results), len(train_claims)
            )
        with timer("BM25 search over train split", log):
            for i, (cid, c) in enumerate(tqdm(train_claims.items(), desc="bm25 train")):
                if cid in bm25_results:
                    continue
                bm25_results[cid] = bm25.search(c.claim_text, top_k=50)
                if (i + 1) % 200 == 0:
                    save_json(bm25_results, partial_cache)
        save_json(bm25_results, bm25_cache)
        partial_cache.unlink(missing_ok=True)
        log.info("Cached BM25 train top-50 -> %s", bm25_cache)
```

Replace with (note: `BM25_BACKEND` constant defined at module top, and we use `bm25.search_batch` since bm25s supports it efficiently):

```python
    bm25_cache = cfg.cache_dir / "bm25_train_top50.json"
    BACKEND_TAG = "bm25s"  # source-of-truth: src/retriever_bm25.py

    bm25_results: dict | None = None
    if bm25_cache.exists():
        cached = load_json(bm25_cache)
        if isinstance(cached, dict) and cached.get("_backend") == BACKEND_TAG:
            log.info("Loading cached BM25 train top-50 (backend=%s) from %s",
                     BACKEND_TAG, bm25_cache)
            bm25_results = cached["results"]
        else:
            log.warning(
                "BM25 cache backend mismatch (got %r, expected %r); rebuilding.",
                cached.get("_backend") if isinstance(cached, dict) else "<legacy>",
                BACKEND_TAG,
            )

    if bm25_results is None:
        with timer("BM25 search over train split", log):
            cids = list(train_claims.keys())
            queries = [train_claims[c].claim_text for c in cids]
            hits = bm25.search_batch(queries, top_k=50)
            bm25_results = dict(zip(cids, hits, strict=True))
        save_json({"_backend": BACKEND_TAG, "results": bm25_results}, bm25_cache)
        log.info("Cached BM25 train top-50 -> %s", bm25_cache)
```

(This drops the legacy partial-cache resume logic because `bm25s.search_batch` runs in seconds, not 20+ minutes — the partial cache was needed for `rank_bm25` only.)

- [ ] **Step 2: Verify `bm25.search_batch` exists and matches signature**

```bash
grep -n "def search_batch\|def search" src/retriever_bm25.py
```
Expected: `search_batch(self, queries: ..., top_k: int) -> list[list[tuple[str, float]]]` (or similar). If the signature differs, adapt the call. If `search_batch` is missing, fall back to a sequential loop (same as old code, but without the partial-cache machinery — bm25s sequential search is also fast).

- [ ] **Step 3: Delete the stale legacy cache**

```bash
rm -f cache/bm25_train_top50.json cache/bm25_train_top50.partial.json
```

- [ ] **Step 4: Smoke-test the script for cache rebuild on a tiny subset**

There is no unit test for `scripts/train_cross_encoder.py`. Sanity-check via:

```bash
python -c "
from src.config import Config
from src.data_loader import load_claims
from src.retriever_bm25 import BM25Retriever
cfg = Config()
bm = BM25Retriever.from_cache(cfg.cache_dir / 'bm25_index')
claims = load_claims(cfg.train_path)
sample_cids = list(claims.keys())[:5]
queries = [claims[c].claim_text for c in sample_cids]
print(bm.search_batch(queries, top_k=5))
"
```
Expected: prints 5 lists of 5 `(evidence_id, score)` tuples each.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_cross_encoder.py
git commit -m "fix(retriever): Stamp BM25 backend in cache file"
```

---

### Task B4: Retrain cross-encoder with all fixes (run in BACKGROUND)

This is the integration test for B1+B2+B3. With batch_size=64 + pre-tokenize cache, expect ~25-30 min on Apple MPS (down from the original ~55 min). Run in the background so Phase C1 can proceed in parallel.

**Files:**
- Modify: `src/config.py:32` (bump `ce_batch_size` to 64)
- (No other code changes; produces new `checkpoints/cross_encoder.pt`)

- [ ] **Step 1: Confirm prerequisites are met**

```bash
git log --oneline -10
```
Verify these commits are present (most recent first): "fix(retriever): Stamp BM25 backend in cache file", "fix(retriever): Deduplicate hard negatives per claim", "perf(retriever): Pre-tokenize cross-encoder training pairs", "fix(retriever): Pass token_type_ids ...". If any are missing, go back.

- [ ] **Step 2: Bump `ce_batch_size` to 64 in `src/config.py:32`**

```python
    ce_batch_size: int = 64  # was 32; doubled for ~30% epoch wall-time saving
```

Halves the number of optimizer steps. lr stays at 2e-5 — the doubling is small enough that lr scaling is unnecessary and prior literature on BERT fine-tuning shows lr is robust at this scale. If MPS OOMs at 64, drop to 48 (commit message updated to match) — but bert-base + max_len=256 + batch=64 fits comfortably in Apple's unified memory.

- [ ] **Step 3: Move the existing checkpoint to a backup name (don't delete — comparison baseline)**

```bash
mv checkpoints/cross_encoder.pt checkpoints/cross_encoder.pre-bugfix.pt
```

- [ ] **Step 4: Launch training in the background**

```bash
mkdir -p logs
nohup python -m scripts.train_cross_encoder > logs/ce-train-bugfix.log 2>&1 &
echo "training PID: $!"
```

Expected first ~60 sec of `logs/ce-train-bugfix.log`: BM25 cache rebuilds, then "Built N training pairs (positives=4122, hard_negs=M)" where M < 16488 (less than before because of dedup), then "Loading evidence corpus", then epoch 1 starts with a tqdm bar.

**Decision gate:** if the background process dies in the first 2 min (check `tail logs/ce-train-bugfix.log`), STOP and debug. Do not proceed to Phase C1 until you have confirmed training is actually progressing through epoch 1.

- [ ] **Step 5: While training runs in background, complete Phase C1 (evaluator fix)**

Switch to Phase C1 below. C1 is fully independent of the retraining (touches `src/evaluator.py` and `tests/test_evaluator.py`, not anything in `src/retriever_cross_enc.py` or the training script).

- [ ] **Step 6: When training finishes, commit the config change**

Wait for training to complete (`tail -f logs/ce-train-bugfix.log` until you see `Saved cross-encoder ckpt -> ...`). Final mean_loss should be < 0.3 (Plan B target).

```bash
git add src/config.py
git commit -m "chore(config): Bump cross-encoder batch_size to 64"
```

**Record** the new pair counts, per-epoch mean_loss, and wall-clock time in your notes for Step 9.

- [ ] **Step 7: Run dev inference with the retrained model (default `bm25_top_k=200`)**

```bash
python -m scripts.run_inference --split dev --mode retriever-only --top-k 4
```

Note F / A / HM.

- [ ] **Step 8: Run dev inference with `bm25_top_k=50` for comparison**

Temporarily edit `src/config.py:23` to `bm25_top_k: int = 50`, run again, then revert:

```bash
# edit src/config.py: bm25_top_k = 50
python -m scripts.run_inference --split dev --mode retriever-only --top-k 4
git checkout -- src/config.py
```

- [ ] **Step 9: Append all results to `m1-m2-baseline-results.md`**

```markdown
## Phase B retrain: token_type_ids + dedup negs + bm25s cache

**Training summary:**
- Pairs: <fill> positives + <fill> hard negs (was 4122 + 16488)
- Epoch 1 mean_loss = <fill>
- Epoch 2 mean_loss = <fill>
- Wall clock: <fill> on <device>

| Setting                                              | F      | A (random) | HM     |
|------------------------------------------------------|--------|------------|--------|
| BM25-only k=4 (M1)                                   | 0.1072 | 0.2468     | 0.1495 |
| Pre-bugfix CE, top-200 -> top-4 (M2)                 | 0.0821 | 0.2468     | 0.1232 |
| Post-bugfix CE, top-200 -> top-4                     | <fill> | 0.2468     | <fill> |
| Post-bugfix CE, top-50  -> top-4                     | <fill> | 0.2468     | <fill> |

**Goal check:** Plan B Task 2.4 step 3 expected F >= 0.18+. Did we hit it?
```

- [ ] **Step 10: Commit doc + checkpoint** (if your repo tracks checkpoints; otherwise just doc)

```bash
git add docs/superpowers/plans/m1-m2-baseline-results.md
git commit -m "docs: Record Phase 2 bug-fix dev F results"
```

(Do NOT `git add` `checkpoints/cross_encoder.pt` unless your repo tracks ckpts — most don't. The pre-bugfix backup stays on disk for manual diffing if needed.)

---

# Phase C: Correctness landmines (independent of M2 metrics)

These don't affect the F number but will bite Phase 4. Do them after Phase B so they don't muddy the regression analysis.

### Task C1: Fix evaluator precision denominator to match official `eval.py`

**Files:**
- Modify: `src/evaluator.py:40-47`
- Modify: `tests/test_evaluator.py`

- [ ] **Step 1: Add a failing test**

Add to `tests/test_evaluator.py`:

```python
def test_precision_uses_list_length_not_set(tmp_path):
    """Official eval.py uses len(predicted_evidence_list); duplicates count.
    Our wrapper must match so Phase 4 oracle/blend modes don't silently
    diverge from the leaderboard."""
    from src.utils import save_json
    from src.evaluator import evaluate_predictions

    pred_path = tmp_path / "pred.json"
    gold_path = tmp_path / "gold.json"
    save_json({
        "c1": {
            "claim_text": "x",
            "claim_label": "SUPPORTS",
            "evidences": ["e-1", "e-1", "e-2", "e-3"],  # duplicate e-1
        }
    }, pred_path)
    save_json({
        "c1": {"claim_text": "x", "claim_label": "SUPPORTS", "evidences": ["e-1"]}
    }, gold_path)

    # Official: precision = 1 (TP=1) / 4 (list length) = 0.25, recall = 1/1 = 1.0
    # F = 2*0.25*1 / 1.25 = 0.4
    m = evaluate_predictions(pred_path, gold_path)
    assert abs(m["evidence_f"] - 0.4) < 1e-6, f"got F={m['evidence_f']}"
```

- [ ] **Step 2: Run; expect FAIL**

```bash
pytest tests/test_evaluator.py::test_precision_uses_list_length_not_set -v
```
Expected: FAIL — current code uses `len(set(...))`=3 instead of `len(list)`=4, so F is overstated (0.4444 instead of 0.4).

- [ ] **Step 3: Fix `src/evaluator.py:40-47`**

Replace:
```python
        retrieved = set(p.get("evidences") or [])
        gold_ev = gold_inst["evidences"]
        if retrieved and gold_ev:
            tp = sum(1 for g in gold_ev if g in retrieved)
            if tp > 0:
                recall = tp / len(gold_ev)
                precision = tp / len(retrieved)
                f = 2 * precision * recall / (precision + recall)
            else:
                f = 0.0
        else:
            f = 0.0
```

With:
```python
        retrieved_list = list(p.get("evidences") or [])
        retrieved_set = set(retrieved_list)
        gold_ev = gold_inst["evidences"]
        if retrieved_list and gold_ev:
            tp = sum(1 for g in gold_ev if g in retrieved_set)
            if tp > 0:
                recall = tp / len(gold_ev)
                precision = tp / len(retrieved_list)  # match official eval.py
                f = 2 * precision * recall / (precision + recall)
            else:
                f = 0.0
        else:
            f = 0.0
```

- [ ] **Step 4: Run all evaluator tests; expect PASS**

```bash
pytest tests/test_evaluator.py -v
```
Expected: all pass. If any pre-existing test relied on the old set-based precision, update the expected value to match the official semantics (and add a comment explaining).

- [ ] **Step 5: Commit**

```bash
git add src/evaluator.py tests/test_evaluator.py
git commit -m "fix(evaluator): Match official eval.py precision denominator"
```

---

### Task C2: Verify all tests still green at HEAD

**Files:**
- (No code changes)

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```
Expected: ALL pass. If anything fails, fix before declaring Phase 2 bug-fix complete.

- [ ] **Step 2: Sanity-check Phase B results have NOT regressed**

Re-run dev inference once more; F must match the value recorded in B4 step 9 within 1e-4 (deterministic seed should make this exact unless evaluator change in C1 affected the metric — which it might, slightly, if any predictions had duplicates, but BM25/CE paths produce none, so the number should be identical).

```bash
python -m scripts.run_inference --split dev --mode retriever-only --top-k 4
```

(No commit — verification only.)

---

# Decision log to fill in during execution

After Task A2, A3, B4, write 1-3 sentences in `m1-m2-baseline-results.md` answering:

1. **Which bug was the dominant cause of the M2 F regression?** (#2 padding, #5 candidate-pool mismatch, #1 segment ids, or compound?)
2. **Did Phase B's retrained CE clear the BM25 baseline (F > 0.107)?**
3. **Did it hit Plan B's original Task 2.4 prediction (F >= 0.18)?**

These answers feed into Phase 3 / Phase 4 decisions about whether to add a BM25+CE blended score (the third option Plan B left open) or proceed with pure CE rerank.

---

# Notes on what's intentionally NOT in this plan

- **Renaming `cfg.bm25_top_k` -> `cfg.candidate_pool_k`:** punted to a separate cleanup commit. It's a churn-only change and would touch every call site.
- **Reproducibility hardening (DataLoader `worker_init_fn`):** `__getitem__` is deterministic given `idx`, so this is latent. Not blocking.
- **Removing the duplicated `_pick_device` helper:** trivial; do it opportunistically next time either script is touched.
- **Re-running BM25-only k=4 baseline under bm25s:** strongly recommended for paper-grade rigor, but doesn't affect the bug-fix scope. Add to a separate verification task if writing the report.
