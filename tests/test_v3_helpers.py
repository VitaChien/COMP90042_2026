import random

from src.v3_helpers import (
    BM25CERetriever,
    build_vocab_full_corpus,
    pick_evidence_ids,
    select_best_epoch,
)


def test_build_vocab_full_corpus_includes_corpus_only_tokens():
    train_claims = {
        "c1": {"claim_text": "alpha beta", "evidences": ["e1"]},
    }
    evidence_corpus = {
        "e1": "alpha gamma",  # gamma appears in gold (and corpus)
        "e2": "delta delta epsilon",  # delta+epsilon only in non-gold corpus
    }
    vocab = build_vocab_full_corpus(train_claims, evidence_corpus, min_freq=1)

    # Special tokens always present
    for sp in ("<PAD>", "<UNK>", "<CLAIM>", "<EVIDENCE>"):
        assert sp in vocab

    # Tokens from non-gold passage e2 must be in vocab
    assert "delta" in vocab, "non-gold corpus tokens must be included"
    assert "epsilon" in vocab

    # Old behaviour (gold-only) would have missed delta/epsilon
    assert "alpha" in vocab and "beta" in vocab and "gamma" in vocab


def test_build_vocab_respects_min_freq():
    # rare_word in claim only (freq=1); common in corpus only (freq=3).
    train_claims = {"c1": {"claim_text": "rare_word", "evidences": []}}
    evidence_corpus = {"e1": "common common common"}

    vocab = build_vocab_full_corpus(train_claims, evidence_corpus, min_freq=2)

    assert "common" in vocab  # freq 3 >= 2 -> included
    assert "rare_word" not in vocab  # freq 1 < 2 -> excluded


def test_build_vocab_respects_max_vocab_size():
    train_claims = {"c1": {"claim_text": "x", "evidences": []}}
    # 100 unique tokens in corpus, all freq 1
    evidence_corpus = {f"e{i}": f"tok{i}" for i in range(100)}
    vocab = build_vocab_full_corpus(train_claims, evidence_corpus, min_freq=1, max_vocab_size=10)
    # 4 special + at most 10 ordinary = 14
    assert len(vocab) <= 14


def test_select_best_epoch_picks_max_retrieved_f1():
    # Each tuple: (epoch, gold_f1, retrieved_f1)
    history = [
        (1, 0.40, 0.20),
        (2, 0.50, 0.30),  # gold peak earlier
        (3, 0.45, 0.35),  # retrieved peak here
        (4, 0.55, 0.32),
    ]
    best = select_best_epoch(history, key="retrieved")
    assert best[0] == 3, "must pick retrieved-F1 peak, not gold-F1 peak"


def test_select_best_epoch_ties_break_by_first():
    history = [(1, 0.5, 0.4), (2, 0.6, 0.4)]
    best = select_best_epoch(history, key="retrieved")
    assert best[0] == 1, "ties must break to earliest epoch"


def test_select_best_epoch_default_is_retrieved():
    history = [(1, 0.99, 0.10), (2, 0.10, 0.50)]
    assert select_best_epoch(history)[0] == 2


def test_mixed_evidence_uses_gold_with_prob_zero():
    """p_retrieved=0 - always gold (degenerate to current behaviour)."""
    gold = ["g1", "g2"]
    fake_retriever_output = ["r1", "r2", "r3"]
    rng = random.Random(0)
    for _ in range(20):
        result = pick_evidence_ids(
            gold=gold, retrieved=fake_retriever_output, p_retrieved=0.0, rng=rng
        )
        assert result == gold


def test_mixed_evidence_uses_retrieved_with_prob_one():
    gold = ["g1", "g2"]
    retrieved = ["r1", "r2", "r3"]
    rng = random.Random(0)
    for _ in range(20):
        result = pick_evidence_ids(gold=gold, retrieved=retrieved, p_retrieved=1.0, rng=rng)
        assert result == retrieved


def test_mixed_evidence_excludes_gold_from_retrieved():
    """Hard negatives must not be gold passages (else they're trivial)."""
    gold = ["e1"]
    retrieved = ["e1", "e2", "e3"]  # retriever happens to return gold too
    rng = random.Random(0)
    result = pick_evidence_ids(gold=gold, retrieved=retrieved, p_retrieved=1.0, rng=rng)
    assert "e1" not in result, "gold passages must be filtered from hard-negatives"
    assert result == ["e2", "e3"]


def test_mixed_evidence_empty_gold_falls_back_to_retrieved():
    """NEI claims often have empty gold; use retrieved unconditionally."""
    rng = random.Random(0)
    result = pick_evidence_ids(gold=[], retrieved=["r1", "r2"], p_retrieved=0.0, rng=rng)
    assert result == ["r1", "r2"]


def test_mixed_evidence_distribution_around_p():
    """With p=0.5 over many trials, ~half should be gold, ~half retrieved."""
    gold = ["g"]
    retrieved = ["r"]
    rng = random.Random(42)
    n = 1000
    n_retrieved = sum(
        pick_evidence_ids(gold=gold, retrieved=retrieved, p_retrieved=0.5, rng=rng) == retrieved
        for _ in range(n)
    )
    assert 400 <= n_retrieved <= 600, f"expected ~500, got {n_retrieved}"


class _StubBM25:
    """Mimics BM25Retriever.search."""

    def __init__(self, return_value):
        self._return = return_value
        self.calls = []

    def search(self, query, top_k=200):
        self.calls.append((query, top_k))
        return self._return[:top_k]


def test_bm25_ce_retriever_returns_ids_only():
    """Adapter must expose .retrieve(claim, top_k=K) -> list[str]."""
    bm25 = _StubBM25([("e1", 0.9), ("e2", 0.5), ("e3", 0.1)])
    retriever = BM25CERetriever(
        bm25=bm25,
        rerank_fn=lambda claim, candidates, top_k: candidates[:top_k],
        bm25_top_k=200,
    )
    result = retriever.retrieve("a claim", top_k=2)
    assert result == ["e1", "e2"], "must return only IDs in rerank order"


def test_lstm_forward_no_pad_contamination():
    """Two batches with identical content but different padding lengths must
    produce identical pooled output (within fp tolerance) when the BiLSTM
    uses pack_padded_sequence."""
    import torch

    from src.v3_helpers import build_minimal_classifier_for_testing

    model = build_minimal_classifier_for_testing(vocab_size=50)
    model.eval()

    # Sequence A: tokens [1,2,3], no padding (length 3)
    # Sequence B: same tokens [1,2,3] padded to length 6 with PAD=0
    a_ids = torch.tensor([[1, 2, 3]])
    a_mask = torch.tensor([[1, 1, 1]])
    b_ids = torch.tensor([[1, 2, 3, 0, 0, 0]])
    b_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])

    with torch.no_grad():
        out_a = model(a_ids, a_mask)
        out_b = model(b_ids, b_mask)

    assert torch.allclose(
        out_a, out_b, atol=1e-5
    ), f"PAD contamination detected: max diff = {(out_a - out_b).abs().max():.2e}"


def test_bm25_ce_retriever_passes_through_to_rerank():
    bm25 = _StubBM25([("e1", 0.5), ("e2", 0.4), ("e3", 0.3)])
    seen = {}

    def fake_rerank(claim, candidates, top_k):
        seen["claim"] = claim
        seen["candidates"] = candidates
        seen["top_k"] = top_k
        return [(eid, -score) for eid, score in candidates][:top_k]  # reverse

    retriever = BM25CERetriever(bm25=bm25, rerank_fn=fake_rerank, bm25_top_k=3)
    out = retriever.retrieve("hello", top_k=2)
    assert seen["claim"] == "hello"
    assert seen["candidates"] == [("e1", 0.5), ("e2", 0.4), ("e3", 0.3)]
    assert seen["top_k"] == 2
    assert out == ["e1", "e2"]


def test_simple_tokenise_strips_trailing_period():
    from src.v3_helpers import simple_tokenise

    assert simple_tokenise("1.5°c.") == ["1.5°c"]


def test_simple_tokenise_strips_trailing_comma():
    from src.v3_helpers import simple_tokenise

    assert simple_tokenise("alpha, beta.") == ["alpha", "beta"]


def test_simple_tokenise_keeps_internal_period():
    """Internal periods (decimals) must NOT be stripped."""
    from src.v3_helpers import simple_tokenise

    tokens = simple_tokenise("a 1.5 b")
    assert "1.5" in tokens, f"got {tokens}"


def test_simple_tokenise_keeps_internal_comma():
    """Internal commas (rare but present) must NOT be stripped."""
    from src.v3_helpers import simple_tokenise

    tokens = simple_tokenise("100,000 dollars")
    # Either "100,000" preserved or split — but never bare "100,"
    assert not any(t.endswith(",") for t in tokens)
    assert not any(t.endswith(".") for t in tokens)


def test_simple_tokenise_handles_double_punctuation():
    from src.v3_helpers import simple_tokenise

    assert simple_tokenise("end..") == ["end"]
    assert simple_tokenise("end,.") == ["end"]
