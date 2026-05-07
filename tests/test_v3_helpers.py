from src.v3_helpers import build_vocab_full_corpus, simple_tokenise


def test_build_vocab_full_corpus_includes_corpus_only_tokens():
    train_claims = {
        "c1": {"claim_text": "alpha beta", "evidences": ["e1"]},
    }
    evidence_corpus = {
        "e1": "alpha gamma",       # gamma appears in gold (and corpus)
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

    assert "common" in vocab     # freq 3 >= 2 -> included
    assert "rare_word" not in vocab  # freq 1 < 2 -> excluded


def test_build_vocab_respects_max_vocab_size():
    train_claims = {"c1": {"claim_text": "x", "evidences": []}}
    # 100 unique tokens in corpus, all freq 1
    evidence_corpus = {f"e{i}": f"tok{i}" for i in range(100)}
    vocab = build_vocab_full_corpus(
        train_claims, evidence_corpus, min_freq=1, max_vocab_size=10
    )
    # 4 special + at most 10 ordinary = 14
    assert len(vocab) <= 14
