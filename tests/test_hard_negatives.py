from src.hard_negatives import build_training_pairs


def test_build_training_pairs_has_pos_and_negs():
    claims = {
        "claim-1": {"claim_text": "Q1", "evidences": ["e-1", "e-2"]},
    }
    bm25_results = {
        "claim-1": [
            ("e-1", 5.0),
            ("e-9", 4.5),
            ("e-8", 4.0),
            ("e-7", 3.5),
            ("e-2", 3.0),
            ("e-6", 2.5),
        ],
    }
    pairs = build_training_pairs(claims, bm25_results, n_neg=2, seed=0)
    labels = [p["label"] for p in pairs]
    assert labels.count(1) == 2  # two gold evidences -> two positives
    assert labels.count(0) == 4  # 2 positives x 2 hard negs each
    neg_eids = {p["evidence_id"] for p in pairs if p["label"] == 0}
    assert neg_eids.issubset({"e-9", "e-8", "e-7", "e-6"})
    assert "e-1" not in neg_eids and "e-2" not in neg_eids


def test_no_pairs_when_bm25_only_returns_gold():
    claims = {"claim-1": {"claim_text": "Q", "evidences": ["e-1"]}}
    bm25_results = {"claim-1": [("e-1", 5.0)]}
    pairs = build_training_pairs(claims, bm25_results, n_neg=2, seed=0)
    # only one positive, no negatives available
    assert sum(p["label"] for p in pairs) == 1
    assert len(pairs) == 1


def test_build_training_pairs_accepts_claim_objects():
    """The miner must work with both raw dicts AND `Claim` dataclasses,
    because train-time call site passes `Claim` objects from data_loader."""
    from src.data_loader import Claim

    claims = {
        "claim-1": Claim(
            claim_id="claim-1",
            claim_text="Q1",
            claim_label="SUPPORTS",
            evidences=["e-1"],
        ),
    }
    bm25_results = {"claim-1": [("e-1", 5.0), ("e-2", 4.0), ("e-3", 3.0)]}
    pairs = build_training_pairs(claims, bm25_results, n_neg=1, seed=0)
    assert sum(p["label"] for p in pairs) == 1
    assert sum(1 for p in pairs if p["label"] == 0) == 1
    assert all(p["claim_text"] == "Q1" for p in pairs)


def test_build_training_pairs_is_deterministic():
    """Same seed -> same negative samples (reproducibility rule)."""
    claims = {"c": {"claim_text": "q", "evidences": ["g"]}}
    bm25 = {"c": [("g", 9.0)] + [(f"n{i}", 5.0 - i) for i in range(20)]}
    a = build_training_pairs(claims, bm25, n_neg=4, seed=42)
    b = build_training_pairs(claims, bm25, n_neg=4, seed=42)
    assert a == b
    c = build_training_pairs(claims, bm25, n_neg=4, seed=43)
    assert c != a  # different seed -> different sample


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
            ("g1", 9.0),
            ("g2", 8.0),
            ("g3", 7.0),
            ("n1", 6.0),
            ("n2", 5.0),
            ("n3", 4.0),
            ("n4", 3.0),
        ],
    }
    # 3 golds * n_neg=4 = 12 negative SLOTS; only 4 unique candidates exist.
    # The miner must cap at 4 unique negatives, not loop and produce 12.
    pairs = build_training_pairs(claims, bm25_results, n_neg=4, seed=0)
    neg_pairs = [(p["claim_id"], p["evidence_id"]) for p in pairs if p["label"] == 0]
    assert len(neg_pairs) == len(set(neg_pairs)), f"duplicate negatives emitted: {neg_pairs}"


def test_miner_uses_full_pool_when_results_span_200():
    """Negatives must be sampled from the full 1-200 range, not just 1-50.

    We pass 200 candidates and verify that at least one sampled negative
    comes from rank > 50 (index >= 50 in the list).  With n_neg=4 and only
    200 non-gold candidates this is virtually certain given any seed.
    """
    gold = ["gold"]
    # 200 non-gold candidates: n0..n49 are ranks 1-50, n50..n199 are ranks 51-200
    candidates = [(f"n{i}", float(200 - i)) for i in range(200)]
    bm25_results = {"c": [("gold", 300.0), *candidates]}
    claims = {"c": {"claim_text": "q", "evidences": gold}}

    pairs = build_training_pairs(claims, bm25_results, n_neg=4, seed=42)
    neg_eids = [p["evidence_id"] for p in pairs if p["label"] == 0]
    assert len(neg_eids) == 4
    # at least one negative from ranks 51-200 (ids n50..n199)
    assert any(
        int(eid[1:]) >= 50 for eid in neg_eids
    ), f"All negatives came from top-50: {neg_eids}"
