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
