from src.config import Config
from src.data_loader import (
    EVIDENCE_KEY_PATTERN,
    Claim,
    load_claims,
    load_evidence_streaming,
)


def test_load_train_claims_returns_dict_of_claims():
    cfg = Config()
    claims = load_claims(cfg.train_path)
    assert len(claims) == 1228
    sample = next(iter(claims.values()))
    assert isinstance(sample, Claim)
    assert sample.claim_text
    assert sample.claim_label in cfg.label_names
    assert isinstance(sample.evidences, list) and len(sample.evidences) > 0


def test_load_test_claims_has_no_label():
    cfg = Config()
    claims = load_claims(cfg.test_path)
    assert len(claims) == 153
    sample = next(iter(claims.values()))
    assert sample.claim_label is None
    assert sample.evidences == []


def test_load_evidence_streaming_yields_all():
    cfg = Config()
    count = 0
    seen_first = None
    for eid, text in load_evidence_streaming(cfg.evidence_path):
        if count == 0:
            seen_first = eid
            assert EVIDENCE_KEY_PATTERN.match(eid)
            assert isinstance(text, str) and text
        count += 1
        if count >= 1000:
            break
    assert count == 1000
    assert seen_first.startswith("evidence-")
