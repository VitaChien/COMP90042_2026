"""Tests for classifier.py bug fixes."""
import pytest
import torch
from unittest.mock import MagicMock, patch


# ── helpers ──────────────────────────────────────────────────────────────────

def make_mock_tokenizer():
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return tok


def make_data_dict(labels=("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED")):
    """Minimal data_dict with one example per label."""
    return {
        f"id_{i}": {
            "claim_text": f"claim {i}",
            "claim_label": label,
            "evidences": [f"ev_{i}"],
        }
        for i, label in enumerate(labels)
    }


def make_evidence_dict():
    return {f"ev_{i}": f"evidence text {i}" for i in range(10)}


# ── C1: FactCheckDataset class_weights ───────────────────────────────────────

def test_factcheckdataset_no_label2id_param():
    """FactCheckDataset should NOT accept label2id parameter."""
    import inspect
    from classifier import FactCheckDataset
    sig = inspect.signature(FactCheckDataset.__init__)
    assert "label2id" not in sig.parameters, \
        "label2id parameter must be removed (dead code, caused C1 bug)"


def test_factcheckdataset_class_weights_4class():
    """class_weights must have exactly 4 entries and be positive."""
    from classifier import FactCheckDataset
    tok = make_mock_tokenizer()
    ds = FactCheckDataset(make_data_dict(), make_evidence_dict(), tok)
    assert ds.class_weights.shape == (4,), \
        f"Expected shape (4,), got {ds.class_weights.shape}"
    assert (ds.class_weights > 0).all(), "All weights must be positive"


def test_factcheckdataset_class_weights_balanced():
    """When all 4 classes have equal counts, weights should be equal."""
    from classifier import FactCheckDataset
    tok = make_mock_tokenizer()
    ds = FactCheckDataset(make_data_dict(), make_evidence_dict(), tok)
    w = ds.class_weights
    assert torch.allclose(w, w[0].expand_as(w), atol=1e-4), \
        f"Equal counts → equal weights, got {w.tolist()}"
