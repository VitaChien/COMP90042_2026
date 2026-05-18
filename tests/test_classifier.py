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


# ── C2: eval_macro_f1_dev uses model.config.id2label ─────────────────────────

def test_eval_macro_f1_dev_uses_model_config_id2label():
    """eval_macro_f1_dev must derive label mapping from model.config, not module constant."""
    import inspect
    from classifier import eval_macro_f1_dev
    sig = inspect.signature(eval_macro_f1_dev)
    assert "id2label" not in sig.parameters, \
        "id2label parameter must be removed; function must read from model.config.id2label"


def test_eval_macro_f1_dev_correct_label_mapping():
    """Predictions are mapped using model.config.id2label, not the module constant."""
    from classifier import eval_macro_f1_dev, LABEL2ID, ID2LABEL

    # Build a mock model that always predicts index 0 (SUPPORTS)
    mock_model = MagicMock()
    mock_model.config.id2label = ID2LABEL  # {0:'SUPPORTS', 1:'REFUTES', ...}
    logits = torch.zeros(2, 4)  # batch of 2, argmax → 0 → SUPPORTS
    mock_model.return_value.logits = logits
    mock_model.eval = MagicMock()

    enc_mock = MagicMock()
    enc_mock.__getitem__ = lambda self, k: (
        torch.zeros(2, 10, dtype=torch.long) if k == "input_ids"
        else torch.ones(2, 10, dtype=torch.long)
    )
    enc_mock.to = lambda d: enc_mock
    mock_tokenizer = MagicMock(return_value=enc_mock)

    dev_data = {
        "a": {"claim_text": "claim a", "claim_label": "SUPPORTS", "evidences": ["ev_0"]},
        "b": {"claim_text": "claim b", "claim_label": "SUPPORTS", "evidences": ["ev_1"]},
    }
    evidence_dict = {"ev_0": "text 0", "ev_1": "text 1"}

    f1, y_true, y_pred = eval_macro_f1_dev(
        mock_model, mock_tokenizer, dev_data, evidence_dict,
        device=torch.device("cpu"), batch_size=2,
    )
    assert all(p == "SUPPORTS" for p in y_pred), \
        f"Expected all SUPPORTS predictions, got {y_pred}"
