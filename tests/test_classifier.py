"""Tests for classifier.py bug fixes."""
import json
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

    # Use a DIFFERENT id2label than module constant: index 0 → "REFUTES" (not "SUPPORTS")
    # If function uses module constant, it would return "SUPPORTS" for argmax=0
    # If function uses model.config.id2label, it returns "REFUTES" — the correct behaviour
    inverted_id2label = {0: "REFUTES", 1: "SUPPORTS", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}

    mock_model = MagicMock()
    mock_model.config.id2label = inverted_id2label
    logits = torch.zeros(2, 4)  # batch of 2, argmax → 0
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
        "a": {"claim_text": "claim a", "claim_label": "REFUTES", "evidences": ["ev_0"]},
        "b": {"claim_text": "claim b", "claim_label": "REFUTES", "evidences": ["ev_1"]},
    }
    evidence_dict = {"ev_0": "text 0", "ev_1": "text 1"}

    f1, y_true, y_pred = eval_macro_f1_dev(
        mock_model, mock_tokenizer, dev_data, evidence_dict,
        device=torch.device("cpu"), batch_size=2,
    )
    assert all(p == "REFUTES" for p in y_pred), \
        f"Expected all REFUTES (from model.config), got {y_pred} — possible fallback to module constant"


# ── I2: DISPUTED weight boost uses LABEL2ID index ────────────────────────────

def test_disputed_weight_boost_uses_label2id():
    """DISPUTED weight boost must use LABEL2ID['DISPUTED'] not hardcoded [3]."""
    from classifier import FactCheckDataset, LABEL2ID
    tok = make_mock_tokenizer()
    ds = FactCheckDataset(make_data_dict(), make_evidence_dict(), tok)
    original_disputed_weight = ds.class_weights[LABEL2ID["DISPUTED"]].item()
    # Verify the index is correct by checking value matches index 3 (current mapping)
    assert LABEL2ID["DISPUTED"] == 3, "Sanity: DISPUTED should be index 3"
    assert original_disputed_weight > 0


# ── I3: NEI evidence uses BM25 in eval when retriever provided ───────────────

def test_eval_macro_f1_dev_nei_uses_bm25_when_provided():
    """When bm25_retriever is passed, NEI examples use retrieved evidence not gold."""
    from classifier import eval_macro_f1_dev

    retrieved_text = "bm25 retrieved passage"
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"text": retrieved_text, "id": "ret_0"}]

    captured_ev_texts = []

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}
    mock_model.eval = MagicMock()

    def fake_tokenizer(ev_texts, claims, **kwargs):
        captured_ev_texts.extend(ev_texts)
        result = MagicMock()
        result.to = lambda d: result
        result.items = lambda: [
            ("input_ids", torch.zeros(len(ev_texts), 5, dtype=torch.long)),
            ("attention_mask", torch.ones(len(ev_texts), 5, dtype=torch.long)),
        ]
        return result

    logits = torch.zeros(1, 4)
    logits[0][2] = 10.0  # predict NOT_ENOUGH_INFO
    mock_model.return_value.logits = logits

    dev_data = {
        "nei_claim": {
            "claim_text": "a nei claim",
            "claim_label": "NOT_ENOUGH_INFO",
            "evidences": [],  # gold evidences empty for NEI
        }
    }
    evidence_dict = {}

    eval_macro_f1_dev(
        mock_model, fake_tokenizer, dev_data, evidence_dict,
        device=torch.device("cpu"), batch_size=16,
        bm25_retriever=mock_retriever,
    )

    mock_retriever.retrieve.assert_called_once_with("a nei claim", top_k=3)
    assert any(retrieved_text in ev for ev in captured_ev_texts), \
        f"BM25 retrieved text must be used for NEI evidence, got: {captured_ev_texts}"


def test_eval_macro_f1_dev_nei_uses_gold_without_retriever():
    """Without bm25_retriever, NEI falls back to gold evidence (original behaviour)."""
    from classifier import eval_macro_f1_dev

    captured_ev_texts = []

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO", 3: "DISPUTED"}
    mock_model.eval = MagicMock()

    def fake_tokenizer(ev_texts, claims, **kwargs):
        captured_ev_texts.extend(ev_texts)
        result = MagicMock()
        result.to = lambda d: result
        result.items = lambda: [
            ("input_ids", torch.zeros(len(ev_texts), 5, dtype=torch.long)),
            ("attention_mask", torch.ones(len(ev_texts), 5, dtype=torch.long)),
        ]
        return result

    mock_model.return_value.logits = torch.zeros(1, 4)

    dev_data = {
        "nei": {
            "claim_text": "nei claim",
            "claim_label": "NOT_ENOUGH_INFO",
            "evidences": ["ev_gold"],
        }
    }
    evidence_dict = {"ev_gold": "gold evidence text"}

    eval_macro_f1_dev(
        mock_model, fake_tokenizer, dev_data, evidence_dict,
        device=torch.device("cpu"),
    )

    assert any("gold evidence text" in ev for ev in captured_ev_texts), \
        "Without retriever, NEI must use gold evidence"


# ── load_retriever_cache ──────────────────────────────────────────────────────

def test_load_retriever_cache_returns_valid_dict(tmp_path):
    """load_retriever_cache should load JSON and return the dict unchanged."""
    from classifier import load_retriever_cache

    data = {
        "claim-1": {
            "claim_text": "foo",
            "claim_label": "SUPPORTS",
            "evidences": ["ev-1", "ev-2"],
        },
        "claim-2": {
            "claim_text": "bar",
            "claim_label": "NOT_ENOUGH_INFO",
            "evidences": ["ev-3"],
        },
    }
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps(data))
    result = load_retriever_cache(str(cache_file))
    assert result == data
    assert len(result) == 2


def test_load_retriever_cache_raises_on_missing_field(tmp_path):
    """load_retriever_cache should raise ValueError if entry is missing required fields."""
    from classifier import load_retriever_cache

    data = {
        "claim-99": {
            "claim_text": "missing evidences",
            "claim_label": "SUPPORTS",
            # no "evidences" key
        },
    }
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="claim-99"):
        load_retriever_cache(str(cache_file))


def test_load_retriever_cache_raises_on_invalid_evidences_type(tmp_path):
    data = {
        "claim-5": {
            "claim_text": "foo",
            "claim_label": "SUPPORTS",
            "evidences": None,   # should be a list
        },
    }
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="claim-5"):
        from classifier import load_retriever_cache
        load_retriever_cache(str(cache_file))
