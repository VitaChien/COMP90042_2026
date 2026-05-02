import pytest
import torch

from src.retriever_cross_enc import (
    CrossEncoderDataset,
    build_cross_encoder,
    rerank,
)

# Tiny BERT (4M params, 2-layer, 128-hidden) keeps these tests CPU-friendly.
# Google's official miniature ships a tokenizer.json so it loads under the
# stricter tokenizer-loading path in transformers >= 5.
TINY_MODEL = "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture(scope="module")
def tokenizer_and_model():
    return build_cross_encoder(TINY_MODEL)


def test_dataset_yields_tensors(tokenizer_and_model):
    tok, _ = tokenizer_and_model
    pairs = [{"claim_text": "claim", "evidence_text": "ev", "label": 1}]
    ds = CrossEncoderDataset(pairs, tok, max_len=32)
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert item["labels"].dtype == torch.float32
    assert item["input_ids"].shape[0] <= 32


def test_dataset_resolves_evidence_via_lookup(tokenizer_and_model):
    """When pairs only contain `evidence_id`, dataset must look up text."""
    tok, _ = tokenizer_and_model
    pairs = [{"claim_text": "c", "evidence_id": "e-1", "label": 0}]
    ds = CrossEncoderDataset(pairs, tok, max_len=16, evidence_lookup={"e-1": "ev text"})
    item = ds[0]
    assert int(item["labels"]) == 0
    assert item["input_ids"].shape[0] <= 16


def test_rerank_orders_by_score(tokenizer_and_model):
    tok, model = tokenizer_and_model
    candidates = [("e-1", 5.0), ("e-2", 4.0), ("e-3", 3.0)]
    evidence_lookup = {"e-1": "alpha", "e-2": "beta", "e-3": "gamma"}
    out = rerank(
        model,
        tok,
        claim_text="q",
        candidates=candidates,
        evidence_lookup=evidence_lookup,
        top_k=2,
        batch_size=2,
        device="cpu",
        max_len=32,
    )
    assert len(out) == 2
    assert all(isinstance(s, float) for _, s in out)
    assert out[0][1] >= out[1][1]


def test_rerank_handles_empty_candidates(tokenizer_and_model):
    tok, model = tokenizer_and_model
    out = rerank(
        model,
        tok,
        claim_text="q",
        candidates=[],
        evidence_lookup={},
        top_k=4,
        device="cpu",
    )
    assert out == []


def test_save_load_roundtrip_preserves_scores(tmp_path, tokenizer_and_model):
    """After save+load, the same (claim, evidence) pair must score identically.

    Guards against checkpoint format drift between train and inference.
    Both models are put in eval() mode so dropout does not introduce noise.
    """
    from src.retriever_cross_enc import load_cross_encoder

    tok, model = tokenizer_and_model
    model.eval()
    ckpt = tmp_path / "ce.pt"
    torch.save(model.state_dict(), ckpt)

    _, model2 = load_cross_encoder(TINY_MODEL, ckpt, device="cpu")

    candidates = [("e-1", 1.0)]
    lookup = {"e-1": "some evidence"}
    s1 = rerank(model, tok, "claim", candidates, lookup, top_k=1, device="cpu", max_len=32)
    s2 = rerank(model2, tok, "claim", candidates, lookup, top_k=1, device="cpu", max_len=32)
    assert abs(s1[0][1] - s2[0][1]) < 1e-5


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

    assert (
        abs(train_score - rerank_score) < 1e-5
    ), f"train vs rerank logit drift: train={train_score:.6f} rerank={rerank_score:.6f}"


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
    enc = tok(
        "claim text",
        "evidence text",
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True,
    )
    with torch.no_grad():
        a = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc["token_type_ids"],
        ).item()
        b = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=torch.zeros_like(enc["token_type_ids"]),
        ).item()
    assert abs(a - b) > 1e-5, "model is ignoring token_type_ids"


def test_train_saves_epoch_checkpoints(tmp_path, tokenizer_and_model):
    """train_cross_encoder must write cross_encoder_epochN.pt after each epoch
    containing model, optimizer, and scheduler state dicts plus epoch number."""
    from src.retriever_cross_enc import train_cross_encoder

    tok, model = tokenizer_and_model
    pairs = [
        {"claim_text": "a", "evidence_text": "b", "label": 1},
        {"claim_text": "c", "evidence_text": "d", "label": 0},
    ]
    train_cross_encoder(
        model,
        tok,
        pairs,
        {},
        max_len=16,
        batch_size=2,
        lr=1e-4,
        epochs=2,
        device="cpu",
        save_path=tmp_path / "ce.pt",
    )
    for ep in (1, 2):
        ckpt_path = tmp_path / f"cross_encoder_epoch{ep}.pt"
        assert ckpt_path.exists(), f"epoch {ep} checkpoint missing"
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert set(ckpt.keys()) == {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
        }
        assert ckpt["epoch"] == ep


def test_resume_starts_from_correct_epoch(tmp_path, tokenizer_and_model):
    """When resume_from points to an epoch-1 checkpoint, only epoch 2 runs
    and cross_encoder_epoch2.pt is created with epoch==2."""
    from src.retriever_cross_enc import train_cross_encoder

    tok, model = tokenizer_and_model
    pairs = [{"claim_text": "a", "evidence_text": "b", "label": 1}]
    save_path = tmp_path / "ce.pt"

    # Phase 1: train only epoch 1 to produce the resume checkpoint
    train_cross_encoder(
        model,
        tok,
        pairs,
        {},
        max_len=16,
        batch_size=1,
        lr=1e-4,
        epochs=1,
        device="cpu",
        save_path=save_path,
    )
    epoch1_ckpt = tmp_path / "cross_encoder_epoch1.pt"
    assert epoch1_ckpt.exists()

    # Phase 2: fresh model resumes from epoch 1 — only epoch 2 should run
    _, model2 = build_cross_encoder(TINY_MODEL)
    train_cross_encoder(
        model2,
        tok,
        pairs,
        {},
        max_len=16,
        batch_size=1,
        lr=1e-4,
        epochs=2,
        device="cpu",
        save_path=save_path,
        resume_from=epoch1_ckpt,
    )
    epoch2_ckpt = tmp_path / "cross_encoder_epoch2.pt"
    assert epoch2_ckpt.exists(), "epoch 2 checkpoint must be created after resume"
    ckpt2 = torch.load(epoch2_ckpt, weights_only=False)
    assert ckpt2["epoch"] == 2
