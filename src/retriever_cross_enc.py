"""BERT cross-encoder for relevance re-ranking (Lecture 11 BERT [CLS] pattern).

Pipeline role: Stage 1B (re-ranker). BM25 reduces 1.2M evidences to ~200
lexically similar candidates; this module re-scores each (claim, evidence)
pair jointly with a BERT cross-encoder so that semantic relevance, not
just word overlap, decides the final top-K.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.utils import get_logger

log = get_logger("cross-enc")


def build_cross_encoder(model_name: str) -> tuple[AutoTokenizer, CrossEncoderHead]:
    """Return ``(tokenizer, model)``.

    Model = pretrained encoder + 1-d linear head over the ``[CLS]`` vector
    producing a single relevance logit. Same factory is used for tests
    (tiny BERT) and training (``bert-base-uncased``).
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    model = CrossEncoderHead(encoder)
    return tok, model


class CrossEncoderHead(nn.Module):
    """1-logit relevance head on top of any HF encoder."""

    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, **_kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls)).squeeze(-1)


class CrossEncoderDataset(Dataset):
    """Dataset of (claim, evidence, label) -> tokenized pair tensors.

    Each ``pair`` dict must contain ``claim_text`` and ``label`` plus
    either ``evidence_text`` (preferred) or ``evidence_id``. When only
    ``evidence_id`` is given, ``evidence_lookup`` resolves the text.
    """

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
            "labels": torch.tensor(float(p["label"]), dtype=torch.float32),
        }


def train_cross_encoder(
    model: CrossEncoderHead,
    tokenizer,
    train_pairs: Sequence[dict],
    evidence_lookup: dict[str, str],
    max_len: int,
    batch_size: int,
    lr: float,
    epochs: int,
    device: str,
    save_path: Path | str,
) -> None:
    """Fine-tune cross-encoder with BCE on (claim, evidence) pairs."""
    ds = CrossEncoderDataset(
        train_pairs,
        tokenizer,
        max_len=max_len,
        evidence_lookup=evidence_lookup,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        running = 0.0
        for batch in tqdm(loader, desc=f"CE epoch {ep + 1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = loss_fn(logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        log.info("epoch %d mean_loss=%.4f", ep + 1, running / len(loader))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    log.info("Saved cross-encoder ckpt -> %s", save_path)


def load_cross_encoder(
    model_name: str,
    ckpt_path: Path | str,
    device: str = "cpu",
) -> tuple[AutoTokenizer, CrossEncoderHead]:
    tok, model = build_cross_encoder(model_name)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return tok, model.to(device).eval()


@torch.no_grad()
def rerank(
    model: CrossEncoderHead,
    tokenizer,
    claim_text: str,
    candidates: Sequence[tuple[str, float]],
    evidence_lookup: dict[str, str],
    top_k: int,
    batch_size: int = 64,
    device: str = "cpu",
    max_len: int = 256,
) -> list[tuple[str, float]]:
    """Re-score candidates with the cross-encoder; return top-K by score.

    ``candidates`` is the BM25 output ``[(evidence_id, bm25_score), ...]``.
    The bm25_score component is discarded - only the id is used to fetch
    text from ``evidence_lookup``. Scores returned are sigmoid-of-logit
    relevance probabilities in [0, 1].
    """
    if not candidates:
        return []
    eids = [eid for eid, _ in candidates]
    texts = [evidence_lookup[e] for e in eids]
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            [claim_text] * len(batch_texts),
            batch_texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    ranked = sorted(zip(eids, scores, strict=True), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
