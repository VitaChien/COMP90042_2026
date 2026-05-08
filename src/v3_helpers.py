"""Helpers used by the v3 CNN+BiLSTM+Multihead+Balanced notebook.

Kept as importable module so unit tests can run without Colab/notebook setup.
"""

from __future__ import annotations

import os
import random as _random
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch


def simple_tokenise(text: str) -> list[str]:
    """Lowercase + strip non-alphanumeric (keep . , - % °) + split + strip
    trailing .,;: from each token.

    The trailing-punctuation strip is the #16 fix: prevents `1.5°c.` and
    `1.5°c` from becoming different vocab entries, which fragments
    climate-domain numeric tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\,\-\%°]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    # Strip *trailing* punctuation only — internal `.` (decimals) and `,`
    # (thousands sep) are preserved.
    return [t.rstrip(".,;:") for t in tokens if t.rstrip(".,;:")]


def build_vocab_full_corpus(
    train_claims: dict,
    evidence_corpus: dict[str, str],
    min_freq: int = 2,
    max_vocab_size: int = 50_000,
) -> dict[str, int]:
    """Build vocab over claims + entire evidence corpus.

    v3 only counted train claims + their gold evidence (~6872 tokens).
    This counts every passage in `evidence_corpus`, drastically reducing
    UNK rate at predict time.
    """
    counter: Counter = Counter()

    for instance in train_claims.values():
        counter.update(simple_tokenise(instance["claim_text"]))

    for text in evidence_corpus.values():
        counter.update(simple_tokenise(text))

    vocab = {"<PAD>": 0, "<UNK>": 1, "<CLAIM>": 2, "<EVIDENCE>": 3}
    for word, freq in counter.most_common(max_vocab_size):
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def select_best_epoch(
    history: Iterable[tuple[int, float, float]],
    key: str = "retrieved",
) -> tuple[int, float, float]:
    """Pick the epoch with the highest F1 by `key`.

    history: iterable of (epoch, gold_f1, retrieved_f1).
    key: 'retrieved' (default — what the leaderboard scores)
         | 'gold'  (legacy, for ablation only)

    Ties break by earliest epoch.
    """
    idx = {"gold": 1, "retrieved": 2}[key]
    best = None
    for entry in history:
        if best is None or entry[idx] > best[idx]:
            best = entry
    if best is None:
        raise ValueError("history was empty")
    return best


def pick_evidence_ids(
    gold: Sequence[str],
    retrieved: Sequence[str],
    p_retrieved: float,
    rng: _random.Random | None = None,
) -> list[str]:
    """Choose evidence IDs for a training example.

    With probability `p_retrieved`, return retrieved-with-gold-filtered-out
    (hard negatives). Otherwise return gold. If gold is empty (e.g. NEI
    claims), fall back to retrieved regardless of p_retrieved.

    Filtering gold from retrieved ensures hard negatives are genuinely
    distractors — passages that look relevant but are not labelled gold.
    """
    rng = rng or _random
    if not gold:
        return list(retrieved)
    use_retrieved = rng.random() < p_retrieved
    if use_retrieved:
        gold_set = set(gold)
        return [eid for eid in retrieved if eid not in gold_set]
    return list(gold)


class BM25CERetriever:
    """Adapter wrapping vita/retriever's BM25 + cross-encoder rerank pipeline.

    Exposes the same `.retrieve(claim_text, top_k) -> list[str]` interface as
    the v3 FAISS retriever, so it's a drop-in replacement.

    Construction is deferred (notebook builds the BM25 + CE objects, then
    wraps them here) so this module stays free of heavyweight imports
    (transformers, faiss, bm25s) and is fast to unit-test.
    """

    def __init__(
        self,
        bm25,
        rerank_fn: Callable[[str, list, int], list],
        bm25_top_k: int = 200,
    ) -> None:
        self.bm25 = bm25
        self.rerank_fn = rerank_fn
        self.bm25_top_k = bm25_top_k

    def retrieve(self, claim_text: str, top_k: int = 5) -> list[str]:
        candidates = self.bm25.search(claim_text, top_k=self.bm25_top_k)
        ranked: list[tuple[str, float]] = self.rerank_fn(claim_text, candidates, top_k)
        return [eid for eid, _score in ranked]


class FAISSDenseRetriever:
    """Sentence-transformer + FAISS dense retrieval — CPU-friendly fallback.

    On first construction encodes the entire evidence corpus (slow:
    ~1 hr on CPU for 1.2M passages) and writes a FAISS index to
    `cache_dir`. Subsequent constructions load from cache instantly.

    Used when no GPU is available, where running BERT cross-encoder over
    BM25 top-200 per claim is prohibitively slow. FAISS dense is hundreds
    of times faster on CPU because the heavy work (encoding the corpus)
    is amortised across runs via the on-disk cache.

    Same `.retrieve(claim_text, top_k) -> list[str]` interface as
    `BM25CERetriever`, so it's a drop-in replacement.

    Heavy deps (`faiss`, `sentence_transformers`) are lazy-imported in
    `__init__` to keep this module fast to import for unit tests.
    """

    def __init__(
        self,
        evidence_corpus: dict,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = "cache/faiss",
        batch_size: int = 64,
        device: str | None = None,
        force_rebuild: bool = False,
    ) -> None:
        import json
        from pathlib import Path

        import faiss
        from sentence_transformers import SentenceTransformer

        self._faiss = faiss
        self.evidence_corpus = evidence_corpus
        self.evidence_ids = list(evidence_corpus.keys())
        self.evidence_texts = [evidence_corpus[eid] for eid in self.evidence_ids]

        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace("/", "_")
        self.index_path = cache / f"{safe_name}.faiss"
        self.ids_path = cache / f"{safe_name}_evidence_ids.json"

        self.model = SentenceTransformer(model_name, device=device)

        if self.index_path.exists() and self.ids_path.exists() and not force_rebuild:
            with self.ids_path.open("r", encoding="utf-8") as f:
                cached_ids = json.load(f)
            if cached_ids == self.evidence_ids:
                self.index = faiss.read_index(str(self.index_path))
                print(f"Loaded cached FAISS index ({self.index.ntotal} vectors).")
                return
            print("Cached IDs mismatch current evidence — rebuilding.")

        print(
            f"Encoding {len(self.evidence_texts)} evidence passages "
            f"(slow: ~1 hr on CPU, one-time)..."
        )
        embeddings = self.model.encode(
            self.evidence_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, str(self.index_path))
        with self.ids_path.open("w", encoding="utf-8") as f:
            json.dump(self.evidence_ids, f)
        print(f"FAISS index built ({self.index.ntotal} vectors) and cached.")

    def retrieve(self, claim_text: str, top_k: int = 5) -> list[str]:
        emb = self.model.encode(
            [claim_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        _scores, indices = self.index.search(emb, top_k)
        return [self.evidence_ids[i] for i in indices[0]]


def build_minimal_classifier_for_testing(vocab_size: int = 50):
    """Tiny CNN+BiLSTM+MHA+Pool model for the PAD-invariance unit test.

    Mirrors the production CNNBiLSTMMultiheadClassifier but with shrunk dims
    (embedding_dim=8, cnn_channels=4, lstm_hidden=8, num_heads=2) so unit
    tests run in <1s on CPU.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class _AttnPool(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = nn.Linear(dim, 1)

        def forward(self, hidden, mask):
            scores = self.attn(hidden).squeeze(-1)
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
            weights = torch.softmax(scores, dim=1)
            return torch.sum(hidden * weights.unsqueeze(-1), dim=1)

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 8, padding_idx=0)
            self.conv = nn.Conv1d(8, 4, kernel_size=3, padding=1)
            self.lstm = nn.LSTM(4, 8, batch_first=True, bidirectional=True)
            self.mha = nn.MultiheadAttention(16, 2, batch_first=True)
            self.norm = nn.LayerNorm(16)
            self.pool = _AttnPool(16)

        def forward(self, input_ids, attention_mask):
            emb = self.embedding(input_ids)
            x = F.relu(self.conv(emb.transpose(1, 2))).transpose(1, 2)
            lengths = attention_mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
            kpm = attention_mask == 0
            attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out, key_padding_mask=kpm)
            attn_out = self.norm(attn_out + lstm_out)
            return self.pool(attn_out, attention_mask)

    torch.manual_seed(42)
    return _Tiny()


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility (CPU + CUDA + numpy + Python)."""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def multi_seed_run(
    runner: Callable[[int], dict[str, float]],
    seeds: Sequence[int],
) -> dict[str, dict]:
    """Run `runner(seed)` for each seed, aggregate to mean / std per metric.

    `runner(seed)` should:
      1. Set the global seed,
      2. Build the model fresh,
      3. Train + evaluate,
      4. Return a flat dict[str, float] of metrics.

    Returns:
      {metric_name: {"mean": float, "std": float, "values": list[float],
                     "seeds": list[int]}}
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")

    per_seed = []
    for s in seeds:
        per_seed.append(runner(s))

    metrics = per_seed[0].keys()
    summary: dict[str, dict] = {}
    for m in metrics:
        values = [r[m] for r in per_seed]
        summary[m] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),  # population — small n
            "values": values,
            "seeds": list(seeds),
        }
    return summary
