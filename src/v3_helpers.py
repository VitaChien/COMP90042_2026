"""Helpers used by the v3 CNN+BiLSTM+Multihead+Balanced notebook.

Kept as importable module so unit tests can run without Colab/notebook setup.
"""

from __future__ import annotations

import random as _random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence


def simple_tokenise(text: str) -> list[str]:
    """Lowercase + strip non-alphanumeric (keep . , - % °) + whitespace split.

    Note: Task 6 (#16) will add trailing-punctuation stripping; for now this
    matches v3 behaviour exactly so the vocab change can be measured in
    isolation.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\,\-\%°]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


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
