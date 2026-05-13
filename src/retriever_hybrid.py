"""Hybrid retriever: RRF fusion over BM25 + dense candidate lists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]: ...


def rrf_fuse(
    bm25_hits: list[tuple[str, float]],
    dense_hits: list[tuple[str, float]],
    k_rrf: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over two ranked lists.

    score(eid) = sum over retrievers of 1 / (k_rrf + rank), where rank is
    1-indexed. eids absent from a list contribute 0 from that retriever.
    Returns top-K unique (eid, score) tuples sorted by score desc.
    """
    scores: dict[str, float] = {}
    for rank, (eid, _) in enumerate(bm25_hits, start=1):
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (k_rrf + rank)
    for rank, (eid, _) in enumerate(dense_hits, start=1):
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (k_rrf + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_k]


@dataclass
class HybridRetriever:
    bm25: _Retriever
    dense: _Retriever
    k_rrf: int = 60
    bm25_top_k: int = 200
    dense_top_k: int = 200

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        bm25_hits = self.bm25.search(query, top_k=self.bm25_top_k)
        dense_hits = self.dense.search(query, top_k=self.dense_top_k)
        return rrf_fuse(bm25_hits, dense_hits, k_rrf=self.k_rrf, top_k=top_k)
