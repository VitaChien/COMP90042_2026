"""Tests for HybridRetriever RRF fusion."""
from __future__ import annotations

import pytest

from src.retriever_hybrid import rrf_fuse


def test_rrf_single_retriever_preserves_order():
    """If only one retriever has results, ranking should match input order."""
    bm25 = [("e-1", 9.0), ("e-2", 8.0), ("e-3", 7.0)]
    dense: list = []
    fused = rrf_fuse(bm25, dense, k_rrf=60, top_k=3)
    assert [eid for eid, _ in fused] == ["e-1", "e-2", "e-3"]


def test_rrf_score_formula():
    """RRF score for an eid at rank 1 in both lists = 2/(60+1) = 2/61."""
    bm25 = [("e-1", 9.0)]
    dense = [("e-1", 0.9)]
    fused = rrf_fuse(bm25, dense, k_rrf=60, top_k=1)
    assert fused[0][0] == "e-1"
    assert fused[0][1] == pytest.approx(2.0 / 61.0)


def test_rrf_dedupe_across_retrievers():
    """An eid appearing in both lists must appear once with summed RRF score."""
    bm25 = [("e-1", 9.0), ("e-2", 8.0)]
    dense = [("e-2", 0.9), ("e-1", 0.8)]
    fused = rrf_fuse(bm25, dense, k_rrf=60, top_k=10)
    eids = [eid for eid, _ in fused]
    assert sorted(eids) == ["e-1", "e-2"]
    # e-1: rank 1 BM25 + rank 2 dense = 1/61 + 1/62
    # e-2: rank 2 BM25 + rank 1 dense = 1/62 + 1/61
    assert fused[0][1] == pytest.approx(1/61 + 1/62)


def test_rrf_eid_only_in_one_list():
    """eid present in only one retriever contributes only its rank term."""
    bm25 = [("e-1", 9.0), ("e-only-bm25", 1.0)]
    dense = [("e-1", 0.9), ("e-only-dense", 0.1)]
    fused = rrf_fuse(bm25, dense, k_rrf=60, top_k=10)
    score_map = dict(fused)
    assert score_map["e-only-bm25"] == pytest.approx(1.0 / 62.0)
    assert score_map["e-only-dense"] == pytest.approx(1.0 / 62.0)
    assert score_map["e-1"] > score_map["e-only-bm25"]


def test_rrf_top_k_clipping():
    """Returned list is clipped to top_k items, ordered by RRF score desc."""
    bm25 = [(f"e-{i}", 100.0 - i) for i in range(5)]
    dense = [(f"e-{i+10}", 1.0 - 0.1 * i) for i in range(5)]
    fused = rrf_fuse(bm25, dense, k_rrf=60, top_k=3)
    assert len(fused) == 3
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_empty_inputs():
    """Both lists empty -> empty result."""
    assert rrf_fuse([], [], k_rrf=60, top_k=10) == []


def test_rrf_ranks_are_one_indexed():
    """Rank 1 (best) must give 1/(k_rrf + 1), not 1/k_rrf."""
    bm25 = [("e-top", 100.0)]
    fused = rrf_fuse(bm25, [], k_rrf=60, top_k=1)
    assert fused[0][1] == pytest.approx(1.0 / 61.0)


def test_hybrid_retriever_search():
    """HybridRetriever calls both retrievers and fuses results."""
    from src.retriever_hybrid import HybridRetriever

    class FakeRet:
        def __init__(self, results):
            self._results = results
        def search(self, query, top_k):
            return self._results[:top_k]

    bm25 = FakeRet([("e-1", 9.0), ("e-2", 8.0)])
    dense = FakeRet([("e-2", 0.9), ("e-3", 0.8)])
    hybrid = HybridRetriever(bm25=bm25, dense=dense, k_rrf=60)
    hits = hybrid.search("any claim", top_k=2)
    assert len(hits) == 2
    assert hits[0][0] == "e-2"  # appears in both -> highest fused RRF score


def test_recall_at_k_with_partial_overlap():
    """recall = |gold ∩ retrieved_topk| / |gold|."""
    from src.retriever_hybrid import recall_at_k

    gold = {"e-1", "e-2", "e-3"}
    retrieved = [("e-1", 9.0), ("e-9", 8.0), ("e-2", 7.0), ("e-8", 6.0)]
    assert recall_at_k(gold, retrieved, k=4) == pytest.approx(2 / 3)
    assert recall_at_k(gold, retrieved, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(set(), retrieved, k=4) == 0.0
