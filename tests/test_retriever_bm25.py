from pathlib import Path

import pytest

from src.retriever_bm25 import BM25Retriever, build_bm25_index


@pytest.fixture(scope="module")
def toy_corpus() -> dict[str, str]:
    return {
        "evidence-0": "CO2 emissions cause global warming",
        "evidence-1": "Coral reefs depend on stable ocean temperatures",
        "evidence-2": "Wind turbines reduce greenhouse gas",
        "evidence-3": "Boston is a city in Massachusetts",
    }


def test_build_bm25_index_persists_and_reloads(tmp_path: Path, toy_corpus):
    cache = tmp_path / "bm25_index"
    build_bm25_index(toy_corpus, cache_path=cache)
    assert cache.is_dir()
    r = BM25Retriever.from_cache(cache)
    assert len(r.evidence_ids) == 4


def test_bm25_retrieves_relevant_first(tmp_path, toy_corpus):
    cache = tmp_path / "bm25_index"
    build_bm25_index(toy_corpus, cache_path=cache)
    r = BM25Retriever.from_cache(cache)
    hits = r.search("greenhouse gas emissions cause warming", top_k=2)
    assert hits[0][0] in {"evidence-0", "evidence-2"}
    assert all(score > 0 for _, score in hits)


def test_bm25_returns_at_most_top_k(tmp_path, toy_corpus):
    cache = tmp_path / "bm25_index"
    build_bm25_index(toy_corpus, cache_path=cache)
    r = BM25Retriever.from_cache(cache)
    assert len(r.search("warming", top_k=2)) == 2


def test_bm25_top_k_clipped_to_corpus_size(tmp_path, toy_corpus):
    """Asking for more than corpus_size results must not crash."""
    cache = tmp_path / "bm25_index"
    build_bm25_index(toy_corpus, cache_path=cache)
    r = BM25Retriever.from_cache(cache)
    assert len(r.search("warming", top_k=999)) <= 4


def test_bm25_empty_query_returns_empty_list(tmp_path, toy_corpus):
    """All-stopword queries tokenize to nothing; must not error."""
    cache = tmp_path / "bm25_index"
    build_bm25_index(toy_corpus, cache_path=cache)
    r = BM25Retriever.from_cache(cache)
    assert r.search("the of a", top_k=2) == []
    assert r.search("", top_k=2) == []


def test_bm25_search_batch_preserves_order_and_handles_empty(tmp_path, toy_corpus):
    """search_batch must return one list per query in input order, with []
    for queries that tokenize to nothing."""
    cache = tmp_path / "bm25_index"
    build_bm25_index(toy_corpus, cache_path=cache)
    r = BM25Retriever.from_cache(cache)
    queries = ["greenhouse gas", "of the a", "ocean temperatures"]
    out = r.search_batch(queries, top_k=2)
    assert len(out) == 3
    assert out[1] == []  # all-stopword query
    assert out[0][0][0] in {"evidence-0", "evidence-2"}
    assert out[2][0][0] == "evidence-1"
