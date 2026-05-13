"""Tests for DenseRetriever (BGE-base + FAISS)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")


def _build_toy_index(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny FAISS index with 4 deterministic 8-dim vectors."""
    vecs = np.array(
        [
            [1.0, 0, 0, 0, 0, 0, 0, 0],
            [0.0, 1.0, 0, 0, 0, 0, 0, 0],
            [0.7, 0.7, 0, 0, 0, 0, 0, 0],
            [0.0, 0.0, 1.0, 0, 0, 0, 0, 0],
        ],
        dtype="float32",
    )
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    index_path = tmp_path / "toy.faiss"
    ids_path = tmp_path / "toy.ids.json"
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(["evidence-A", "evidence-B", "evidence-C", "evidence-D"]))
    return index_path, ids_path


class _StubEncoder:
    """Stand-in for SentenceTransformer.encode used in unit tests."""
    def __init__(self, vec: np.ndarray):
        self._vec = vec.astype("float32")
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        out = np.tile(self._vec, (len(texts), 1))
        if normalize_embeddings:
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out


def test_dense_retriever_returns_top_k_by_inner_product(tmp_path):
    from src.retriever_dense import DenseRetriever

    index_path, ids_path = _build_toy_index(tmp_path)
    encoder = _StubEncoder(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    ret = DenseRetriever(
        index_path=index_path,
        ids_path=ids_path,
        encoder=encoder,
        query_prefix="",
    )
    hits = ret.search("anything", top_k=2)
    assert hits[0][0] == "evidence-A"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert hits[1][0] == "evidence-C"


def test_dense_retriever_applies_query_prefix(tmp_path):
    """The BGE retrieval prefix must be prepended to the query before encoding."""
    from src.retriever_dense import DenseRetriever

    index_path, ids_path = _build_toy_index(tmp_path)

    captured = {}

    class CapturingEncoder(_StubEncoder):
        def encode(self, texts, **kwargs):
            captured["texts"] = list(texts)
            return super().encode(texts, **kwargs)

    encoder = CapturingEncoder(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    ret = DenseRetriever(
        index_path=index_path,
        ids_path=ids_path,
        encoder=encoder,
        query_prefix="QPREFIX: ",
    )
    ret.search("how cold is space", top_k=1)
    assert captured["texts"] == ["QPREFIX: how cold is space"]


def test_dense_retriever_top_k_clipped_to_index_size(tmp_path):
    from src.retriever_dense import DenseRetriever

    index_path, ids_path = _build_toy_index(tmp_path)
    encoder = _StubEncoder(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    ret = DenseRetriever(
        index_path=index_path,
        ids_path=ids_path,
        encoder=encoder,
        query_prefix="",
    )
    hits = ret.search("q", top_k=999)
    assert len(hits) <= 4


def test_build_dense_index_smoke(tmp_path, monkeypatch):
    """End-to-end: stub encoder + tiny corpus -> readable FAISS index."""
    from src.retriever_dense import DenseRetriever, build_dense_index

    corpus = {
        "evidence-A": "ice melting",
        "evidence-B": "ocean temperature",
        "evidence-C": "carbon emissions",
        "evidence-D": "boston city",
    }
    rng = np.random.default_rng(42)

    class RandomEncoder:
        def encode(self, texts, batch_size=128, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=False):
            out = rng.standard_normal((len(texts), 8)).astype("float32")
            if normalize_embeddings:
                out /= np.linalg.norm(out, axis=1, keepdims=True)
            return out

    index_path = tmp_path / "smoke.faiss"
    ids_path = tmp_path / "smoke.ids.json"
    build_dense_index(corpus, RandomEncoder(), index_path, ids_path, batch_size=2)
    assert index_path.exists()
    assert ids_path.exists()

    # Reload and search
    ret = DenseRetriever.from_cache(index_path, ids_path, RandomEncoder(), query_prefix="")
    hits = ret.search("anything", top_k=4)
    assert len(hits) == 4
    assert {eid for eid, _ in hits} == set(corpus)
