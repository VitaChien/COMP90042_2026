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


def test_build_dense_index_smoke(tmp_path):
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


def test_build_dense_index_resume_from_checkpoint(tmp_path):
    """A partial .faiss + .progress.json should resume, only encoding remaining docs."""
    from src.retriever_dense import build_dense_index

    corpus = {f"evidence-{i:03d}": f"text {i}" for i in range(8)}
    rng = np.random.default_rng(0)

    class CountingEncoder:
        def __init__(self):
            self.encoded = 0

        def encode(self, texts, batch_size=128, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=False):
            self.encoded += len(texts)
            out = rng.standard_normal((len(texts), 8)).astype("float32")
            if normalize_embeddings:
                out /= np.linalg.norm(out, axis=1, keepdims=True)
            return out

    index_path = tmp_path / "resume.faiss"
    ids_path = tmp_path / "resume.ids.json"
    progress_path = index_path.with_suffix(".progress.json")

    # Pre-build a partial index covering the first 4 docs and write a progress marker.
    pre_index = faiss.IndexFlatIP(8)
    pre_emb = rng.standard_normal((4, 8)).astype("float32")
    pre_emb /= np.linalg.norm(pre_emb, axis=1, keepdims=True)
    pre_index.add(pre_emb)
    faiss.write_index(pre_index, str(index_path))
    ids_list = list(corpus.keys())
    progress_path.write_text(json.dumps({
        "next_doc_idx": 4, "n_total": 8,
        "first_id": ids_list[0], "last_id": ids_list[-1],
    }))

    # Resume: should only encode the remaining 4 docs.
    encoder = CountingEncoder()
    build_dense_index(corpus, encoder, index_path, ids_path, batch_size=2)

    assert encoder.encoded == 4, f"resumed run should encode only 4 docs, got {encoder.encoded}"
    assert not progress_path.exists(), "progress marker should be removed on completion"
    assert ids_path.exists()
    # Final index should contain all 8 vectors (4 pre + 4 newly encoded).
    final = faiss.read_index(str(index_path))
    assert final.ntotal == 8, f"expected 8 vectors in resumed index, got {final.ntotal}"
    # ids.json should list all 8 evidence ids in corpus order.
    persisted_ids = json.loads(ids_path.read_text())
    assert persisted_ids == ids_list


def test_build_dense_index_n_total_mismatch_starts_fresh(tmp_path):
    """If progress.json identity fields don't match current corpus, start fresh."""
    from src.retriever_dense import build_dense_index

    rng = np.random.default_rng(0)

    class RE:
        def __init__(self):
            self.encoded = 0
        def encode(self, texts, batch_size=128, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=False):
            self.encoded += len(texts)
            out = rng.standard_normal((len(texts), 8)).astype("float32")
            if normalize_embeddings:
                out /= np.linalg.norm(out, axis=1, keepdims=True)
            return out

    corpus = {f"e-{i:03d}": f"t{i}" for i in range(6)}
    index_path = tmp_path / "mis.faiss"
    ids_path = tmp_path / "mis.ids.json"
    progress_path = index_path.with_suffix(".progress.json")

    # Stale checkpoint claims 999 docs — must be discarded.
    pre_index = faiss.IndexFlatIP(8)
    pre_index.add(rng.standard_normal((3, 8)).astype("float32"))
    faiss.write_index(pre_index, str(index_path))
    progress_path.write_text(json.dumps({
        "next_doc_idx": 3, "n_total": 999, "first_id": "wrong", "last_id": "wrong",
    }))

    encoder = RE()
    build_dense_index(corpus, encoder, index_path, ids_path, batch_size=2)

    # Mismatch should have triggered fresh build over all 6 docs.
    assert encoder.encoded == 6, f"fresh build should encode all 6 docs, got {encoder.encoded}"
    assert faiss.read_index(str(index_path)).ntotal == 6
    assert not progress_path.exists()


def test_build_dense_index_writes_mid_run_checkpoint(tmp_path):
    """With multiple chunks, checkpoint files appear mid-run (between chunks)."""
    from src.retriever_dense import build_dense_index

    # chunk_size = batch_size * 32. batch_size=1 -> chunk_size=32.
    # 100 docs -> 4 chunks. checkpoint_every=2 -> checkpoint fires after chunk 2 (doc 64),
    # then finalisation handles the remaining chunks 3-4 and cleans progress marker.
    rng = np.random.default_rng(0)

    class CheckpointSpy:
        """Encoder that records progress.json contents after each chunk encode."""

        def __init__(self, progress_path):
            self.progress_path = progress_path
            self.snapshots: list[dict] = []
            self.encoded_so_far = 0

        def encode(self, texts, batch_size=128, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=False):
            # snapshot progress state BEFORE encoding this chunk
            if self.progress_path.exists():
                self.snapshots.append(json.loads(self.progress_path.read_text()))
            out = rng.standard_normal((len(texts), 8)).astype("float32")
            if normalize_embeddings:
                out /= np.linalg.norm(out, axis=1, keepdims=True)
            self.encoded_so_far += len(texts)
            return out

    corpus = {f"e-{i:03d}": f"t{i}" for i in range(100)}
    index_path = tmp_path / "ckpt.faiss"
    ids_path = tmp_path / "ckpt.ids.json"
    progress_path = index_path.with_suffix(".progress.json")
    spy = CheckpointSpy(progress_path)
    build_dense_index(corpus, spy, index_path, ids_path, batch_size=1, checkpoint_every=2)

    # The snapshot taken after chunk 2 (before chunk 3 encodes) should show next_doc_idx=64.
    assert any(s.get("next_doc_idx") == 64 and s.get("n_total") == 100 for s in spy.snapshots), \
        f"expected a mid-run checkpoint at doc 64; saw snapshots: {spy.snapshots}"
    # End-of-run state: final index written, progress marker cleaned.
    assert index_path.exists()
    assert ids_path.exists()
    assert not progress_path.exists()
