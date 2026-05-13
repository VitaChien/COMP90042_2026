"""Dense first-stage retriever over the evidence corpus (BGE-base + FAISS).

Build phase (one-shot, ``build_dense_index``):
  1. Encode every evidence text with a SentenceTransformer (no query prefix).
  2. L2-normalize, build ``faiss.IndexFlatIP``.
  3. Persist .faiss + .ids.json (evidence-id ordering aligned with FAISS rows).

Search phase (``DenseRetriever.search``):
  1. Prepend the BGE retrieval query prefix (per model card).
  2. Encode + L2-normalize.
  3. FAISS top-K, map row indices back to evidence ids.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.utils import get_logger

log = get_logger("dense")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_dense_index(
    evidence: dict[str, str],
    encoder: Any,
    index_path: Path | str,
    ids_path: Path | str,
    batch_size: int = 128,
) -> None:
    """Encode evidence corpus, build a FAISS IndexFlatIP, persist to disk.

    ``encoder`` must expose ``.encode(texts, normalize_embeddings=True,
    convert_to_numpy=True)`` (sentence-transformers SentenceTransformer
    matches this contract).
    """
    index_path = Path(index_path)
    ids_path = Path(ids_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    ids = list(evidence.keys())
    texts = [evidence[i] for i in ids]
    n = len(ids)
    log.info("Encoding %d evidences for dense retrieval ...", n)

    chunks: list[np.ndarray] = []
    for start in range(0, n, batch_size * 32):  # 32 batches per chunk
        end = min(start + batch_size * 32, n)
        emb = encoder.encode(
            texts[start:end],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        chunks.append(emb.astype("float32"))
        log.info("  encoded %d / %d", end, n)
    embeddings = np.vstack(chunks)
    dim = embeddings.shape[1]
    log.info("Building FAISS IndexFlatIP (dim=%d, n=%d) ...", dim, n)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(ids))
    log.info("Saved dense index -> %s (%.1f MB)", index_path, index_path.stat().st_size / 1e6)


@dataclass
class DenseRetriever:
    index_path: Path
    ids_path: Path
    encoder: Any
    query_prefix: str = BGE_QUERY_PREFIX
    _index: Any = None
    _ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self._index is None:
            self._index = faiss.read_index(str(self.index_path))
        if self._ids is None:
            self._ids = json.loads(Path(self.ids_path).read_text())

    @classmethod
    def from_cache(
        cls,
        index_path: Path | str,
        ids_path: Path | str,
        encoder: Any,
        query_prefix: str = BGE_QUERY_PREFIX,
    ) -> DenseRetriever:
        return cls(
            index_path=Path(index_path),
            ids_path=Path(ids_path),
            encoder=encoder,
            query_prefix=query_prefix,
        )

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        prefixed = self.query_prefix + query
        emb = self.encoder.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        k = min(top_k, self._index.ntotal)
        scores, idxs = self._index.search(emb, k)
        return [
            (self._ids[int(i)], float(s))
            for i, s in zip(idxs[0], scores[0], strict=True)
            if i >= 0
        ]
