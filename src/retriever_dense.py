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
from dataclasses import dataclass, field
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
    checkpoint_every: int = 50,
) -> None:
    """Encode evidence corpus, build a FAISS IndexFlatIP, persist to disk.

    Streams chunks directly into FAISS to bound peak RAM (np.vstack on 1.2M x
    768 would briefly hold ~7 GB and OOM on Colab T4). Writes a checkpoint
    every ``checkpoint_every`` chunks so a Colab disconnect mid-encode loses
    at most that many chunks of work.

    Resume: if ``<index_path>`` and ``<index_path>.progress.json`` both exist
    with a matching ``n_total``, load the partial index and continue from
    ``next_doc_idx``. On normal completion the progress file is deleted, so a
    subsequent ``mod.main()`` sees a finished index and skips.
    """
    index_path = Path(index_path)
    ids_path = Path(ids_path)
    progress_path = index_path.with_suffix(".progress.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    ids = list(evidence.keys())
    texts = [evidence[i] for i in ids]
    n = len(ids)

    index: faiss.Index | None = None
    start_idx = 0
    if index_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        # Identity guard: n_total alone is just a count and would silently accept
        # a same-length but mutated corpus. Pin to first+last evidence id too so
        # any rename / re-order / mutation forces a fresh build.
        matches = (
            progress.get("n_total") == n
            and progress.get("first_id") == ids[0]
            and progress.get("last_id") == ids[-1]
        )
        if matches:
            start_idx = int(progress["next_doc_idx"])
            index = faiss.read_index(str(index_path))
            log.info(
                "Resuming from checkpoint: %d / %d docs already indexed", start_idx, n
            )
        else:
            log.warning(
                "Progress file mismatch (n=%s first=%s last=%s vs n=%d first=%s last=%s)"
                " — discarding checkpoint, starting fresh",
                progress.get("n_total"), progress.get("first_id"), progress.get("last_id"),
                n, ids[0], ids[-1],
            )
            # Remove the stale marker so a fresh interrupt doesn't churn through
            # the same mismatch on every restart.
            progress_path.unlink()

    log.info("Encoding %d remaining evidences for dense retrieval ...", n - start_idx)
    chunk_size = batch_size * 32
    chunks_done_this_session = 0
    for start in range(start_idx, n, chunk_size):
        end = min(start + chunk_size, n)
        emb = encoder.encode(
            texts[start:end],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        if index is None:
            dim = emb.shape[1]
            log.info("Building FAISS IndexFlatIP (dim=%d) ...", dim)
            index = faiss.IndexFlatIP(dim)
        index.add(emb)
        log.info("  encoded + indexed %d / %d", end, n)
        del emb
        chunks_done_this_session += 1
        # Periodic checkpoint (skip the final partial save — finalisation below handles it).
        if chunks_done_this_session % checkpoint_every == 0 and end < n:
            log.info("Checkpointing at doc %d / %d ...", end, n)
            faiss.write_index(index, str(index_path))
            progress_path.write_text(json.dumps({
                "next_doc_idx": end, "n_total": n,
                "first_id": ids[0], "last_id": ids[-1],
            }))

    assert index is not None, "evidence corpus was empty"
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(ids))
    # Mark completion by removing the progress file so future runs see "done".
    if progress_path.exists():
        progress_path.unlink()
    log.info("Saved dense index -> %s (%.1f MB)", index_path, index_path.stat().st_size / 1e6)


@dataclass
class DenseRetriever:
    index_path: Path
    ids_path: Path
    encoder: Any
    query_prefix: str = BGE_QUERY_PREFIX
    _index: Any = field(default=None, init=False, repr=False)
    _ids: list[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = faiss.read_index(str(self.index_path))
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
