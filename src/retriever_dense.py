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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.utils import get_logger

log = get_logger("dense")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# On Colab, the /content/COMP90042_2026/cache symlink points into the Drive FUSE
# mount. Drive FUSE silently truncates large file writes (>2 GB) when buffered
# data hasn't finished syncing to cloud at session end — observed multiple
# times producing 2.5 GB / 3.1 GB partial .faiss files instead of the full
# 3.7 GB. Local SSD writes are direct disk I/O and don't suffer this. The
# trade-off is the index doesn't survive a session disconnect (must rebuild,
# ~12-15 min on T4), which is far preferable to silent corruption.
COLAB_LOCAL_DENSE_DIR = Path("/content/dense_cache")


def _is_colab() -> bool:
    return "google.colab" in sys.modules


def resolve_dense_paths(drive_index_path: Path, drive_ids_path: Path) -> tuple[Path, Path]:
    """Pick where the dense index physically lives.

    Colab → local SSD (avoids Drive FUSE silently truncating 3.7 GB writes).
    Anywhere else (local Mac, CI, tests) → the path the caller asked for.
    """
    if _is_colab():
        COLAB_LOCAL_DENSE_DIR.mkdir(parents=True, exist_ok=True)
        return (
            COLAB_LOCAL_DENSE_DIR / drive_index_path.name,
            COLAB_LOCAL_DENSE_DIR / drive_ids_path.name,
        )
    return drive_index_path, drive_ids_path


# Drive FUSE truncates large (>~2 GB) streaming writes, but sub-1 GB files sync
# reliably. Splitting the index into parts lets us persist it to Drive so a
# session disconnect doesn't force a 40-min rebuild — reassembly is a fast
# local read.
DENSE_INDEX_CHUNK_SIZE = 900 * 1024 * 1024  # 900 MB
_RESTORE_READ_BLOCK = 64 * 1024 * 1024  # 64 MB streaming block for reassembly


def save_index_to_drive(
    local_index_path: Path,
    local_ids_path: Path,
    drive_index_path: Path,
    drive_ids_path: Path,
    chunk_size: int = DENSE_INDEX_CHUNK_SIZE,
) -> None:
    """Persist a locally-built FAISS index to Drive as verified sub-1 GB parts.

    Writes ``<drive_index>.partNN`` files plus a ``.manifest.json``. The ids
    JSON is small enough to copy whole. The manifest is written LAST, so a
    crash mid-save leaves no manifest and ``restore_index_from_drive`` treats
    the partial save as absent (caller rebuilds).
    """
    drive_dir = drive_index_path.parent
    drive_dir.mkdir(parents=True, exist_ok=True)
    stem = drive_index_path.name
    total_size = local_index_path.stat().st_size

    part_sizes: list[int] = []
    with open(local_index_path, "rb") as f:
        part_idx = 0
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            part_path = drive_dir / f"{stem}.part{part_idx:02d}"
            for attempt in range(3):
                part_path.write_bytes(block)
                if part_path.stat().st_size == len(block):
                    break
                log.warning(
                    "Drive part %s truncated (attempt %d) — retrying",
                    part_path.name, attempt + 1,
                )
            else:
                raise RuntimeError(f"Drive part {part_path} kept truncating after 3 tries")
            part_sizes.append(len(block))
            log.info("  wrote %s (%.0f MB)", part_path.name, len(block) / 1e6)
            part_idx += 1

    drive_ids_path.write_bytes(local_ids_path.read_bytes())
    manifest = {"n_parts": len(part_sizes), "part_sizes": part_sizes, "total_size": total_size}
    (drive_dir / f"{stem}.manifest.json").write_text(json.dumps(manifest))
    log.info("Saved dense index to Drive in %d parts -> %s", len(part_sizes), drive_dir)


def restore_index_from_drive(
    drive_index_path: Path,
    drive_ids_path: Path,
    local_index_path: Path,
    local_ids_path: Path,
) -> bool:
    """Reassemble a chunk-persisted FAISS index from Drive onto local SSD.

    Returns True only if a complete, size-verified set of parts was found and
    reassembled; False otherwise (caller should then build from scratch).
    """
    drive_dir = drive_index_path.parent
    stem = drive_index_path.name
    manifest_path = drive_dir / f"{stem}.manifest.json"
    if not manifest_path.exists() or not drive_ids_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        n_parts = int(manifest["n_parts"])
        part_sizes = list(manifest["part_sizes"])
        total_size = int(manifest["total_size"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        log.warning("Drive dense-index manifest unreadable (%s) — will rebuild", e)
        return False

    parts: list[Path] = []
    for i in range(n_parts):
        p = drive_dir / f"{stem}.part{i:02d}"
        if not p.exists() or p.stat().st_size != part_sizes[i]:
            log.warning("Drive part %s missing or wrong size — cannot restore", p.name)
            return False
        parts.append(p)

    local_index_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Restoring dense index from %d Drive parts ...", n_parts)
    with open(local_index_path, "wb") as out:
        for p in parts:
            with open(p, "rb") as pf:
                while True:
                    blk = pf.read(_RESTORE_READ_BLOCK)
                    if not blk:
                        break
                    out.write(blk)
    if local_index_path.stat().st_size != total_size:
        log.warning("Reassembled index size mismatch — discarding, will rebuild")
        local_index_path.unlink()
        return False
    local_ids_path.parent.mkdir(parents=True, exist_ok=True)
    local_ids_path.write_bytes(drive_ids_path.read_bytes())
    log.info("Restored dense index -> %s (%.1f MB)", local_index_path, total_size / 1e6)
    return True


def _atomic_write_index(index: Any, target_path: Path) -> None:
    """Write a FAISS index via tmp + rename so an interrupted write never
    leaves a half-written file at ``target_path``.

    Without this, a Colab disconnect or OOM kill mid-write produces a
    truncated .faiss that fools the .exists() check and crashes downstream
    with ``ret == size`` when read.
    """
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    faiss.write_index(index, str(tmp_path))
    # Verify size > 0 before promoting; an empty file would also be wrong.
    if tmp_path.stat().st_size == 0:
        tmp_path.unlink()
        raise RuntimeError(f"FAISS write produced empty file at {tmp_path}")
    tmp_path.replace(target_path)


def _atomic_write_text(text: str, target_path: Path) -> None:
    """Same idea for small JSON sidecars (progress, ids)."""
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(target_path)


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
            _atomic_write_index(index, index_path)
            _atomic_write_text(
                json.dumps({
                    "next_doc_idx": end, "n_total": n,
                    "first_id": ids[0], "last_id": ids[-1],
                }),
                progress_path,
            )

    assert index is not None, "evidence corpus was empty"
    _atomic_write_index(index, index_path)
    _atomic_write_text(json.dumps(ids), ids_path)
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
