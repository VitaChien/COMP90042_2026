"""Build the dense FAISS index over the evidence corpus.

One-shot script: encodes data/evidence.json with the configured
SentenceTransformer (default BGE-base-en-v1.5), saves
cache/dense_index_bge.{faiss, ids.json}.

Idempotent: skips if both output files already exist.
"""
from __future__ import annotations

import argparse
import sys

import faiss
from sentence_transformers import SentenceTransformer

from src.config import Config
from src.data_loader import load_evidence
from src.retriever_dense import (
    build_dense_index,
    resolve_dense_paths,
    restore_index_from_drive,
    save_index_to_drive,
)
from src.utils import get_logger, timer

log = get_logger("build-dense")


def _existing_index_is_loadable(index_path) -> bool:
    """Try to read the FAISS header so a truncated/corrupt file forces a rebuild.

    Without this guard, a write interrupted by Colab disconnect / OOM kill leaves
    a partial .faiss on disk that passes the .exists() check but fails downstream
    in load_hybrid_components with a cryptic 'ret == size' error.
    """
    try:
        idx = faiss.read_index(str(index_path))
        return idx.ntotal > 0
    except Exception as e:
        log.warning("Existing dense index failed to load (%s) — rebuilding", e)
        return False


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    if argv is None:
        # Inside Jupyter/IPython, sys.argv contains the kernel launcher's
        # `-f kernel.json` which argparse can't parse. Detect and ignore.
        argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    args = p.parse_args(argv)

    cfg = Config()
    # On Colab, redirect from Drive paths to local SSD (Drive FUSE silently
    # truncates the 3.7 GB write). Other environments keep cfg paths unchanged.
    index_path, ids_path = resolve_dense_paths(cfg.dense_index_path, cfg.dense_ids_path)
    on_colab = index_path != cfg.dense_index_path
    if on_colab:
        log.info("Detected Colab — building to local SSD: %s", index_path.parent)

    # 1. Already on local SSD this session?
    if (
        index_path.exists()
        and ids_path.exists()
        and not args.force
        and _existing_index_is_loadable(index_path)
    ):
        log.info("Dense index already on local SSD %s — skipping", index_path)
        return

    # 2. Persisted on Drive from an earlier session? Restore instead of the
    #    ~40-min rebuild. The index was saved as verified sub-1 GB parts.
    if on_colab and not args.force and restore_index_from_drive(
        cfg.dense_index_path, cfg.dense_ids_path, index_path, ids_path
    ):
        if _existing_index_is_loadable(index_path):
            log.info("Restored dense index from Drive — skipping rebuild")
            return
        log.warning("Restored index failed to load — rebuilding from scratch")

    # 3. Build from scratch.
    log.info("Loading evidence corpus ...")
    with timer("load_evidence", log):
        evidence = load_evidence(cfg.evidence_path)

    log.info("Loading encoder %s ...", cfg.dense_encoder)
    encoder = SentenceTransformer(cfg.dense_encoder)
    # fp16 on GPU: ~1.7x faster encoding, negligible effect on cosine ranking.
    import torch

    if torch.cuda.is_available():
        encoder = encoder.half()
        log.info("fp16 encoding enabled (CUDA detected)")

    with timer("build_dense_index", log):
        build_dense_index(
            evidence,
            encoder,
            index_path=index_path,
            ids_path=ids_path,
            batch_size=args.batch_size,
        )

    # 4. Persist to Drive in chunks so the next session restores instead of
    #    rebuilding. Skipped off-Colab (the index is already at its final path).
    if on_colab:
        log.info("Persisting dense index to Drive in chunks for cross-session reuse ...")
        with timer("save_index_to_drive", log):
            save_index_to_drive(
                index_path, ids_path, cfg.dense_index_path, cfg.dense_ids_path
            )


if __name__ == "__main__":
    main()
