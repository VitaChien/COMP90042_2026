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
from src.retriever_dense import build_dense_index, resolve_dense_paths
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
    if index_path != cfg.dense_index_path:
        log.info("Detected Colab — building to local SSD: %s", index_path.parent)
        log.info("(Drive FUSE truncates >2 GB writes; rebuilding each session is safer.)")

    if (
        index_path.exists()
        and ids_path.exists()
        and not args.force
        and _existing_index_is_loadable(index_path)
    ):
        log.info("Dense index already exists at %s — skipping (use --force to rebuild)", index_path)
        return

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


if __name__ == "__main__":
    main()
