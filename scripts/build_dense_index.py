"""Build the dense FAISS index over the evidence corpus.

One-shot script: encodes data/evidence.json with the configured
SentenceTransformer (default BGE-base-en-v1.5), saves
cache/dense_index_bge.{faiss, ids.json}.

Idempotent: skips if both output files already exist.
"""
from __future__ import annotations

import argparse
import sys

from sentence_transformers import SentenceTransformer

from src.config import Config
from src.data_loader import load_evidence
from src.retriever_dense import build_dense_index
from src.utils import get_logger, timer

log = get_logger("build-dense")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    if argv is None:
        # Inside Jupyter/IPython, sys.argv contains the kernel launcher's
        # `-f kernel.json` which argparse can't parse. Detect and ignore.
        argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    args = p.parse_args(argv)

    cfg = Config()
    if cfg.dense_index_path.exists() and cfg.dense_ids_path.exists() and not args.force:
        log.info("Dense index already exists at %s — skipping (use --force to rebuild)", cfg.dense_index_path)
        return

    log.info("Loading evidence corpus ...")
    with timer("load_evidence", log):
        evidence = load_evidence(cfg.evidence_path)

    log.info("Loading encoder %s ...", cfg.dense_encoder)
    encoder = SentenceTransformer(cfg.dense_encoder)

    with timer("build_dense_index", log):
        build_dense_index(
            evidence,
            encoder,
            index_path=cfg.dense_index_path,
            ids_path=cfg.dense_ids_path,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
