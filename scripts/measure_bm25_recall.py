"""Measure BM25 retrieval recall@K on dev claims.

Recall@K = average over claims of (|gold ∩ top_K| / |gold|).
This is the right metric for first-stage retrieval; the official eval reports
F at the final top-K which conflates retrieval and reranking.
"""

from __future__ import annotations

import argparse

from src.config import Config
from src.data_loader import load_claims
from src.retriever_bm25 import BM25Retriever
from src.utils import get_logger, timer

log = get_logger("bm25-recall")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=200)
    args = p.parse_args()

    cfg = Config()
    bm25 = BM25Retriever.from_cache(cfg.cache_dir / "bm25_index")
    claims = load_claims(cfg.dev_path)

    recalls: list[float] = []
    with timer(f"BM25 retrieval x{len(claims)}", log):
        for claim in claims.values():
            gold = set(claim.evidences)
            if not gold:
                continue
            hits = bm25.search(claim.claim_text, top_k=args.top_k)
            retrieved = {eid for eid, _ in hits}
            recalls.append(len(gold & retrieved) / len(gold))

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    log.info("Recall@%d on dev (n=%d): %.4f", args.top_k, len(recalls), mean_recall)
    print(f"Recall@{args.top_k} = {mean_recall:.4f}  (n={len(recalls)})")


if __name__ == "__main__":
    main()
