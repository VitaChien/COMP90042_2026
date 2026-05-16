"""Measure BM25 / Dense / Hybrid recall@K on dev claims.

Validates Gate 1 from the spec: hybrid recall@200 should beat BM25-only by
10+ percentage points before we pay for CE rerank inference. Prints a
3-column table for K in {50, 100, 200, 500}.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.config import Config
from src.data_loader import load_claims
from src.retriever_bm25 import BM25Retriever
from src.retriever_dense import (
    DenseRetriever,
    resolve_dense_paths,
    restore_index_from_drive,
)
from src.retriever_hybrid import HybridRetriever, recall_at_k
from src.utils import get_logger, timer

log = get_logger("hybrid-recall")

K_VALUES = [50, 100, 200, 500]


def _sweep(name: str, retriever, claims, k_values, fetch_k):
    per_k: dict[int, list[float]] = {k: [] for k in k_values}
    for claim in claims.values():
        gold = set(claim.evidences)
        if not gold:
            continue
        hits = retriever.search(claim.claim_text, top_k=fetch_k)
        for k in k_values:
            per_k[k].append(recall_at_k(gold, hits, k))
    return {k: sum(v) / len(v) if v else 0.0 for k, v in per_k.items()}


def main() -> None:
    cfg = Config()
    claims = load_claims(cfg.dev_path)
    log.info("Loaded %d dev claims", len(claims))

    bm25 = BM25Retriever.from_cache(cfg.cache_dir / "bm25_index")
    encoder = SentenceTransformer(cfg.dense_encoder)
    dense_index_path, dense_ids_path = resolve_dense_paths(
        cfg.dense_index_path, cfg.dense_ids_path
    )
    if not dense_index_path.exists():
        # Fresh Colab session — restore from Drive chunks rather than rebuild.
        if dense_index_path != cfg.dense_index_path:
            restore_index_from_drive(
                cfg.dense_index_path, cfg.dense_ids_path, dense_index_path, dense_ids_path
            )
        if not dense_index_path.exists():
            raise FileNotFoundError(
                f"Dense index not found at {dense_index_path} and no Drive backup "
                "to restore. Run cell 1.4 (build dense index) first."
            )
    dense = DenseRetriever.from_cache(dense_index_path, dense_ids_path, encoder)
    hybrid = HybridRetriever(
        bm25=bm25, dense=dense, k_rrf=cfg.rrf_k,
        bm25_top_k=max(K_VALUES), dense_top_k=max(K_VALUES),
    )

    max_k = max(K_VALUES)
    with timer("BM25 sweep", log):
        bm25_r = _sweep("BM25", bm25, claims, K_VALUES, fetch_k=max_k)
    with timer("Dense sweep", log):
        dense_r = _sweep("Dense", dense, claims, K_VALUES, fetch_k=max_k)
    with timer("Hybrid sweep", log):
        hybrid_r = _sweep("Hybrid", hybrid, claims, K_VALUES, fetch_k=max_k)

    print()
    print(f"{'K':>5}  {'BM25':>8}  {'Dense':>8}  {'Hybrid':>8}")
    for k in K_VALUES:
        print(f"{k:>5}  {bm25_r[k]:>8.4f}  {dense_r[k]:>8.4f}  {hybrid_r[k]:>8.4f}")
    print()
    print(f"Gate 1 (hybrid recall@200 >= 0.65): {'PASS' if hybrid_r[200] >= 0.65 else 'FAIL'} "
          f"(got {hybrid_r[200]:.4f})")


if __name__ == "__main__":
    main()
