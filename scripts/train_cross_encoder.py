"""Train BERT cross-encoder on (claim, gold / hard-neg evidence) pairs.

Steps:
1. Load train claims + BM25 index (built in M1).
2. Run BM25 top-50 retrieval over train (cached to JSON for re-use).
3. Build (claim, gold) positives + (claim, BM25 top-N \\ gold) hard negatives.
4. Fine-tune cross-encoder with BCE for ``cfg.ce_epochs`` epochs.
5. Save checkpoint to ``checkpoints/cross_encoder.pt``.
"""

from __future__ import annotations

import torch

from src.config import Config
from src.data_loader import load_claims, load_evidence
from src.hard_negatives import build_training_pairs
from src.retriever_bm25 import BM25Retriever
from src.retriever_cross_enc import build_cross_encoder, train_cross_encoder
from src.utils import get_logger, load_json, save_json, set_seed, timer

log = get_logger("ce-train")


def pick_device() -> str:
    """Prefer CUDA (Colab T4) > MPS (local Mac) > CPU. Defensive helper because
    Plan B was originally spec'd cuda-or-cpu only; MPS support lets us iterate
    locally before submitting to Colab.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    device = pick_device()
    log.info("Training on device: %s", device)

    train_claims = load_claims(cfg.train_path)
    log.info("Loaded %d train claims", len(train_claims))

    bm25 = BM25Retriever.from_cache(cfg.cache_dir / "bm25_index")

    bm25_cache = cfg.cache_dir / "bm25_train_top50.json"
    backend_tag = "bm25s"  # source-of-truth: src/retriever_bm25.py

    bm25_results: dict | None = None
    if bm25_cache.exists():
        cached = load_json(bm25_cache)
        if isinstance(cached, dict) and cached.get("_backend") == backend_tag:
            log.info(
                "Loading cached BM25 train top-50 (backend=%s) from %s", backend_tag, bm25_cache
            )
            bm25_results = cached["results"]
        else:
            log.warning(
                "BM25 cache backend mismatch (got %r, expected %r); rebuilding.",
                cached.get("_backend") if isinstance(cached, dict) else "<legacy>",
                backend_tag,
            )

    if bm25_results is None:
        with timer("BM25 search over train split", log):
            cids = list(train_claims.keys())
            queries = [train_claims[c].claim_text for c in cids]
            hits = bm25.search_batch(queries, top_k=50)
            bm25_results = dict(zip(cids, hits, strict=True))
        save_json({"_backend": backend_tag, "results": bm25_results}, bm25_cache)
        log.info("Cached BM25 train top-50 -> %s", bm25_cache)

    pairs = build_training_pairs(
        train_claims,
        bm25_results,
        n_neg=cfg.hard_negatives_per_pos,
        seed=cfg.seed,
    )
    n_pos = sum(1 for p in pairs if p["label"] == 1)
    n_neg = len(pairs) - n_pos
    log.info("Built %d training pairs (positives=%d, hard_negs=%d)", len(pairs), n_pos, n_neg)

    log.info("Loading evidence corpus into memory (~1.2M items, ~1GB) ...")
    with timer("load_evidence", log):
        evidence = load_evidence(cfg.evidence_path)
    log.info("Loaded %d evidences", len(evidence))

    tok, model = build_cross_encoder(cfg.cross_encoder_model)
    train_cross_encoder(
        model,
        tok,
        pairs,
        evidence,
        max_len=cfg.ce_max_len,
        batch_size=cfg.ce_batch_size,
        lr=cfg.ce_lr,
        epochs=cfg.ce_epochs,
        device=device,
        save_path=cfg.ckpt_dir / "cross_encoder.pt",
    )


if __name__ == "__main__":
    main()
