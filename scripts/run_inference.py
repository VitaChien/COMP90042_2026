"""Two-stage retrieval / inference entry point.

Modes:
- ``bm25-random``     : BM25 top-K only, label is random (M1 sanity baseline).
- ``retriever-only``  : BM25 top-200 -> cross-encoder rerank top-K, label random
                        (M2 retrieval-quality measurement, isolates label noise).

Phase 4 will add ``full`` and ``oracle`` modes.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import Config
from src.data_loader import load_claims, load_evidence
from src.evaluator import evaluate_predictions
from src.retriever_bm25 import BM25Retriever
from src.retriever_cross_enc import load_cross_encoder, rerank
from src.utils import get_logger, save_json, set_seed, timer

log = get_logger("infer")


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_baseline(claims_path: Path, output_path: Path, top_k: int) -> dict:
    """BM25-only retrieval + random label. The M1 baseline."""
    cfg = Config()
    set_seed(cfg.seed)
    bm25 = BM25Retriever.from_cache(cfg.cache_dir / "bm25_index")
    claims = load_claims(claims_path)

    preds: dict[str, dict] = {}
    with timer(f"BM25 retrieval x{len(claims)}", log):
        for cid, claim in claims.items():
            hits = bm25.search(claim.claim_text, top_k=top_k)
            preds[cid] = {
                "claim_text": claim.claim_text,
                "claim_label": random.choice(cfg.label_names),
                "evidences": [eid for eid, _ in hits],
            }
    save_json(preds, output_path)
    log.info("Saved %d predictions -> %s", len(preds), output_path)
    return preds


def run_retriever_only(
    claims_path: Path, output_path: Path, top_k: int, bm25_top_k: int | None = None
) -> dict:
    """BM25 top-200 -> cross-encoder rerank top-K. Label still random.

    Random label keeps A constant (~0.25) so any movement in F or HM
    relative to the bm25-random baseline is attributable to the cross-encoder
    re-ranker alone.
    """
    cfg = Config()
    set_seed(cfg.seed)
    device = _pick_device()
    log.info("Reranking on device: %s", device)

    effective_bm25_top_k = bm25_top_k if bm25_top_k is not None else cfg.bm25_top_k
    log.info("BM25 pool size: %d", effective_bm25_top_k)

    bm25 = BM25Retriever.from_cache(cfg.cache_dir / "bm25_index")
    ce_tok, ce_model = load_cross_encoder(
        cfg.cross_encoder_model,
        cfg.ckpt_dir / "cross_encoder.pt",
        device=device,
    )
    log.info("Loading evidence corpus ...")
    with timer("load_evidence", log):
        evidence = load_evidence(cfg.evidence_path)
    claims = load_claims(claims_path)

    preds: dict[str, dict] = {}
    with timer(f"Retriever pipeline x{len(claims)}", log):
        for cid, claim in tqdm(claims.items(), desc="rerank"):
            cand = bm25.search(claim.claim_text, top_k=effective_bm25_top_k)
            ranked = rerank(
                ce_model,
                ce_tok,
                claim.claim_text,
                cand,
                evidence,
                top_k=top_k,
                batch_size=64,
                device=device,
                max_len=cfg.ce_max_len,
            )
            preds[cid] = {
                "claim_text": claim.claim_text,
                "claim_label": random.choice(cfg.label_names),
                "evidences": [eid for eid, _ in ranked],
            }
    save_json(preds, output_path)
    log.info("Saved %d predictions -> %s", len(preds), output_path)
    return preds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--top-k", type=int, default=Config().final_top_k)
    p.add_argument(
        "--mode",
        choices=["bm25-random", "retriever-only"],
        default="bm25-random",
        help="Phase 4 will add 'full' and 'oracle'.",
    )
    p.add_argument(
        "--bm25-top-k",
        type=int,
        default=None,
        help="BM25 pool size (default: Config.bm25_top_k=200)",
    )
    args = p.parse_args()

    cfg = Config()
    claims_path = cfg.dev_path if args.split == "dev" else cfg.test_path
    bm25_suffix = f"-bm25{args.bm25_top_k}" if args.bm25_top_k is not None else ""
    output_path = cfg.output_dir / f"{args.split}-{args.mode}-k{args.top_k}{bm25_suffix}.json"

    if args.mode == "bm25-random":
        run_baseline(claims_path, output_path, args.top_k)
    elif args.mode == "retriever-only":
        run_retriever_only(claims_path, output_path, args.top_k, bm25_top_k=args.bm25_top_k)

    if args.split == "dev":
        m = evaluate_predictions(output_path, cfg.dev_path)
        log.info(
            "F=%.4f  A=%.4f  HM=%.4f",
            m["evidence_f"],
            m["claim_accuracy"],
            m["harmonic_mean"],
        )


if __name__ == "__main__":
    main()
