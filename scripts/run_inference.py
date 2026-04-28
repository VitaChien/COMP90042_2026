"""Run BM25-only retrieval + (initially) random-label classification.
This is the M1 sanity baseline to confirm the pipeline produces a valid JSON."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.config import Config
from src.data_loader import load_claims
from src.evaluator import evaluate_predictions
from src.retriever_bm25 import BM25Retriever
from src.utils import get_logger, save_json, set_seed, timer

log = get_logger("infer")


def run_baseline(claims_path: Path, output_path: Path, top_k: int) -> dict:
    cfg = Config()
    set_seed(cfg.seed)
    bm25 = BM25Retriever.from_cache(cfg.cache_dir / "bm25.pkl")
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--top-k", type=int, default=Config().final_top_k)
    p.add_argument(
        "--mode",
        choices=["bm25-random"],
        default="bm25-random",
        help="Phase 4 will add 'full-pipeline'.",
    )
    args = p.parse_args()

    cfg = Config()
    claims_path = cfg.dev_path if args.split == "dev" else cfg.test_path
    output_path = cfg.output_dir / f"{args.split}-{args.mode}-k{args.top_k}.json"
    run_baseline(claims_path, output_path, args.top_k)

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
