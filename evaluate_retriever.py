"""
evaluate_retriever.py — Standalone retriever evaluation for COMP90042.

Measures the retriever's evidence F-score, precision, recall, and MRR at
multiple k values — the same metric eval.py uses for the full pipeline's F.

Usage (from notebook):
    from evaluate_retriever import evaluate_retriever
    evaluate_retriever(hybrid_retriever, dev_data, top_k_list=[5, 10, 20, 30])

Usage (CLI on Colab/local):
    python evaluate_retriever.py \
        --dev dev-claims.json \
        --evidence evidence.json \
        --bm25_cache bm25_index \
        --top_k 5 10 20 30
"""

import json
import os
import argparse
from statistics import mean


def evaluate_retriever(retriever, dev_data, top_k_list=None, cache_path=None):
    """Measure retriever-only evidence P/R/F/MRR at multiple k values.

    This exactly mirrors eval.py's evidence F-score computation, but applied
    only to the retriever output (ignoring the classifier label).

    Args:
        retriever:    any object with a .retrieve(claim_text, top_k=k) method
                      returning a list of dicts with "id" keys.
        dev_data:     dict of {claim_id: {"claim_text": ..., "evidences": [...], ...}}
        top_k_list:   list of k values to evaluate (default: [5, 10, 20, 30])
        cache_path:   optional path to a JSON file for per-claim checkpointing.
                      Saves after every claim; resumes automatically if the file exists.
                      Pass a unique name per retriever config, e.g. "eval_cache_hybrid.json".

    Returns:
        dict: {k: {"precision": float, "recall": float, "f": float, "mrr": float}}
    """
    if top_k_list is None:
        top_k_list = [5, 10, 20, 30]

    max_k = max(top_k_list)

    # Load existing checkpoint if available
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as _f:
            cache = json.load(_f)
        print(f"  Resuming from checkpoint: {len(cache)} / {len(dev_data)} claims already done.")

    per_k      = {k: {"p": [], "r": [], "f": [], "mrr": []} for k in top_k_list}
    n_has_gold = 0

    claims_to_run = {cid: item for cid, item in dev_data.items() if cid not in cache}
    print(f"  Retrieving top-{max_k} for {len(claims_to_run)} claims "
          f"({'truncating per k' if len(top_k_list) > 1 else f'k={max_k}'}) ...")

    for claim_id, item in dev_data.items():
        gold = set(item.get("evidences", []))
        if not gold:
            continue
        n_has_gold += 1

        if claim_id in cache:
            retrieved_ids_by_k = {int(k): v for k, v in cache[claim_id].items()}
        else:
            retrieved_all = retriever.retrieve(item["claim_text"], top_k=max_k)
            retrieved_ids_by_k = {k: [r["id"] for r in retrieved_all[:k]] for k in top_k_list}
            if cache_path:
                cache[claim_id] = {str(k): ids for k, ids in retrieved_ids_by_k.items()}
                with open(cache_path, "w") as _f:
                    json.dump(cache, _f)

        for k in top_k_list:
            retrieved_ids = retrieved_ids_by_k[k]
            retrieved_set = set(retrieved_ids)

            if retrieved_set and gold:
                p = len(retrieved_set & gold) / len(retrieved_set)
                r = len(retrieved_set & gold) / len(gold)
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            else:
                p = r = f = 0.0

            mrr = 0.0
            for rank, rid in enumerate(retrieved_ids, start=1):
                if rid in gold:
                    mrr = 1.0 / rank
                    break

            per_k[k]["p"].append(p)
            per_k[k]["r"].append(r)
            per_k[k]["f"].append(f)
            per_k[k]["mrr"].append(mrr)

    results = {}
    print(f"{'k':>4}  {'Precision':>10}  {'Recall':>8}  {'F-score':>8}  {'MRR':>8}  {'Coverage':>9}")
    print("-" * 60)
    for k in top_k_list:
        stats    = per_k[k]
        coverage = sum(1 for f in stats["f"] if f > 0) / n_has_gold if n_has_gold else 0.0
        avg_p    = mean(stats["p"]) if stats["p"] else 0.0
        avg_r    = mean(stats["r"]) if stats["r"] else 0.0
        avg_f    = mean(stats["f"]) if stats["f"] else 0.0
        avg_mrr  = mean(stats["mrr"]) if stats["mrr"] else 0.0

        print(f"{k:>4}  {avg_p:>10.3f}  {avg_r:>8.3f}  {avg_f:>8.3f}  {avg_mrr:>8.3f}  {coverage:>8.1%}")
        results[k] = {"precision": avg_p, "recall": avg_r, "f": avg_f, "mrr": avg_mrr, "coverage": coverage}

    print()
    print("  F-score = eval.py's evidence F component.")
    print("  Coverage = fraction of claims where at least one gold passage was retrieved.")
    return results


def evaluate_retriever_by_label(retriever, dev_data, top_k=10):
    """Break down retriever F@k by claim label. Useful for debugging label-specific failures."""
    from collections import defaultdict
    label_stats = defaultdict(lambda: {"p": [], "r": [], "f": []})

    for item in dev_data.values():
        gold  = set(item.get("evidences", []))
        label = item.get("claim_label", "UNKNOWN")
        if not gold:
            continue
        retrieved = {r["id"] for r in retriever.retrieve(item["claim_text"], top_k=top_k)}
        if retrieved and gold:
            p = len(retrieved & gold) / len(retrieved)
            r = len(retrieved & gold) / len(gold)
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        else:
            p = r = f = 0.0
        label_stats[label]["p"].append(p)
        label_stats[label]["r"].append(r)
        label_stats[label]["f"].append(f)

    print(f"\nRetriever F@{top_k} by label:")
    print(f"{'Label':>16}  {'P':>6}  {'R':>6}  {'F':>6}  {'N':>5}")
    print("-" * 45)
    for lbl, stats in sorted(label_stats.items()):
        print(f"{lbl:>16}  {mean(stats['p']):>6.3f}  {mean(stats['r']):>6.3f}  {mean(stats['f']):>6.3f}  {len(stats['f']):>5}")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from retriever import BM25Retriever, DenseRetriever, HybridRetriever, load_bm25_index

    parser = argparse.ArgumentParser(description="Evaluate retriever evidence F-score")
    parser.add_argument("--dev",         default="dev-claims.json")
    parser.add_argument("--evidence",    default="evidence.json")
    parser.add_argument("--bm25_cache",  default="bm25_index")
    parser.add_argument("--dense_cache", default="dense_embeddings.npy", help="pass '' to skip dense")
    parser.add_argument("--ce_model",    default="", help="CrossEncoder path/name; empty = BM25-only")
    parser.add_argument("--top_k",       nargs="+", type=int, default=[5, 10, 20, 30])
    parser.add_argument("--by_label",    action="store_true")
    args = parser.parse_args()

    print("Loading evidence corpus ...")
    with open(args.evidence) as f:
        evidence_dict = json.load(f)

    print("Loading dev data ...")
    with open(args.dev) as f:
        dev_data = json.load(f)

    print("Loading BM25 index ...")
    bm25 = load_bm25_index(args.bm25_cache, evidence_dict)

    dense = None
    if args.dense_cache:
        print(f"Loading DenseRetriever (cache={args.dense_cache}) ...")
        dense = DenseRetriever(evidence_dict)
        dense.build_index(cache_path=args.dense_cache if os.path.exists(args.dense_cache) else None)

    hybrid = HybridRetriever(evidence_dict, bm25, dense_retriever=dense)
    if args.ce_model:
        import torch
        from sentence_transformers import CrossEncoder
        dev_ce = "cuda" if torch.cuda.is_available() else "cpu"
        hybrid.cross_encoder = CrossEncoder(args.ce_model, device=dev_ce)
        print(f"CrossEncoder loaded: {args.ce_model} on {dev_ce}")
    else:
        print("No CrossEncoder specified — evaluating BM25+Dense only (no re-ranking).")

    print(f"\nEvaluating on {len(dev_data)} dev claims ...")
    evaluate_retriever(hybrid, dev_data, top_k_list=args.top_k)

    if args.by_label:
        evaluate_retriever_by_label(hybrid, dev_data, top_k=args.top_k[-1])
