"""Mine hard negatives for cross-encoder training: BM25 top-N \\ gold."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any


def _claim_field(claim: Any, key: str) -> Any:
    """Read a field from either a `Claim` dataclass or a raw dict."""
    if isinstance(claim, dict):
        return claim[key]
    return getattr(claim, key)


def build_training_pairs(
    claims: dict[str, Any],
    bm25_results: dict[str, Sequence[tuple[str, float]]],
    n_neg: int = 4,
    seed: int = 42,
) -> list[dict]:
    """Build (claim, evidence, label) training pairs for the cross-encoder.

    For every gold evidence we emit one positive pair (label=1) and up to
    ``n_neg`` hard negatives (label=0) sampled uniformly from the BM25
    top-N candidates *minus* the gold set. Hard negatives are lexically
    similar to the claim (so they survived BM25) but are known to be
    irrelevant - the exact failure mode the cross-encoder must learn to
    fix at re-rank time.

    Returns a flat list of dicts ``{claim_id, claim_text, evidence_id, label}``.
    Both ``Claim`` dataclasses and raw dicts (``{"claim_text", "evidences"}``)
    are accepted as ``claims`` values so the same helper works in tests
    and from `data_loader.load_claims`.
    """
    rng = random.Random(seed)
    pairs: list[dict] = []
    for cid, claim in claims.items():
        ctext = _claim_field(claim, "claim_text")
        gold_list = _claim_field(claim, "evidences")
        gold_set = set(gold_list)
        bm_hits = bm25_results.get(cid, [])
        candidates = [eid for eid, _ in bm_hits if eid not in gold_set]

        # Sample ALL negatives for this claim in one shot so each candidate
        # appears at most once. Cap at n_neg * len(gold_list) total slots.
        target_n_neg = n_neg * len(gold_list)
        sampled_negs = (
            rng.sample(candidates, k=min(target_n_neg, len(candidates))) if candidates else []
        )

        for ge in gold_list:
            pairs.append({"claim_id": cid, "claim_text": ctext, "evidence_id": ge, "label": 1})
        for ne in sampled_negs:
            pairs.append({"claim_id": cid, "claim_text": ctext, "evidence_id": ne, "label": 0})
    return pairs
