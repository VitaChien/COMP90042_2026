"""BM25 first-stage retriever over the evidence corpus."""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.preprocessing import tokenize_for_bm25
from src.utils import get_logger

log = get_logger("bm25")


def build_bm25_index(evidence: dict[str, str], cache_path: Path | str) -> None:
    """Tokenize all evidences and persist (BM25Okapi, ids) to disk."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(evidence.keys())
    log.info("Tokenizing %d evidences for BM25 ...", len(ids))
    corpus_tokens = [tokenize_for_bm25(evidence[i]) for i in tqdm(ids, desc="tokenize")]
    log.info("Building BM25Okapi index ...")
    bm25 = BM25Okapi(corpus_tokens)
    with cache_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    log.info(
        "Saved BM25 index -> %s (%.1f MB)",
        cache_path,
        cache_path.stat().st_size / 1e6,
    )


@dataclass
class BM25Retriever:
    bm25: BM25Okapi
    evidence_ids: list[str]

    @classmethod
    def from_cache(cls, cache_path: Path | str) -> BM25Retriever:
        with Path(cache_path).open("rb") as f:
            blob = pickle.load(f)
        return cls(bm25=blob["bm25"], evidence_ids=blob["ids"])

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.evidence_ids[int(i)], float(scores[i])) for i in top_idx]

    def search_batch(
        self, queries: Iterable[str], top_k: int = 200
    ) -> list[list[tuple[str, float]]]:
        return [self.search(q, top_k=top_k) for q in queries]
