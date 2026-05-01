"""BM25 first-stage retriever over the evidence corpus.

Backed by ``bm25s`` (sparse-matrix BM25 with multi-threaded retrieval, ~100x
faster than ``rank_bm25`` on our 1.2M-doc corpus). Public API is preserved
so existing callers and tests don't change.

On-disk layout: ``cache_path`` is treated as a directory because bm25s
stores multiple shards (``data.csc.index.npy``, ``vocab.index.json`` ...).
The original ``evidence_ids`` ordering is pickled alongside as
``evidence_ids.pkl`` so that bm25s integer-index outputs can be mapped
back to the ``evidence-XXXX`` strings the rest of the project expects.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import bm25s
from tqdm import tqdm

from src.preprocessing import tokenize_for_bm25
from src.utils import get_logger

log = get_logger("bm25")

_EVIDENCE_IDS_FILE = "evidence_ids.pkl"


def build_bm25_index(evidence: dict[str, str], cache_path: Path | str) -> None:
    """Tokenize all evidences and build + save a bm25s index at ``cache_path``."""
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    ids = list(evidence.keys())
    log.info("Tokenizing %d evidences for BM25 ...", len(ids))
    corpus_tokens = [tokenize_for_bm25(evidence[i]) for i in tqdm(ids, desc="tokenize")]
    log.info("Building bm25s index ...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    retriever.save(str(cache_path), show_progress=False)
    with (cache_path / _EVIDENCE_IDS_FILE).open("wb") as f:
        pickle.dump(ids, f)
    n_files = sum(1 for _ in cache_path.iterdir())
    total_size = sum(p.stat().st_size for p in cache_path.iterdir())
    log.info(
        "Saved BM25 index -> %s (%.1f MB across %d files)",
        cache_path,
        total_size / 1e6,
        n_files,
    )


@dataclass
class BM25Retriever:
    bm25: bm25s.BM25
    evidence_ids: list[str]

    @classmethod
    def from_cache(cls, cache_path: Path | str) -> BM25Retriever:
        cache_path = Path(cache_path)
        bm25 = bm25s.BM25.load(str(cache_path), load_corpus=False)
        with (cache_path / _EVIDENCE_IDS_FILE).open("rb") as f:
            ids = pickle.load(f)
        return cls(bm25=bm25, evidence_ids=ids)

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []
        k = min(top_k, len(self.evidence_ids))
        results, scores = self.bm25.retrieve([tokens], k=k, show_progress=False)
        return [
            (self.evidence_ids[int(idx)], float(score))
            for idx, score in zip(results[0], scores[0], strict=True)
        ]

    def search_batch(
        self,
        queries: Iterable[str],
        top_k: int = 200,
        show_progress: bool = False,
    ) -> list[list[tuple[str, float]]]:
        """Tokenize and retrieve for many queries in one bm25s call.

        Returns one list per input query; queries that tokenize to nothing
        get an empty result list (preserving input ordering).
        """
        token_lists = [tokenize_for_bm25(q) for q in queries]
        non_empty_idx = [i for i, t in enumerate(token_lists) if t]
        if not non_empty_idx:
            return [[] for _ in token_lists]
        non_empty_tokens = [token_lists[i] for i in non_empty_idx]
        k = min(top_k, len(self.evidence_ids))
        results, scores = self.bm25.retrieve(
            non_empty_tokens,
            k=k,
            show_progress=show_progress,
        )
        out: list[list[tuple[str, float]]] = [[] for _ in token_lists]
        for j, i in enumerate(non_empty_idx):
            out[i] = [
                (self.evidence_ids[int(idx)], float(score))
                for idx, score in zip(results[j], scores[j], strict=True)
            ]
        return out
