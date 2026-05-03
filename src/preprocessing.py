"""BM25-friendly tokenizer: regex split + stopword drop + climate-domain
synonym expansion + Snowball (Porter2) English stemming."""

from __future__ import annotations

import re

import Stemmer

# Minimal stopword list (subset of NLTK english) - kept inline to avoid runtime download.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "will",
        "with",
        "you",
        "your",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Climate-domain abbreviations -> tokens to add alongside (NOT replace).
# Keys are post-lowercase tokens; values are extra tokens injected before stemming.
# Data justification (counts in evidence.json):
#   "CO2": 472 docs, "carbon dioxide": 967 docs, overlap only 42.
#   Same asymmetry holds for CH4/methane, N2O/nitrous oxide, H2O/water, GHG/greenhouse gas.
SYNONYM_MAP: dict[str, tuple[str, ...]] = {
    "co2": ("carbon", "dioxide"),
    "ch4": ("methane",),
    "n2o": ("nitrous", "oxide"),
    "h2o": ("water",),
    "ghg": ("greenhouse", "gas"),
}

# PyStemmer Stemmer instances are not documented as thread-safe; we only use
# the tokenizer from the main thread (bm25s indexing is single-threaded
# in build_bm25_index, search is per-call), so a module singleton is fine.
_STEMMER = Stemmer.Stemmer("english")


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase, regex-tokenize, drop stopwords/single-char noise (digits kept),
    expand domain abbreviations, then Snowball-stem."""
    if not text:
        return []
    raw = _TOKEN_RE.findall(text.lower())
    expanded: list[str] = []
    for t in raw:
        if t in _STOPWORDS:
            continue
        if len(t) < 2 and not t.isdigit():
            continue
        expanded.append(t)
        if t in SYNONYM_MAP:
            expanded.extend(SYNONYM_MAP[t])
    # Stem in batch — PyStemmer's stemWords is faster than per-token calls.
    # Pure-digit tokens stem to themselves under Snowball, so years are preserved.
    return _STEMMER.stemWords(expanded)
