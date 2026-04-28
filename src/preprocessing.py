"""Lightweight BM25-friendly tokenizer (lecture 2 preprocessing concepts)."""

from __future__ import annotations

import re

# minimal stopword list (subset of NLTK english) - kept here to avoid runtime download
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


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase, regex-tokenize, drop stopwords/empties."""
    if not text:
        return []
    return [
        t
        for t in _TOKEN_RE.findall(text.lower())
        if (t not in _STOPWORDS and len(t) > 1) or t.isdigit()
    ]
