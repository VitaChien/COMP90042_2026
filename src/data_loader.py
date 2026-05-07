"""Load claim and evidence files into typed objects."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from src.utils import load_json

EVIDENCE_KEY_PATTERN = re.compile(r"^evidence-\d+$")


@dataclass
class Claim:
    claim_id: str
    claim_text: str
    claim_label: str | None = None
    evidences: list[str] = field(default_factory=list)


def load_claims(path: Path | str) -> dict[str, Claim]:
    """Load a {train,dev,test}-claims JSON into a dict of Claim objects."""
    raw = load_json(path)
    out: dict[str, Claim] = {}
    for cid, body in raw.items():
        out[cid] = Claim(
            claim_id=cid,
            claim_text=body["claim_text"],
            claim_label=body.get("claim_label"),
            evidences=list(body.get("evidences", [])),
        )
    return out


def load_evidence(path: Path | str) -> dict[str, str]:
    """Load full evidence corpus into memory (~1.2M items, ~1GB resident)."""
    return load_json(path)


def load_evidence_streaming(path: Path | str) -> Iterator[tuple[str, str]]:
    """Yield (evidence_id, text) one at a time.

    Useful for very low-memory hosts. JSON file is a single object so we
    still parse fully, but expose a generator API to keep call sites
    streaming-style. Swap body for ijson if RAM becomes a problem.
    """
    data = load_json(path)
    yield from data.items()
