"""Text helpers shared by every detector, so no detector sees a different
rendering of the same row than another one did."""
from __future__ import annotations

import re

# Split after . ! ? when the next sentence starts like one. Abbreviations that
# PubMed prose is full of ("e.g.", "vs.", "et al.") would otherwise cut a claim
# in half, so a split that leaves a fragment under three words is undone.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
_MIN_WORDS = 3


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _BOUNDARY.split(text.strip()) if p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1].split()) < _MIN_WORDS:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    if len(merged) > 1 and len(merged[-1].split()) < _MIN_WORDS:
        tail = merged.pop()
        merged[-1] = f"{merged[-1]} {tail}"
    return merged or [text.strip()]


def render_documents(documents: list[str]) -> str:
    """One string for detectors that take a single premise."""
    if len(documents) == 1:
        return documents[0]
    return "\n\n".join(f"Document {i + 1}:\n{d}" for i, d in enumerate(documents))
