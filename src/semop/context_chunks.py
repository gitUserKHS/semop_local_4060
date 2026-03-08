from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source: str = "context"


def split_context_into_chunks(text: str) -> List[TextChunk]:
    raw = text.strip()
    if not raw:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", raw) if piece.strip()]

    chunks: List[TextChunk] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        chunk_id = f"ctx_{index:03d}"
        chunks.append(TextChunk(chunk_id=chunk_id, text=paragraph))
    return chunks
