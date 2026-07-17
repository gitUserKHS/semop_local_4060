from __future__ import annotations

import re


def split_statements(text: str) -> tuple[str, ...]:
    return tuple(
        statement.strip()
        for statement in re.split(r"[\n;]+|(?<=[.!?。！？])\s*", text)
        if statement.strip()
    )


def strip_sentence_punctuation(value: str) -> str:
    return value.strip().strip(" \t\r\n.!?。！？")


def split_items(value: str) -> tuple[str, ...]:
    parts = re.split(
        r"\s*(?:,|，|\band also\b|\band\b)\s*|\s+및\s+|(?:와|과)\s+",
        value,
        flags=re.IGNORECASE,
    )
    return tuple(
        normalized
        for item in parts
        if (normalized := normalize_identifier(item))
    )


def normalize_identifier(value: str) -> str:
    normalized = strip_sentence_punctuation(value).strip("'\"")
    normalized = re.sub(r"^(?:to|the)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", "_", normalized.strip().lower())
    normalized = re.sub(r"[^0-9a-z_가-힣-]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_-")
