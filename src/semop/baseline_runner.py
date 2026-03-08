from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Protocol

from .context_chunks import split_context_into_chunks
from .rag_baseline import PlainRagBaseline, RagBaselineResult


class BaselineProtocol(Protocol):
    def answer(self, query: str, context: str, domain: str = "general") -> RagBaselineResult: ...


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    description: str


class FirstChunkBaseline:
    def answer(self, query: str, context: str, domain: str = "general") -> RagBaselineResult:
        chunks = split_context_into_chunks(context)
        if not chunks:
            return RagBaselineResult(
                query=query,
                domain=domain,
                answer_text="No context was provided, so the first-chunk baseline cannot answer.",
                retrieved_chunks=[],
                confidence=0.0,
                warnings=["no context provided"],
            )
        top = chunks[0]
        return RagBaselineResult(
            query=query,
            domain=domain,
            answer_text=f"First-chunk baseline answer from {top.chunk_id}: {top.text}",
            retrieved_chunks=[top],
            confidence=0.3,
            warnings=["first chunk only"],
        )


class ConfigurableKeywordBaseline:
    def __init__(self, config_path: str | Path):
        path = Path(config_path)
        with path.open("r", encoding="utf-8-sig") as handle:
            self.config = json.load(handle)
        self.name = self.config.get("name", path.stem)
        self.top_k = int(self.config.get("top_k", 2))
        self.keyword_bonus = self.config.get("keyword_bonus", {})
        self.required_any = [token.lower() for token in self.config.get("required_any", [])]
        self.preferred_order = [token.lower() for token in self.config.get("preferred_order", [])]

    def answer(self, query: str, context: str, domain: str = "general") -> RagBaselineResult:
        chunks = split_context_into_chunks(context)
        if not chunks:
            return RagBaselineResult(
                query=query,
                domain=domain,
                answer_text="No context was provided, so the configured baseline cannot answer.",
                retrieved_chunks=[],
                confidence=0.0,
                warnings=["no context provided"],
            )
        scored = [(chunk, self._score(query, chunk.text, index)) for index, chunk in enumerate(chunks)]
        if self.required_any:
            scored = [item for item in scored if any(token in item[0].text.lower() for token in self.required_any)] or scored
        scored.sort(key=lambda item: (-item[1], len(item[0].text)))
        selected = [chunk for chunk, score in scored[: self.top_k] if score > 0]
        if not selected:
            selected = [chunks[0]]
        joined = " ".join(chunk.text for chunk in selected)
        warnings: List[str] = []
        if self.required_any and not any(token in joined.lower() for token in self.required_any):
            warnings.append("configured required_any terms were not matched")
        confidence = min(0.88, round(0.32 + 0.07 * max(1.0, scored[0][1]), 2))
        return RagBaselineResult(
            query=query,
            domain=domain,
            answer_text=f"Configured baseline answer from {', '.join(chunk.chunk_id for chunk in selected)}: {joined}",
            retrieved_chunks=selected,
            confidence=confidence,
            warnings=warnings,
        )

    def _score(self, query: str, chunk: str, index: int) -> float:
        query_tokens = self._tokens(query)
        chunk_tokens = self._tokens(chunk)
        overlap = len(query_tokens & chunk_tokens)
        bonus = 0.0
        lowered = chunk.lower()
        for token, value in self.keyword_bonus.items():
            if token.lower() in lowered:
                bonus += float(value)
        for order, token in enumerate(self.preferred_order):
            if token in lowered:
                bonus += max(0.0, 1.0 - 0.1 * order)
        bonus += max(0.0, 0.2 - 0.01 * index)
        return overlap + bonus

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z가-힣0-9]+", text.lower()))


BASELINE_SPECS: Dict[str, BaselineSpec] = {
    "lexical_rag": BaselineSpec(name="lexical_rag", description="Lexical overlap chunk retrieval baseline."),
    "first_chunk": BaselineSpec(name="first_chunk", description="Naive customer baseline that reads only the first chunk."),
    "configurable_keyword": BaselineSpec(name="configurable_keyword", description="Keyword-weighted baseline defined by a customer JSON config."),
}


class BaselineRunner:
    def __init__(self, name: str = "lexical_rag", config_path: str | None = None):
        self.name = name
        self.config_path = config_path
        self._baseline = self._build(name, config_path=config_path)

    def answer(self, query: str, context: str, domain: str = "general") -> RagBaselineResult:
        return self._baseline.answer(query, context, domain=domain)

    @staticmethod
    def _build(name: str, config_path: str | None = None) -> BaselineProtocol:
        if name == "lexical_rag":
            return PlainRagBaseline()
        if name == "first_chunk":
            return FirstChunkBaseline()
        if name == "configurable_keyword":
            if not config_path:
                raise ValueError("configurable_keyword baseline requires --baseline-config")
            return ConfigurableKeywordBaseline(config_path)
        raise ValueError(f"Unknown baseline: {name}")
