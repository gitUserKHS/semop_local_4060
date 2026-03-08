from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Dict, List

from .context_chunks import TextChunk, split_context_into_chunks


@dataclass
class RagBaselineResult:
    query: str
    domain: str
    answer_text: str
    retrieved_chunks: List[TextChunk]
    confidence: float
    warnings: List[str]

    def model_dump(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "domain": self.domain,
            "answer_text": self.answer_text,
            "retrieved_chunks": [asdict(chunk) for chunk in self.retrieved_chunks],
            "confidence": self.confidence,
            "warnings": self.warnings,
        }

    def model_dump_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=ensure_ascii)


class PlainRagBaseline:
    def answer(self, query: str, context: str, domain: str = "general") -> RagBaselineResult:
        chunks = split_context_into_chunks(context)
        if not chunks:
            return RagBaselineResult(
                query=query,
                domain=domain,
                answer_text="No context was provided, so the baseline RAG answer is unavailable.",
                retrieved_chunks=[],
                confidence=0.0,
                warnings=["no context provided"],
            )

        scored = [(chunk, self._score(query, chunk.text)) for chunk in chunks]
        scored = [item for item in scored if item[1] > 0]
        if not scored:
            return RagBaselineResult(
                query=query,
                domain=domain,
                answer_text="The baseline RAG retriever found no clearly relevant chunk.",
                retrieved_chunks=chunks[:1],
                confidence=0.2,
                warnings=["retrieval was weak"],
            )

        scored.sort(key=lambda item: (-item[1], len(item[0].text)))
        top_chunks = [chunk for chunk, _score in scored[:2]]
        answer = self._compose_answer(query, top_chunks)
        confidence = min(0.85, round(0.35 + 0.08 * scored[0][1], 2))
        warnings: List[str] = []
        if len(top_chunks) == 1:
            warnings.append("single supporting chunk only")
        return RagBaselineResult(
            query=query,
            domain=domain,
            answer_text=answer,
            retrieved_chunks=top_chunks,
            confidence=confidence,
            warnings=warnings,
        )

    @staticmethod
    def _compose_answer(query: str, chunks: List[TextChunk]) -> str:
        top = chunks[0].text
        if len(chunks) == 1:
            return f"Baseline RAG answer from {chunks[0].chunk_id}: {top}"
        joined_ids = ", ".join(chunk.chunk_id for chunk in chunks)
        second = chunks[1].text
        return f"Baseline RAG answer from {joined_ids}: {top} {second}"

    def _score(self, query: str, chunk: str) -> float:
        query_tokens = self._tokens(query)
        chunk_tokens = self._tokens(chunk)
        if not query_tokens or not chunk_tokens:
            return 0.0
        overlap = len(query_tokens & chunk_tokens)
        numeric_overlap = len(self._numbers(query) & self._numbers(chunk)) * 0.5
        question_bonus = 0.5 if any(token in chunk.lower() for token in ["승인", "approval", "불일치", "mismatch", "통로", "aisle", "보류", "hold"]) else 0.0
        return overlap + numeric_overlap + question_bonus

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z가-힣0-9]+", text.lower()))

    @staticmethod
    def _numbers(text: str) -> set[str]:
        return set(re.findall(r"\d+(?:\.\d+)?", text))
