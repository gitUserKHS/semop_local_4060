from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import List, Sequence

from .model_cache import resolve_embedding_model_id
from .structures import StructuredMeaningGraph


@dataclass
class RetrievalHit:
    query: str
    intent: str
    score: float
    source: str = "memory"


class QueryEmbeddingIndex:
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_id = resolve_embedding_model_id(model_id)
        self._model = None
        self._embedding_backend = False
        self._load_attempted = False
        self._indexed_graphs: List[StructuredMeaningGraph] = []
        self._indexed_embeddings = None
        self._indexed_tokens: List[set[str]] = []

    def _lazy_load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model = SentenceTransformer(self.model_id, device=device, local_files_only=True)
        except Exception:
            self._model = None
            self._embedding_backend = False
            return
        self._embedding_backend = True

    def build(self, graphs: Sequence[StructuredMeaningGraph]) -> None:
        self._lazy_load()
        self._indexed_graphs = list(graphs)
        if self._embedding_backend and self._model is not None:
            texts = [graph.query for graph in self._indexed_graphs]
            self._indexed_embeddings = self._model.encode(texts, normalize_embeddings=True)
            self._indexed_tokens = []
        else:
            self._indexed_embeddings = None
            self._indexed_tokens = [self._tokenize(graph.query) for graph in self._indexed_graphs]

    def search(self, query: str, graphs: Sequence[StructuredMeaningGraph], top_k: int = 3) -> List[RetrievalHit]:
        if not graphs:
            return []
        self.build(graphs)
        return self.search_indexed(query, top_k=top_k)

    def search_indexed(self, query: str, top_k: int = 3) -> List[RetrievalHit]:
        if not self._indexed_graphs:
            return []
        if self._embedding_backend and self._model is not None and self._indexed_embeddings is not None:
            return self._search_with_indexed_embeddings(query, top_k=top_k)
        return self._search_with_indexed_lexical_similarity(query, top_k=top_k)

    def search_indexed_graphs(self, query: str, top_k: int = 3) -> List[StructuredMeaningGraph]:
        hits = self.search_indexed(query, top_k=top_k)
        graph_map = {graph.query: graph for graph in self._indexed_graphs}
        selected: List[StructuredMeaningGraph] = []
        for hit in hits:
            graph = graph_map.get(hit.query)
            if graph is not None:
                selected.append(graph)
        return selected

    def _search_with_embeddings(self, query: str, graphs: Sequence[StructuredMeaningGraph], top_k: int) -> List[RetrievalHit]:
        self.build(graphs)
        return self._search_with_indexed_embeddings(query, top_k=top_k)

    def _search_with_indexed_embeddings(self, query: str, top_k: int) -> List[RetrievalHit]:
        query_embedding = self._model.encode([query], normalize_embeddings=True)[0]
        hits: List[RetrievalHit] = []
        for graph, embedding in zip(self._indexed_graphs, self._indexed_embeddings):
            score = float(sum(a * b for a, b in zip(query_embedding, embedding)))
            hits.append(RetrievalHit(query=graph.query, intent=graph.intent, score=round(score, 4)))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def _search_with_lexical_similarity(self, query: str, graphs: Sequence[StructuredMeaningGraph], top_k: int) -> List[RetrievalHit]:
        self.build(graphs)
        return self._search_with_indexed_lexical_similarity(query, top_k=top_k)

    def _search_with_indexed_lexical_similarity(self, query: str, top_k: int) -> List[RetrievalHit]:
        query_tokens = self._tokenize(query)
        hits: List[RetrievalHit] = []
        for graph, tokens in zip(self._indexed_graphs, self._indexed_tokens):
            score = self._lexical_similarity(query_tokens, tokens)
            hits.append(RetrievalHit(query=graph.query, intent=graph.intent, score=round(score, 4)))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z가-힣0-9]+", text.lower()))

    @staticmethod
    def _lexical_similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
