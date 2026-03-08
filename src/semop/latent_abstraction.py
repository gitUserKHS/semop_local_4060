from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Iterable, List

from .model_cache import resolve_embedding_model_id


@dataclass
class PhraseCluster:
    label: str
    members: List[str]


class OperatorAbstractionClustering:
    """Paraphrase clustering with an embedding path when available and a lexical fallback otherwise."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_id = resolve_embedding_model_id(model_id)
        self._model = None
        self._embedding_backend = False
        self._load_attempted = False

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

    def cluster(self, phrases: Iterable[str], threshold: float = 0.66) -> List[PhraseCluster]:
        phrases = list(dict.fromkeys(phrase.strip() for phrase in phrases if phrase.strip()))
        if not phrases:
            return []

        self._lazy_load()
        if self._embedding_backend and self._model is not None:
            return self._cluster_with_embeddings(phrases, threshold)
        return self._cluster_with_lexical_similarity(phrases, threshold)

    def _cluster_with_embeddings(self, phrases: List[str], threshold: float) -> List[PhraseCluster]:
        embeddings = self._model.encode(phrases, normalize_embeddings=True)
        clusters: List[PhraseCluster] = []
        used: set[int] = set()

        for index, phrase in enumerate(phrases):
            if index in used:
                continue
            members: List[str] = []
            for other_index, other_phrase in enumerate(phrases):
                score = float(sum(a * b for a, b in zip(embeddings[index], embeddings[other_index])))
                if score >= threshold:
                    used.add(other_index)
                    members.append(other_phrase)
            clusters.append(PhraseCluster(label=self._suggest_label(members), members=members))
        return clusters

    def _cluster_with_lexical_similarity(self, phrases: List[str], threshold: float) -> List[PhraseCluster]:
        clusters: List[PhraseCluster] = []
        used: set[int] = set()

        for index, phrase in enumerate(phrases):
            if index in used:
                continue
            members = [phrase]
            used.add(index)
            for other_index, other_phrase in enumerate(phrases[index + 1 :], start=index + 1):
                if other_index in used:
                    continue
                if self._lexical_similarity(phrase, other_phrase) >= threshold:
                    members.append(other_phrase)
                    used.add(other_index)
            clusters.append(PhraseCluster(label=self._suggest_label(members), members=members))
        return clusters

    def _lexical_similarity(self, left: str, right: str) -> float:
        left_tokens = self._tokenize(left)
        right_tokens = self._tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0

        overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        left_bigrams = self._char_bigrams(left)
        right_bigrams = self._char_bigrams(right)
        if not left_bigrams or not right_bigrams:
            return overlap
        bigram_overlap = len(left_bigrams & right_bigrams) / len(left_bigrams | right_bigrams)
        return (overlap + bigram_overlap) / 2

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z가-힣0-9]+", text.lower()))

    @staticmethod
    def _char_bigrams(text: str) -> set[str]:
        compact = re.sub(r"\s+", "", text.lower())
        return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}

    @staticmethod
    def _suggest_label(members: List[str]) -> str:
        text = " ".join(members)
        if any(token in text for token in ["보류", "미루", "다음에", "나중"]):
            return "SUSPEND"
        if any(token in text for token in ["우선", "먼저", "중요", "핵심"]):
            return "PRIORITIZE"
        if any(token in text for token in ["조건", "경우", "라면", "하면"]):
            return "CONDITION"
        if any(token in text for token in ["확인", "점검", "검증", "맞는지"]):
            return "VERIFY"
        if any(token in text for token in ["대안", "다른", "바꿔", "우회"]):
            return "ALTERNATIVE"
        return "ABSTRACT_OPERATOR"
