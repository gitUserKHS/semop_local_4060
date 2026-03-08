from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .corpus_learning import CorpusReasoningLearner, LearnedOperatorFamily


@dataclass
class FamilyTransferScore:
    family: str
    train_support: int
    purity: float
    test_matches: int
    test_total: int
    transfer_rate: float


@dataclass
class TransferEvaluationResult:
    train_size: int
    test_size: int
    family_reuse_rate: float
    grammar_transfer_rate: float
    average_family_purity: float
    per_family: List[FamilyTransferScore]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)


class TransferEvaluator:
    def __init__(self, mode: str = "heuristic", model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.learner = CorpusReasoningLearner(mode=mode, model_id=model_id)

    def evaluate_queries(self, queries: Sequence[str], train_ratio: float = 0.7) -> TransferEvaluationResult:
        train_queries, test_queries = self._split_queries(queries, train_ratio=train_ratio)
        return self.evaluate_train_test_queries(train_queries, test_queries)

    def evaluate_train_test_queries(self, train_queries: Sequence[str], test_queries: Sequence[str]) -> TransferEvaluationResult:
        train_result = self.learner.learn_from_queries(train_queries)
        train_family_map = {self._extract_base_family(family): family for family in train_result.learned_families}
        test_graphs = [self.learner.pipeline.run(query) for query in test_queries]

        total_candidates = 0
        matched_candidates = 0
        total_grammar = 0
        matched_grammar = 0
        family_hits: Dict[str, int] = {}
        family_totals: Dict[str, int] = {}

        for graph in test_graphs:
            for candidate in graph.induced_operators:
                base_family = candidate.family
                total_candidates += 1
                family_totals[base_family] = family_totals.get(base_family, 0) + 1
                if base_family in train_family_map:
                    matched_candidates += 1
                    family_hits[base_family] = family_hits.get(base_family, 0) + 1
            for rule in graph.grammar_hypotheses:
                total_grammar += 1
                if any(base_family in rule for base_family in train_family_map):
                    matched_grammar += 1

        per_family: List[FamilyTransferScore] = []
        for base_family, family in sorted(train_family_map.items()):
            hits = family_hits.get(base_family, 0)
            total = family_totals.get(base_family, 0)
            per_family.append(
                FamilyTransferScore(
                    family=base_family,
                    train_support=family.support,
                    purity=family.purity,
                    test_matches=hits,
                    test_total=total,
                    transfer_rate=round(hits / total, 2) if total else 0.0,
                )
            )

        average_purity = round(sum(family.purity for family in train_result.learned_families) / max(1, len(train_result.learned_families)), 2)
        return TransferEvaluationResult(
            train_size=len(train_queries),
            test_size=len(test_queries),
            family_reuse_rate=round(matched_candidates / max(1, total_candidates), 2),
            grammar_transfer_rate=round(matched_grammar / max(1, total_grammar), 2),
            average_family_purity=average_purity,
            per_family=per_family,
        )

    def evaluate_jsonl(self, path: str | Path, train_ratio: float = 0.7) -> TransferEvaluationResult:
        queries = self._load_queries(path)
        return self.evaluate_queries(queries, train_ratio=train_ratio)

    @staticmethod
    def _split_queries(queries: Sequence[str], train_ratio: float) -> tuple[List[str], List[str]]:
        train: List[str] = []
        test: List[str] = []
        for query in queries:
            bucket = int(hashlib.sha256(query.encode("utf-8")).hexdigest(), 16) % 100
            if bucket < int(train_ratio * 100):
                train.append(query)
            else:
                test.append(query)
        if not train and queries:
            train.append(queries[0])
        if not test and len(queries) > 1:
            test.append(queries[-1])
            if queries[-1] in train:
                train.remove(queries[-1])
        return train, test

    @staticmethod
    def _load_queries(path: str | Path) -> List[str]:
        queries: List[str] = []
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    item = json.loads(line)
                    query = item.get("query") or item.get("text") or item.get("input")
                    if query:
                        queries.append(query)
                else:
                    queries.append(line)
        return queries

    @staticmethod
    def _extract_base_family(family: LearnedOperatorFamily) -> str:
        label = family.name
        if label.startswith("FAMILY_"):
            label = label[len("FAMILY_"):]
        if "_" in label:
            return label.split("_", 1)[0]
        return label