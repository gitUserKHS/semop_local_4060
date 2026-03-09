from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, List

from .contest_programmer import CompetitiveProgrammingReasoner
from .pipeline import StructuredMeaningPipeline
from .structures import StructuredMeaningGraph, SymbolicResult


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    score: float
    detail: str


@dataclass
class HardProblemCandidate:
    source: str
    kind: str
    answer: str
    confidence: float
    verification_score: float
    verified: bool
    detail: str


@dataclass
class HardProblemReport:
    query: str
    graph: StructuredMeaningGraph
    candidates: List[HardProblemCandidate]
    checks: List[VerificationCheck]
    chosen_answer: str
    solved: bool
    verification_score: float
    matched_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["graph"] = self.graph.model_dump()
        return payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class PatternOutcomeTrainer:
    def __init__(self, path: str | Path = "data/logical_pattern_weights.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Dict[str, float]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload.get("weights", {}) if isinstance(payload, dict) else {}

    def update(self, matched_patterns: List[str], success: bool) -> Dict[str, Dict[str, float]]:
        weights = self.load()
        for key in matched_patterns:
            item = weights.setdefault(key, {"success": 0.0, "failure": 0.0, "weight": 1.0})
            if success:
                item["success"] = float(item.get("success", 0.0)) + 1.0
            else:
                item["failure"] = float(item.get("failure", 0.0)) + 1.0
            total = item["success"] + item["failure"]
            success_rate = item["success"] / total if total else 0.5
            item["weight"] = round(0.7 + 0.8 * success_rate, 2)
        self.path.write_text(json.dumps({"weights": weights}, ensure_ascii=False, indent=2), encoding="utf-8")
        return weights


class HardProblemEngine:
    def __init__(self, mode: str = "heuristic", model_id: str = "Qwen/Qwen2.5-3B-Instruct", memory_store_path: str | None = None, memory_source: str | None = None, logical_weight_path: str | None = "data/logical_pattern_weights.json") -> None:
        self.pipeline = StructuredMeaningPipeline(
            mode=mode,
            model_id=model_id,
            memory_store_path=memory_store_path,
            memory_source=memory_source,
            logical_weight_path=logical_weight_path,
        )
        self.contest = CompetitiveProgrammingReasoner()
        self.trainer = PatternOutcomeTrainer(logical_weight_path or "data/logical_pattern_weights.json")

    def solve(self, query: str) -> HardProblemReport:
        graph = self.pipeline.run(query)
        candidates: List[HardProblemCandidate] = []
        checks: List[VerificationCheck] = []

        for result in graph.symbolic_results:
            candidate, candidate_checks = self._candidate_from_symbolic(result)
            candidates.append(candidate)
            checks.extend(candidate_checks)

        contest_candidate = self.contest.solve(query)
        if contest_candidate is not None and self._looks_like_contest_problem(query):
            verified = contest_candidate.category != "generic_contest_analysis"
            score = 0.76 if verified else 0.46
            candidates.append(
                HardProblemCandidate(
                    source="contest_programmer",
                    kind=contest_candidate.category,
                    answer=contest_candidate.approach,
                    confidence=contest_candidate.confidence,
                    verification_score=score,
                    verified=verified,
                    detail=contest_candidate.complexity,
                )
            )
            checks.append(
                VerificationCheck(
                    name="contest_template_check",
                    passed=verified,
                    score=score,
                    detail=f"category={contest_candidate.category}, complexity={contest_candidate.complexity}",
                )
            )

        structural_score = 0.35
        if graph.plan:
            structural_score += 0.2
        if graph.induced_operators:
            structural_score += 0.15
        if not graph.invalid_advice:
            structural_score += 0.1
        checks.append(
            VerificationCheck(
                name="structured_reasoning_check",
                passed=structural_score >= 0.55,
                score=round(min(0.9, structural_score), 2),
                detail="Checks whether a plan, operator trace, and no obvious invalid advice are present.",
            )
        )

        if not candidates:
            candidates.append(
                HardProblemCandidate(
                    source="pipeline",
                    kind="structured_reasoning",
                    answer="No direct solver answer was produced; use the graph, plan, and operator trace as the current strategy state.",
                    confidence=0.42,
                    verification_score=round(min(0.8, structural_score), 2),
                    verified=structural_score >= 0.6,
                    detail="Strategy-only fallback",
                )
            )

        candidates.sort(key=lambda item: (-item.verification_score, -item.confidence, item.source))
        best = candidates[0]
        matched_patterns = self._matched_patterns(graph)
        solved = best.verified or best.verification_score >= 0.75
        return HardProblemReport(
            query=query,
            graph=graph,
            candidates=candidates,
            checks=checks,
            chosen_answer=best.answer,
            solved=solved,
            verification_score=best.verification_score,
            matched_patterns=matched_patterns,
        )

    def learn_from_report(self, report: HardProblemReport, success: bool | None = None) -> Dict[str, Dict[str, float]]:
        outcome = report.solved if success is None else success
        return self.trainer.update(report.matched_patterns, outcome)

    @staticmethod
    def _matched_patterns(graph: StructuredMeaningGraph) -> List[str]:
        patterns: List[str] = []
        for warning in graph.warnings:
            prefix = "logical grammar prior matched: "
            if warning.startswith(prefix):
                rhs = warning[len(prefix):]
                if " -> " in rhs:
                    connector, family = rhs.split(" -> ", 1)
                    patterns.append(f"{family}::{connector.lower()}")
        return list(dict.fromkeys(patterns))

    @staticmethod
    def _looks_like_contest_problem(query: str) -> bool:
        lowered = query.lower()
        return any(token in lowered for token in ["graph", "array", "queries", "roads", "nodes", "shortest path", "capacity", "knapsack", "grid"])

    @staticmethod
    def _candidate_from_symbolic(result: SymbolicResult) -> tuple[HardProblemCandidate, List[VerificationCheck]]:
        checks: List[VerificationCheck] = []
        if result.domain == "arithmetic":
            score = 0.95 if result.equations else 0.72
            checks.append(VerificationCheck(name="arithmetic_equation_check", passed=bool(result.equations), score=score, detail="Arithmetic symbolic result includes explicit equations."))
            return (
                HardProblemCandidate(source=result.source or "symbolic", kind="arithmetic", answer=result.answer, confidence=result.confidence, verification_score=score, verified=score >= 0.9, detail="equation-backed arithmetic"),
                checks,
            )
        if result.domain == "olympiad_proof":
            incomplete = "incomplete" in result.answer.lower()
            score = 0.84 if not incomplete else 0.48
            checks.append(VerificationCheck(name="proof_trace_check", passed=bool(result.evidence) and bool(result.equations), score=score, detail="Olympiad solver produced a proof outline and operator trace."))
            return (
                HardProblemCandidate(source=result.source or "symbolic", kind="olympiad_proof", answer=result.answer, confidence=result.confidence, verification_score=score, verified=not incomplete, detail="proof-search result"),
                checks,
            )
        if result.domain == "document_grounding":
            score = 0.78 if result.evidence else 0.55
            checks.append(VerificationCheck(name="document_evidence_check", passed=bool(result.evidence), score=score, detail="Document grounding includes supporting evidence spans."))
            return (
                HardProblemCandidate(source=result.source or "symbolic", kind="document_grounding", answer=result.answer, confidence=result.confidence, verification_score=score, verified=bool(result.evidence), detail="evidence-backed document answer"),
                checks,
            )
        score = max(0.4, result.confidence)
        checks.append(VerificationCheck(name="symbolic_presence_check", passed=True, score=score, detail="Generic symbolic result is present."))
        return (
            HardProblemCandidate(source=result.source or "symbolic", kind=result.domain, answer=result.answer, confidence=result.confidence, verification_score=score, verified=score >= 0.75, detail="generic symbolic result"),
            checks,
        )
