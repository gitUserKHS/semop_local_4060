from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Dict, List, Sequence

from .cp_knowledge import CpAlgorithmKnowledge, CpDslOperator, CpKnowledgeLoader, CpLogicalFrame
from .cpp_compiler import CppSyntaxChecker
from .cp_validation import CpSolutionValidator
from .cp_repair import CpRepairAttempt, CppRepairEngine
from .cp_episode_store import CpEpisodeRecord, CpEpisodeStore


@dataclass
class ContestProblemStructure:
    normalized_query: str
    goal_types: List[str]
    domain_tags: List[str]
    extracted_constraints: List[str]
    detected_phrases: List[str]
    hidden_concepts: List[str]
    logical_frames: List[str]
    dsl_operators: List[str]
    candidate_algorithms: List[str]
    episode_priors: Dict[str, object]
    memory_projection: Dict[str, object]


@dataclass
class ContestSolution:
    category: str
    approach: str
    cpp_code: str
    time_complexity: str
    memory_complexity: str
    cues: List[str] = field(default_factory=list)
    confidence: float = 0.6
    reasoning_steps: List[str] = field(default_factory=list)
    hidden_concepts: List[str] = field(default_factory=list)
    logical_frames: List[str] = field(default_factory=list)
    goal_types: List[str] = field(default_factory=list)
    domain_tags: List[str] = field(default_factory=list)
    dsl_operators: List[str] = field(default_factory=list)
    memory_projection: Dict[str, object] = field(default_factory=dict)
    extracted_constraints: List[str] = field(default_factory=list)
    compile_ok: bool = False
    compile_command: str = ""
    compile_stderr: str = ""
    validation_report: Dict[str, object] = field(default_factory=dict)
    knowledge_sources: List[str] = field(default_factory=list)
    repaired: bool = False
    repair_attempts: List[dict] = field(default_factory=list)
    search_trace: List[dict] = field(default_factory=list)
    selection_strategy: str = 'single_best'

    def model_dump(self) -> dict:
        return {
            "category": self.category,
            "approach": self.approach,
            "cpp_code": self.cpp_code,
            "time_complexity": self.time_complexity,
            "memory_complexity": self.memory_complexity,
            "cues": self.cues,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "hidden_concepts": self.hidden_concepts,
            "logical_frames": self.logical_frames,
            "goal_types": self.goal_types,
            "domain_tags": self.domain_tags,
            "dsl_operators": self.dsl_operators,
            "memory_projection": self.memory_projection,
            "extracted_constraints": self.extracted_constraints,
            "compile_ok": self.compile_ok,
            "compile_command": self.compile_command,
            "compile_stderr": self.compile_stderr,
            "validation_report": self.validation_report,
            "knowledge_sources": self.knowledge_sources,
            "repaired": self.repaired,
            "repair_attempts": self.repair_attempts,
            "search_trace": self.search_trace,
            "selection_strategy": self.selection_strategy,
        }

    def model_dump_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=ensure_ascii)


class CompetitiveProgrammingReasoner:
    def __init__(
        self,
        knowledge_path: str = "data/knowledge/cp_knowledge.json",
        compiler: str = "g++",
        episode_store_path: str | None = None,
    ) -> None:
        self.knowledge = CpKnowledgeLoader(knowledge_path).load()
        standard = str(self.knowledge.compiler_profile.get("language_standard", "c++17"))
        self.compiler = CppSyntaxChecker(compiler=compiler, standard=standard)
        self.validator = CpSolutionValidator(compiler=compiler, standard=standard)
        self.repair_engine = CppRepairEngine()
        self.episode_store = CpEpisodeStore(episode_store_path) if episode_store_path else None

    def solve(self, query: str) -> ContestSolution | None:
        normalized = " ".join(query.lower().split())
        if not normalized:
            return None

        structure = self._structure_problem(normalized)
        ranked = self._rank_algorithms(structure)
        chosen = ranked[0] if ranked else None
        memory_projection = self._memory_projection(structure, chosen)
        if chosen is None:
            solution = self._finalize_solution(
                category="generic_contest_analysis",
                approach="Read constraints, identify the state space, and only then choose a data structure or algorithm family.",
                cpp_code=self._generic_template(),
                time_complexity="Depends on the final reduction",
                memory_complexity="Depends on the chosen representation",
                cues=structure.detected_phrases,
                confidence=0.45,
                reasoning_steps=self._reasoning_steps(structure, None),
                hidden_concepts=structure.hidden_concepts,
                logical_frames=structure.logical_frames,
                goal_types=structure.goal_types,
                domain_tags=structure.domain_tags,
                dsl_operators=structure.dsl_operators,
                memory_projection=memory_projection,
                extracted_constraints=structure.extracted_constraints,
                source_ids=[],
            )
            self._record_episode(query, normalized, structure, solution)
            return solution

        search_trace: list[dict] = []
        best_solution = None
        best_score = -1.0
        for rank_index, candidate in enumerate(ranked[:3], start=1):
            candidate_projection = self._memory_projection(structure, candidate)
            candidate_projection['procedural']['candidate_rank'] = rank_index
            candidate_projection['procedural']['search_mode'] = 'verifier_rerank'
            cpp_code = self._template_for(candidate.template_kind)
            approach = self._approach_for(candidate, structure)
            trial = self._finalize_solution(
                category=candidate.id,
                approach=approach,
                cpp_code=cpp_code,
                time_complexity=candidate.time_complexity,
                memory_complexity=candidate.memory_complexity,
                cues=candidate.triggers[:4],
                confidence=self._confidence(candidate, structure),
                reasoning_steps=self._reasoning_steps(structure, candidate),
                hidden_concepts=list(dict.fromkeys(structure.hidden_concepts + candidate.hidden_concepts))[:10],
                logical_frames=structure.logical_frames,
                goal_types=structure.goal_types,
                domain_tags=structure.domain_tags,
                dsl_operators=structure.dsl_operators,
                memory_projection=candidate_projection,
                extracted_constraints=structure.extracted_constraints,
                source_ids=candidate.source_ids,
            )
            selection_score = self._selection_score(trial, candidate, structure, rank_index)
            search_trace.append({
                'rank': rank_index,
                'category': trial.category,
                'selection_score': round(selection_score, 4),
                'compile_ok': trial.compile_ok,
                'overall_ok': bool(trial.validation_report.get('overall_ok')) if isinstance(trial.validation_report, dict) else False,
                'failure_type': str(trial.validation_report.get('failure_type', '')) if isinstance(trial.validation_report, dict) else '',
            })
            if selection_score > best_score:
                best_solution = trial
                best_score = selection_score

        assert best_solution is not None
        best_solution.search_trace = search_trace
        best_solution.selection_strategy = 'verifier_rerank_top3'
        best_solution.memory_projection.setdefault('procedural', {})['search_trace'] = search_trace
        self._record_episode(query, normalized, structure, best_solution)
        return best_solution

    def _finalize_solution(
        self,
        category: str,
        approach: str,
        cpp_code: str,
        time_complexity: str,
        memory_complexity: str,
        cues: List[str],
        confidence: float,
        reasoning_steps: List[str],
        hidden_concepts: List[str],
        logical_frames: List[str],
        goal_types: List[str],
        domain_tags: List[str],
        dsl_operators: List[str],
        memory_projection: Dict[str, object],
        extracted_constraints: List[str],
        source_ids: Sequence[str],
    ) -> ContestSolution:
        active_code = cpp_code
        compile_result = self.compiler.check(active_code)
        source_links = [self.knowledge.sources[item]["url"] for item in source_ids if item in self.knowledge.sources]
        validation_report = {}
        repair_attempts: List[CpRepairAttempt] = []
        repaired = False
        if compile_result.ok:
            validation_report = self.validator.validate(active_code, category).model_dump()
        if (not compile_result.ok) or validation_report.get("overall_ok") is False:
            repaired_code, attempts = self.repair_engine.repair(
                category=category,
                code=active_code,
                compile_stderr=compile_result.stderr,
                validation_notes=validation_report.get("notes", []),
                failure_type=str(validation_report.get("failure_type", "compile_error" if not compile_result.ok else "")),
                counterexample_input=str(validation_report.get("counterexample_input", "")),
                fallback_code=cpp_code,
            )
            repair_attempts.extend(attempts)
            if repaired_code != active_code:
                active_code = repaired_code
                repaired = True
                compile_result = self.compiler.check(active_code)
                if compile_result.ok:
                    validation_report = self.validator.validate(active_code, category).model_dump()
                else:
                    validation_report = {}
        if compile_result.ok and validation_report:
            if validation_report.get("overall_ok") is False:
                confidence = max(0.2, confidence - 0.15)
        elif not compile_result.ok:
            confidence = max(0.2, confidence - 0.2)
        return ContestSolution(
            category=category,
            approach=approach,
            cpp_code=active_code,
            time_complexity=time_complexity,
            memory_complexity=memory_complexity,
            cues=cues,
            confidence=round(confidence, 2),
            reasoning_steps=reasoning_steps,
            hidden_concepts=hidden_concepts,
            logical_frames=logical_frames,
            goal_types=goal_types,
            domain_tags=domain_tags,
            dsl_operators=dsl_operators,
            memory_projection=memory_projection,
            extracted_constraints=extracted_constraints,
            compile_ok=compile_result.ok,
            compile_command=compile_result.command,
            compile_stderr=compile_result.stderr,
            validation_report=validation_report,
            knowledge_sources=source_links,
            repaired=repaired,
            repair_attempts=[attempt.model_dump() for attempt in repair_attempts],
        )

    def _record_episode(
        self,
        statement: str,
        normalized_statement: str,
        structure: ContestProblemStructure,
        solution: ContestSolution,
    ) -> None:
        if self.episode_store is None:
            return
        record = CpEpisodeRecord(
            statement=statement,
            normalized_statement=normalized_statement,
            category=solution.category,
            approach=solution.approach,
            cpp_code=solution.cpp_code,
            time_complexity=solution.time_complexity,
            memory_complexity=solution.memory_complexity,
            confidence=solution.confidence,
            goal_types=solution.goal_types,
            domain_tags=solution.domain_tags,
            logical_frames=solution.logical_frames,
            dsl_operators=solution.dsl_operators,
            hidden_concepts=solution.hidden_concepts,
            extracted_constraints=solution.extracted_constraints,
            candidate_algorithms=structure.candidate_algorithms,
            memory_projection=solution.memory_projection,
            reasoning_steps=solution.reasoning_steps,
            compile_ok=solution.compile_ok,
            compile_command=solution.compile_command,
            compile_stderr=solution.compile_stderr,
            validation_report=solution.validation_report,
            repaired=solution.repaired,
            repair_attempts=solution.repair_attempts,
            knowledge_sources=solution.knowledge_sources,
            outcome_label="AC" if solution.compile_ok and solution.validation_report.get("overall_ok") else "WA",
            failure_kind=str(solution.validation_report.get("failure_type", "")),
            source_kind="generated",
        )
        self.episode_store.record_episode(record)

    def _structure_problem(self, normalized: str) -> ContestProblemStructure:
        constraints = self._extract_constraints(normalized)
        phrases = self._detected_phrases(normalized)
        goals = self._goal_types(normalized)
        frames = self._matched_frames(normalized)
        domains = self._domain_tags(normalized, frames)
        hidden = self._hidden_concepts_from_text(normalized, frames)
        operators = self._matched_operators(normalized, frames, goals, domains)
        candidate_algorithms = [item.id for item in self._rank_algorithms_from_text(normalized, frames)]
        structure = ContestProblemStructure(
            normalized_query=normalized,
            goal_types=goals,
            domain_tags=domains,
            extracted_constraints=constraints,
            detected_phrases=phrases,
            hidden_concepts=hidden,
            logical_frames=[frame.id for frame in frames],
            dsl_operators=operators,
            candidate_algorithms=candidate_algorithms,
            episode_priors=self._episode_priors(normalized),
            memory_projection={},
        )
        structure.memory_projection = self._memory_projection(structure, None)
        return structure



    def parse_problem(self, query: str) -> ContestProblemStructure | None:
        normalized = " ".join(query.lower().split())
        if not normalized:
            return None
        return self._structure_problem(normalized)

    @staticmethod
    def _contains_trigger(normalized: str, trigger: str) -> bool:
        if " " in trigger:
            return trigger in normalized
        return re.search(r"\b" + re.escape(trigger) + r"\b", normalized) is not None

    def _has_geometry_context(self, normalized: str) -> bool:
        strong_terms = [
            'line', 'segment', 'triangle', 'polygon', 'parallel', 'perpendicular', 'distance',
            'area', 'rectangle', 'circle', 'angle', 'coordinate', 'coordinates', 'vector', 'quadrilateral', 'parallelogram',
            'convex', 'intersect', 'intersection', 'bounding rectangle',
        ]
        graph_context = any(
            self._contains_trigger(normalized, term)
            for term in ('graph', 'node', 'nodes', 'edge', 'edges', 'road', 'roads', 'path')
        )
        geometry_terms = [term for term in strong_terms if term != 'distance']
        if any(self._contains_trigger(normalized, term) for term in geometry_terms):
            return True
        if self._contains_trigger(normalized, 'distance') and not graph_context:
            return True
        if any(term in normalized for term in ['point update', 'point updates']):
            return False
        point_patterns = [
            'given a point', 'given points', 'three points', 'set of points', 'point lies',
            'point inside', 'coordinates of three points', 'point and an axis-aligned rectangle',
            'point and a rectangle',
        ]
        return any(pattern in normalized for pattern in point_patterns)

    def _rank_algorithms(self, structure: ContestProblemStructure) -> List[CpAlgorithmKnowledge]:
        base_scores = {
            item.id: score
            for score, item in self._score_algorithms_from_text(
                structure.normalized_query,
                self._matched_frames(structure.normalized_query),
            )
        }
        prior_scores = dict(structure.episode_priors.get("category_scores", {}))
        combined: List[tuple[float, CpAlgorithmKnowledge]] = []
        for item in self.knowledge.algorithms:
            score = base_scores.get(item.id, 0.0) + float(prior_scores.get(item.id, 0.0))
            if score > 0:
                combined.append((score, item))
        combined.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [item for _, item in combined]

    def _goal_types(self, normalized: str) -> List[str]:
        goals: List[str] = []
        mapping = {
            "count": ["how many", "count", "number of"],
            "optimize": ["minimum", "maximum", "maximize", "minimize", "best"],
            "decision": ["is it possible", "possible", "can we", "exists", "whether"],
            "construct": ["construct", "build", "output any", "find one"],
            "query": ["queries", "for each query", "answer the query", "output the sum"],
        }
        for goal, triggers in mapping.items():
            if any(self._contains_trigger(normalized, trigger) for trigger in triggers):
                goals.append(goal)
        if not goals and "output" in normalized:
            goals.append("query")
        return goals[:4]

    def _domain_tags(self, normalized: str, frames: Sequence[CpLogicalFrame]) -> List[str]:
        tags: List[str] = []
        mapping = {
            "graph": ["graph", "node", "edge", "road", "path"],
            "tree": ["tree", "subtree", "parent", "root"],
            "array": ["array", "sequence", "prefix", "range", "interval"],
            "grid": ["grid", "maze", "board", "cell"],
            "dp": ["capacity", "weight", "value", "state"],
            "string": ["string", "substring", "character"],
            "math": ["mod", "prime", "gcd", "divisible"],
        }
        for tag, triggers in mapping.items():
            if any(self._contains_trigger(normalized, trigger) for trigger in triggers):
                tags.append(tag)
        if self._has_geometry_context(normalized):
            tags.append("geometry")
        for frame in frames:
            if "graph" in frame.id and "graph" not in tags:
                tags.append("graph")
        return tags[:5]

    def _matched_frames(self, normalized: str) -> List[CpLogicalFrame]:
        matched: List[CpLogicalFrame] = []
        geometry_context = self._has_geometry_context(normalized)
        for frame in self.knowledge.logical_frames:
            if frame.id == 'geometry_configuration' and not geometry_context:
                continue
            if any(self._contains_trigger(normalized, trigger) for trigger in frame.triggers):
                matched.append(frame)
        return matched

    def _matched_operators(
        self,
        normalized: str,
        frames: Sequence[CpLogicalFrame],
        goal_types: Sequence[str],
        domain_tags: Sequence[str],
    ) -> List[str]:
        matched: List[str] = []
        frame_tokens = " ".join(frame.reasoning_operator for frame in frames)
        auto_ops = {
            "count": ["COUNT", "AGGREGATE"],
            "optimize": ["OPTIMIZE"],
            "decision": ["DECIDE"],
            "construct": ["CONSTRUCT"],
            "query": ["RANGE_QUERY", "QUERY_LOOP"],
            "graph": ["STATE_GRAPH"],
            "tree": ["SUBTREE", "TRAVERSE"],
            "array": ["ARRAY_INDEX", "PREFIX"],
            "grid": ["GRID_TO_GRAPH", "NEIGHBOR_EXPAND"],
            "dp": ["STATE", "DP_TRANSITION"],
            "math": ["CONSTRAINT", "INVARIANT"],
            "geometry": ["POINT", "SEGMENT", "ORIENTATION_TEST", "CROSS_PRODUCT"],
        }
        for operator in self.knowledge.dsl_operators:
            if any(self._contains_trigger(normalized, trigger) for trigger in operator.trigger_hints):
                matched.append(operator.id)
                continue
            if operator.id in frame_tokens:
                matched.append(operator.id)
        for item in list(goal_types) + list(domain_tags):
            for operator_id in auto_ops.get(item, []):
                if operator_id not in matched:
                    matched.append(operator_id)
        return matched[:12]

    def _score_algorithms_from_text(self, normalized: str, frames: Sequence[CpLogicalFrame] | None = None) -> List[tuple[float, CpAlgorithmKnowledge]]:
        frames = frames or []
        scored: List[tuple[float, CpAlgorithmKnowledge]] = []
        geometry_context = self._has_geometry_context(normalized)
        for item in self.knowledge.algorithms:
            score = 0.0
            for trigger in item.triggers:
                if item.id == 'computational_geometry_analysis' and trigger in {'point', 'points'} and not geometry_context:
                    continue
                if self._contains_trigger(normalized, trigger):
                    score += 1.2 if len(trigger.split()) > 1 else 0.6
            if item.id == 'computational_geometry_analysis' and any(
                term in normalized for term in ['quadrilateral', 'parallelogram', 'bounding rectangle']
            ):
                score += 1.0
            for concept in item.hidden_concepts:
                for token in re.findall(r"[a-z]+", concept.lower()):
                    if len(token) > 3 and token in normalized:
                        score += 0.18
            for frame in frames:
                frame_text = " ".join(frame.inferred_concepts + [frame.reduction_goal, frame.reasoning_operator]).lower()
                haystack = item.id.lower() + " " + item.family.lower() + " " + " ".join(item.hidden_concepts).lower()
                for token in re.findall(r"[a-z_]+", frame_text):
                    if len(token) > 3 and token in haystack:
                        score += 0.12
            if (
                item.id == "dijkstra_shortest_path"
                and self._contains_trigger(normalized, "negative")
            ):
                score -= 1.0
            if item.id == "prefix_sum_range_query" and any(term in normalized for term in ["update", "modify"]):
                score -= 0.8
            if item.id == "fenwick_tree" and "query" in normalized and any(term in normalized for term in ["update", "modify", "change"]):
                score += 0.7
            if item.id == "segment_tree" and any(term in normalized for term in ["min", "max", "range minimum", "range maximum"]):
                score += 0.8
            if item.id == "lazy_segment_tree" and any(term in normalized for term in ["range update", "interval add", "range add", "lazy propagation"]):
                score += 1.0
            if item.id == "binary_search_answer" and any(term in normalized for term in ["minimum possible", "maximum possible", "maximize the minimum", "minimize the maximum"]):
                score += 0.9
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return scored

    def _rank_algorithms_from_text(self, normalized: str, frames: Sequence[CpLogicalFrame] | None = None) -> List[CpAlgorithmKnowledge]:
        return [item for _, item in self._score_algorithms_from_text(normalized, frames)]

    @staticmethod
    def _extract_constraints(normalized: str) -> List[str]:
        patterns = [
            r"\b[nmqwhk]\s*[<>=]+\s*\d+",
            r"\d+\s*(?:nodes|edges|queries|rows|columns|items|weights|capacity)",
            r"time limit[^.,;]*",
            r"memory limit[^.,;]*",
        ]
        found: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, normalized):
                item = match.group(0).strip()
                if item not in found:
                    found.append(item)
        return found[:8]

    @staticmethod
    def _detected_phrases(normalized: str) -> List[str]:
        phrases: List[str] = []
        phrase_bank = [
            "weighted graph",
            "shortest path",
            "range sum",
            "many queries",
            "point update",
            "range update",
            "connectivity queries",
            "union find",
            "grid",
            "minimum steps",
            "capacity",
            "maximize value",
            "minimum possible",
            "maximum possible",
            "triangle",
            "polygon",
            "parallel",
            "perpendicular",
            "distance",
            "area",
        ]
        for phrase in phrase_bank:
            if phrase in normalized and phrase not in phrases:
                phrases.append(phrase)
        return phrases[:8]

    def _episode_priors(self, normalized: str) -> Dict[str, object]:
        if self.episode_store is None or self.episode_store.count() == 0:
            return {
                "category_scores": {},
                "success_counts": {},
                "failure_counts": {},
                "failure_kinds": {},
                "similar_episodes": [],
            }
        return self.episode_store.build_rerank_priors(normalized, limit=8)

    @staticmethod
    def _hidden_concepts_from_text(normalized: str, frames: Sequence[CpLogicalFrame]) -> List[str]:
        concepts: List[str] = []
        mapping = {
            "weighted graph": "nonnegative weighted shortest path",
            "shortest path": "graph state reduction",
            "many queries": "preprocessing vs online query tradeoff",
            "point update": "dynamic prefix/range structure",
            "range update": "lazy propagation",
            "capacity": "state compression DP",
            "maximize value": "optimization dynamic programming",
            "minimum possible": "binary search on answer",
            "grid": "implicit graph traversal",
            "connectivity": "component maintenance",
            "triangle": "coordinate or Euclidean geometry structure",
            "polygon": "geometric boundary reasoning",
            "parallel": "slope or cross-product invariant",
            "perpendicular": "dot-product or right-angle reasoning",
        }
        for trigger, concept in mapping.items():
            if trigger in normalized and concept not in concepts:
                concepts.append(concept)
        for frame in frames:
            for concept in frame.inferred_concepts:
                if concept not in concepts:
                    concepts.append(concept)
        return concepts[:10]

    def _memory_projection(self, structure: ContestProblemStructure, item: CpAlgorithmKnowledge | None) -> Dict[str, object]:
        coding_rule_ids = [rule.get("id", "") for rule in self.knowledge.coding_rules[:4] if rule.get("id")]
        semantic = list(dict.fromkeys(structure.hidden_concepts + ([item.family] if item is not None else [])))[:10]
        structural = {
            "goals": structure.goal_types,
            "domains": structure.domain_tags,
            "frames": structure.logical_frames,
            "operators": structure.dsl_operators,
            "constraints": structure.extracted_constraints,
        }
        episodic = {
            "retrieval_keys": list(dict.fromkeys(structure.logical_frames + structure.domain_tags + structure.goal_types))[:8],
            "candidate_algorithms": structure.candidate_algorithms[:5],
            "selected_algorithm": item.id if item is not None else None,
            "category_scores": structure.episode_priors.get("category_scores", {}),
            "success_counts": structure.episode_priors.get("success_counts", {}),
            "failure_counts": structure.episode_priors.get("failure_counts", {}),
            "failure_kinds": structure.episode_priors.get("failure_kinds", {}),
            "similar_episodes": structure.episode_priors.get("similar_episodes", []),
        }
        procedural = {
            "coding_rules": coding_rule_ids,
            "checklist": [
                "check complexity against inferred constraints",
                "check indexing and initialization",
                "run syntax-only compile check",
                "plan sample and edge-case validation",
            ],
        }
        return {
            "semantic": semantic,
            "structural": structural,
            "episodic": episodic,
            "procedural": procedural,
        }

    def _selection_score(
        self,
        solution: ContestSolution,
        item: CpAlgorithmKnowledge,
        structure: ContestProblemStructure,
        rank_index: int,
    ) -> float:
        score = float(solution.confidence)
        score += self._structural_alignment_score(item, structure)
        if solution.compile_ok:
            score += 0.2
        if isinstance(solution.validation_report, dict) and solution.validation_report.get('overall_ok'):
            score += 0.35
        elif (
            isinstance(solution.validation_report, dict)
            and solution.validation_report.get('checked') is True
        ):
            score -= 0.35
        elif item.family in structure.domain_tags:
            score += 0.1
        failure = str(solution.validation_report.get('failure_type', '')) if isinstance(solution.validation_report, dict) else ''
        if failure in {'output_mismatch', 'time_limit'}:
            score -= 0.25
        if not solution.compile_ok:
            score -= 0.35
        score -= 0.08 * max(0, rank_index - 1)
        return score

    @staticmethod
    def _structural_alignment_score(item: CpAlgorithmKnowledge, structure: ContestProblemStructure) -> float:
        score = 0.0
        if item.family in structure.domain_tags:
            score += 0.7
        if item.family == 'geometry' and 'geometry_configuration' in structure.logical_frames:
            score += 0.35
        if item.id == 'computational_geometry_analysis' and any(
            operator in structure.dsl_operators
            for operator in ['POINT', 'SEGMENT', 'ORIENTATION_TEST', 'CROSS_PRODUCT', 'DOT_PRODUCT']
        ):
            score += 0.2
        if item.family == 'range_data_structure' and 'geometry' in structure.domain_tags:
            score -= 0.25
        if item.family == 'graph' and 'geometry' in structure.domain_tags:
            score -= 0.2
        return score

    @staticmethod
    def _confidence(item: CpAlgorithmKnowledge, structure: ContestProblemStructure) -> float:
        base = 0.58
        base += min(0.18, 0.03 * len(structure.detected_phrases))
        base += min(0.12, 0.02 * len(structure.extracted_constraints))
        base += 0.08 if item.id in structure.candidate_algorithms[:1] else 0.0
        base += min(0.09, 0.03 * len(structure.logical_frames))
        base += min(0.06, 0.01 * len(structure.dsl_operators))
        prior = float(structure.episode_priors.get("category_scores", {}).get(item.id, 0.0))
        if prior > 0:
            base += min(0.09, 0.08 * prior)
        elif prior < 0:
            base -= min(0.12, 0.08 * abs(prior))
        return min(0.95, max(0.2, base))

    @staticmethod
    def _reasoning_steps(structure: ContestProblemStructure, item: CpAlgorithmKnowledge | None) -> List[str]:
        steps = [
            "Extract constraints, objects, and query types from the statement.",
            "Map phrases into logical problem frames such as shortest-path, offline-range-query, dynamic-connectivity, or geometric-configuration.",
            "Project the statement into CP DSL operators and memory layers.",
            "Identify hidden concepts such as monotonicity, dynamic updates, or graph-state modeling.",
        ]
        if structure.episode_priors.get("similar_episodes"):
            steps.append("Consult episodic memory from similar solved and failed contest episodes before final ranking.")
        if structure.goal_types:
            steps.append(f"Infer goal types: {', '.join(structure.goal_types[:4])}.")
        if structure.domain_tags:
            steps.append(f"Infer domain tags: {', '.join(structure.domain_tags[:4])}.")
        if structure.logical_frames:
            steps.append(f"Matched logical frames: {', '.join(structure.logical_frames[:4])}.")
        if structure.dsl_operators:
            steps.append(f"Matched DSL operators: {', '.join(structure.dsl_operators[:6])}.")
        if item is not None:
            steps.append(f"Map the problem to algorithm family `{item.id}`.")
            steps.append(f"Check whether the claimed complexity `{item.time_complexity}` fits the inferred constraints.")
            steps.append("Generate a C++17 template and run a syntax-only compile check with g++.")
        else:
            steps.append("No strong algorithm family match was found, so keep the analysis at the structural level.")
        return steps

    @staticmethod
    def _approach_for(item: CpAlgorithmKnowledge, structure: ContestProblemStructure) -> str:
        hidden = ", ".join(item.hidden_concepts[:3])
        constraints = ", ".join(structure.extracted_constraints[:3]) if structure.extracted_constraints else "no explicit numeric constraint detected"
        frame_text = ", ".join(structure.logical_frames[:3]) if structure.logical_frames else "no strong logical frame match"
        operator_text = ", ".join(structure.dsl_operators[:5]) if structure.dsl_operators else "no strong operator match"
        return (
            f"Reduce the statement to `{item.id}` because the detected phrases, logical frames ({frame_text}), and DSL operators ({operator_text}) point to {hidden}. "
            f"Then choose a C++17 implementation whose time complexity is {item.time_complexity} and whose memory complexity is {item.memory_complexity}. "
            f"Constraint summary: {constraints}."
        )

    def _template_for(self, kind: str) -> str:
        templates = {
            "prefix_sum": self._prefix_sum_template(),
            "fenwick": self._fenwick_template(),
            "segtree": self._segment_tree_template(),
            "lazy_segtree": self._lazy_segment_tree_template(),
            "dsu": self._dsu_template(),
            "dijkstra": self._dijkstra_template(),
            "grid_bfs": self._grid_bfs_template(),
            "binary_search": self._binary_search_template(),
            "knapsack": self._knapsack_template(),
            "geometry": self._geometry_template(),
        }
        return templates.get(kind, self._generic_template())

    @staticmethod
    def _prefix_sum_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    cin >> n >> q;
    vector<long long> pref(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        long long x;
        cin >> x;
        pref[i] = pref[i - 1] + x;
    }
    while (q--) {
        int l, r;
        cin >> l >> r;
        cout << pref[r] - pref[l - 1] << '\\n';
    }
    return 0;
}
'''

    @staticmethod
    def _fenwick_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

struct Fenwick {
    int n;
    vector<long long> bit;
    Fenwick(int n) : n(n), bit(n + 1, 0) {}
    void add(int idx, long long val) {
        for (; idx <= n; idx += idx & -idx) bit[idx] += val;
    }
    long long sum_prefix(int idx) const {
        long long ans = 0;
        for (; idx > 0; idx -= idx & -idx) ans += bit[idx];
        return ans;
    }
    long long sum_range(int l, int r) const {
        return sum_prefix(r) - sum_prefix(l - 1);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    cin >> n >> q;
    Fenwick fw(n);
    for (int i = 1; i <= n; ++i) {
        long long x;
        cin >> x;
        fw.add(i, x);
    }
    while (q--) {
        int type;
        cin >> type;
        if (type == 1) {
            int idx; long long delta;
            cin >> idx >> delta;
            fw.add(idx, delta);
        } else {
            int l, r;
            cin >> l >> r;
            cout << fw.sum_range(l, r) << '\\n';
        }
    }
    return 0;
}
'''

    @staticmethod
    def _segment_tree_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

struct SegTree {
    int n;
    const long long INF = (1LL << 60);
    vector<long long> tree;
    SegTree(int size) : n(1) {
        while (n < size) n <<= 1;
        tree.assign(2 * n, INF);
    }
    void build(const vector<long long>& arr) {
        for (int i = 0; i < (int)arr.size(); ++i) tree[n + i] = arr[i];
        for (int i = n - 1; i >= 1; --i) tree[i] = min(tree[2 * i], tree[2 * i + 1]);
    }
    void set_value(int idx, long long value) {
        idx += n;
        tree[idx] = value;
        for (idx >>= 1; idx >= 1; idx >>= 1) {
            tree[idx] = min(tree[2 * idx], tree[2 * idx + 1]);
            if (idx == 1) break;
        }
    }
    long long query(int l, int r, int x, int lx, int rx) const {
        if (r <= lx || rx <= l) return INF;
        if (l <= lx && rx <= r) return tree[x];
        int mid = (lx + rx) / 2;
        return min(query(l, r, 2 * x, lx, mid), query(l, r, 2 * x + 1, mid, rx));
    }
    long long query(int l, int r) const {
        return query(l, r, 1, 0, n);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    cin >> n >> q;
    vector<long long> arr(n);
    for (int i = 0; i < n; ++i) cin >> arr[i];
    SegTree st(n);
    st.build(arr);
    while (q--) {
        int type;
        cin >> type;
        if (type == 1) {
            int idx;
            long long value;
            cin >> idx >> value;
            st.set_value(idx - 1, value);
        } else {
            int l, r;
            cin >> l >> r;
            cout << st.query(l - 1, r) << '\\n';
        }
    }
    return 0;
}
'''

    @staticmethod
    def _lazy_segment_tree_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

struct LazySegTree {
    int n;
    vector<long long> tree, lazy;
    LazySegTree(int size) : n(1) {
        while (n < size) n <<= 1;
        tree.assign(2 * n, 0);
        lazy.assign(2 * n, 0);
    }
    void build(const vector<long long>& arr) {
        for (int i = 0; i < (int)arr.size(); ++i) tree[n + i] = arr[i];
        for (int i = n - 1; i >= 1; --i) tree[i] = tree[2 * i] + tree[2 * i + 1];
    }
    void apply(int x, int lx, int rx, long long value) {
        tree[x] += (rx - lx) * value;
        lazy[x] += value;
    }
    void push(int x, int lx, int rx) {
        if (rx - lx == 1 || lazy[x] == 0) return;
        int mid = (lx + rx) / 2;
        apply(2 * x, lx, mid, lazy[x]);
        apply(2 * x + 1, mid, rx, lazy[x]);
        lazy[x] = 0;
    }
    void add(int l, int r, long long value, int x, int lx, int rx) {
        if (r <= lx || rx <= l) return;
        if (l <= lx && rx <= r) {
            apply(x, lx, rx, value);
            return;
        }
        push(x, lx, rx);
        int mid = (lx + rx) / 2;
        add(l, r, value, 2 * x, lx, mid);
        add(l, r, value, 2 * x + 1, mid, rx);
        tree[x] = tree[2 * x] + tree[2 * x + 1];
    }
    long long sum(int l, int r, int x, int lx, int rx) {
        if (r <= lx || rx <= l) return 0;
        if (l <= lx && rx <= r) return tree[x];
        push(x, lx, rx);
        int mid = (lx + rx) / 2;
        return sum(l, r, 2 * x, lx, mid) + sum(l, r, 2 * x + 1, mid, rx);
    }
    void add(int l, int r, long long value) { add(l, r, value, 1, 0, n); }
    long long sum(int l, int r) { return sum(l, r, 1, 0, n); }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    cin >> n >> q;
    vector<long long> arr(n);
    for (int i = 0; i < n; ++i) cin >> arr[i];
    LazySegTree st(n);
    st.build(arr);
    while (q--) {
        int type;
        cin >> type;
        if (type == 1) {
            int l, r;
            long long delta;
            cin >> l >> r >> delta;
            st.add(l - 1, r, delta);
        } else {
            int l, r;
            cin >> l >> r;
            cout << st.sum(l - 1, r) << '\\n';
        }
    }
    return 0;
}
'''

    @staticmethod
    def _dijkstra_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    cin >> n >> m >> s;
    vector<vector<pair<int, long long>>> g(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        long long w;
        cin >> u >> v >> w;
        g[u].push_back({v, w});
        g[v].push_back({u, w});
    }

    const long long INF = (1LL << 62);
    vector<long long> dist(n + 1, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;
    dist[s] = 0;
    pq.push({0, s});

    while (!pq.empty()) {
        pair<long long, int> cur = pq.top();
        pq.pop();
        long long d = cur.first;
        int u = cur.second;
        if (d != dist[u]) continue;
        for (size_t i = 0; i < g[u].size(); ++i) {
            int v = g[u][i].first;
            long long w = g[u][i].second;
            if (dist[v] > d + w) {
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }

    for (int i = 1; i <= n; ++i) {
        cout << (dist[i] == INF ? -1 : dist[i]) << '\\n';
    }
    return 0;
}
'''

    @staticmethod
    def _grid_bfs_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int h, w;
    cin >> h >> w;
    vector<string> grid(h);
    for (auto &row : grid) cin >> row;
    int sx, sy, tx, ty;
    cin >> sx >> sy >> tx >> ty;
    --sx; --sy; --tx; --ty;

    vector<vector<int>> dist(h, vector<int>(w, -1));
    queue<pair<int, int>> q;
    q.push({sx, sy});
    dist[sx][sy] = 0;
    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};

    while (!q.empty()) {
        pair<int, int> cur = q.front();
        q.pop();
        int x = cur.first;
        int y = cur.second;
        for (int dir = 0; dir < 4; ++dir) {
            int nx = x + dx[dir];
            int ny = y + dy[dir];
            if (nx < 0 || nx >= h || ny < 0 || ny >= w) continue;
            if (grid[nx][ny] == '#') continue;
            if (dist[nx][ny] != -1) continue;
            dist[nx][ny] = dist[x][y] + 1;
            q.push({nx, ny});
        }
    }

    cout << dist[tx][ty] << '\\n';
    return 0;
}
'''

    @staticmethod
    def _dsu_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, sz;
    DSU(int n) : p(n + 1), sz(n + 1, 1) {
        iota(p.begin(), p.end(), 0);
    }
    int find(int x) {
        if (p[x] == x) return x;
        return p[x] = find(p[x]);
    }
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (sz[a] < sz[b]) swap(a, b);
        p[b] = a;
        sz[a] += sz[b];
        return true;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    cin >> n >> q;
    DSU dsu(n);
    while (q--) {
        int type, a, b;
        cin >> type >> a >> b;
        if (type == 1) dsu.unite(a, b);
        else cout << (dsu.find(a) == dsu.find(b) ? "YES" : "NO") << '\\n';
    }
    return 0;
}
'''

    @staticmethod
    def _binary_search_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

bool feasible(const vector<long long>& arr, int k, long long limit) {
    int groups = 1;
    long long current = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        if (arr[i] > limit) return false;
        if (current + arr[i] > limit) {
            ++groups;
            current = arr[i];
        } else {
            current += arr[i];
        }
    }
    return groups <= k;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, k;
    cin >> n >> k;
    vector<long long> arr(n);
    long long lo = 0;
    long long hi = 0;
    for (int i = 0; i < n; ++i) {
        cin >> arr[i];
        lo = max(lo, arr[i]);
        hi += arr[i];
    }
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(arr, k, mid)) hi = mid;
        else lo = mid + 1;
    }
    cout << lo << '\\n';
    return 0;
}
'''

    @staticmethod
    def _knapsack_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, W;
    cin >> n >> W;
    vector<long long> dp(W + 1, 0);
    for (int i = 0; i < n; ++i) {
        int weight;
        long long value;
        cin >> weight >> value;
        for (int w = W; w >= weight; --w) {
            dp[w] = max(dp[w], dp[w - weight] + value);
        }
    }
    cout << *max_element(dp.begin(), dp.end()) << '\\n';
    return 0;
}
'''

    @staticmethod
    def _geometry_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

struct Point {
    long double x, y;
};

Point operator-(const Point& a, const Point& b) {
    return {a.x - b.x, a.y - b.y};
}

long double cross(const Point& a, const Point& b) {
    return a.x * b.y - a.y * b.x;
}

long double dot(const Point& a, const Point& b) {
    return a.x * b.x + a.y * b.y;
}

int sgn(long double value) {
    const long double EPS = 1e-12L;
    if (value > EPS) return 1;
    if (value < -EPS) return -1;
    return 0;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Geometry skeleton:
    // 1. Read points, segments, or polygons.
    // 2. Convert the problem into orientation / cross-product / dot-product predicates.
    // 3. Derive intersections, parallelism, perpendicularity, area, or containment from those predicates.

    return 0;
}
'''

    @staticmethod
    def _generic_template() -> str:
        return '''#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 1. Extract constraints.
    // 2. Identify the hidden structure.
    // 3. Choose the algorithm family.
    // 4. Implement and compile-check.

    return 0;
}
'''

