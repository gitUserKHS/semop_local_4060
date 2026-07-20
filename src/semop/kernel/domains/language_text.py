from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import TYPE_CHECKING

from ..catalog import register_all_requirements_ready
from ..grounding import (
    GroundingAuthority,
    GroundingDisposition,
    GroundingTrace,
    grounding_payload_digest,
    make_grounding_record,
)
from ..model import (
    AssertionStatus,
    Atom,
    EvidenceStatus,
    Fact,
    FactStatus,
    Goal,
    SolveResult,
    Symbol,
    WorldState,
)
from .base import DomainInstance
from .hidden_premise import create_hidden_premise_registry
from .language_common import (
    normalize_identifier as _normalize_slot,
    split_items as _split_items,
    split_statements as _split_statements,
    strip_sentence_punctuation as _strip_sentence_punctuation,
)
from .language_logic import (
    LanguageLogicAdapter,
    LanguageLogicParser,
    LanguageLogicProblem,
)


if TYPE_CHECKING:
    from semop.structures import StructuredMeaningGraph

    from ..adapters import StructuredMeaningGraphAdapter


_CLAIM_ARITY = {
    "GOAL": 1,
    "REQUIRES": 2,
    "SATISFIED": 1,
    "BLOCKED": 1,
}


def _status_claims(
    relation: str,
    value: str,
) -> list[tuple[str, tuple[str, ...]]]:
    return [(relation, (item,)) for item in _split_items(value)]


@dataclass(frozen=True)
class LanguageClaim:
    relation: str
    arguments: tuple[str, ...]
    verified: bool
    confidence: float
    source: str
    statement: str

    def __post_init__(self) -> None:
        relation = self.relation.strip().upper()
        expected = _CLAIM_ARITY.get(relation)
        if expected is None:
            raise ValueError(f"unsupported language claim: {relation}")
        if len(self.arguments) != expected:
            raise ValueError(
                f"{relation} requires {expected} arguments, got {len(self.arguments)}"
            )
        if any(not argument.strip() for argument in self.arguments):
            raise ValueError("language claim arguments cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("language claim confidence must be between 0 and 1")
        object.__setattr__(self, "relation", relation)

    @property
    def parser_verified(self) -> bool:
        """Whether a controlled parser rule matched, not whether reality was checked."""

        return self.verified


@dataclass(frozen=True)
class LanguageTextProblem:
    text: str
    source_context: str = ""
    use_legacy_heuristics: bool = True

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("language text cannot be empty")


@dataclass(frozen=True)
class LanguageParse:
    claims: tuple[LanguageClaim, ...]
    unparsed_statements: tuple[str, ...] = ()
    heuristic_used: bool = False
    warnings: tuple[str, ...] = ()


class LanguageTextParser:
    """High-precision Korean/English premise parser with proposed fallback."""

    def parse(self, value: str | LanguageTextProblem) -> LanguageParse:
        problem = value if isinstance(value, LanguageTextProblem) else LanguageTextProblem(value)
        claims: dict[tuple[str, tuple[str, ...]], LanguageClaim] = {}
        unparsed: list[str] = []
        warnings: list[str] = []
        current_goal = ""
        pending_requirements: list[tuple[str, str]] = []

        def add_claim(
            relation: str,
            arguments: tuple[str, ...],
            *,
            verified: bool,
            confidence: float,
            source: str,
            statement: str,
        ) -> None:
            normalized = tuple(_normalize_slot(argument) for argument in arguments)
            if any(not argument for argument in normalized):
                return
            claim = LanguageClaim(
                relation,
                normalized,
                verified,
                confidence,
                source,
                statement,
            )
            key = (claim.relation, claim.arguments)
            previous = claims.get(key)
            if previous is None or (claim.verified, claim.confidence) > (
                previous.verified,
                previous.confidence,
            ):
                claims[key] = claim

        for statement in _split_statements(problem.text):
            clean = _strip_sentence_punctuation(statement)
            label = re.fullmatch(
                r"(?P<label>goal|목표|requires?|필요|전제|satisfied|met|"
                r"충족|완료|blocked|missing|차단|누락)\s*[:：]\s*(?P<value>.+)",
                clean,
                flags=re.IGNORECASE,
            )
            if label:
                kind = label.group("label").lower()
                values = _split_items(label.group("value"))
                if kind in {"goal", "목표"}:
                    for item in values:
                        current_goal = _normalize_slot(item)
                        add_claim(
                            "GOAL",
                            (current_goal,),
                            verified=True,
                            confidence=1.0,
                            source="explicit_label",
                            statement=statement,
                        )
                    if current_goal and pending_requirements:
                        for premise, original in pending_requirements:
                            add_claim(
                                "REQUIRES",
                                (current_goal, premise),
                                verified=True,
                                confidence=1.0,
                                source="explicit_label",
                                statement=original,
                            )
                        pending_requirements.clear()
                elif kind in {"require", "requires", "필요", "전제"}:
                    for item in values:
                        premise = _normalize_slot(item)
                        if current_goal:
                            add_claim(
                                "REQUIRES",
                                (current_goal, premise),
                                verified=True,
                                confidence=1.0,
                                source="explicit_label",
                                statement=statement,
                            )
                        else:
                            pending_requirements.append((premise, statement))
                elif kind in {"satisfied", "met", "충족", "완료"}:
                    for item in values:
                        add_claim(
                            "SATISFIED",
                            (item,),
                            verified=True,
                            confidence=1.0,
                            source="explicit_label",
                            statement=statement,
                        )
                else:
                    for item in values:
                        add_claim(
                            "BLOCKED",
                            (item,),
                            verified=True,
                            confidence=1.0,
                            source="explicit_label",
                            statement=statement,
                        )
                continue

            parsed, parsed_goal = self._parse_sentence(clean)
            if parsed:
                if parsed_goal:
                    current_goal = parsed_goal
                for relation, arguments in parsed:
                    add_claim(
                        relation,
                        arguments,
                        verified=True,
                        confidence=0.95,
                        source="explicit_sentence",
                        statement=statement,
                    )
                continue
            unparsed.append(statement)

        has_goal = any(
            claim.verified and claim.relation == "GOAL" for claim in claims.values()
        )
        has_requirement = any(
            claim.verified and claim.relation == "REQUIRES"
            for claim in claims.values()
        )
        heuristic_used = False
        if problem.use_legacy_heuristics and (not has_goal or not has_requirement):
            heuristic_used = True
            try:
                from semop.pipeline import StructuredMeaningPipeline

                graph = StructuredMeaningPipeline(
                    operator_backend="legacy"
                ).run(problem.text, source_context=problem.source_context)
                heuristic_goals = tuple(
                    _normalize_slot(goal) for goal in graph.hidden_goals if goal
                )
                for goal in heuristic_goals:
                    add_claim(
                        "GOAL",
                        (goal,),
                        verified=False,
                        confidence=0.6,
                        source="legacy_heuristic",
                        statement=problem.text,
                    )
                fallback_goal = heuristic_goals[0] if heuristic_goals else ""
                if fallback_goal:
                    for premise in graph.required_premises:
                        add_claim(
                            "REQUIRES",
                            (fallback_goal, premise),
                            verified=False,
                            confidence=0.6,
                            source="legacy_heuristic",
                            statement=problem.text,
                        )
                for premise in graph.satisfied_premises:
                    add_claim(
                        "SATISFIED",
                        (premise,),
                        verified=False,
                        confidence=0.6,
                        source="legacy_heuristic",
                        statement=problem.text,
                    )
                for premise in graph.missing_premises:
                    add_claim(
                        "BLOCKED",
                        (premise,),
                        verified=False,
                        confidence=0.6,
                        source="legacy_heuristic",
                        statement=problem.text,
                    )
                warnings.extend(graph.warnings)
            except Exception as exc:
                warnings.append(
                    f"legacy heuristic fallback failed: {type(exc).__name__}: {exc}"
                )

        if pending_requirements:
            warnings.append("requirements appeared before any explicit goal")
        return LanguageParse(
            claims=tuple(
                claims[key]
                for key in sorted(claims, key=lambda item: (item[0], item[1]))
            ),
            unparsed_statements=tuple(unparsed),
            heuristic_used=heuristic_used,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _parse_sentence(
        statement: str,
    ) -> tuple[list[tuple[str, tuple[str, ...]]], str]:
        parsed: list[tuple[str, tuple[str, ...]]] = []

        english_requires = re.fullmatch(
            r"(?P<goal>.+?)\s+requires?\s+(?P<premises>.+)",
            statement,
            flags=re.IGNORECASE,
        )
        if english_requires:
            goal = _normalize_slot(english_requires.group("goal"))
            parsed.append(("GOAL", (goal,)))
            parsed.extend(
                ("REQUIRES", (goal, premise))
                for premise in _split_items(english_requires.group("premises"))
            )
            return parsed, goal

        english_to = re.fullmatch(
            r"to\s+(?P<goal>.+?),\s*(?P<premises>.+?)\s+(?:is|are)\s+required",
            statement,
            flags=re.IGNORECASE,
        )
        if english_to:
            goal = _normalize_slot(english_to.group("goal"))
            parsed.append(("GOAL", (goal,)))
            parsed.extend(
                ("REQUIRES", (goal, premise))
                for premise in _split_items(english_to.group("premises"))
            )
            return parsed, goal

        korean_requires = re.fullmatch(
            r"(?P<goal>.+?)(?:하려면|하기\s*위해서는?|하려고\s*하면)\s*"
            r"(?P<premises>.+?)(?:이|가|은|는)?\s+필요(?:하다|해|합니다|해요|함)?",
            statement,
        )
        if korean_requires:
            goal = _normalize_slot(korean_requires.group("goal"))
            parsed.append(("GOAL", (goal,)))
            parsed.extend(
                ("REQUIRES", (goal, premise))
                for premise in _split_items(korean_requires.group("premises"))
            )
            return parsed, goal

        korean_without = re.fullmatch(
            r"(?P<premises>.+?)\s+없이\s+"
            r"(?P<goal>.+?)(?:을|를)?\s+진행하지\s*않는다",
            statement,
        )
        if korean_without:
            goal = _normalize_slot(korean_without.group("goal"))
            parsed.append(("GOAL", (goal,)))
            parsed.extend(
                ("REQUIRES", (goal, premise))
                for premise in _split_items(korean_without.group("premises"))
            )
            return parsed, goal

        korean_contrast = re.fullmatch(
            r"(?P<positive>.+?)(?:이|가|은|는|을|를)?\s*"
            r"(?:충족되었|준비되었|통과했|완료했|확보했)지만\s+"
            r"(?P<negative>.+?)(?:이|가|은|는)?\s*"
            r"(?:충족되지\s*않았다|준비되지\s*않았다|누락되었다|"
            r"막혔다|불가능하다|실패했다)",
            statement,
        )
        if korean_contrast:
            return [
                *_status_claims("SATISFIED", korean_contrast.group("positive")),
                *_status_claims("BLOCKED", korean_contrast.group("negative")),
            ], ""

        english_negative = re.fullmatch(
            r"(?P<premises>.+?)\s+(?:is|are)\s+"
            r"(?:not\s+satisfied|not\s+met|missing|blocked|unavailable|absent)",
            statement,
            flags=re.IGNORECASE,
        )
        if english_negative:
            return _status_claims(
                "BLOCKED", english_negative.group("premises")
            ), ""

        korean_negative = re.fullmatch(
            r"(?P<premises>.+?)(?:이|가|은|는|을|를)?\s*"
            r"(?:충족되지\s*않았다|준비되지\s*않았다|없다|누락되었다|"
            r"막혔다|불가능하다|실패했다)",
            statement,
        )
        if korean_negative:
            return _status_claims(
                "BLOCKED", korean_negative.group("premises")
            ), ""

        english_positive = re.fullmatch(
            r"(?P<premises>.+?)\s+(?:is|are)\s+"
            r"(?:satisfied|met|available|ready|present)",
            statement,
            flags=re.IGNORECASE,
        )
        if english_positive:
            return _status_claims(
                "SATISFIED", english_positive.group("premises")
            ), ""

        korean_positive = re.fullmatch(
            r"(?P<premises>.+?)(?:만)?(?:이|가|은|는|을|를)?\s*"
            r"(?:충족되었다|준비되었다|통과했다|완료되었다|있다|확보되었다|"
            r"충족했다|준비했다|완료했다|확보했다)",
            statement,
        )
        if korean_positive:
            return _status_claims(
                "SATISFIED", korean_positive.group("premises")
            ), ""

        english_question = re.fullmatch(
            r"can\s+(?:(?:we|i|the\s+system|this\s+system)\s+)?(?P<goal>.+)",
            statement,
            flags=re.IGNORECASE,
        )
        if english_question:
            goal = _normalize_slot(english_question.group("goal"))
            return [("GOAL", (goal,))], goal

        english_intent = re.fullmatch(
            r"(?:i|we)\s+(?:want|need)\s+to\s+(?P<goal>.+)",
            statement,
            flags=re.IGNORECASE,
        )
        if english_intent:
            goal = _normalize_slot(english_intent.group("goal"))
            return [("GOAL", (goal,))], goal

        korean_question = re.fullmatch(
            r"(?P<goal>.+?)(?:할|될)\s*수\s*있(?:나|을까|습니까|나요|어)",
            statement,
        )
        if korean_question:
            goal = _normalize_slot(korean_question.group("goal"))
            return [("GOAL", (goal,))], goal

        korean_how = re.fullmatch(
            r"(?P<goal>.+?)하려면\s*어떻게(?:\s*해야\s*(?:해|하나|합니까))?",
            statement,
        )
        if korean_how:
            goal = _normalize_slot(korean_how.group("goal"))
            return [("GOAL", (goal,))], goal

        korean_intent = re.fullmatch(
            r"(?P<goal>.+?)(?:고|하고)\s*싶(?:다|어|습니다)",
            statement,
        )
        if korean_intent:
            goal = _normalize_slot(korean_intent.group("goal"))
            return [("GOAL", (goal,))], goal
        return [], ""


class LanguageTextAdapter:
    """Compile explicit text claims into replayable hidden-premise operators."""

    def __init__(self, parser: LanguageTextParser | None = None) -> None:
        self.parser = parser or LanguageTextParser()

    def adapt(self, value: str | LanguageTextProblem) -> DomainInstance:
        problem = (
            value
            if isinstance(value, LanguageTextProblem)
            else LanguageTextProblem(value)
        )
        parsed = self.parser.parse(problem)
        registry = create_hidden_premise_registry()
        goal_symbols: dict[str, Symbol] = {}
        premise_symbols: dict[str, Symbol] = {}

        def goal_symbol(name: str) -> Symbol:
            if name not in goal_symbols:
                goal_symbols[name] = registry.symbol(name, "ReasoningGoal")
            return goal_symbols[name]

        def premise_symbol(name: str) -> Symbol:
            if name not in premise_symbols:
                premise_symbols[name] = registry.symbol(name, "Premise")
            return premise_symbols[name]

        verified_satisfied = {
            claim.arguments[0]
            for claim in parsed.claims
            if claim.verified and claim.relation == "SATISFIED"
        }
        verified_blocked = {
            claim.arguments[0]
            for claim in parsed.claims
            if claim.verified and claim.relation == "BLOCKED"
        }
        contradictions = tuple(sorted(verified_satisfied & verified_blocked))
        contradiction_set = set(contradictions)
        facts: list[Fact] = []
        grounding_records = []
        seen_facts: set[tuple[Atom, FactStatus]] = set()
        input_digest = grounding_payload_digest(problem.text)

        for claim in parsed.claims:
            if claim.relation == "GOAL":
                atom = registry.atom("GOAL", goal_symbol(claim.arguments[0]))
            elif claim.relation == "REQUIRES":
                atom = registry.atom(
                    "REQUIRES",
                    goal_symbol(claim.arguments[0]),
                    premise_symbol(claim.arguments[1]),
                )
            elif claim.relation == "SATISFIED":
                atom = registry.atom(
                    "SATISFIED", premise_symbol(claim.arguments[0])
                )
            else:
                atom = registry.atom("BLOCKED", premise_symbol(claim.arguments[0]))

            if claim.verified and any(
                argument in contradiction_set for argument in claim.arguments
            ) and claim.relation in {"SATISFIED", "BLOCKED"}:
                status = FactStatus.CONTRADICTED
                source = "language_text_conflict"
            else:
                status = FactStatus.OBSERVED if claim.verified else FactStatus.PROPOSED
                source = f"language_text:{claim.source}"
            key = (atom, status)
            if key in seen_facts:
                continue
            seen_facts.add(key)
            disposition = {
                FactStatus.OBSERVED: GroundingDisposition.OBSERVED,
                FactStatus.PROPOSED: GroundingDisposition.PROPOSED,
                FactStatus.CONTRADICTED: GroundingDisposition.CONTRADICTED,
            }[status]
            authority = (
                GroundingAuthority.EXPLICIT_INPUT
                if claim.verified
                else GroundingAuthority.HEURISTIC_PROPOSAL
            )
            record = make_grounding_record(
                domain="language",
                statement=claim.statement,
                atom=atom,
                producer_id=f"language_text:{claim.source}",
                source=source,
                disposition=disposition,
                authority=authority,
                assertion_status=(
                    AssertionStatus.EXPLICIT
                    if claim.source.startswith("explicit_")
                    else AssertionStatus.INFERRED
                ),
                evidence_status=EvidenceStatus.UNVERIFIED,
                rationale=(
                    "controlled language parser matched an explicit assertion"
                    if claim.verified
                    else "legacy language heuristic proposed a typed assertion"
                ),
                input_digest=input_digest,
                evidence=(f"input:{input_digest}",),
                confidence=claim.confidence,
                sensor_features=(
                    ("parser.controlled_match", float(claim.verified)),
                    ("parser.heuristic_match", float(not claim.verified)),
                    ("claim.arity", len(claim.arguments) / 3.0),
                    ("claim.confidence", claim.confidence),
                ),
            )
            grounding_records.append(record)
            if record.fact is not None:
                facts.append(record.fact)

        observed_goals = tuple(
            dict.fromkeys(
                claim.arguments[0]
                for claim in parsed.claims
                if claim.verified and claim.relation == "GOAL"
            )
        )
        required_by_goal: dict[str, list[str]] = {
            goal: [] for goal in observed_goals
        }
        for claim in parsed.claims:
            if not claim.verified or claim.relation != "REQUIRES":
                continue
            goal, premise = claim.arguments
            if goal in required_by_goal and premise not in required_by_goal[goal]:
                required_by_goal[goal].append(premise)

        goals: list[Goal] = []
        for index, goal_name in enumerate(observed_goals):
            required = tuple(required_by_goal[goal_name])
            if not required:
                continue
            if any(
                premise in verified_blocked and premise not in contradiction_set
                for premise in required
            ):
                goals.append(Goal(registry.atom("NOT_READY", goal_symbol(goal_name))))
            else:
                register_all_requirements_ready(
                    registry,
                    goal_symbol(goal_name),
                    tuple(premise_symbol(premise) for premise in required),
                    operator_name=f"all_text_requirements_ready_{index}",
                )
                goals.append(Goal(registry.atom("READY", goal_symbol(goal_name))))

        grounding_trace = GroundingTrace(tuple(grounding_records))
        return DomainInstance(
            registry=registry,
            state=WorldState(tuple(facts)),
            goals=tuple(goals),
            domain="language",
            metadata={
                "source": "natural_language_text",
                "input_kind": "text",
                "text": problem.text,
                "source_context": problem.source_context,
                "claims": tuple(
                    {
                        **asdict(claim),
                        "parser_verified": claim.parser_verified,
                        "assertion_status": (
                            AssertionStatus.EXPLICIT.value
                            if claim.source.startswith("explicit_")
                            else AssertionStatus.INFERRED.value
                        ),
                        "evidence_status": EvidenceStatus.UNVERIFIED.value,
                    }
                    for claim in parsed.claims
                ),
                "explicit_claim_count": sum(claim.verified for claim in parsed.claims),
                "proposed_claim_count": sum(not claim.verified for claim in parsed.claims),
                "contradictions": contradictions,
                "unparsed_statements": parsed.unparsed_statements,
                "heuristic_used": parsed.heuristic_used,
                "parser_warnings": parsed.warnings,
                "reviewed_examples": 0,
                "grounding": grounding_trace.to_dict(include_records=False),
            },
            grounding_trace=grounding_trace,
        )

    @staticmethod
    def project(_value: str | LanguageTextProblem, _result: SolveResult) -> bool:
        return False


class LanguageInputAdapter:
    """Dispatch structured graphs and raw text through one language boundary."""

    def __init__(
        self,
        *,
        graph: "StructuredMeaningGraphAdapter" | None = None,
        text: LanguageTextAdapter | None = None,
        logic: LanguageLogicAdapter | None = None,
    ) -> None:
        if graph is None:
            # Import lazily because adapters.py imports domains.base while the
            # domains package is still being initialized.
            from ..adapters import StructuredMeaningGraphAdapter

            graph = StructuredMeaningGraphAdapter()
        self.graph = graph
        self.text = text or LanguageTextAdapter()
        self.logic = logic or LanguageLogicAdapter()

    def adapt(
        self,
        value: "StructuredMeaningGraph" | str | LanguageTextProblem | LanguageLogicProblem,
    ) -> DomainInstance:
        from semop.structures import StructuredMeaningGraph

        if isinstance(value, StructuredMeaningGraph):
            return self.graph.adapt(value)
        if isinstance(value, LanguageLogicProblem):
            return self.logic.adapt(value)
        if isinstance(value, str) and LanguageLogicParser.looks_like(value):
            return self.logic.adapt(value)
        if isinstance(value, (str, LanguageTextProblem)):
            return self.text.adapt(value)
        raise TypeError(
            "language payload must be StructuredMeaningGraph, str, "
            "LanguageTextProblem, or LanguageLogicProblem"
        )

    def project(
        self,
        value: "StructuredMeaningGraph" | str | LanguageTextProblem | LanguageLogicProblem,
        result: SolveResult,
    ) -> bool:
        from semop.structures import StructuredMeaningGraph

        if isinstance(value, StructuredMeaningGraph):
            self.graph.project(value, result)
            return True
        if isinstance(value, LanguageLogicProblem):
            return self.logic.project(value, result)
        if isinstance(value, str) and LanguageLogicParser.looks_like(value):
            return self.logic.project(value, result)
        return self.text.project(value, result)
