from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from ..catalog import register_transitive_relation
from ..model import (
    AssertionStatus,
    EvidenceStatus,
    Fact,
    FactStatus,
    Goal,
    OperatorFamily,
    Rule,
    SolveResult,
    Symbol,
    WorldState,
)
from ..registry import KernelRegistry
from .base import DomainInstance
from .language_common import (
    normalize_identifier,
    split_statements,
    strip_sentence_punctuation,
)


_LOGIC_RELATIONS = {"SUBCLASS_OF", "INSTANCE_OF", "NOT_INSTANCE_OF"}


@dataclass(frozen=True)
class LanguageLogicProblem:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("language logic text must be a string")
        if not self.text.strip():
            raise ValueError("language logic text cannot be empty")


@dataclass(frozen=True)
class LanguageLogicClaim:
    relation: str
    arguments: tuple[str, str]
    is_goal: bool
    statement: str

    def __post_init__(self) -> None:
        relation = self.relation.strip().upper()
        if relation not in _LOGIC_RELATIONS:
            raise ValueError(f"unsupported language logic relation: {relation}")
        if len(self.arguments) != 2 or any(not item for item in self.arguments):
            raise ValueError("language logic claims require two non-empty arguments")
        if self.is_goal and relation == "SUBCLASS_OF":
            raise ValueError("subclass declarations cannot be proof goals")
        object.__setattr__(self, "relation", relation)


@dataclass(frozen=True)
class LanguageLogicRule:
    antecedents: tuple[str, ...]
    consequent: str
    statement: str

    def __post_init__(self) -> None:
        antecedents = tuple(item for item in self.antecedents if item)
        if not 2 <= len(antecedents) <= 8:
            raise ValueError("language conjunction rules require 2 to 8 antecedents")
        if len(set(antecedents)) != len(antecedents):
            raise ValueError("language conjunction rule antecedents must be unique")
        if not self.consequent:
            raise ValueError("language conjunction rule requires a consequent")
        object.__setattr__(self, "antecedents", antecedents)


@dataclass(frozen=True)
class LanguageLogicParse:
    claims: tuple[LanguageLogicClaim, ...]
    unparsed_statements: tuple[str, ...] = ()
    rules: tuple[LanguageLogicRule, ...] = ()


class LanguageLogicParser:
    """Parse a small Korean/English universal-classification language."""

    @staticmethod
    def looks_like(text: str) -> bool:
        explicit_proof = re.search(
            r"(?:prove|증명)\s*[:：]",
            text,
            flags=re.IGNORECASE,
        )
        universal = re.search(r"\bevery\b|모든", text, flags=re.IGNORECASE)
        typed_goal = re.search(
            r"(?:goal|목표)\s*[:：].*(?:\bis\b|이다|아니다)",
            text,
            flags=re.IGNORECASE,
        )
        explicit_rule = re.search(
            r"(?:rule|규칙)\s*[:：]|\bif\s+(?:something|someone)\b|만약",
            text,
            flags=re.IGNORECASE,
        )
        return bool(explicit_proof or explicit_rule or (universal and typed_goal))

    def parse(self, value: str | LanguageLogicProblem) -> LanguageLogicParse:
        problem = (
            value
            if isinstance(value, LanguageLogicProblem)
            else LanguageLogicProblem(value)
        )
        claims: dict[tuple[str, tuple[str, str], bool], LanguageLogicClaim] = {}
        rules: dict[tuple[tuple[str, ...], str], LanguageLogicRule] = {}
        unparsed: list[str] = []

        for statement in split_statements(problem.text):
            clean = strip_sentence_punctuation(statement)
            goal_match = re.fullmatch(
                r"(?:prove|goal|증명|목표)\s*[:：]\s*(?P<body>.+)",
                clean,
                flags=re.IGNORECASE,
            )
            is_goal = goal_match is not None
            body = goal_match.group("body") if goal_match else clean

            if not is_goal:
                parsed_rule = self._parse_conjunction_rule(body)
                if parsed_rule is not None:
                    raw_antecedents, raw_consequent = parsed_rule
                    antecedents = tuple(
                        normalize_identifier(item) for item in raw_antecedents
                    )
                    consequent = normalize_identifier(raw_consequent)
                    try:
                        rule = LanguageLogicRule(
                            antecedents,
                            consequent,
                            statement,
                        )
                    except ValueError:
                        unparsed.append(statement)
                        continue
                    rules[(rule.antecedents, rule.consequent)] = rule
                    continue

            parsed = None if is_goal else self._parse_universal(body)
            if parsed is None:
                parsed = self._parse_assertion(body)
            if parsed is None or (is_goal and parsed[0] == "SUBCLASS_OF"):
                unparsed.append(statement)
                continue

            relation, raw_arguments = parsed
            arguments = tuple(normalize_identifier(item) for item in raw_arguments)
            if any(not item for item in arguments):
                unparsed.append(statement)
                continue
            claim = LanguageLogicClaim(
                relation,
                (arguments[0], arguments[1]),
                is_goal,
                statement,
            )
            claims[(claim.relation, claim.arguments, claim.is_goal)] = claim

        return LanguageLogicParse(
            claims=tuple(
                claims[key]
                for key in sorted(claims, key=lambda item: (item[2], item[0], item[1]))
            ),
            unparsed_statements=tuple(unparsed),
            rules=tuple(
                rules[key] for key in sorted(rules, key=lambda item: (item[1], item[0]))
            ),
        )

    @staticmethod
    def _parse_conjunction_rule(
        statement: str,
    ) -> tuple[tuple[str, ...], str] | None:
        explicit = re.fullmatch(
            r"(?:rule|규칙)\s*[:：]\s*(?P<conditions>.+?)\s*(?:->|→)\s*"
            r"(?P<result>.+)",
            statement,
            flags=re.IGNORECASE,
        )
        if explicit:
            conditions = tuple(
                _clean_rule_concept(item)
                for item in re.split(
                    r"\s*(?:&|그리고)\s*|\s+and\s+",
                    explicit.group("conditions"),
                    flags=re.IGNORECASE,
                )
            )
            return conditions, _clean_rule_concept(explicit.group("result"))

        english = re.fullmatch(
            r"if\s+(?:something|someone|an?\s+object)\s+is\s+"
            r"(?P<conditions>.+?)\s*,?\s*then\s+"
            r"(?:it|that\s+thing|that\s+object)\s+is\s+(?P<result>.+)",
            statement,
            flags=re.IGNORECASE,
        )
        if english:
            conditions = tuple(
                _clean_rule_concept(item)
                for item in re.split(
                    r"\s+and\s+",
                    english.group("conditions"),
                    flags=re.IGNORECASE,
                )
            )
            return conditions, _clean_rule_concept(english.group("result"))

        korean = re.fullmatch(
            r"만약\s+(?:어떤\s+)?(?:것|대상)(?:이|가)\s+"
            r"(?P<first>.+?)(?:이고|이며)\s+(?P<second>.+?)(?:이면|라면)"
            r"\s*,?\s*(?:그것(?:은|이|가)?\s*)?"
            r"(?P<result>.+?)(?:입니다|이다|다)",
            statement,
        )
        if korean:
            return (
                (
                    _clean_rule_concept(korean.group("first")),
                    _clean_rule_concept(korean.group("second")),
                ),
                _clean_rule_concept(korean.group("result")),
            )
        return None

    @staticmethod
    def _parse_universal(
        statement: str,
    ) -> tuple[str, tuple[str, str]] | None:
        english = re.fullmatch(
            r"every\s+(?P<child>.+?)\s+is\s+(?:an?\s+)?(?P<parent>.+)",
            statement,
            flags=re.IGNORECASE,
        )
        if english:
            return "SUBCLASS_OF", (english.group("child"), english.group("parent"))

        korean = re.fullmatch(
            r"모든\s+(?P<child>.+?)(?:은|는)\s+"
            r"(?P<parent>.+?)(?:입니다|이다|다)",
            statement,
        )
        if korean:
            return "SUBCLASS_OF", (korean.group("child"), korean.group("parent"))
        return None

    @staticmethod
    def _parse_assertion(
        statement: str,
    ) -> tuple[str, tuple[str, str]] | None:
        english_negative = re.fullmatch(
            r"(?P<subject>.+?)\s+is\s+not\s+(?:an?\s+)?(?P<concept>.+)",
            statement,
            flags=re.IGNORECASE,
        )
        if english_negative:
            return "NOT_INSTANCE_OF", (
                english_negative.group("subject"),
                english_negative.group("concept"),
            )

        korean_negative = re.fullmatch(
            r"(?P<subject>.+?)(?:은|는)\s+"
            r"(?P<concept>.+?)(?:이|가)?\s*(?:아닙니다|아니다)",
            statement,
        )
        if korean_negative:
            return "NOT_INSTANCE_OF", (
                korean_negative.group("subject"),
                korean_negative.group("concept"),
            )

        english_positive = re.fullmatch(
            r"(?P<subject>.+?)\s+is\s+(?:an?\s+)?(?P<concept>.+)",
            statement,
            flags=re.IGNORECASE,
        )
        if english_positive:
            return "INSTANCE_OF", (
                english_positive.group("subject"),
                english_positive.group("concept"),
            )

        korean_positive = re.fullmatch(
            r"(?P<subject>.+?)(?:은|는)\s+"
            r"(?P<concept>.+?)(?:입니다|이다|다)",
            statement,
        )
        if korean_positive:
            return "INSTANCE_OF", (
                korean_positive.group("subject"),
                korean_positive.group("concept"),
            )
        return None


class LanguageLogicAdapter:
    """Compile explicit class facts and universal rules into typed Horn operators."""

    def __init__(self, parser: LanguageLogicParser | None = None) -> None:
        self.parser = parser or LanguageLogicParser()

    def adapt(self, value: str | LanguageLogicProblem) -> DomainInstance:
        problem = (
            value
            if isinstance(value, LanguageLogicProblem)
            else LanguageLogicProblem(value)
        )
        parsed = self.parser.parse(problem)
        registry = self._create_registry()
        individual_symbols: dict[str, Symbol] = {}
        concept_symbols: dict[str, Symbol] = {}

        def individual(name: str) -> Symbol:
            if name not in individual_symbols:
                individual_symbols[name] = registry.symbol(name, "Individual")
            return individual_symbols[name]

        def concept(name: str) -> Symbol:
            if name not in concept_symbols:
                concept_symbols[name] = registry.symbol(name, "Concept")
            return concept_symbols[name]

        positive = {
            claim.arguments
            for claim in parsed.claims
            if not claim.is_goal and claim.relation == "INSTANCE_OF"
        }
        negative = {
            claim.arguments
            for claim in parsed.claims
            if not claim.is_goal and claim.relation == "NOT_INSTANCE_OF"
        }
        contradictions = tuple(sorted(positive & negative))
        contradiction_set = set(contradictions)
        facts: list[Fact] = []
        goals: list[Goal] = []

        for claim in parsed.claims:
            left, right = claim.arguments
            if claim.relation == "SUBCLASS_OF":
                atom = registry.atom("SUBCLASS_OF", concept(left), concept(right))
            else:
                atom = registry.atom(
                    claim.relation,
                    individual(left),
                    concept(right),
                )
            if claim.is_goal:
                goals.append(Goal(atom, label=claim.statement))
                continue
            status = (
                FactStatus.CONTRADICTED
                if claim.relation in {"INSTANCE_OF", "NOT_INSTANCE_OF"}
                and claim.arguments in contradiction_set
                else FactStatus.OBSERVED
            )
            facts.append(
                Fact(
                    atom,
                    status,
                    source=(
                        "language_logic_conflict"
                        if status is FactStatus.CONTRADICTED
                        else "language_logic_parser"
                    ),
                    assertion_status=AssertionStatus.EXPLICIT,
                    evidence_status=EvidenceStatus.UNVERIFIED,
                )
            )

        for index, rule in enumerate(parsed.rules):
            item = registry.variable("item", "Individual")
            consequent = concept(rule.consequent)
            guard_name = f"conjunction_not_explicitly_negated_{index:03d}"
            registry.register_guard(
                guard_name,
                _conjunction_not_negated_guard(registry, consequent),
            )
            registry.register_operator(
                Rule(
                    name=f"classify_conjunction_{index:03d}",
                    parameters=(item,),
                    preconditions=tuple(
                        registry.atom("INSTANCE_OF", item, concept(antecedent))
                        for antecedent in rule.antecedents
                    ),
                    effects=(registry.atom("INSTANCE_OF", item, consequent),),
                    guards=(guard_name,),
                    description_ko=(
                        f"대상 {{item}}이 {', '.join(rule.antecedents)} 조건을 모두 "
                        f"만족하므로 {rule.consequent}로 분류한다."
                    ),
                ),
                family=OperatorFamily.COMPOSE.value,
                tags=("language", "logic", "conjunction"),
            )

        return DomainInstance(
            registry=registry,
            state=WorldState(tuple(facts)),
            goals=tuple(goals),
            domain="language",
            metadata={
                "source": "controlled_language_logic",
                "input_kind": "logic_text",
                "text": problem.text,
                "claims": tuple(asdict(claim) for claim in parsed.claims),
                "rules": tuple(asdict(rule) for rule in parsed.rules),
                "contradictions": contradictions,
                "unparsed_statements": parsed.unparsed_statements,
                "reviewed_examples": 0,
            },
        )

    @staticmethod
    def _create_registry() -> KernelRegistry:
        registry = KernelRegistry()
        entity = registry.types.register("Entity")
        individual = registry.types.register("Individual", entity)
        concept = registry.types.register("Concept", entity)
        registry.register_predicate("INSTANCE_OF", (individual, concept))
        registry.register_predicate("NOT_INSTANCE_OF", (individual, concept))
        registry.register_predicate("SUBCLASS_OF", (concept, concept))
        register_transitive_relation(
            registry,
            "SUBCLASS_OF",
            concept,
            operator_name="subclass_transitivity",
            tags=("language", "logic"),
            description_ko="개념 {x}에서 {y}, {y}에서 {z}로 이어지는 포함 관계를 합성한다.",
        )
        item = registry.variable("item", individual)
        child = registry.variable("child", concept)
        parent = registry.variable("parent", concept)
        registry.register_guard(
            "classification_not_explicitly_negated",
            lambda binding, state: not state.contains(
                registry.atom(
                    "NOT_INSTANCE_OF",
                    binding["item"],
                    binding["parent"],
                ),
                proof_eligible=False,
            ),
        )
        registry.register_operator(
            Rule(
                name="inherit_instance_through_subclass",
                parameters=(item, child, parent),
                preconditions=(
                    registry.atom("INSTANCE_OF", item, child),
                    registry.atom("SUBCLASS_OF", child, parent),
                ),
                effects=(registry.atom("INSTANCE_OF", item, parent),),
                guards=("classification_not_explicitly_negated",),
                description_ko="대상 {item}의 개념 {child}와 상위 개념 {parent}를 이어 분류한다.",
            ),
            family=OperatorFamily.COMPOSE.value,
            tags=("language", "logic", "inheritance"),
        )
        return registry

    @staticmethod
    def project(_value: str | LanguageLogicProblem, _result: SolveResult) -> bool:
        return False


def _clean_rule_concept(value: str) -> str:
    cleaned = strip_sentence_punctuation(value).strip()
    cleaned = re.sub(
        r"^(?:it\s+is\s+|(?:an?|the)\s+)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _conjunction_not_negated_guard(registry: KernelRegistry, consequent: Symbol):
    def verify(binding, state: WorldState) -> bool:
        return not state.contains(
            registry.atom("NOT_INSTANCE_OF", binding["item"], consequent),
            proof_eligible=False,
        )

    return verify
