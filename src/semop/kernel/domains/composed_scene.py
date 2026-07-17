from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re

from ..composition import CompositionComponent, compose_domain_instances
from ..model import Fact, FactStatus, Goal, OperatorFamily, Rule, SolveResult, WorldState
from .base import DomainInstance
from .language_common import (
    normalize_identifier,
    split_statements,
    strip_sentence_punctuation,
)
from .language_logic import LanguageLogicAdapter
from .raster_vision import (
    RasterImage,
    RasterVisionAdapter,
    RasterVisionConfig,
    RasterVisionProblem,
)


class SceneThresholdError(ValueError):
    def __init__(self, message: str, line: int = 1, column: int = 1) -> None:
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"<scene-threshold>:{line}:{column}: {message}")


@dataclass(frozen=True)
class SceneThresholdProblem:
    image: RasterImage
    text: str
    config: RasterVisionConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, RasterImage):
            raise TypeError("scene threshold problem requires a RasterImage")
        if not isinstance(self.text, str):
            raise TypeError("scene threshold text must be a string")
        if not self.text.strip():
            raise ValueError("scene threshold text cannot be empty")


@dataclass(frozen=True)
class SceneCountCondition:
    selector: str
    comparator: str
    threshold: Fraction

    def __post_init__(self) -> None:
        if not self.selector:
            raise ValueError("scene count selector cannot be empty")
        if self.comparator not in {"<", "<=", ">", ">=", "==", "!="}:
            raise ValueError(f"unsupported scene comparator: {self.comparator}")
        if not isinstance(self.threshold, Fraction):
            raise TypeError("scene count threshold must be a Fraction")


@dataclass(frozen=True)
class SceneThresholdParse:
    conditions: tuple[SceneCountCondition, ...]
    rule_property: str
    goal_property: str
    context_statements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", tuple(self.conditions))
        if not 1 <= len(self.conditions) <= 8:
            raise ValueError("scene rule requires between one and eight conditions")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("scene rule contains duplicate count conditions")

    @property
    def selector(self) -> str:
        return self.conditions[0].selector

    @property
    def comparator(self) -> str:
        return self.conditions[0].comparator

    @property
    def threshold(self) -> Fraction:
        return self.conditions[0].threshold


class SceneThresholdParser:
    """Parse a controlled language rule joining visual counts to a scene class."""

    _ENGLISH_COMPARATORS = {
        "greater than": ">",
        "at least": ">=",
        "less than": "<",
        "at most": "<=",
        "equal to": "==",
        "not equal to": "!=",
    }
    _KOREAN_COMPARATORS = {
        "크면": ">",
        "크고": ">",
        "크거나 같으면": ">=",
        "크거나 같고": ">=",
        "작으면": "<",
        "작고": "<",
        "작거나 같으면": "<=",
        "작거나 같고": "<=",
        "같으면": "==",
        "같고": "==",
        "다르면": "!=",
        "다르고": "!=",
    }
    _KOREAN_SELECTORS = {
        "모든": "all",
        "전체": "all",
        "빨간": "red",
        "빨강": "red",
        "적색": "red",
        "파란": "blue",
        "파랑": "blue",
        "청색": "blue",
        "초록": "green",
        "녹색": "green",
        "노란": "yellow",
        "노랑": "yellow",
        "검은": "black",
        "검정": "black",
        "흰": "white",
        "하얀": "white",
        "흰색": "white",
    }

    def parse(self, value: str | SceneThresholdProblem) -> SceneThresholdParse:
        text = value.text if isinstance(value, SceneThresholdProblem) else value
        if not isinstance(text, str):
            raise TypeError("scene threshold text must be a string")
        statements = split_statements(text)
        rule = None
        goal_property = None
        context: list[str] = []
        cursor = 0
        for statement in statements:
            offset = text.find(statement, cursor)
            cursor = max(cursor, offset + len(statement))
            clean = strip_sentence_punctuation(statement)
            parsed_rule = self._parse_rule(clean)
            if parsed_rule is not None:
                if rule is not None:
                    line, column = _line_column(text, max(0, offset))
                    raise SceneThresholdError(
                        "scene question contains more than one count rule",
                        line,
                        column,
                    )
                rule = parsed_rule
                continue
            parsed_goal = self._parse_goal(clean)
            if parsed_goal is not None:
                if goal_property is not None:
                    line, column = _line_column(text, max(0, offset))
                    raise SceneThresholdError(
                        "scene question contains more than one proof goal",
                        line,
                        column,
                    )
                goal_property = parsed_goal
                continue
            if _looks_like_proof_directive(clean):
                line, column = _line_column(text, max(0, offset))
                raise SceneThresholdError(
                    "only scene classification proof goals are supported",
                    line,
                    column,
                )
            context.append(statement)
        if rule is None:
            raise SceneThresholdError("scene question requires one count threshold rule")
        if goal_property is None:
            raise SceneThresholdError("scene question requires Prove:/증명: goal")
        conditions, rule_property = rule
        return SceneThresholdParse(
            conditions=conditions,
            rule_property=rule_property,
            goal_property=goal_property,
            context_statements=tuple(context),
        )

    def _parse_rule(
        self,
        statement: str,
    ) -> tuple[tuple[SceneCountCondition, ...], str] | None:
        english = re.fullmatch(
            r"if\s+(?P<conditions>.+?)\s*,?\s*(?:then\s+)?"
            r"(?:the\s+)?scene\s+is\s+(?:an?\s+)?(?P<property>.+)",
            statement,
            flags=re.IGNORECASE,
        )
        if english:
            return (
                self._parse_english_conditions(english.group("conditions")),
                _normalize_scene_property(english.group("property")),
            )

        korean = re.fullmatch(
            r"(?P<conditions>.+?물체의\s*개수가.+?)\s*,?\s*"
            r"장면(?:은|이)\s+(?P<property>.+)",
            statement,
        )
        if korean:
            return (
                self._parse_korean_conditions(korean.group("conditions")),
                _normalize_scene_property(korean.group("property")),
            )
        return None

    def _parse_english_conditions(
        self,
        text: str,
    ) -> tuple[SceneCountCondition, ...]:
        parts = re.split(
            r"\s+and\s+(?=(?:the\s+)?(?:number|count)\s+of\s+)",
            text.strip(),
            flags=re.IGNORECASE,
        )
        conditions: list[SceneCountCondition] = []
        pattern = re.compile(
            r"(?:the\s+)?(?:number|count)\s+of\s+(?P<selector>.+?)\s+"
            r"objects?\s+is\s+(?P<comparator>greater\s+than|at\s+least|"
            r"less\s+than|at\s+most|equal\s+to|not\s+equal\s+to)\s+"
            r"(?P<threshold>[+-]?\d+(?:/\d+)?)",
            flags=re.IGNORECASE,
        )
        for part in parts:
            match = pattern.fullmatch(part.strip())
            if match is None:
                raise SceneThresholdError(
                    f"invalid English count condition: {part.strip()!r}"
                )
            comparator_key = re.sub(
                r"\s+", " ", match.group("comparator").strip().lower()
            )
            conditions.append(
                SceneCountCondition(
                    _normalize_scene_selector(match.group("selector")),
                    self._ENGLISH_COMPARATORS[comparator_key],
                    _parse_threshold(match.group("threshold")),
                )
            )
        return _validate_scene_conditions(conditions)

    def _parse_korean_conditions(
        self,
        text: str,
    ) -> tuple[SceneCountCondition, ...]:
        pattern = re.compile(
            r"\s*(?P<selector>.+?)\s*물체의\s*개수가\s*"
            r"(?P<threshold>[+-]?\d+(?:/\d+)?)\s*보다\s*"
            r"(?P<comparator>크거나\s*같으면|크거나\s*같고|"
            r"작거나\s*같으면|작거나\s*같고|크면|크고|작으면|작고|"
            r"같으면|같고|다르면|다르고)"
        )
        conditions: list[SceneCountCondition] = []
        position = 0
        while position < len(text):
            match = pattern.match(text, position)
            if match is None:
                raise SceneThresholdError(
                    f"invalid Korean count condition near {text[position:].strip()!r}"
                )
            comparator_key = re.sub(r"\s+", " ", match.group("comparator"))
            selector_text = match.group("selector").strip()
            conditions.append(
                SceneCountCondition(
                    self._KOREAN_SELECTORS.get(
                        selector_text,
                        _normalize_scene_selector(selector_text),
                    ),
                    self._KOREAN_COMPARATORS[comparator_key],
                    _parse_threshold(match.group("threshold")),
                )
            )
            position = match.end()
            terminal = comparator_key.endswith("면")
            remainder = text[position:].strip()
            if terminal:
                if remainder:
                    raise SceneThresholdError(
                        "a terminal Korean count condition must be last"
                    )
                break
            if not remainder:
                raise SceneThresholdError(
                    "the final Korean count condition must end with '면'"
                )
            connector = re.match(r"\s*(?:그리고\s+)?", text[position:])
            position += connector.end() if connector is not None else 0
        return _validate_scene_conditions(conditions)

    @staticmethod
    def _parse_goal(statement: str) -> str | None:
        english = re.fullmatch(
            r"(?:prove|goal)\s*[:：]\s*(?:the\s+)?scene\s+is\s+"
            r"(?:an?\s+)?(?P<property>.+)",
            statement,
            flags=re.IGNORECASE,
        )
        if english:
            return _normalize_scene_property(english.group("property"))
        korean = re.fullmatch(
            r"(?:증명|목표)\s*[:：]\s*장면(?:은|이)\s+(?P<property>.+)",
            statement,
        )
        if korean:
            return _normalize_scene_property(korean.group("property"))
        return None


class SceneThresholdAdapter:
    """Build one vision -> math -> language proof over an immutable raster."""

    def __init__(
        self,
        *,
        parser: SceneThresholdParser | None = None,
        raster: RasterVisionAdapter | None = None,
        language: LanguageLogicAdapter | None = None,
    ) -> None:
        self.parser = parser or SceneThresholdParser()
        self.raster = raster or RasterVisionAdapter()
        self.language = language or LanguageLogicAdapter()

    def adapt(self, value: SceneThresholdProblem) -> DomainInstance:
        if not isinstance(value, SceneThresholdProblem):
            raise TypeError("composed scene payload must be a SceneThresholdProblem")
        parsed = self.parser.parse(value)
        vision = self.raster.adapt(
            RasterVisionProblem(
                value.image,
                query=value.text,
                config=value.config,
                count_selectors=tuple(
                    dict.fromkeys(
                        condition.selector for condition in parsed.conditions
                    )
                ),
            )
        )
        context = ". ".join(parsed.context_statements)
        language_text = (
            f"{context}. " if context else ""
        ) + f"Prove: Scene is {parsed.goal_property}."
        language = self.language.adapt(language_text)
        if len(language.goals) != 1:
            raise SceneThresholdError(
                "scene language context must produce exactly one proof goal"
            )
        composition = compose_domain_instances(
            (
                CompositionComponent(
                    vision,
                    alias="vision",
                    include_goals=False,
                ),
                CompositionComponent(
                    language,
                    alias="language",
                    include_goals=True,
                ),
            ),
            domain="composed",
        )
        registry = composition.instance.registry
        registry.types.ensure("Condition", "Entity")
        registry.types.ensure("Comparator", "Entity")
        registry.register_predicate(
            "SCENE_COUNT_RULE",
            ("Condition", "Color", "Comparator", "Number", "Concept"),
        )
        registry.register_predicate(
            "COUNT_CONDITION_MET",
            ("Condition", "Concept"),
        )

        rule_property = registry.symbol(parsed.rule_property, "Concept")
        rule_atoms = []
        condition_atoms = []
        multiple = len(parsed.conditions) > 1
        for index, condition in enumerate(parsed.conditions, start=1):
            suffix = f"_{index:03d}" if multiple else ""
            condition_id = registry.symbol(f"condition_{index:03d}", "Condition")
            scope = registry.symbol(condition.selector, "Color")
            comparator = registry.symbol(condition.comparator, "Comparator")
            threshold = registry.symbol(str(condition.threshold), "Number")
            rule_atom = registry.atom(
                "SCENE_COUNT_RULE",
                condition_id,
                scope,
                comparator,
                threshold,
                rule_property,
            )
            condition_atom = registry.atom(
                "COUNT_CONDITION_MET",
                condition_id,
                rule_property,
            )
            count_name = f"count_{index:03d}" if multiple else "count"
            count = registry.variable(count_name, "Number")
            guard_name = f"verify_scene_count_threshold{suffix}"
            registry.register_guard(
                guard_name,
                _scene_count_guard(
                    condition.comparator,
                    condition.threshold,
                    count_name,
                ),
            )
            registry.register_operator(
                Rule(
                    name=f"compare_scene_object_count{suffix}",
                    parameters=(count,),
                    preconditions=(
                        registry.atom("OBJECT_COUNT", scope, count),
                        rule_atom,
                    ),
                    effects=(condition_atom,),
                    guards=(guard_name,),
                    description_ko=(
                        f"장면의 {condition.selector} 물체 개수 "
                        f"{{{count_name}}}를 기준 {condition.threshold}와 "
                        f"비교해 {parsed.rule_property}의 {index}번 조건을 검산한다."
                    ),
                ),
                family=OperatorFamily.COMPARE.value,
                tags=(
                    "math",
                    "vision",
                    "cross_domain",
                    "count_threshold",
                ),
            )
            rule_atoms.append(rule_atom)
            condition_atoms.append(condition_atom)

        language_goal = composition.instance.goals[0]
        scene = language_goal.atom.arguments[0]
        conclusion = registry.atom("INSTANCE_OF", scene, rule_property)
        registry.register_guard(
            "scene_property_not_explicitly_negated",
            _scene_property_not_negated_guard(registry, conclusion),
        )
        registry.register_operator(
            Rule(
                name=(
                    "classify_scene_from_verified_counts"
                    if multiple
                    else "classify_scene_from_verified_count"
                ),
                parameters=(),
                preconditions=tuple(condition_atoms),
                effects=(conclusion,),
                guards=("scene_property_not_explicitly_negated",),
                description_ko=(
                    f"검증된 개수 조건 {len(condition_atoms)}개에 따라 장면의 분류를 "
                    f"{parsed.rule_property} 상태로 정한다."
                ),
            ),
            family=OperatorFamily.COMPOSE.value,
            tags=(
                "language",
                "vision",
                "cross_domain",
                "classification",
                "conjunction",
            ),
        )
        goal = Goal(language_goal.atom, label=value.text.strip())
        return DomainInstance(
            registry=registry,
            state=WorldState(
                composition.instance.state.facts
                + tuple(
                    Fact(atom, FactStatus.OBSERVED, "scene_rule_parser")
                    for atom in rule_atoms
                )
            ),
            goals=(goal,),
            domain="composed",
            metadata={
                **composition.instance.metadata,
                "source": "composed_scene_threshold",
                "input_kind": "scene_threshold",
                "text": value.text,
                "parsed_rule": _serialize_scene_parse(parsed),
                "condition_count": len(parsed.conditions),
                "vision": vision.metadata,
                "language": language.metadata,
                "operator_domains": ("vision", "math", "language"),
                "reviewed_examples": 0,
            },
        )

    @staticmethod
    def project(_value: SceneThresholdProblem, _result: SolveResult) -> bool:
        return False


def _normalize_scene_selector(value: str) -> str:
    normalized = normalize_identifier(value)
    normalized = re.sub(r"^the_", "", normalized)
    normalized = re.sub(r"_(?:object|objects|물체)$", "", normalized)
    return "all" if normalized in {"all", "every", "모든", "전체"} else normalized


def _normalize_scene_property(value: str) -> str:
    cleaned = strip_sentence_punctuation(value).strip()
    cleaned = re.sub(r"(?:입니다|이다|하다|다)$", "", cleaned).strip()
    return normalize_identifier(cleaned)


def _parse_threshold(value: str) -> Fraction:
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise SceneThresholdError(f"invalid threshold {value!r}") from exc


def _validate_scene_conditions(
    conditions: list[SceneCountCondition],
) -> tuple[SceneCountCondition, ...]:
    result = tuple(conditions)
    if not result:
        raise SceneThresholdError("scene rule requires at least one count condition")
    if len(result) > 8:
        raise SceneThresholdError("scene rule supports at most eight count conditions")
    if len(set(result)) != len(result):
        raise SceneThresholdError("scene rule contains duplicate count conditions")
    return result


def _serialize_scene_parse(parsed: SceneThresholdParse) -> dict[str, object]:
    return {
        "conditions": tuple(
            {
                "selector": condition.selector,
                "comparator": condition.comparator,
                "threshold": str(condition.threshold),
            }
            for condition in parsed.conditions
        ),
        "selector": parsed.selector,
        "comparator": parsed.comparator,
        "threshold": str(parsed.threshold),
        "rule_property": parsed.rule_property,
        "goal_property": parsed.goal_property,
        "context_statements": parsed.context_statements,
    }


def _scene_count_guard(
    comparator: str,
    threshold: Fraction,
    binding_name: str = "count",
):
    def verify(binding, _state: WorldState) -> bool:
        try:
            count = Fraction(str(binding[binding_name]))
        except (KeyError, ValueError, ZeroDivisionError):
            return False
        if comparator == ">":
            return count > threshold
        if comparator == ">=":
            return count >= threshold
        if comparator == "<":
            return count < threshold
        if comparator == "<=":
            return count <= threshold
        if comparator == "==":
            return count == threshold
        if comparator == "!=":
            return count != threshold
        return False

    return verify


def _scene_property_not_negated_guard(registry, conclusion):
    def verify(_binding, state: WorldState) -> bool:
        negative = registry.atom(
            "NOT_INSTANCE_OF",
            conclusion.arguments[0],
            conclusion.arguments[1],
        )
        return not state.contains(negative, proof_eligible=False)

    return verify


def _line_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    previous = text.rfind("\n", 0, offset)
    return line, offset - previous


def _looks_like_proof_directive(statement: str) -> bool:
    return bool(
        re.match(r"^(?:prove|goal)\s*[:\uff1a]", statement, re.IGNORECASE)
        or re.match(
            r"^(?:\uc99d\uba85|\ubaa9\ud45c)\s*[:\uff1a]",
            statement,
        )
    )
