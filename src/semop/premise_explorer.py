from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .commonsense_kb import concept_label, lookup_concept
from .structures import Edge, GoalPreservationCheck, Node, StructuredMeaningGraph


BOOKING_TOKENS = (
    "\uc608\uc57d",
    "\ubb38\uc758",
    "\ucde8\uc18c",
    "\ud655\uc778",
    "\uc811\uc218",
    "booking",
    "inquiry",
    "cancel",
    "confirm",
    "check in",
)

WALK_TOKENS = ("\uac78\uc5b4", "\ub3c4\ubcf4", "walk")
CLOSED_TOKENS = ("\ub2eb", "closed")
LEFT_VEHICLE_TOKENS = ("\ub9e1\uaca8\ub46c", "\ub9e1\uaca8\ub1a8", "already left", "drop off")
OPEN_STATE_TOKENS = ("\uc5f4\ub9b0", "open")


@dataclass
class HiddenPremiseResult:
    hidden_goals: List[str]
    hidden_assumptions: List[str]
    required_premises: List[str]
    optional_interpretations: List[str]
    goal_preservation_checks: List[GoalPreservationCheck]
    clarification_needed: bool
    audit_trace: List[str]


class HiddenPremiseExplorer:
    def enrich(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        result = self.explore(graph)
        self._apply(graph, result)
        return graph

    def explore(self, graph: StructuredMeaningGraph) -> HiddenPremiseResult:
        lowered = graph.query.lower()
        if 'car_wash' in graph.node_ids():
            return self._car_wash_premises(graph, lowered)
        if {'bag', 'book'}.issubset(graph.node_ids()):
            return self._bag_book_premises(lowered)
        return self._generic_premises(graph)

    def _car_wash_premises(self, graph: StructuredMeaningGraph, lowered: str) -> HiddenPremiseResult:
        booking_like = any(token in lowered for token in BOOKING_TOKENS)
        walking_like = any(token in lowered for token in WALK_TOKENS)
        left_vehicle = any(token in lowered for token in LEFT_VEHICLE_TOKENS)

        if left_vehicle:
            checks: List[GoalPreservationCheck] = []
            if walking_like:
                checks.append(
                    GoalPreservationCheck(
                        action='walk_without_car',
                        hidden_goal='reservation_check_goal',
                        status='valid',
                        rationale='If the car is already at the service site, walking may preserve a follow-up or pickup-related goal.',
                        confidence=0.78,
                    )
                )
            return HiddenPremiseResult(
                hidden_goals=['reservation_check_goal'],
                hidden_assumptions=[
                    'The vehicle may already be at the service site, so the remaining goal can shift to pickup, follow-up, or status confirmation.',
                ],
                required_premises=[],
                optional_interpretations=['clean_car_goal'],
                goal_preservation_checks=checks,
                clarification_needed=False,
                audit_trace=['hidden premise explorer: detected already-left-vehicle counterfactual'],
            )

        if booking_like:
            checks: List[GoalPreservationCheck] = []
            if walking_like:
                checks.append(
                    GoalPreservationCheck(
                        action='walk_without_car',
                        hidden_goal='booking_or_inquiry_goal',
                        status='conditionally_valid',
                        rationale='Walking can preserve a booking or inquiry goal, but not a normal wash-service goal.',
                        confidence=0.74,
                    )
                )
            return HiddenPremiseResult(
                hidden_goals=['booking_or_inquiry_goal'],
                hidden_assumptions=[
                    'The visible destination may be a car wash, but the real goal could be booking, cancellation, or inquiry.',
                    'Contact-oriented service goals can sometimes be satisfied without bringing the vehicle.',
                ],
                required_premises=['goal_clarity'],
                optional_interpretations=['clean_car_goal'],
                goal_preservation_checks=checks,
                clarification_needed=True,
                audit_trace=['hidden premise explorer: detected service-contact interpretation'],
            )

        checks: List[GoalPreservationCheck] = []
        if walking_like:
            checks.append(
                GoalPreservationCheck(
                    action='walk_without_car',
                    hidden_goal='clean_car_goal',
                    status='risk_high',
                    rationale='Walking may achieve arrival, but it usually fails the hidden goal of getting the car washed because the car is not present.',
                    confidence=0.92,
                )
            )
        return HiddenPremiseResult(
            hidden_goals=['clean_car_goal'],
            hidden_assumptions=[
                'A car wash is typically a place where a vehicle is physically brought for washing.',
                'The true goal is usually to get the car cleaned, not merely to arrive at the site.',
                'A normal wash-service script usually requires vehicle presence at the site.',
            ],
            required_premises=['vehicle_present'],
            optional_interpretations=['booking_or_inquiry_goal', 'reservation_check_goal'],
            goal_preservation_checks=checks,
            clarification_needed=True,
            audit_trace=['hidden premise explorer: recovered wash-service goal and vehicle-presence premise'],
        )

    def _bag_book_premises(self, lowered: str) -> HiddenPremiseResult:
        checks: List[GoalPreservationCheck] = []
        explicit_open = any(token in lowered for token in OPEN_STATE_TOKENS)
        if any(token in lowered for token in CLOSED_TOKENS) and not explicit_open:
            checks.append(
                GoalPreservationCheck(
                    action='insert_without_opening',
                    hidden_goal='store_book_in_bag_goal',
                    status='risk_high',
                    rationale='Direct insertion without opening access usually breaks the hidden containment goal.',
                    confidence=0.87,
                )
            )
        return HiddenPremiseResult(
            hidden_goals=['store_book_in_bag_goal'],
            hidden_assumptions=[
                'The real goal is to place the book inside the bag interior, not just near the bag.',
                'Successful containment usually requires open access to the bag interior.',
                'Containment also requires enough available interior space.',
            ],
            required_premises=['open_access', 'available_space'],
            optional_interpretations=['use_other_compartment'],
            goal_preservation_checks=checks,
            clarification_needed=False,
            audit_trace=['hidden premise explorer: recovered containment, access, and space assumptions'],
        )

    def _generic_premises(self, graph: StructuredMeaningGraph) -> HiddenPremiseResult:
        assumptions: List[str] = []
        required: List[str] = []
        hidden_goals: List[str] = []
        for edge in graph.edges:
            if edge.relation == 'REQUIRES':
                required.append(edge.target)
                assumptions.append(f'{concept_label(edge.source)} requires {concept_label(edge.target)}.')
            if edge.relation == 'GOAL_OF':
                hidden_goals.append(edge.source)
        for script in graph.inferred_scripts[:2]:
            assumptions.append(f'The query may depend on the script step: {script.replace("_", " ")}')
        return HiddenPremiseResult(
            hidden_goals=list(dict.fromkeys(hidden_goals)),
            hidden_assumptions=list(dict.fromkeys(assumptions))[:4],
            required_premises=list(dict.fromkeys(required))[:4],
            optional_interpretations=[],
            goal_preservation_checks=[],
            clarification_needed=False,
            audit_trace=['hidden premise explorer: derived generic prerequisite assumptions'] if assumptions or hidden_goals else [],
        )

    def _apply(self, graph: StructuredMeaningGraph, result: HiddenPremiseResult) -> None:
        graph.hidden_goals = self._merge_unique(graph.hidden_goals, result.hidden_goals)
        graph.hidden_assumptions = self._merge_unique(graph.hidden_assumptions, result.hidden_assumptions)
        graph.required_premises = self._merge_unique(graph.required_premises, result.required_premises)
        graph.optional_interpretations = self._merge_unique(graph.optional_interpretations, result.optional_interpretations)
        graph.clarification_needed = graph.clarification_needed or result.clarification_needed
        for check in result.goal_preservation_checks:
            if not any(existing.action == check.action and existing.hidden_goal == check.hidden_goal for existing in graph.goal_preservation_checks):
                graph.goal_preservation_checks.append(check)
        graph.audit_trace.extend(item for item in result.audit_trace if item not in graph.audit_trace)

        for hidden_goal in result.hidden_goals:
            graph.add_node(Node(id=hidden_goal, label=concept_label(hidden_goal), kind='hidden_goal', provenance=['premise:goal']))
        for premise in result.required_premises:
            graph.add_node(Node(id=premise, label=concept_label(premise), kind=lookup_concept(premise).get('kind', 'premise'), provenance=['premise:requirement']))
        for premise in result.required_premises:
            for hidden_goal in result.hidden_goals:
                graph.add_edge(Edge(source=hidden_goal, relation='REQUIRES', target=premise, confidence=0.88, provenance=['premise_explorer:requires']))
        for hidden_goal in result.hidden_goals:
            if 'car_wash' in graph.node_ids():
                graph.add_edge(Edge(source='car_wash', relation='TYPICAL_FOR', target=hidden_goal, confidence=0.82, provenance=['premise_explorer:script']))
            if 'bag' in graph.node_ids():
                graph.add_edge(Edge(source='bag', relation='TYPICAL_FOR', target=hidden_goal, confidence=0.82, provenance=['premise_explorer:script']))

        for item in result.hidden_assumptions:
            warning = f'hidden premise: {item}'
            if warning not in graph.warnings:
                graph.warnings.append(warning)
        if result.clarification_needed:
            warning = 'The surface action may not reveal the real goal. Clarifying the true goal should improve the decision.'
            if warning not in graph.warnings:
                graph.warnings.append(warning)
        for check in result.goal_preservation_checks:
            note = f'goal preservation: {check.action} -> {check.status}'
            if note not in graph.warnings:
                graph.warnings.append(note)
            if check.status == 'risk_high':
                advice = self._invalid_advice_from_check(check)
                if advice and advice not in graph.invalid_advice:
                    graph.invalid_advice.append(advice)

    @staticmethod
    def _invalid_advice_from_check(check: GoalPreservationCheck) -> str:
        if check.action == 'walk_without_car' and check.hidden_goal == 'clean_car_goal':
            return '\ucc28 \uc5c6\uc774 \uac78\uc5b4\uc11c \uc138\ucc28\uc7a5\uae4c\uc9c0 \uac00\ub77c\uace0 \ud558\ub294 \uc870\uc5b8'
        if check.action == 'insert_without_opening':
            return '\uac00\ubc29\uc744 \uc5f4\uc9c0 \uc54a\uace0 \ubc14\ub85c \ucc45\uc744 \ub123\uc73c\ub77c\uace0 \ud558\ub294 \uc870\uc5b8'
        return ''

    @staticmethod
    def _merge_unique(left: Iterable[str], right: Iterable[str]) -> List[str]:
        return list(dict.fromkeys(list(left) + list(right)))
