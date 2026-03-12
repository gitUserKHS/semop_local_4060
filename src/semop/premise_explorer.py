from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, TYPE_CHECKING

from .commonsense_kb import concept_label, lookup_concept, match_concepts, scripts_for_concept
from .script_compatibility import ScriptCompatibilityScorer
from .structures import (
    Edge,
    GoalPreservationCheck,
    Node,
    OperatorCandidate,
    PremiseCandidate,
    PremiseValidation,
    StructuredMeaningGraph,
)

if TYPE_CHECKING:
    from .corpus_store import CorpusMemoryStore


BOOKING_TOKENS = ("\uc608\uc57d", "\ubb38\uc758", "\ucde8\uc18c", "\ud655\uc778", "booking", "inquiry", "cancel", "confirm", "check in", "reservation")
WALK_TOKENS = ("\uac78\uc5b4", "\ub3c4\ubcf4", "walk", "on foot")
CLOSED_TOKENS = ("\ub2eb\ud78c", "\ub2eb\ud600", "closed", "shut", "locked")
LEFT_VEHICLE_TOKENS = ("\ub9e1\uaca8", "\ub450\uace0", "already left", "already at", "drop off", "dropped off")
OPEN_STATE_TOKENS = ("\uc5f4\ub9b0", "\uc5f4\ub824", "\uc5f4\uace0", "open", "already open", "unzipped", "unlocked")
RETRIEVE_TOKENS = ("\uaebc\ub0b4", "\ube7c", "take", "get", "pull", "retrieve")
PASS_THROUGH_TOKENS = ("\ud1b5\uacfc", "\uc9c0\ub098", "\ub4e4\uc5b4\uac00", "enter", "go through", "pass through", "walk through")
RANGE_QUERY_TOKENS = ("range sum", "range query", "many queries", "q queries", "prefix sum")
LARGE_INPUT_TOKENS = ("2e5", "10^5", "100000", "200000", "large n", "large q")
FROM_SCRATCH_TOKENS = ("from scratch", "recompute", "naive", "brute force")
SMALL_QUERY_TOKENS = ("only 5 queries", "five queries", "few queries", "small q")
PRECOMPUTED_TOKENS = ("prefix sums already precomputed", "prefix sums are precomputed", "precomputed", "o(1)", "constant time")
NO_SPACE_TOKENS = ("\uc790\ub9ac\uac00 \uc5c6", "\uacf5\uac04\uc774 \uc5c6", "no room", "no space", "full", "packed")
POUR_TOKENS = ("pour", "drink", "empty", "spill", "use", "??", "??")
CAP_TOKENS = ("cap", "lid", "??", "??")
FILE_TOKENS = ("file", "folder", "document", "??", "??", "??")


@dataclass
class HiddenPremiseResult:
    hidden_goals: List[str]
    hidden_assumptions: List[str]
    required_premises: List[str]
    optional_interpretations: List[str]
    premise_candidates: List[PremiseCandidate]
    premise_validations: List[PremiseValidation]
    goal_preservation_checks: List[GoalPreservationCheck]
    clarification_needed: bool
    clarification_score: float
    clarification_reasons: List[str]
    audit_trace: List[str]


class HiddenPremiseExplorer:
    def __init__(self, memory_store: CorpusMemoryStore | None = None, memory_source: str | None = None, compatibility_model_path: str | None = None) -> None:
        self.memory_store = memory_store
        self.memory_source = memory_source
        self.compatibility_scorer = ScriptCompatibilityScorer(memory_store=memory_store, memory_source=memory_source, model_path=compatibility_model_path)

    def enrich(self, graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
        result = self.explore(graph)
        self._apply(graph, result)
        return graph

    def explore(self, graph: StructuredMeaningGraph) -> HiddenPremiseResult:
        candidates = self.retrieve_candidates(graph)
        proposed = self.propose(graph, candidates)
        validations = self.validate(graph, proposed.premise_candidates, proposed.hidden_goals)

        required_candidate_premises = {
            candidate.premise
            for candidate in proposed.premise_candidates
            if candidate.candidate_type == "required"
        }
        supported_required = [
            item.premise
            for item in validations
            if item.premise in required_candidate_premises
            and item.status in {"supported", "uncertain", "missing"}
            and item.requirement_state != "satisfied"
            and item.goal_relevance >= 0.5
        ]
        if not supported_required and not validations:
            supported_required = proposed.required_premises

        clarification_score, clarification_reasons = self._calibrate_clarification(
            graph,
            proposed.premise_candidates,
            validations,
            proposed.hidden_goals,
            proposed.goal_preservation_checks,
        )

        return HiddenPremiseResult(
            hidden_goals=proposed.hidden_goals,
            hidden_assumptions=proposed.hidden_assumptions,
            required_premises=list(dict.fromkeys(supported_required)),
            optional_interpretations=proposed.optional_interpretations,
            premise_candidates=proposed.premise_candidates,
            premise_validations=validations,
            goal_preservation_checks=proposed.goal_preservation_checks,
            clarification_needed=clarification_score >= 0.5,
            clarification_score=clarification_score,
            clarification_reasons=clarification_reasons,
            audit_trace=proposed.audit_trace,
        )

    def retrieve_candidates(self, graph: StructuredMeaningGraph) -> List[PremiseCandidate]:
        lowered = graph.query.lower()
        candidates: List[PremiseCandidate] = []
        candidates.extend(self._rule_candidates(graph, lowered))
        candidates.extend(self._script_candidates(graph, lowered))
        candidates.extend(self._memory_candidates(graph.query))
        candidates.extend(self._script_memory_candidates(graph.query, lowered))
        candidates.extend(self._operator_memory_candidates(graph.query))
        deduplicated = self._deduplicate_candidates(candidates)
        return self._rerank_candidates_with_compatibility(graph, deduplicated)

    def propose(self, graph: StructuredMeaningGraph, candidates: List[PremiseCandidate]) -> HiddenPremiseResult:
        grouped_goals: List[str] = []
        required: List[str] = []
        optional: List[str] = []
        assumptions: List[str] = []
        audit: List[str] = []

        for candidate in candidates:
            should_group_goal = False
            if candidate.hidden_goal:
                if candidate.candidate_type in {'goal', 'required', 'action_probe'} and candidate.support_score >= 0.55:
                    should_group_goal = True
                elif candidate.candidate_type == 'alternative' and candidate.support_score >= 0.65:
                    should_group_goal = True
            if should_group_goal:
                grouped_goals.append(candidate.hidden_goal)
            if candidate.candidate_type == 'required' and candidate.support_score >= 0.58:
                required.append(candidate.premise)
            elif candidate.candidate_type in {'optional', 'alternative'}:
                optional.append(candidate.premise)
            assumptions.extend(candidate.evidence[:2])

        hidden_goals = list(dict.fromkeys(grouped_goals))[:4]
        goal_checks = self._goal_preservation_checks(graph, hidden_goals)
        if candidates:
            audit.append('hidden premise explorer: retrieved premise candidates')
        if self.memory_store is not None and any(item.source == 'memory' for item in candidates):
            audit.append('hidden premise explorer: premise memory contributed candidate support')
        if self.memory_store is not None and any(item.source == 'operator_memory' for item in candidates):
            audit.append('hidden premise explorer: operator memory contributed structural premise hints')
        if any(item.source == 'script' for item in candidates):
            audit.append('hidden premise explorer: script and commonsense retrieval expanded hidden-goal candidates')
        if goal_checks:
            audit.append('hidden premise explorer: scored goal-preservation risks from hidden goals')

        return HiddenPremiseResult(
            hidden_goals=hidden_goals,
            hidden_assumptions=list(dict.fromkeys(assumptions))[:8],
            required_premises=list(dict.fromkeys(required))[:6],
            optional_interpretations=list(dict.fromkeys(optional))[:6],
            premise_candidates=candidates,
            premise_validations=[],
            goal_preservation_checks=goal_checks,
            clarification_needed=False,
            clarification_score=0.0,
            clarification_reasons=[],
            audit_trace=audit,
        )

    def validate(self, graph: StructuredMeaningGraph, candidates: List[PremiseCandidate], hidden_goals: List[str]) -> List[PremiseValidation]:
        lowered = graph.query.lower()
        validations: List[PremiseValidation] = []
        for candidate in candidates:
            contradiction = 0.0
            requirement_state = 'unknown'
            rationale_bits: List[str] = []
            goal_relevance = 0.85 if not candidate.hidden_goal or candidate.hidden_goal in hidden_goals else 0.35
            if candidate.premise == 'vehicle_present' and any(token in lowered for token in LEFT_VEHICLE_TOKENS):
                requirement_state = 'satisfied'
                rationale_bits.append('The query explicitly suggests the vehicle may already be at the site.')
            if candidate.premise == 'open_access' and (any(token in lowered for token in OPEN_STATE_TOKENS) or ('cap' in lowered and any(token in lowered for token in ('off', 'removed'))) or ('lid' in lowered and any(token in lowered for token in ('off', 'removed', 'lifted')))):
                requirement_state = 'satisfied'
                rationale_bits.append('The query already states that access is open, unlocked, or the closure is removed.')
            if candidate.premise == 'available_space' and any(token in lowered for token in NO_SPACE_TOKENS):
                status = 'missing'
                requirement_state = 'missing'
                rationale_bits.append('The query says there is no room left, so the space requirement is currently unsatisfied.')
                validations.append(
                    PremiseValidation(
                        premise=candidate.premise,
                        hidden_goal=candidate.hidden_goal,
                        status=status,
                        support_score=round(candidate.support_score, 2),
                        contradiction_score=round(contradiction, 2),
                        goal_relevance=round(goal_relevance, 2),
                        rationale=' '.join(rationale_bits),
                        requirement_state=requirement_state,
                    )
                )
                continue
            if candidate.premise == 'subquadratic_complexity' and 'only 5 queries' in lowered:
                contradiction = 0.45
                rationale_bits.append('The query may be too small for strong efficiency assumptions to be strictly necessary.')
            if candidate.premise == 'subquadratic_complexity' and any(token in lowered for token in PRECOMPUTED_TOKENS):
                requirement_state = 'satisfied'
                rationale_bits.append('The query already provides a precomputed or O(1)-style path that satisfies the hidden efficiency requirement.')
            status = 'supported'
            if requirement_state == 'satisfied':
                status = 'satisfied'
            elif candidate.premise == 'open_access' and (any(token in lowered for token in CLOSED_TOKENS) or (any(token in lowered for token in CAP_TOKENS) and not any(token in lowered for token in OPEN_STATE_TOKENS))) and not any(token in lowered for token in OPEN_STATE_TOKENS):
                status = 'missing'
                requirement_state = 'missing'
                rationale_bits.append('The query indicates that access is still closed, so the requirement is not yet satisfied.')
            elif candidate.premise == 'vehicle_present' and any(token in lowered for token in WALK_TOKENS) and not any(token in lowered for token in LEFT_VEHICLE_TOKENS):
                status = 'missing'
                requirement_state = 'missing'
                rationale_bits.append('Walking without the vehicle leaves the vehicle-presence requirement unsatisfied for the service goal.')
            elif candidate.premise == 'subquadratic_complexity' and any(token in lowered for token in FROM_SCRATCH_TOKENS):
                status = 'missing'
                requirement_state = 'missing'
                rationale_bits.append('The proposed from-scratch plan does not satisfy the hidden efficiency requirement.')
            elif contradiction >= 0.6:
                status = 'contradicted'
            elif candidate.support_score < 0.58:
                status = 'uncertain'
            if not rationale_bits:
                rationale_bits.append('The premise remains consistent with the current script and retrieved support.')
            validations.append(
                PremiseValidation(
                    premise=candidate.premise,
                    hidden_goal=candidate.hidden_goal,
                    status=status,
                    support_score=round(candidate.support_score, 2),
                    contradiction_score=round(contradiction, 2),
                    goal_relevance=round(goal_relevance, 2),
                    rationale=' '.join(rationale_bits),
                    requirement_state=requirement_state,
                )
            )
        return self._deduplicate_validations(validations)

    def _rule_candidates(self, graph: StructuredMeaningGraph, lowered: str) -> List[PremiseCandidate]:
        node_ids = graph.node_ids()
        candidates: List[PremiseCandidate] = []
        if 'car_wash' in node_ids:
            booking_like = any(token in lowered for token in BOOKING_TOKENS)
            walking_like = any(token in lowered for token in WALK_TOKENS)
            left_vehicle = any(token in lowered for token in LEFT_VEHICLE_TOKENS)
            if left_vehicle:
                candidates.extend([
                    PremiseCandidate('reservation_check_goal', hidden_goal='reservation_check_goal', candidate_type='goal', source='rule', support_score=0.82, evidence=['If the car is already there, the remaining goal can shift to pickup or status confirmation.']),
                    PremiseCandidate('vehicle_present', hidden_goal='reservation_check_goal', candidate_type='optional', source='rule', support_score=0.42, evidence=['Vehicle presence may already be satisfied at the site.']),
                ])
            elif booking_like:
                candidates.extend([
                    PremiseCandidate('booking_or_inquiry_goal', hidden_goal='booking_or_inquiry_goal', candidate_type='goal', source='rule', support_score=0.84, evidence=['The destination may be a car wash, but the immediate goal looks like booking, cancellation, or inquiry.']),
                    PremiseCandidate('goal_clarity', hidden_goal='booking_or_inquiry_goal', candidate_type='required', source='rule', support_score=0.71, evidence=['The question should confirm whether the user wants contact, booking, or an actual wash.']),
                    PremiseCandidate('clean_car_goal', hidden_goal='clean_car_goal', candidate_type='alternative', source='rule', support_score=0.39, evidence=['A service-contact reading competes with the default wash-service reading.']),
                ])
            else:
                candidates.extend([
                    PremiseCandidate('clean_car_goal', hidden_goal='clean_car_goal', candidate_type='goal', source='rule', support_score=0.9, evidence=['A car wash is typically a place where a vehicle is physically brought for washing.']),
                    PremiseCandidate('vehicle_present', hidden_goal='clean_car_goal', candidate_type='required', source='rule', support_score=0.91, evidence=['A normal wash-service script usually requires vehicle presence at the site.']),
                    PremiseCandidate('booking_or_inquiry_goal', hidden_goal='booking_or_inquiry_goal', candidate_type='alternative', source='rule', support_score=0.28, evidence=['A weaker alternative interpretation is that the user only needs booking or inquiry.']),
                ])
            if walking_like:
                candidates.append(PremiseCandidate('walk_without_car', hidden_goal='clean_car_goal', candidate_type='action_probe', source='rule', support_score=0.77, evidence=['The candidate action removes the car from the plan and may break the service goal.']))
        if {'bag', 'book'}.issubset(node_ids):
            candidates.extend([
                PremiseCandidate('store_book_in_bag_goal', hidden_goal='store_book_in_bag_goal', candidate_type='goal', source='rule', support_score=0.9, evidence=['The real goal is to place the book inside the bag interior, not merely near the bag.']),
                PremiseCandidate('open_access', hidden_goal='store_book_in_bag_goal', candidate_type='required', source='rule', support_score=0.83, evidence=['Successful containment usually requires open access to the bag interior.']),
                PremiseCandidate('available_space', hidden_goal='store_book_in_bag_goal', candidate_type='required', source='rule', support_score=0.81, evidence=['Containment also requires enough available interior space.']),
                PremiseCandidate('use_other_compartment', hidden_goal='store_book_in_bag_goal', candidate_type='alternative', source='rule', support_score=0.37, evidence=['A side compartment or different compartment may satisfy the same goal.']),
            ])
            if any(token in lowered for token in CLOSED_TOKENS):
                candidates.append(PremiseCandidate('insert_without_opening', hidden_goal='store_book_in_bag_goal', candidate_type='action_probe', source='rule', support_score=0.8, evidence=['Direct insertion without opening access usually breaks the hidden containment goal.']))
        if 'box' in node_ids and ('inside' in lowered or any(token in lowered for token in RETRIEVE_TOKENS) or any(token in lowered for token in FILE_TOKENS)):
            candidates.extend([
                PremiseCandidate('retrieve_item_from_box_goal', hidden_goal='retrieve_item_from_box_goal', candidate_type='goal', source='rule', support_score=0.86, evidence=['A box query often hides the goal of accessing or retrieving something stored inside.']),
                PremiseCandidate('open_access', hidden_goal='retrieve_item_from_box_goal', candidate_type='required', source='rule', support_score=0.84, evidence=['Retrieving an item from a box normally requires opening the lid or access panel first.']),
            ])
            if any(token in lowered for token in CLOSED_TOKENS):
                candidates.append(PremiseCandidate('retrieve_without_opening', hidden_goal='retrieve_item_from_box_goal', candidate_type='action_probe', source='rule', support_score=0.82, evidence=['Trying to reach into a closed box usually fails the access goal.']))
        if 'bin' in node_ids and ('inside' in lowered or any(token in lowered for token in RETRIEVE_TOKENS) or any(token in lowered for token in FILE_TOKENS)):
            candidates.extend([
                PremiseCandidate('retrieve_item_from_bin_goal', hidden_goal='retrieve_item_from_bin_goal', candidate_type='goal', source='rule', support_score=0.84, evidence=['A bin query often hides the goal of accessing or retrieving contents.']),
                PremiseCandidate('open_access', hidden_goal='retrieve_item_from_bin_goal', candidate_type='required', source='rule', support_score=0.82, evidence=['Accessing a bin usually requires opening or lifting the cover first.']),
            ])
            if any(token in lowered for token in CLOSED_TOKENS):
                candidates.append(PremiseCandidate('retrieve_without_opening', hidden_goal='retrieve_item_from_bin_goal', candidate_type='action_probe', source='rule', support_score=0.8, evidence=['Trying to reach into a closed bin usually fails the access goal.']))
        if 'pouch' in node_ids and ('inside' in lowered or any(token in lowered for token in RETRIEVE_TOKENS) or any(token in lowered for token in FILE_TOKENS)):
            candidates.extend([
                PremiseCandidate('retrieve_item_from_pouch_goal', hidden_goal='retrieve_item_from_pouch_goal', candidate_type='goal', source='rule', support_score=0.86, evidence=['A pouch query often hides the goal of accessing or retrieving contents.']),
                PremiseCandidate('open_access', hidden_goal='retrieve_item_from_pouch_goal', candidate_type='required', source='rule', support_score=0.84, evidence=['Retrieving an item from a pouch normally requires opening the zipper or access band first.']),
            ])
            if any(token in lowered for token in CLOSED_TOKENS):
                candidates.append(PremiseCandidate('retrieve_without_opening', hidden_goal='retrieve_item_from_pouch_goal', candidate_type='action_probe', source='rule', support_score=0.82, evidence=['Trying to retrieve an item from a closed pouch usually fails the access goal.']))
        if 'suitcase' in node_ids and ('inside' in lowered or any(token in lowered for token in RETRIEVE_TOKENS) or any(token in lowered for token in FILE_TOKENS)):
            candidates.extend([
                PremiseCandidate('retrieve_item_from_suitcase_goal', hidden_goal='retrieve_item_from_suitcase_goal', candidate_type='goal', source='rule', support_score=0.87, evidence=['A suitcase query often hides the goal of accessing or retrieving contents.']),
                PremiseCandidate('open_access', hidden_goal='retrieve_item_from_suitcase_goal', candidate_type='required', source='rule', support_score=0.85, evidence=['Retrieving an item from a suitcase normally requires opening its closure first.']),
            ])
            if any(token in lowered for token in CLOSED_TOKENS):
                candidates.append(PremiseCandidate('retrieve_without_opening', hidden_goal='retrieve_item_from_suitcase_goal', candidate_type='action_probe', source='rule', support_score=0.83, evidence=['Trying to retrieve an item from a closed suitcase usually fails the access goal.']))
        if 'drawer' in node_ids and ('folder' in node_ids or 'document' in lowered or 'inside' in lowered or any(token in lowered for token in RETRIEVE_TOKENS)):
            candidates.extend([
                PremiseCandidate('retrieve_item_from_drawer_goal', hidden_goal='retrieve_item_from_drawer_goal', candidate_type='goal', source='rule', support_score=0.86, evidence=['A drawer query usually hides the goal of accessing its contents.']),
                PremiseCandidate('open_access', hidden_goal='retrieve_item_from_drawer_goal', candidate_type='required', source='rule', support_score=0.84, evidence=['Retrieving an item from a drawer normally requires opening the drawer.']),
            ])
            if any(token in lowered for token in CLOSED_TOKENS):
                candidates.append(PremiseCandidate('retrieve_without_opening', hidden_goal='retrieve_item_from_drawer_goal', candidate_type='action_probe', source='rule', support_score=0.81, evidence=['Trying to pull an item from a closed drawer usually fails the access goal.']))
        if 'cabinet' in node_ids and ((any(token in lowered for token in FILE_TOKENS)) or 'inside' in lowered or any(token in lowered for token in RETRIEVE_TOKENS)):
            candidates.extend([
                PremiseCandidate('retrieve_item_from_cabinet_goal', hidden_goal='retrieve_item_from_cabinet_goal', candidate_type='goal', source='rule', support_score=0.88, evidence=['A cabinet query often hides the goal of retrieving or accessing something stored inside.']),
                PremiseCandidate('open_access', hidden_goal='retrieve_item_from_cabinet_goal', candidate_type='required', source='rule', support_score=0.86, evidence=['Retrieving an item from a cabinet normally requires opening the cabinet first.']),
            ])
            if any(token in lowered for token in CLOSED_TOKENS):
                candidates.append(PremiseCandidate('retrieve_without_opening', hidden_goal='retrieve_item_from_cabinet_goal', candidate_type='action_probe', source='rule', support_score=0.83, evidence=['Trying to reach into a closed cabinet usually fails the access goal.']))
        if ('bottle' in node_ids or 'jar' in node_ids) and (any(token in lowered for token in POUR_TOKENS) or any(token in lowered for token in CAP_TOKENS) or 'inside' in lowered):
            candidates.extend([
                PremiseCandidate('pour_from_bottle_goal', hidden_goal='pour_from_bottle_goal', candidate_type='goal', source='rule', support_score=0.87, evidence=['A bottle or jar query usually hides the goal of pouring or accessing contents.']),
                PremiseCandidate('open_access', hidden_goal='pour_from_bottle_goal', candidate_type='required', source='rule', support_score=0.85, evidence=['Pouring from a capped container normally requires removing or opening the cap first.']),
            ])
            if any(token in lowered for token in CLOSED_TOKENS) or any(token in lowered for token in CAP_TOKENS):
                candidates.append(PremiseCandidate('pour_without_uncapping', hidden_goal='pour_from_bottle_goal', candidate_type='action_probe', source='rule', support_score=0.83, evidence=['Trying to pour while the cap or lid is still on usually fails the access goal.']))
        if 'door' in node_ids and any(token in lowered for token in PASS_THROUGH_TOKENS):
            candidates.extend([
                PremiseCandidate('pass_through_door_goal', hidden_goal='pass_through_door_goal', candidate_type='goal', source='rule', support_score=0.86, evidence=['A door query usually hides the goal of passing through or entering.']),
                PremiseCandidate('open_access', hidden_goal='pass_through_door_goal', candidate_type='required', source='rule', support_score=0.84, evidence=['Passing through a closed door normally requires opening access first.']),
            ])
            if any(token in lowered for token in CLOSED_TOKENS):
                candidates.append(PremiseCandidate('proceed_without_opening', hidden_goal='pass_through_door_goal', candidate_type='action_probe', source='rule', support_score=0.81, evidence=['Walking through a closed door usually fails the passage goal.']))
        if self._looks_like_cp_complexity_case(lowered):
            candidates.extend([
                PremiseCandidate('efficient_solution_goal', hidden_goal='efficient_solution_goal', candidate_type='goal', source='rule', support_score=0.83, evidence=['The statement implies that scaling the method is part of the real task, not only obtaining the right answer.']),
                PremiseCandidate('subquadratic_complexity', hidden_goal='efficient_solution_goal', candidate_type='required', source='rule', support_score=0.82, evidence=['Large query counts or large constraints usually require subquadratic processing.']),
            ])
            if 'range sum' in lowered or 'prefix sum' in lowered:
                candidates.append(PremiseCandidate('precompute_prefix_information', hidden_goal='efficient_solution_goal', candidate_type='alternative', source='rule', support_score=0.74, evidence=['Range-sum structure often supports a prefix-style compression of repeated work.']))
            if any(token in lowered for token in FROM_SCRATCH_TOKENS):
                candidates.append(PremiseCandidate('recompute_each_query', hidden_goal='efficient_solution_goal', candidate_type='action_probe', source='rule', support_score=0.82, evidence=['A from-scratch plan may violate the hidden efficiency goal under many-query constraints.']))
        if not candidates:
            for edge in graph.edges:
                if edge.relation == 'REQUIRES':
                    candidates.append(PremiseCandidate(edge.target, hidden_goal=edge.source, candidate_type='required', source='graph', support_score=max(0.62, edge.confidence), evidence=[f'{concept_label(edge.source)} requires {concept_label(edge.target)}.']))
                if edge.relation == 'GOAL_OF':
                    candidates.append(PremiseCandidate(edge.source, hidden_goal=edge.source, candidate_type='goal', source='graph', support_score=max(0.62, edge.confidence), evidence=[f'{concept_label(edge.source)} appears as an implicit goal in the current graph.']))
            for script in graph.inferred_scripts[:2]:
                candidates.append(PremiseCandidate(script, hidden_goal='', candidate_type='optional', source='graph', support_score=0.5, evidence=[f'The query may depend on the script step: {script.replace("_", " ")}']))
        return candidates

    def _script_candidates(self, graph: StructuredMeaningGraph, lowered: str) -> List[PremiseCandidate]:
        candidates: List[PremiseCandidate] = []
        for concept in self._query_concepts(graph):
            data = lookup_concept(concept)
            default_goal = self._default_goal_for_concept(concept, lowered)
            for requirement in data.get('requires', []):
                if default_goal:
                    candidates.append(PremiseCandidate(requirement, hidden_goal=default_goal, candidate_type='required', source='script', support_score=0.67, evidence=[f'{concept_label(concept)} commonly requires {concept_label(requirement)}.']))
            for script in scripts_for_concept(concept)[:3]:
                candidates.append(PremiseCandidate(script, hidden_goal=default_goal or concept, candidate_type='optional', source='script', support_score=0.54, evidence=[f'Retrieved script memory step for {concept_label(concept)}: {script.replace("_", " ")}']))
        return candidates

    def _memory_candidates(self, query: str) -> List[PremiseCandidate]:
        if self.memory_store is None:
            return []
        rows = self.memory_store.search_premise_support(query, top_k=6, source=self.memory_source)
        candidates: List[PremiseCandidate] = []
        for row in rows:
            candidates.append(
                PremiseCandidate(
                    premise=row['premise'],
                    hidden_goal=row.get('hidden_goal', ''),
                    candidate_type=row.get('candidate_type', 'required'),
                    source='memory',
                    support_score=float(row.get('support_score', 0.6)),
                    evidence=[f"Retrieved from similar query: {row.get('graph_query', '')}"],
                )
            )
        return candidates


    def _script_memory_candidates(self, query: str, lowered: str) -> List[PremiseCandidate]:
        if self.memory_store is None:
            return []
        rows = self.memory_store.search_script_support(query, top_k=6, source=self.memory_source)
        candidates: List[PremiseCandidate] = []
        access_goal = self._default_access_goal(lowered)
        for row in rows:
            script_step = str(row.get('script_step', ''))
            support = float(row.get('support_score', 0.55))
            evidence = [f"Retrieved script step from similar query: {row.get('graph_query', '')}"]
            match_bonus = 0.0
            if 'drawer' in lowered and 'open_drawer' in script_step:
                match_bonus = 0.2
            elif 'door' in lowered and 'open_door' in script_step:
                match_bonus = 0.2
            elif 'cabinet' in lowered and 'open_cabinet' in script_step:
                match_bonus = 0.2
            elif 'bag' in lowered and 'open_bag' in script_step:
                match_bonus = 0.2
            elif 'box' in lowered and 'open_box' in script_step:
                match_bonus = 0.2
            elif ('bottle' in lowered or 'jar' in lowered) and ('uncap_container' in script_step or 'lift_lid_or_cap' in script_step):
                match_bonus = 0.2
            elif 'pouch' in lowered and 'open_pouch' in script_step:
                match_bonus = 0.2
            elif 'suitcase' in lowered and 'open_suitcase' in script_step:
                match_bonus = 0.2
            elif any(token in lowered for token in RETRIEVE_TOKENS) and 'retrieve_item' in script_step:
                match_bonus = 0.18
            elif 'car wash' in lowered and 'drive_vehicle_to_site' in script_step:
                match_bonus = 0.16
            if 'drive_vehicle_to_site' in script_step:
                candidates.append(PremiseCandidate('vehicle_present', hidden_goal='clean_car_goal', candidate_type='required', source='script_memory', support_score=min(0.86, support + 0.08 + match_bonus), evidence=evidence))
            if any(step in script_step for step in ('open_bag', 'open_drawer', 'open_door', 'open_cabinet', 'open_box', 'open_pouch', 'open_suitcase', 'uncap_container', 'lift_lid_or_cap')):
                candidates.append(PremiseCandidate('open_access', hidden_goal=access_goal, candidate_type='required', source='script_memory', support_score=min(0.86, support + 0.08 + match_bonus), evidence=evidence))
            if 'wash_vehicle' in script_step:
                candidates.append(PremiseCandidate('clean_car_goal', hidden_goal='clean_car_goal', candidate_type='goal', source='script_memory', support_score=min(0.88, support + 0.06 + match_bonus), evidence=evidence))
            if 'retrieve_item' in script_step:
                if 'cabinet' in lowered:
                    goal_name = 'retrieve_item_from_cabinet_goal'
                elif 'box' in lowered:
                    goal_name = 'retrieve_item_from_box_goal'
                elif 'bin' in lowered:
                    goal_name = 'retrieve_item_from_bin_goal'
                elif 'pouch' in lowered:
                    goal_name = 'retrieve_item_from_pouch_goal'
                elif 'suitcase' in lowered:
                    goal_name = 'retrieve_item_from_suitcase_goal'
                else:
                    goal_name = 'retrieve_item_from_drawer_goal'
                candidates.append(PremiseCandidate(goal_name, hidden_goal=goal_name, candidate_type='goal', source='script_memory', support_score=min(0.86, support + 0.06 + match_bonus), evidence=evidence))
            if 'pass_through' in script_step:
                candidates.append(PremiseCandidate('pass_through_door_goal', hidden_goal='pass_through_door_goal', candidate_type='goal', source='script_memory', support_score=min(0.86, support + 0.06 + match_bonus), evidence=evidence))
            if 'uncap_container' in script_step or 'lift_lid_or_cap' in script_step or 'pour_contents' in script_step:
                candidates.append(PremiseCandidate('pour_from_bottle_goal', hidden_goal='pour_from_bottle_goal', candidate_type='goal', source='script_memory', support_score=min(0.86, support + 0.06 + match_bonus), evidence=evidence))
        return candidates

    def _operator_memory_candidates(self, query: str) -> List[PremiseCandidate]:
        if self.memory_store is None:
            return []
        rows = self.memory_store.search_operator_support(query, top_k=6, source=self.memory_source)
        candidates: List[PremiseCandidate] = []
        for row in rows:
            family = str(row.get('operator_family', '')).upper()
            name = str(row.get('operator_name', '')).upper()
            support = float(row.get('support_score', 0.55))
            evidence = [f"Retrieved from operator memory: {row.get('operator_name', '')}"]
            hidden_goal = self._default_access_goal(query.lower())
            if 'ACCESS' in family or 'ACCESS' in name or 'CONTROL' in family or 'CONTROL' in name or 'CAPPED' in family or 'CAPPED' in name:
                candidates.append(PremiseCandidate('open_access', hidden_goal=hidden_goal, candidate_type='required', source='operator_memory', support_score=min(0.84, support + 0.1), evidence=evidence))
            if 'CONTAINER' in family or 'CONTAINER' in name:
                candidates.append(PremiseCandidate('available_space', hidden_goal=hidden_goal, candidate_type='required', source='operator_memory', support_score=min(0.8, support + 0.05), evidence=evidence))
            if 'SERVICE' in family or 'SERVICE_GOAL' in name:
                candidates.append(PremiseCandidate('goal_clarity', hidden_goal='service_goal', candidate_type='required', source='operator_memory', support_score=min(0.74, support + 0.04), evidence=evidence))
        return candidates

    def _goal_preservation_checks(self, graph: StructuredMeaningGraph, hidden_goals: List[str]) -> List[GoalPreservationCheck]:
        lowered = graph.query.lower()
        checks: List[GoalPreservationCheck] = []
        if 'clean_car_goal' in hidden_goals and any(token in lowered for token in WALK_TOKENS):
            checks.append(GoalPreservationCheck(action='walk_without_car', hidden_goal='clean_car_goal', status='risk_high', rationale='Walking may achieve arrival, but it usually fails the hidden goal of getting the car washed because the car is not present.', confidence=0.92))
        if 'booking_or_inquiry_goal' in hidden_goals and any(token in lowered for token in WALK_TOKENS):
            checks.append(GoalPreservationCheck(action='walk_without_car', hidden_goal='booking_or_inquiry_goal', status='conditionally_valid', rationale='Walking can preserve a booking or inquiry goal, but not a normal wash-service goal.', confidence=0.74))
        if 'reservation_check_goal' in hidden_goals and any(token in lowered for token in WALK_TOKENS):
            checks.append(GoalPreservationCheck(action='walk_without_car', hidden_goal='reservation_check_goal', status='valid', rationale='If the car is already at the service site, walking may preserve a follow-up or pickup-related goal.', confidence=0.78))
        if 'store_book_in_bag_goal' in hidden_goals and any(token in lowered for token in CLOSED_TOKENS) and not any(token in lowered for token in OPEN_STATE_TOKENS):
            checks.append(GoalPreservationCheck(action='insert_without_opening', hidden_goal='store_book_in_bag_goal', status='risk_high', rationale='Direct insertion without opening access usually breaks the hidden containment goal.', confidence=0.87))
        if 'retrieve_item_from_drawer_goal' in hidden_goals and any(token in lowered for token in CLOSED_TOKENS):
            checks.append(GoalPreservationCheck(action='retrieve_without_opening', hidden_goal='retrieve_item_from_drawer_goal', status='risk_high', rationale='A closed drawer blocks retrieval unless access is opened first.', confidence=0.88))
        if 'retrieve_item_from_cabinet_goal' in hidden_goals and any(token in lowered for token in CLOSED_TOKENS):
            checks.append(GoalPreservationCheck(action='retrieve_without_opening', hidden_goal='retrieve_item_from_cabinet_goal', status='risk_high', rationale='A closed cabinet blocks retrieval unless access is opened first.', confidence=0.89))
        if 'retrieve_item_from_box_goal' in hidden_goals and any(token in lowered for token in CLOSED_TOKENS):
            checks.append(GoalPreservationCheck(action='retrieve_without_opening', hidden_goal='retrieve_item_from_box_goal', status='risk_high', rationale='A closed box blocks retrieval unless access is opened first.', confidence=0.88))
        if 'retrieve_item_from_bin_goal' in hidden_goals and any(token in lowered for token in CLOSED_TOKENS):
            checks.append(GoalPreservationCheck(action='retrieve_without_opening', hidden_goal='retrieve_item_from_bin_goal', status='risk_high', rationale='A closed bin blocks retrieval unless access is opened first.', confidence=0.86))
        if 'retrieve_item_from_pouch_goal' in hidden_goals and any(token in lowered for token in CLOSED_TOKENS):
            checks.append(GoalPreservationCheck(action='retrieve_without_opening', hidden_goal='retrieve_item_from_pouch_goal', status='risk_high', rationale='A closed pouch blocks retrieval unless access is opened first.', confidence=0.88))
        if 'retrieve_item_from_suitcase_goal' in hidden_goals and any(token in lowered for token in CLOSED_TOKENS):
            checks.append(GoalPreservationCheck(action='retrieve_without_opening', hidden_goal='retrieve_item_from_suitcase_goal', status='risk_high', rationale='A closed suitcase blocks retrieval unless access is opened first.', confidence=0.89))
        if 'pour_from_bottle_goal' in hidden_goals and (any(token in lowered for token in CLOSED_TOKENS) or any(token in lowered for token in CAP_TOKENS)) and not any(token in lowered for token in OPEN_STATE_TOKENS):
            checks.append(GoalPreservationCheck(action='pour_without_uncapping', hidden_goal='pour_from_bottle_goal', status='risk_high', rationale='A capped bottle or jar usually must be uncapped before pouring or accessing its contents.', confidence=0.9))
        if 'pass_through_door_goal' in hidden_goals and any(token in lowered for token in CLOSED_TOKENS):
            checks.append(GoalPreservationCheck(action='proceed_without_opening', hidden_goal='pass_through_door_goal', status='risk_high', rationale='Moving through a closed door usually fails until access is opened.', confidence=0.88))
        if 'efficient_solution_goal' in hidden_goals and any(token in lowered for token in FROM_SCRATCH_TOKENS):
            checks.append(GoalPreservationCheck(action='recompute_each_query', hidden_goal='efficient_solution_goal', status='risk_high', rationale='Recomputing each query from scratch may violate the hidden efficiency goal under repeated-query constraints.', confidence=0.86))
        return checks

    def _apply(self, graph: StructuredMeaningGraph, result: HiddenPremiseResult) -> None:
        graph.hidden_goals = self._merge_unique(graph.hidden_goals, result.hidden_goals)
        graph.hidden_assumptions = self._merge_unique(graph.hidden_assumptions, result.hidden_assumptions)
        graph.required_premises = self._merge_unique(graph.required_premises, result.required_premises)
        graph.optional_interpretations = self._merge_unique(graph.optional_interpretations, result.optional_interpretations)
        graph.clarification_needed = graph.clarification_needed or result.clarification_needed
        graph.clarification_score = max(graph.clarification_score, result.clarification_score)
        graph.clarification_reasons = self._merge_unique(graph.clarification_reasons, result.clarification_reasons)
        for candidate in result.premise_candidates:
            if not any(existing.premise == candidate.premise and existing.hidden_goal == candidate.hidden_goal and existing.source == candidate.source for existing in graph.premise_candidates):
                graph.premise_candidates.append(candidate)
        for validation in result.premise_validations:
            if not any(existing.premise == validation.premise and existing.hidden_goal == validation.hidden_goal for existing in graph.premise_validations):
                graph.premise_validations.append(validation)
        graph.satisfied_premises = self._merge_unique(graph.satisfied_premises, [item.premise for item in result.premise_validations if item.status == 'satisfied'])
        graph.missing_premises = self._merge_unique(graph.missing_premises, [item.premise for item in result.premise_validations if item.status == 'missing'])
        for check in result.goal_preservation_checks:
            if not any(existing.action == check.action and existing.hidden_goal == check.hidden_goal for existing in graph.goal_preservation_checks):
                graph.goal_preservation_checks.append(check)
        self._inject_premise_support_operators(graph, result.hidden_goals)
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
            if 'drawer' in graph.node_ids():
                graph.add_edge(Edge(source='drawer', relation='TYPICAL_FOR', target=hidden_goal, confidence=0.82, provenance=['premise_explorer:script']))
            if 'cabinet' in graph.node_ids():
                graph.add_edge(Edge(source='cabinet', relation='TYPICAL_FOR', target=hidden_goal, confidence=0.82, provenance=['premise_explorer:script']))
            if 'box' in graph.node_ids():
                graph.add_edge(Edge(source='box', relation='TYPICAL_FOR', target=hidden_goal, confidence=0.82, provenance=['premise_explorer:script']))
            if 'bin' in graph.node_ids():
                graph.add_edge(Edge(source='bin', relation='TYPICAL_FOR', target=hidden_goal, confidence=0.82, provenance=['premise_explorer:script']))
            if 'pouch' in graph.node_ids():
                graph.add_edge(Edge(source='pouch', relation='TYPICAL_FOR', target=hidden_goal, confidence=0.82, provenance=['premise_explorer:script']))
            if 'suitcase' in graph.node_ids():
                graph.add_edge(Edge(source='suitcase', relation='TYPICAL_FOR', target=hidden_goal, confidence=0.82, provenance=['premise_explorer:script']))
            if 'door' in graph.node_ids():
                graph.add_edge(Edge(source='door', relation='TYPICAL_FOR', target=hidden_goal, confidence=0.82, provenance=['premise_explorer:script']))

        for item in result.hidden_assumptions:
            warning = f'hidden premise: {item}'
            if warning not in graph.warnings:
                graph.warnings.append(warning)
        for validation in result.premise_validations[:8]:
            note = f'premise validation: {validation.premise} -> {validation.status}'
            if note not in graph.warnings:
                graph.warnings.append(note)
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
            return 'Walking to the car wash without the car usually breaks the real service goal.'
        if check.action == 'insert_without_opening':
            return 'Pushing the item in without opening access usually fails the containment goal.'
        if check.action == 'retrieve_without_opening':
            return 'Trying to retrieve the item without opening the container usually fails the access goal.'
        if check.action == 'pour_without_uncapping':
            return 'Trying to pour while the cap or lid is still on usually fails the access goal.'
        if check.action == 'proceed_without_opening':
            return 'Trying to pass through a closed door without opening it usually fails the passage goal.'
        if check.action == 'recompute_each_query':
            return 'Recomputing every query from scratch usually fails the hidden efficiency goal.'
        return ''

    def _query_concepts(self, graph: StructuredMeaningGraph) -> List[str]:
        concepts = list(graph.node_ids())
        for concept in match_concepts(graph.query):
            if concept not in concepts:
                concepts.append(concept)
        return concepts

    @staticmethod
    def _looks_like_cp_complexity_case(lowered: str) -> bool:
        if any(token in lowered for token in SMALL_QUERY_TOKENS):
            return False
        has_range = any(token in lowered for token in RANGE_QUERY_TOKENS)
        has_large = any(token in lowered for token in LARGE_INPUT_TOKENS)
        has_many_queries = 'many queries' in lowered or 'q queries' in lowered or 'queries' in lowered
        return has_range and (has_large or has_many_queries)

    @staticmethod
    def _default_access_goal(lowered: str) -> str:
        if 'cabinet' in lowered:
            return 'retrieve_item_from_cabinet_goal'
        if 'box' in lowered:
            return 'retrieve_item_from_box_goal'
        if 'bin' in lowered:
            return 'retrieve_item_from_bin_goal'
        if 'pouch' in lowered:
            return 'retrieve_item_from_pouch_goal'
        if 'suitcase' in lowered:
            return 'retrieve_item_from_suitcase_goal'
        if 'drawer' in lowered or any(token in lowered for token in RETRIEVE_TOKENS):
            return 'retrieve_item_from_drawer_goal'
        if 'door' in lowered or any(token in lowered for token in PASS_THROUGH_TOKENS):
            return 'pass_through_door_goal'
        if 'bottle' in lowered or 'jar' in lowered or any(token in lowered for token in CAP_TOKENS) or any(token in lowered for token in POUR_TOKENS):
            return 'pour_from_bottle_goal'
        if 'bag' in lowered and 'book' in lowered:
            return 'store_book_in_bag_goal'
        return 'access_goal'

    @staticmethod
    def _default_goal_for_concept(concept: str, lowered: str) -> str:
        if concept == 'car_wash':
            return 'booking_or_inquiry_goal' if any(token in lowered for token in BOOKING_TOKENS) else 'clean_car_goal'
        if concept == 'bag' and 'book' in lowered:
            return 'store_book_in_bag_goal'
        if concept == 'drawer' and any(token in lowered for token in RETRIEVE_TOKENS):
            return 'retrieve_item_from_drawer_goal'
        if concept == 'cabinet' and (any(token in lowered for token in RETRIEVE_TOKENS) or any(token in lowered for token in FILE_TOKENS)):
            return 'retrieve_item_from_cabinet_goal'
        if concept == 'box' and (any(token in lowered for token in RETRIEVE_TOKENS) or 'inside' in lowered or any(token in lowered for token in FILE_TOKENS)):
            return 'retrieve_item_from_box_goal'
        if concept == 'bin' and (any(token in lowered for token in RETRIEVE_TOKENS) or 'inside' in lowered or any(token in lowered for token in FILE_TOKENS)):
            return 'retrieve_item_from_bin_goal'
        if concept == 'pouch' and (any(token in lowered for token in RETRIEVE_TOKENS) or 'inside' in lowered or any(token in lowered for token in FILE_TOKENS)):
            return 'retrieve_item_from_pouch_goal'
        if concept == 'suitcase' and (any(token in lowered for token in RETRIEVE_TOKENS) or 'inside' in lowered or any(token in lowered for token in FILE_TOKENS)):
            return 'retrieve_item_from_suitcase_goal'
        if concept == 'door' and any(token in lowered for token in PASS_THROUGH_TOKENS):
            return 'pass_through_door_goal'
        if concept in {'bottle', 'jar'} and (any(token in lowered for token in POUR_TOKENS) or any(token in lowered for token in CAP_TOKENS)):
            return 'pour_from_bottle_goal'
        return ''

    def _calibrate_clarification(
        self,
        graph: StructuredMeaningGraph,
        candidates: List[PremiseCandidate],
        validations: List[PremiseValidation],
        hidden_goals: List[str],
        goal_checks: List[GoalPreservationCheck],
    ) -> tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        goal_candidates = sorted(
            [item for item in candidates if item.candidate_type == 'goal'],
            key=lambda item: item.support_score,
            reverse=True,
        )
        if len(goal_candidates) >= 2:
            gap = goal_candidates[0].support_score - goal_candidates[1].support_score
            if gap < 0.12:
                score += 0.35
                reasons.append('top hidden goals remain close in support')
            elif gap < 0.2:
                score += 0.2
                reasons.append('goal ranking is only weakly separated')
        strong_alternatives = [item for item in candidates if item.candidate_type in {'optional', 'alternative'} and item.support_score >= 0.3]
        if strong_alternatives:
            score += min(0.3, 0.12 + 0.04 * len(strong_alternatives))
            reasons.append('multiple plausible alternative interpretations are still active')
        uncertain = [item for item in validations if item.status == 'uncertain']
        if uncertain:
            score += 0.15
            reasons.append('some required premises remain uncertain')
        conditional = [item for item in goal_checks if item.status == 'conditionally_valid']
        if conditional:
            score += 0.15
            reasons.append('action validity changes across hidden-goal readings')
        if any(item.status == 'risk_high' for item in goal_checks) and strong_alternatives:
            score += 0.18
            reasons.append('the current action is risky under at least one competing hidden-goal reading')
        if len(hidden_goals) > 1:
            score += 0.18
            reasons.append('more than one hidden goal remains viable')
        if not hidden_goals:
            score += 0.1
            reasons.append('no dominant hidden goal was recovered')
        if len(hidden_goals) == 1 and goal_candidates and goal_candidates[0].support_score >= 0.82 and not uncertain and not strong_alternatives:
            score -= 0.2
        if any(item.status == 'satisfied' for item in validations) and not strong_alternatives:
            score -= 0.05
        if any(item.premise == 'vehicle_present' and item.status == 'satisfied' for item in validations) and not any(item.status == 'missing' for item in validations):
            score -= 0.25
            reasons.append('core service precondition is already satisfied')
        score = round(max(0.0, min(1.0, score)), 4)
        return score, reasons[:4]

    def _rerank_candidates_with_compatibility(self, graph: StructuredMeaningGraph, candidates: List[PremiseCandidate]) -> List[PremiseCandidate]:
        if not candidates:
            return []
        support_profile = self._query_support_profile(graph)
        reranked: List[PremiseCandidate] = []
        for item in candidates:
            symbolic = self._candidate_compatibility_score(item, support_profile)
            breakdown = self.compatibility_scorer.score(graph, item, support_profile, symbolic)
            if breakdown.combined_score > 0.0:
                item.support_score = round(min(0.99, item.support_score * 0.74 + breakdown.combined_score * 0.26), 4)
                evidence = (
                    f"Compatibility support: symbolic={breakdown.symbolic_score:.2f} learned={breakdown.learned_score:.2f} "
                    f"concepts={','.join(sorted(support_profile['concepts'])[:3]) or 'none'}"
                )
                if evidence not in item.evidence:
                    item.evidence.append(evidence)
                for reason in breakdown.reasons:
                    tagged = f"Compatibility rationale: {reason}"
                    if tagged not in item.evidence:
                        item.evidence.append(tagged)
            reranked.append(item)
        return sorted(reranked, key=lambda item: (-item.support_score, item.premise, item.hidden_goal))

    def _candidate_compatibility_score(self, candidate: PremiseCandidate, support_profile: dict[str, set[str] | str]) -> float:
        concepts = support_profile['concepts']
        scripts = support_profile['scripts']
        requirements = support_profile['requirements']
        goal_hints = support_profile['goal_hints']
        score = 0.0
        if candidate.premise in requirements:
            score += 0.9
        if candidate.premise in scripts:
            score += 0.6
        if candidate.hidden_goal and candidate.hidden_goal in goal_hints:
            score += 0.55
        if candidate.premise == 'open_access' and concepts & {'bag', 'drawer', 'door', 'cabinet', 'box', 'bottle', 'jar', 'bin', 'pouch', 'suitcase'}:
            score += 0.45
        if candidate.premise == 'available_space' and concepts & {'bag', 'box', 'bin', 'pouch', 'suitcase'}:
            score += 0.35
        if candidate.premise == 'vehicle_present' and concepts & {'car_wash', 'car'}:
            score += 0.5
        if candidate.premise == 'subquadratic_complexity' and 'efficient_solution_goal' in goal_hints:
            score += 0.45
        return round(min(1.0, score), 4)

    def _query_support_profile(self, graph: StructuredMeaningGraph) -> dict[str, set[str] | str]:
        concepts = set(self._query_concepts(graph))
        scripts: set[str] = set()
        requirements: set[str] = set()
        goal_hints: set[str] = set()
        for concept in concepts:
            data = lookup_concept(concept)
            scripts.update(data.get('scripts', []))
            requirements.update(data.get('requires', []))
            goal = self._default_goal_for_concept(concept, graph.query.lower())
            if goal:
                goal_hints.add(goal)
        if self._looks_like_cp_complexity_case(graph.query.lower()):
            goal_hints.add('efficient_solution_goal')
            requirements.add('subquadratic_complexity')
        return {
            'concepts': concepts,
            'scripts': scripts,
            'requirements': requirements,
            'goal_hints': goal_hints,
            'query_text': graph.query.lower(),
        }

    @staticmethod
    def _deduplicate_candidates(candidates: Iterable[PremiseCandidate]) -> List[PremiseCandidate]:
        priority = {'script_memory': 5, 'memory': 4, 'operator_memory': 4, 'script': 3, 'rule': 2, 'graph': 1}
        best: dict[tuple[str, str, str], PremiseCandidate] = {}
        for item in candidates:
            key = (item.premise, item.hidden_goal, item.candidate_type)
            existing = best.get(key)
            if existing is None:
                best[key] = item
                continue
            replace = item.support_score > existing.support_score
            close_competitor = abs(item.support_score - existing.support_score) <= 0.2 and priority.get(item.source, 0) > priority.get(existing.source, 0)
            if replace or close_competitor:
                item.evidence = list(dict.fromkeys(existing.evidence + item.evidence))
                best[key] = item
            else:
                existing.evidence = list(dict.fromkeys(existing.evidence + item.evidence))
        return sorted(best.values(), key=lambda item: (-item.support_score, item.premise, item.hidden_goal))

    @staticmethod
    def _deduplicate_validations(items: Iterable[PremiseValidation]) -> List[PremiseValidation]:
        best: dict[tuple[str, str], PremiseValidation] = {}
        for item in items:
            key = (item.premise, item.hidden_goal)
            existing = best.get(key)
            if existing is None or item.support_score > existing.support_score:
                best[key] = item
        return sorted(best.values(), key=lambda item: (-item.goal_relevance, -item.support_score, item.premise))

    @staticmethod
    def _merge_unique(existing: Iterable[str], new_items: Iterable[str]) -> List[str]:
        merged = list(existing)
        for item in new_items:
            if item not in merged:
                merged.append(item)
        return merged


    @staticmethod
    def _inject_premise_support_operators(graph: StructuredMeaningGraph, hidden_goals: List[str]) -> None:
        operator_map = {
            'clean_car_goal': ['SERVICE_GOAL_OPERATOR', 'GOAL_PRESERVATION_OPERATOR'],
            'booking_or_inquiry_goal': ['SERVICE_GOAL_OPERATOR', 'GOAL_PRESERVATION_OPERATOR'],
            'reservation_check_goal': ['SERVICE_GOAL_OPERATOR', 'GOAL_PRESERVATION_OPERATOR'],
            'store_book_in_bag_goal': ['CONTAINMENT_GOAL_OPERATOR', 'GOAL_PRESERVATION_OPERATOR'],
            'retrieve_item_from_drawer_goal': ['GOAL_PRESERVATION_OPERATOR'],
            'retrieve_item_from_cabinet_goal': ['GOAL_PRESERVATION_OPERATOR'],
            'retrieve_item_from_box_goal': ['GOAL_PRESERVATION_OPERATOR'],
            'retrieve_item_from_bin_goal': ['GOAL_PRESERVATION_OPERATOR'],
            'retrieve_item_from_pouch_goal': ['GOAL_PRESERVATION_OPERATOR'],
            'retrieve_item_from_suitcase_goal': ['GOAL_PRESERVATION_OPERATOR'],
            'pour_from_bottle_goal': ['GOAL_PRESERVATION_OPERATOR'],
            'pass_through_door_goal': ['GOAL_PRESERVATION_OPERATOR'],
            'efficient_solution_goal': ['GOAL_PRESERVATION_OPERATOR'],
        }
        for goal in hidden_goals:
            for operator_name in operator_map.get(goal, []):
                if any(existing.name == operator_name for existing in graph.induced_operators):
                    continue
                family = operator_name.lower().replace('_operator', '')
                graph.induced_operators.append(
                    OperatorCandidate(
                        name=operator_name,
                        family=family,
                        arity=2,
                        input_types=['hidden_goal', 'premise'],
                        output_type='operator',
                        description='Premise-layer support operator recovered from hidden-goal structure.',
                        confidence=0.72,
                        provenance=['premise_explorer:support_operator'],
                    )
                )
