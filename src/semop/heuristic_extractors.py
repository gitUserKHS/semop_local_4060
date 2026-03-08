from __future__ import annotations

import re
from typing import Iterable, List

from .commonsense_kb import concept_label, kb_relations_for_concept, match_concepts, normalize_concept, scripts_for_concept
from .ops_heuristics import OpsHeuristicExtractor
from .semantic_operators import apply_many, apply_operator
from .structures import Node, PlanStep, StructuredMeaningGraph


class HeuristicExtractor:
    def __init__(self) -> None:
        self.ops = OpsHeuristicExtractor()

    def extract(self, query: str) -> StructuredMeaningGraph:
        lowered = query.lower()
        if self._looks_like_bag_case(lowered):
            return self._bag_book_case(query)
        if self._looks_like_car_wash_case(lowered):
            return self._car_wash_case(query)
        if self.ops.matches(query):
            return self.ops.extract(query)
        return self._generic_case(query)

    @staticmethod
    def _looks_like_bag_case(lowered: str) -> bool:
        return (("가방" in lowered) or ("bag" in lowered)) and (("책" in lowered) or ("book" in lowered))

    @staticmethod
    def _looks_like_car_wash_case(lowered: str) -> bool:
        has_car_wash = ("세차장" in lowered) or ("세차" in lowered) or ("car wash" in lowered)
        has_constraint = any(token in lowered for token in ["막히", "정체", "traffic", "멀", "멀고"])
        return has_car_wash and has_constraint

    def _seed_graph(self, query: str, intent: str, concepts: Iterable[str], provenance_prefix: str = "heuristic") -> StructuredMeaningGraph:
        graph = StructuredMeaningGraph(query=query, intent=intent)
        for concept_id in concepts:
            graph.add_node(
                Node(
                    id=concept_id,
                    label=concept_label(concept_id),
                    kind=self._kind_for(concept_id),
                    provenance=[f"{provenance_prefix}:entity:{concept_id}"],
                )
            )
            apply_many(graph, kb_relations_for_concept(concept_id))
            for script in scripts_for_concept(concept_id):
                if script not in graph.inferred_scripts:
                    graph.inferred_scripts.append(script)
        return graph

    @staticmethod
    def _kind_for(concept_id: str) -> str:
        from .commonsense_kb import lookup_concept

        return lookup_concept(concept_id).get("kind", "concept")

    def _bag_book_case(self, query: str) -> StructuredMeaningGraph:
        concepts = ["user", "bag", "book", "zipper", "compartment", "open_access", "available_space"]
        graph = self._seed_graph(query, "insert_object_into_container", concepts)
        graph.add_node(Node(id="insert_book", label="책 넣기", kind="task", provenance=["heuristic:task:insert_book"]))

        apply_operator(graph, "REQUIRES", "insert_book", "open_access", provenance=["heuristic:bag_case"])
        apply_operator(graph, "REQUIRES", "insert_book", "available_space", provenance=["heuristic:bag_case"])
        apply_operator(graph, "GOAL_OF", "insert_book", "user", provenance=["heuristic:bag_case"])
        apply_operator(graph, "CONTAINS", "bag", "book", provenance=["heuristic:bag_case"])

        graph.candidate_actions.extend(["가방을 연다", "빈 공간을 확인한다", "책을 넣는다", "다시 닫는다"])
        graph.creative_alternatives.extend(
            [
                "지퍼가 아닌 다른 개폐 구조라면 먼저 열리는 방향을 확인한다.",
                "책이 크면 방향을 바꾸거나 다른 수납칸을 찾는다.",
            ]
        )
        graph.plan = [
            PlanStep(
                id="step1",
                action="가방의 지퍼나 덮개를 열어 수납공간에 접근 가능한 상태를 만든다.",
                rationale="책을 넣으려면 먼저 open_access 전제조건이 충족되어야 한다.",
                requires=["open_access"],
            ),
            PlanStep(
                id="step2",
                action="수납공간에 책이 들어갈 만큼 빈 공간이 있는지 확인한다.",
                rationale="containment 동작은 접근 가능성과 공간 확보를 함께 요구한다.",
                requires=["available_space"],
            ),
            PlanStep(
                id="step3",
                action="책을 수납공간 안으로 넣고 걸리는 부분이 없는지 확인한다.",
                rationale="목표 동작은 객체를 컨테이너 내부 공간으로 이동시키는 것이다.",
                requires=["open_access", "available_space"],
            ),
            PlanStep(
                id="step4",
                action="필요하면 다시 닫아 내용물이 빠지지 않게 고정한다.",
                rationale="가방의 closure는 보관 상태를 유지하기 위한 후속 단계다.",
                requires=["book inserted"],
            ),
        ]
        graph.invalid_advice.append("가방 구조를 무시하고 책을 그냥 눌러 넣으라고 하는 조언")
        return graph

    def _car_wash_case(self, query: str) -> StructuredMeaningGraph:
        concepts = ["user", "car", "car_wash", "traffic", "nearby_car_wash", "mobile_detailer", "delay_departure"]
        graph = self._seed_graph(query, "plan_under_travel_constraint", concepts)
        graph.add_node(Node(id="drive_to_car_wash", label="세차장으로 이동", kind="task", provenance=["heuristic:task:drive_to_car_wash"]))
        graph.add_node(Node(id="detour_route", label="우회 경로", kind="route_strategy", provenance=["heuristic:entity:detour_route"]))

        apply_many(graph, kb_relations_for_concept("detour_route"))
        apply_operator(graph, "REQUIRES", "drive_to_car_wash", "vehicle_present", provenance=["heuristic:carwash_case"])
        apply_operator(graph, "BLOCKED_BY", "drive_to_car_wash", "traffic", provenance=["heuristic:carwash_case"])
        apply_operator(graph, "GOAL_OF", "drive_to_car_wash", "user", provenance=["heuristic:carwash_case"])
        apply_operator(graph, "ALTERNATIVE", "drive_to_car_wash", "nearby_car_wash", provenance=["heuristic:carwash_case"])
        apply_operator(graph, "ALTERNATIVE", "drive_to_car_wash", "mobile_detailer", provenance=["heuristic:carwash_case"])
        apply_operator(graph, "ALTERNATIVE", "drive_to_car_wash", "delay_departure", provenance=["heuristic:carwash_case"])
        apply_operator(graph, "ALTERNATIVE", "drive_to_car_wash", "detour_route", provenance=["heuristic:carwash_case"])

        graph.candidate_actions.extend(["출발을 미룬다", "가까운 세차장을 찾는다", "출장 세차를 부른다", "우회 경로를 본다"])
        graph.creative_alternatives.extend(
            [
                "세차가 급하지 않다면 교통이 완화되는 시간대로 미룬다.",
                "세차 목적이 외관 관리라면 출장 세차 같은 비이동형 서비스로 목표를 재설정한다.",
            ]
        )
        graph.plan = [
            PlanStep(
                id="step1",
                action="세차가 지금 꼭 필요한지 먼저 판단한다.",
                rationale="교통체증이 큰 상황에서는 urgency 판단이 계획 분기점이 된다.",
                requires=["goal clarity"],
            ),
            PlanStep(
                id="step2",
                action="급하지 않다면 교통이 덜한 시간대로 출발을 미루고 다시 확인한다.",
                rationale="교통은 이동 계획을 막는 장애물이므로 시간 이동이 가장 낮은 비용의 대응이다.",
                requires=["flexible schedule"],
            ),
            PlanStep(
                id="step3",
                action="급하면 더 가까운 세차장, 출장 세차, 우회 경로 중 이동 비용이 가장 낮은 대안을 고른다.",
                rationale="ALTERNATIVE 연산자는 같은 목표를 다른 수단으로 달성하게 해 준다.",
                requires=["alternative provider exists"],
            ),
            PlanStep(
                id="step4",
                action="반드시 현재 세차장에 가야 한다면 차량을 타고 우회 경로와 예상 시간을 확인한 뒤 이동한다.",
                rationale="세차장 이용은 일반적으로 vehicle_present 전제조건을 가진다.",
                requires=["vehicle_present"],
            ),
        ]
        graph.invalid_advice.append("차 없이 걸어서 세차장까지 가라고 하는 조언")
        graph.warnings.append("세차장 이용 스크립트는 보통 차량 동반을 전제로 한다.")
        return graph

    def _generic_case(self, query: str) -> StructuredMeaningGraph:
        concepts = match_concepts(query)
        if not concepts:
            concepts = self._fallback_tokens(query)
        graph = self._seed_graph(query, self._infer_intent(query), concepts[:8])

        if "car_wash" in graph.node_ids() and "traffic" in graph.node_ids():
            graph.add_node(Node(id="drive_to_car_wash", label="세차장으로 이동", kind="task", provenance=["heuristic:task:drive_to_car_wash"]))
            apply_operator(graph, "BLOCKED_BY", "drive_to_car_wash", "traffic", provenance=["heuristic:generic_traffic"])
            apply_operator(graph, "ALTERNATIVE", "drive_to_car_wash", "nearby_car_wash", provenance=["heuristic:generic_traffic"])
            apply_operator(graph, "ALTERNATIVE", "drive_to_car_wash", "mobile_detailer", provenance=["heuristic:generic_traffic"])

        graph.candidate_actions.extend(self._candidate_actions_from_graph(graph))
        graph.creative_alternatives.extend(self._creative_alternatives_from_graph(graph))
        graph.plan = self._generic_plan(graph)
        if not graph.warnings:
            graph.warnings.append("일반 질의는 휴리스틱 추론으로 처리했다. 더 정교한 구조 추출은 llm 모드가 적합하다.")
        return graph

    @staticmethod
    def _infer_intent(query: str) -> str:
        lowered = query.lower()
        if any(token in lowered for token in ["어떻게", "하려", "why", "how"]):
            return "goal_directed_reasoning"
        if any(token in lowered for token in ["막히", "정체", "불가", "못"]):
            return "constraint_reasoning"
        return "generic_reasoning"

    @staticmethod
    def _fallback_tokens(query: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z가-힣0-9_]+", query)
        return [normalize_concept(token) for token in tokens]

    def _generic_plan(self, graph: StructuredMeaningGraph) -> List[PlanStep]:
        relations = graph.relation_tuples()
        steps: List[PlanStep] = [
            PlanStep(
                id="step1",
                action="질문의 목표, 제약, 장애물을 먼저 분리한다.",
                rationale="구조적 추론은 개체와 관계 분해에서 시작한다.",
            )
        ]

        for source, relation, target in sorted(relations):
            if relation == "REQUIRES":
                steps.append(
                    PlanStep(
                        id=f"require_{source}_{target}",
                        action=f"{concept_label(source)} 관련 행동 전에 {concept_label(target)} 조건을 먼저 충족한다.",
                        rationale="REQUIRES 관계는 실행 가능성을 결정한다.",
                        requires=[target],
                    )
                )
            if relation == "BLOCKED_BY":
                steps.append(
                    PlanStep(
                        id=f"block_{source}_{target}",
                        action=f"{concept_label(target)} 때문에 막히는 경로를 줄이거나 우회한다.",
                        rationale="BLOCKED_BY 관계는 직접 실행보다 장애 처리 우선순위를 뜻한다.",
                        requires=[target],
                    )
                )

        steps.append(
            PlanStep(
                id="step_final",
                action="실행 가능한 후보만 남기고 비용과 대안을 비교해 선택한다.",
                rationale="논리적 추론의 목적은 제약을 통과하는 실행 계획을 고르는 것이다.",
            )
        )
        return steps

    def _candidate_actions_from_graph(self, graph: StructuredMeaningGraph) -> List[str]:
        actions: List[str] = []
        for edge in graph.edges:
            if edge.relation == "AFFORDS":
                phrase = f"{concept_label(edge.source)}에서 {edge.target} 가능성을 본다"
                if phrase not in actions:
                    actions.append(phrase)
            if edge.relation == "ALTERNATIVE":
                phrase = f"{concept_label(edge.target)} 대안을 검토한다"
                if phrase not in actions:
                    actions.append(phrase)
        return actions[:8]

    def _creative_alternatives_from_graph(self, graph: StructuredMeaningGraph) -> List[str]:
        alternatives: List[str] = []
        blocked_targets = [edge.target for edge in graph.edges if edge.relation == "BLOCKED_BY"]
        if blocked_targets:
            alternatives.append("시간, 장소, 수단 중 하나를 바꿔 같은 목표를 우회 달성할 수 있다.")
        if any(edge.relation == "ALTERNATIVE" for edge in graph.edges):
            alternatives.append("정답 하나를 고집하지 말고 대안들의 비용과 제약을 기준으로 재평가한다.")
        if not alternatives:
            alternatives.append("지식 KB에 없는 대상이면 더 작은 하위 문제로 쪼개서 관계를 다시 만든다.")
        return alternatives
