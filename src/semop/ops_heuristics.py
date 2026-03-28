from __future__ import annotations

from typing import Iterable

from .commonsense_kb import concept_label, kb_relations_for_concept, scripts_for_concept
from .semantic_operators import apply_many, apply_operator
from .structures import Node, PlanStep, StructuredMeaningGraph


class OpsHeuristicExtractor:
    def matches(self, query: str) -> bool:
        lowered = query.lower()
        keywords = [
            "forklift",
            "pallet",
            "rack",
            "aisle",
            "barcode",
            "label",
            "scan",
            "warehouse",
            "지게차",
            "팔레트",
            "랙",
            "통로",
            "바코드",
            "라벨",
            "피킹",
            "포장",
            "창고",
            "sop",
            "manual",
        ]
        return any(token in lowered for token in keywords)

    def extract(self, query: str) -> StructuredMeaningGraph:
        lowered = query.lower()
        if self._looks_like_blocked_access_case(lowered):
            return self._blocked_access_case(query)
        if self._looks_like_forklift_case(lowered):
            return self._forklift_exception_case(query)
        if self._looks_like_label_case(lowered):
            return self._label_mismatch_case(query)
        return self._generic_ops_case(query)

    @staticmethod
    def _looks_like_blocked_access_case(lowered: str) -> bool:
        mentions_route = any(token in lowered for token in [
            "aisle", "lane", "corridor", "route", "path",
            "통로", "경로", "진입", "접근",
        ])
        mentions_blocker = any(token in lowered for token in [
            "blocked", "blocking", "closed",
            "차단", "막혀", "막혔", "막힌", "봉쇄", "불가",
        ])
        mentions_approval = any(token in lowered for token in [
            "approval", "approved", "permit", "permission",
            "승인", "허가", "결재",
        ])
        asks_next_step = any(token in lowered for token in [
            "what should", "should i", "can i", "how do i", "how should",
            "어떻게", "해야", "되나", "되나요", "괜찮", "가능", "해도", "어쩌",
        ])
        return mentions_route and (mentions_blocker or mentions_approval) and asks_next_step

    @staticmethod
    def _looks_like_forklift_case(lowered: str) -> bool:
        return any(token in lowered for token in ["forklift", "지게차", "pallet", "팔레트"]) and any(
            token in lowered for token in ["aisle", "통로", "approval", "승인", "rack", "랙", "blocked", "막"]
        )

    @staticmethod
    def _looks_like_label_case(lowered: str) -> bool:
        return any(token in lowered for token in ["barcode", "바코드", "label", "라벨", "scan", "스캔"]) and any(
            token in lowered for token in ["mismatch", "불일치", "packing", "포장", "shipment", "출고", "pick", "피킹"]
        )

    def _seed_graph(self, query: str, intent: str, concepts: Iterable[str]) -> StructuredMeaningGraph:
        graph = StructuredMeaningGraph(query=query, intent=intent, domain="warehouse_ops")
        for concept_id in concepts:
            graph.add_node(
                Node(
                    id=concept_id,
                    label=concept_label(concept_id),
                    kind=self._kind_for(concept_id),
                    provenance=[f"ops:entity:{concept_id}"],
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

    def _blocked_access_case(self, query: str) -> StructuredMeaningGraph:
        concepts = [
            "worker",
            "aisle",
            "blocked_aisle",
            "supervisor_approval",
            "safety_clearance",
            "staging_area",
            "incident_report",
            "stop_work",
        ]
        graph = self._seed_graph(query, "warehouse_access_exception_response", concepts)
        graph.add_node(Node(id="move_through_blocked_aisle", label="막힌 통로로 직접 진입", kind="task", provenance=["ops:task:move_through_blocked_aisle"]))

        apply_operator(graph, "REQUIRES", "move_through_blocked_aisle", "supervisor_approval", provenance=["ops:blocked_access_case"])
        apply_operator(graph, "REQUIRES", "move_through_blocked_aisle", "safety_clearance", provenance=["ops:blocked_access_case"])
        apply_operator(graph, "BLOCKED_BY", "move_through_blocked_aisle", "blocked_aisle", provenance=["ops:blocked_access_case"])
        apply_operator(graph, "ALTERNATIVE", "move_through_blocked_aisle", "staging_area", provenance=["ops:blocked_access_case"])
        apply_operator(graph, "ALTERNATIVE", "move_through_blocked_aisle", "incident_report", provenance=["ops:blocked_access_case"])
        apply_operator(graph, "ALTERNATIVE", "move_through_blocked_aisle", "stop_work", provenance=["ops:blocked_access_case"])
        apply_operator(graph, "GOAL_OF", "move_through_blocked_aisle", "worker", provenance=["ops:blocked_access_case"])

        graph.candidate_actions.extend([
            "통로 차단 상태와 실제 위험 요인을 먼저 확인한다",
            "승인 없이는 직접 진입이나 진행을 멈춘다",
            "우회 경로, 스테이징, hold 중 가능한 대체 흐름을 고른다",
            "위험이 지속되면 보고하고 stop work로 전환한다",
        ])
        graph.creative_alternatives.extend([
            "승인이 늦으면 같은 목표를 유지한 채 대체 작업 순서를 먼저 재배치한다.",
            "통로 복구 전까지 스테이징 구역이나 hold 상태로 전환해 현장 충돌을 막는다.",
        ])
        graph.plan = [
            PlanStep(
                id="step1",
                action="통로가 실제로 막혀 있는지, 왜 막혔는지, 안전 구역이 함께 영향을 받는지 먼저 확인한다.",
                rationale="차단 상태는 BLOCKED_BY 관계의 근거이므로 먼저 사실로 잠가야 한다.",
                requires=["blocked_aisle"],
            ),
            PlanStep(
                id="step2",
                action="승인이 없으면 직접 진입이나 진행을 멈추고 승인 요청 또는 상위 보고로 올린다.",
                rationale="REQUIRES 관계가 충족되지 않으면 실행보다 승인 확보가 먼저다.",
                requires=["supervisor_approval"],
            ),
            PlanStep(
                id="step3",
                action="작업이 급하면 우회 경로를 찾거나 스테이징 구역으로 전환해서 목표를 보류된 형태로 유지한다.",
                rationale="ALTERNATIVE 연산자는 목표를 버리지 않고 위험한 직접 실행만 피하게 한다.",
                requires=["staging_area"],
            ),
            PlanStep(
                id="step4",
                action="차단과 승인 문제가 계속되면 incident report를 남기고 stop work 상태로 묶는다.",
                rationale="차단과 승인 누락이 동시에 남아 있으면 현장 예외 대응은 보고와 작업 중지가 우선이다.",
                requires=["incident_report", "stop_work"],
            ),
        ]
        graph.invalid_advice.extend([
            "승인 없이 그냥 들어가 보라는 조언",
            "막힌 통로라도 일단 진행하라는 조언",
        ])
        graph.warnings.extend([
            "blocked aisle 상태에서는 직접 진입보다 차단 해소와 우회 판단이 먼저다.",
            "승인 없는 진행은 현장 SOP와 안전 규칙을 동시에 어길 수 있다.",
        ])
        return graph

    def _forklift_exception_case(self, query: str) -> StructuredMeaningGraph:
        concepts = [
            "worker",
            "forklift",
            "pallet",
            "rack",
            "blocked_aisle",
            "supervisor_approval",
            "forklift_certification",
            "safety_clearance",
            "staging_area",
            "incident_report",
            "stop_work",
        ]
        graph = self._seed_graph(query, "warehouse_exception_response", concepts)
        graph.add_node(Node(id="move_pallet_to_rack", label="팔레트를 랙으로 이동", kind="task", provenance=["ops:task:move_pallet_to_rack"]))

        apply_operator(graph, "REQUIRES", "move_pallet_to_rack", "forklift_certification", provenance=["ops:forklift_case"])
        apply_operator(graph, "REQUIRES", "move_pallet_to_rack", "supervisor_approval", provenance=["ops:forklift_case"])
        apply_operator(graph, "REQUIRES", "move_pallet_to_rack", "safety_clearance", provenance=["ops:forklift_case"])
        apply_operator(graph, "BLOCKED_BY", "move_pallet_to_rack", "blocked_aisle", provenance=["ops:forklift_case"])
        apply_operator(graph, "ALTERNATIVE", "move_pallet_to_rack", "staging_area", provenance=["ops:forklift_case"])
        apply_operator(graph, "ALTERNATIVE", "move_pallet_to_rack", "incident_report", provenance=["ops:forklift_case"])
        apply_operator(graph, "ALTERNATIVE", "move_pallet_to_rack", "stop_work", provenance=["ops:forklift_case"])
        apply_operator(graph, "GOAL_OF", "move_pallet_to_rack", "worker", provenance=["ops:forklift_case"])

        graph.candidate_actions.extend([
            "통로와 안전 구역 상태를 확인한다",
            "승인과 자격 상태를 확인한다",
            "막힌 통로면 스테이징 구역에 임시 보관한다",
            "사고 가능성이 있으면 작업을 중지하고 보고한다",
        ])
        graph.creative_alternatives.extend([
            "통로가 복구될 때까지 스테이징 구역에서 작업 순서를 재배치한다.",
            "즉시 적재가 불가하면 incident report와 hold 태그를 남겨 다음 교대조가 같은 실수를 반복하지 않게 한다.",
        ])
        graph.plan = [
            PlanStep(
                id="step1",
                action="팔레트 이동 전에 지게차 자격, 관리자 승인, 안전 확인이 모두 있는지 먼저 점검한다.",
                rationale="고위험 현장 작업은 prerequisite 누락이 곧 사고로 이어지므로 승인과 자격을 먼저 잠근다.",
                requires=["forklift_certification", "supervisor_approval", "safety_clearance"],
            ),
            PlanStep(
                id="step2",
                action="통로가 막혀 있으면 즉시 직접 이동을 중단하고 차단 원인을 기록한다.",
                rationale="BLOCKED_BY 관계는 억지 실행이 아니라 중지와 우회 판단을 의미한다.",
                requires=["blocked_aisle"],
            ),
            PlanStep(
                id="step3",
                action="긴급 출고가 아니면 스테이징 구역에 임시 보관하고 복구 시점이나 대체 작업 순서를 잡는다.",
                rationale="ALTERNATIVE 연산자는 목표를 유지하면서 위험한 직접 실행을 피하게 한다.",
                requires=["staging_area"],
            ),
            PlanStep(
                id="step4",
                action="안전 이슈가 지속되면 incident report를 남기고 stop work 상태로 전환한다.",
                rationale="현장 SOP는 문제를 숨긴 채 진행하는 것보다 작업 중지와 보고를 우선한다.",
                requires=["incident_report", "stop_work"],
            ),
        ]
        graph.invalid_advice.extend([
            "승인 없이 그냥 지게차로 밀어 넣으라는 조언",
            "통로가 막혔는데도 바로 랙으로 이동하라는 조언",
        ])
        graph.warnings.extend([
            "forklift 이동은 관리자 승인과 안전 확인이 빠지면 실행 불가로 봐야 한다.",
            "blocked aisle 상태에서 직접 이동을 계속하면 현장 예외 대응 규칙을 위반할 수 있다.",
        ])
        return graph

    def _label_mismatch_case(self, query: str) -> StructuredMeaningGraph:
        concepts = [
            "worker",
            "barcode",
            "order_label",
            "label_mismatch",
            "verified_label_match",
            "rescan_item",
            "hold_shipment",
            "supervisor_approval",
            "pick_ticket",
            "package",
        ]
        graph = self._seed_graph(query, "warehouse_quality_gate", concepts)
        graph.add_node(Node(id="pack_item_for_shipment", label="출고 포장 진행", kind="task", provenance=["ops:task:pack_item_for_shipment"]))

        apply_operator(graph, "REQUIRES", "pack_item_for_shipment", "verified_label_match", provenance=["ops:label_case"])
        apply_operator(graph, "BLOCKED_BY", "pack_item_for_shipment", "label_mismatch", provenance=["ops:label_case"])
        apply_operator(graph, "ALTERNATIVE", "pack_item_for_shipment", "rescan_item", provenance=["ops:label_case"])
        apply_operator(graph, "ALTERNATIVE", "pack_item_for_shipment", "hold_shipment", provenance=["ops:label_case"])
        apply_operator(graph, "ALTERNATIVE", "pack_item_for_shipment", "supervisor_approval", provenance=["ops:label_case"])
        apply_operator(graph, "GOAL_OF", "pack_item_for_shipment", "worker", provenance=["ops:label_case"])

        graph.candidate_actions.extend([
            "바코드와 피킹 티켓을 다시 스캔한다",
            "라벨 불일치면 출고를 보류한다",
            "관리자 승인 전에는 포장을 닫지 않는다",
        ])
        graph.creative_alternatives.extend([
            "문제 물건을 hold 구역으로 빼서 정상 출고 라인을 오염시키지 않게 한다.",
            "재스캔 이력이 반복되면 교육용 케이스로 남겨 신입 온보딩 퀴즈에 재사용한다.",
        ])
        graph.plan = [
            PlanStep(
                id="step1",
                action="바코드, 피킹 티켓, 주문 라벨을 다시 대조해서 verified_label_match 상태를 먼저 만든다.",
                rationale="출고 포장은 일치 확인이 prerequisite다.",
                requires=["verified_label_match"],
            ),
            PlanStep(
                id="step2",
                action="라벨 불일치가 있으면 바로 포장을 진행하지 말고 label mismatch 원인을 분리한다.",
                rationale="BLOCKED_BY 관계는 품질 게이트 실패를 뜻하므로 direct execution을 멈춰야 한다.",
                requires=["label_mismatch"],
            ),
            PlanStep(
                id="step3",
                action="재스캔으로 해결되지 않으면 hold shipment 상태로 바꾸고 관리자 승인을 요청한다.",
                rationale="품질 불일치는 대체 흐름과 escalation을 같이 써야 한다.",
                requires=["hold_shipment", "supervisor_approval"],
            ),
        ]
        graph.invalid_advice.extend([
            "라벨이 달라도 일단 포장해서 보내라는 조언",
            "재확인 없이 출고를 진행하라는 조언",
        ])
        graph.warnings.extend([
            "label mismatch는 context misread보다 심한 출고 오류로 이어질 수 있다.",
            "검증되지 않은 포장은 executability보다 먼저 품질 차단 대상으로 봐야 한다.",
        ])
        return graph

    def _generic_ops_case(self, query: str) -> StructuredMeaningGraph:
        concepts = ["worker", "supervisor_approval", "safety_clearance", "staging_area"]
        graph = self._seed_graph(query, "operations_sop_reasoning", concepts)
        graph.candidate_actions.extend([
            "작업 목표와 위험 조건을 분리한다",
            "SOP prerequisite와 승인 조건을 먼저 확인한다",
            "직접 실행이 막히면 hold 또는 escalation 경로를 연다",
        ])
        graph.plan = [
            PlanStep(
                id="step1",
                action="질문을 작업 목표, 위험 요인, 승인 필요 여부로 나눠 본다.",
                rationale="현장 운영 Copilot은 문장을 그대로 답하지 않고 SOP 구조로 다시 읽어야 한다.",
            ),
            PlanStep(
                id="step2",
                action="승인, 안전, 품질 확인 같은 prerequisite가 빠졌는지 확인한다.",
                rationale="운영 도메인에서는 prerequisite 누락이 바로 invalid advice로 이어진다.",
                requires=["supervisor_approval", "safety_clearance"],
            ),
            PlanStep(
                id="step3",
                action="직접 실행이 위험하면 staging, hold, escalation 중 하나로 우회한다.",
                rationale="예외 대응은 실행보다 통제가 우선이다.",
                requires=["staging_area"],
            ),
        ]
        graph.warnings.append("ops heuristic fallback was used; attach SOP context for stronger document grounding.")
        return graph
