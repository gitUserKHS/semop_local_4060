from __future__ import annotations

from .structures import StructuredMeaningGraph


def validate_graph(graph: StructuredMeaningGraph) -> StructuredMeaningGraph:
    relations = graph.relation_tuples()

    for step in graph.plan:
        if "걸어" in step.action and ("car_wash", "REQUIRES", "vehicle_present") in relations:
            step.status = "invalid"
            if "차 없이 걸어서 세차장까지 가라고 하는 조언" not in graph.invalid_advice:
                graph.invalid_advice.append("차 없이 걸어서 세차장까지 가라고 하는 조언")

        if ("insert_book", "REQUIRES", "open_access") in relations and "넣" in step.action:
            has_open_step = any(("열" in candidate.action) or ("open" in candidate.action.lower()) for candidate in graph.plan)
            if not has_open_step:
                step.status = "warning"
                _append_unique(graph.warnings, "책을 넣기 전에 가방을 여는 단계가 빠져 있다.")

        if ("drive_to_car_wash", "BLOCKED_BY", "traffic") in relations and "바로" in step.action and "우회" not in step.action:
            if step.status == "valid":
                step.status = "warning"
            _append_unique(graph.warnings, "교통체증이 큰 경우에는 시간 조정이나 우회 경로 검토가 먼저 필요하다.")

    if any(step.status == "invalid" for step in graph.plan):
        _append_unique(graph.warnings, "계획 안에 실행 불가능한 단계가 포함되어 있다.")

    graph.invalid_advice = _dedupe(graph.invalid_advice)
    graph.warnings = _dedupe(graph.warnings)
    graph.candidate_actions = _dedupe(graph.candidate_actions)
    graph.creative_alternatives = _dedupe(graph.creative_alternatives)
    graph.inferred_scripts = _dedupe(graph.inferred_scripts)
    return graph


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
