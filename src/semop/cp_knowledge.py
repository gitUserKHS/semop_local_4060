from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List


@dataclass
class CpAlgorithmKnowledge:
    id: str
    family: str
    triggers: List[str]
    hidden_concepts: List[str]
    time_complexity: str
    memory_complexity: str
    source_ids: List[str]
    template_kind: str


@dataclass
class CpLogicalFrame:
    id: str
    triggers: List[str]
    inferred_concepts: List[str]
    reduction_goal: str
    reasoning_operator: str


@dataclass
class CpDslOperator:
    id: str
    category: str
    description: str
    trigger_hints: List[str]


@dataclass
class CpKnowledgeBase:
    sources: Dict[str, Dict[str, str]]
    hardware_profile: Dict[str, object]
    compiler_profile: Dict[str, object]
    algorithms: List[CpAlgorithmKnowledge]
    logical_frames: List[CpLogicalFrame]
    dsl_operators: List[CpDslOperator]
    memory_schema: Dict[str, object]
    coding_rules: List[Dict[str, object]]


class CpKnowledgeLoader:
    def __init__(self, path: str | Path = "data/knowledge/cp_knowledge.json") -> None:
        self.path = Path(path)

    def load(self) -> CpKnowledgeBase:
        payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        sources = {item["id"]: item for item in payload.get("sources", [])}
        algorithms = [CpAlgorithmKnowledge(**item) for item in payload.get("algorithms", [])]
        logical_frames = [CpLogicalFrame(**item) for item in payload.get("logical_frames", [])]
        dsl_operators = [CpDslOperator(**item) for item in payload.get("dsl_operators", [])]
        return CpKnowledgeBase(
            sources=sources,
            hardware_profile=payload.get("hardware_profile", {}),
            compiler_profile=payload.get("compiler_profile", {}),
            algorithms=algorithms,
            logical_frames=logical_frames,
            dsl_operators=dsl_operators,
            memory_schema=payload.get("memory_schema", {}),
            coding_rules=payload.get("coding_rules", []),
        )
