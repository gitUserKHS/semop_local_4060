from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.kernel.adapters import (
    StructuredMeaningGraphAdapter,
    TypedKernelBridge,
)
from semop.structures import Edge, Node, PremiseValidation, StructuredMeaningGraph


class TypedAdapterTests(unittest.TestCase):
    def _graph(self) -> StructuredMeaningGraph:
        return StructuredMeaningGraph(
            query="배포해도 될까?",
            intent="decision",
            nodes=[Node("service", "service", "concept")],
            edges=[Edge("service", "mystery relation", "external")],
            hidden_goals=["deploy"],
            required_premises=["tests_pass"],
            satisfied_premises=["tests_pass"],
            premise_validations=[
                PremiseValidation(
                    premise="tests_pass",
                    hidden_goal="deploy",
                    status="satisfied",
                    requirement_state="satisfied",
                )
            ],
        )

    def test_unknown_relations_are_preserved_but_not_proof_eligible(self) -> None:
        adapted = StructuredMeaningGraphAdapter().adapt(self._graph())
        unknown = [
            fact
            for fact in adapted.state.facts
            if fact.atom.predicate.name == "MYSTERY_RELATION"
        ]

        self.assertEqual(adapted.unverified_relations, ("MYSTERY_RELATION",))
        self.assertEqual(len(unknown), 1)
        self.assertFalse(unknown[0].proof_eligible)

    def test_shadow_keeps_legacy_report_and_typed_mode_projects(self) -> None:
        shadow_graph = self._graph()
        shadow = TypedKernelBridge().run(shadow_graph, mode="shadow")
        self.assertTrue(shadow.typed_result.success)
        self.assertIsNone(shadow.graph.operator_execution)
        self.assertTrue(any("typed_shadow:" in item for item in shadow.graph.audit_trace))

        typed_graph = self._graph()
        typed = TypedKernelBridge().run(typed_graph, mode="typed")
        self.assertTrue(typed.typed_result.verified)
        self.assertIsNotNone(typed.graph.operator_execution)
        self.assertTrue(typed.graph.operator_execution.support_trace)

    def test_legacy_mode_does_not_run_typed_kernel(self) -> None:
        graph = self._graph()
        result = TypedKernelBridge().run(graph, mode="legacy")
        self.assertIsNone(result.typed_result)
        self.assertEqual(graph.audit_trace, [])


if __name__ == "__main__":
    unittest.main()
