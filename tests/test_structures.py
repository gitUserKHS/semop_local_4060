from __future__ import annotations

import os
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.structures import StructuredMeaningGraph


class StructuresTests(unittest.TestCase):
    def test_structured_meaning_graph_roundtrip_preserves_scenario(self) -> None:
        graph = StructuredMeaningGraph(
            query='What should I verify before moving?',
            intent='goal_directed_reasoning',
            domain='warehouse_exception',
            scenario='exception_response',
            source_context='Stop first and confirm approval and alternate route.',
        )
        restored = StructuredMeaningGraph.from_dict(graph.model_dump())
        self.assertEqual(restored.domain, 'warehouse_exception')
        self.assertEqual(restored.scenario, 'exception_response')


if __name__ == '__main__':
    unittest.main()
