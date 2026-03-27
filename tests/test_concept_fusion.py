from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop.concept_fusion import ConceptFusionEngine


class ConceptFusionTests(unittest.TestCase):
    def test_build_summary_emits_hypotheses_for_math_route(self) -> None:
        engine = ConceptFusionEngine()
        summary = engine.build_summary(
            prompt='Prove the geometry claim and connect the diagram to symbolic proof search.',
            route='math',
            prompt_understanding={
                'likely_domain': 'mathematics',
                'likely_scenario': 'geometry_proof',
                'hidden_constraints': ['Keep the answer grounded by symbolic checks.'],
            },
            payload={'notes': ['Use the diagram world model first.']},
        )
        self.assertTrue(summary.hypotheses)
        self.assertIn('Creative proof', summary.hypotheses[0].label)
        self.assertGreater(summary.hypotheses[0].novelty_score, 0.0)


if __name__ == '__main__':
    unittest.main()
