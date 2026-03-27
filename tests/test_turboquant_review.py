from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semop import TurboQuantReviewPlanner


class TurboQuantReviewTests(unittest.TestCase):
    def test_quantizer_spreads_distinct_domains_into_multiple_clusters(self) -> None:
        planner = TurboQuantReviewPlanner(bit_width=4, max_clusters=8, spawn_threshold=0.55)
        planner.add(label='ops-1', text='warehouse blocked approval route reroute manager', domain='warehouse_exception', scenario='exception_response')
        planner.add(label='doc-1', text='document evidence grounded claim cite sop sentence', domain='service', scenario='document_grounding')
        planner.add(label='vis-1', text='visual opening container topology geometry reconstruction', domain='vlso', scenario='geometry_visual')
        summary = planner.summarize()
        self.assertGreaterEqual(summary.used_clusters, 2)
        self.assertGreater(summary.coverage_score, 0.0)
        self.assertIn('service/document_grounding', summary.domain_scenario_pairs)


if __name__ == '__main__':
    unittest.main()
