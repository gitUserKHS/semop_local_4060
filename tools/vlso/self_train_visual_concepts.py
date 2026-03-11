from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import PseudoLabelAcceptanceConfig, VisualConceptSelfTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description='Run unsupervised / weakly supervised visual self-training from candidate JSONL')
    parser.add_argument('--candidates', required=True, help='candidate JSONL from build_visual_concept_candidates.py')
    parser.add_argument('--store', required=True, help='output SQLite prototype store path')
    parser.add_argument('--summary-output', help='optional JSON summary output path')
    parser.add_argument('--reference-store', help='optional existing concept store used as prior memory')
    parser.add_argument('--weights', help='optional classifier weights JSON path')
    parser.add_argument('--cluster-threshold', type=float, default=0.9)
    parser.add_argument('--pseudo-threshold', type=float, default=0.72)
    parser.add_argument('--consensus-threshold', type=float, default=0.55)
    parser.add_argument('--concept-match-threshold', type=float, default=0.86)
    parser.add_argument('--min-cluster-size', type=int, default=2)
    args = parser.parse_args()

    config = PseudoLabelAcceptanceConfig(
        cluster_similarity_threshold=args.cluster_threshold,
        pseudo_confidence_threshold=args.pseudo_threshold,
        cluster_consensus_threshold=args.consensus_threshold,
        concept_match_threshold=args.concept_match_threshold,
        min_cluster_size=args.min_cluster_size,
    )
    summary = VisualConceptSelfTrainer(
        concept_store_path=args.reference_store,
        weights_path=args.weights,
    ).train_candidates_jsonl(
        args.candidates,
        args.store,
        summary_output=args.summary_output,
        config=config,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
