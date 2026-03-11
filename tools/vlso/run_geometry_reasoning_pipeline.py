from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import PseudoLabelAcceptanceConfig
from semop.vlso.geometry_pipeline import VisualGeometryBootstrapPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the full VLSO geometry pipeline: candidates -> pseudo labels -> concept self-train -> operator train -> eval')
    parser.add_argument('--inputs', nargs='+', required=True, help='image files or directories')
    parser.add_argument('--workspace', default='data/vlso_geometry_pipeline', help='output workspace directory')
    parser.add_argument('--weights', help='optional affordance weights json')
    parser.add_argument('--eval-input', default='examples/vlso_geometry_eval.jsonl', help='grounded QA eval jsonl')
    parser.add_argument('--eval-mode', choices=['heuristic', 'hybrid', 'deep'], default='heuristic')
    parser.add_argument('--answer-mode', choices=['structured', 'llm'], default='structured')
    parser.add_argument('--cluster-threshold', type=float, default=0.9)
    parser.add_argument('--pseudo-threshold', type=float, default=0.72)
    parser.add_argument('--consensus-threshold', type=float, default=0.55)
    parser.add_argument('--concept-match-threshold', type=float, default=0.86)
    parser.add_argument('--min-cluster-size', type=int, default=2)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    summary = VisualGeometryBootstrapPipeline(weights_path=args.weights).run(
        inputs=args.inputs,
        candidates_path=workspace / 'geometry_candidates.jsonl',
        pseudo_labels_path=workspace / 'geometry_pseudo_labels.jsonl',
        concept_store_path=workspace / 'geometry_concepts.db',
        operator_store_path=workspace / 'geometry_operators.db',
        eval_input=args.eval_input if Path(args.eval_input).exists() else None,
        eval_mode=args.eval_mode,
        answer_mode=args.answer_mode,
        config=PseudoLabelAcceptanceConfig(
            cluster_similarity_threshold=args.cluster_threshold,
            pseudo_confidence_threshold=args.pseudo_threshold,
            cluster_consensus_threshold=args.consensus_threshold,
            concept_match_threshold=args.concept_match_threshold,
            min_cluster_size=args.min_cluster_size,
        ),
        concept_summary_output=workspace / 'geometry_cluster_summary.json',
    )
    (workspace / 'geometry_pipeline_summary.json').write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
