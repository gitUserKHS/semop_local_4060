from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import CpLearnedParser, CpParserEvaluator, OperatorIntelligenceProgressEstimator, SemOpCommonEvaluator, VlsoGroundedEvaluator
from evaluate_hidden_premises import load_cases as load_hidden_cases


def main() -> None:
    parser = argparse.ArgumentParser(description='Estimate current progress toward the operator-intelligence target from SemOp evaluators.')
    parser.add_argument('--hidden-premises')
    parser.add_argument('--cp-input')
    parser.add_argument('--vlso-input')
    parser.add_argument('--cp-mode', choices=['heuristic', 'model', 'compare'], default='heuristic')
    parser.add_argument('--cp-model')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()

    common = SemOpCommonEvaluator()
    hidden_cases = load_hidden_cases(Path(args.hidden_premises)) if args.hidden_premises else None
    cp_examples = CpParserEvaluator.load_examples(Path(args.cp_input)) if args.cp_input else None
    vlso_cases = VlsoGroundedEvaluator.load_cases(Path(args.vlso_input)) if args.vlso_input else None
    cp_model = None
    if args.cp_mode in {'model', 'compare'}:
        if not args.cp_model:
            parser.error('--cp-model is required for --cp-mode model or compare')
        cp_model = CpLearnedParser(args.cp_model, local_files_only=args.local_files_only)

    snapshot = common.evaluate(
        hidden_premise_cases=hidden_cases,
        cp_examples=cp_examples,
        cp_model=cp_model,
        vlso_cases=vlso_cases,
        cp_mode=args.cp_mode,
    )
    progress = OperatorIntelligenceProgressEstimator().estimate(snapshot)
    print(json.dumps({'snapshot': snapshot.model_dump(), 'progress': progress.model_dump()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
