from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import CpLearnedParser, CpParserEvaluator, SemOpUnderstandingEvaluator, VlsoGroundedEvaluator
from evaluate_hidden_premises import load_cases as load_hidden_cases


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a broad SemOp understanding benchmark and summary report.')
    parser.add_argument('--hidden-premises', default='examples/hidden_premise_eval.jsonl')
    parser.add_argument('--cp-input', default='examples/cp_parser_eval.jsonl')
    parser.add_argument('--cp-hidden-input', default='examples/cp_hidden_constraint_eval.jsonl')
    parser.add_argument('--vlso-input', default='examples/vlso_eval.jsonl')
    parser.add_argument('--vlso-real-image-input', default='examples/vlso_real_image_eval_gold.jsonl')
    parser.add_argument('--cp-mode', choices=['heuristic', 'model', 'compare'], default='heuristic')
    parser.add_argument('--cp-model')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()

    hidden_cases = load_hidden_cases(Path(args.hidden_premises)) if args.hidden_premises else None
    cp_examples = CpParserEvaluator.load_examples(Path(args.cp_input)) if args.cp_input else None
    cp_hidden_examples = CpParserEvaluator.load_examples(Path(args.cp_hidden_input)) if args.cp_hidden_input else None
    vlso_cases = VlsoGroundedEvaluator.load_cases(Path(args.vlso_input)) if args.vlso_input else None
    vlso_real_image_cases = VlsoGroundedEvaluator.load_cases(Path(args.vlso_real_image_input)) if args.vlso_real_image_input else None

    cp_model = None
    if args.cp_mode in {'model', 'compare'}:
        if not args.cp_model:
            parser.error('--cp-model is required for --cp-mode model or compare')
        cp_model = CpLearnedParser(args.cp_model, local_files_only=args.local_files_only)

    summary = SemOpUnderstandingEvaluator().evaluate(
        hidden_premise_cases=hidden_cases,
        cp_examples=cp_examples,
        cp_hidden_examples=cp_hidden_examples,
        cp_model=cp_model,
        vlso_cases=vlso_cases,
        vlso_real_image_cases=vlso_real_image_cases,
        cp_mode=args.cp_mode,
    )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
