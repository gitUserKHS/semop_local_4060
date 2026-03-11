from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import HiddenPremiseEvalCase, HiddenPremiseEvaluator, StructuredMeaningPipeline


def load_cases(path: Path) -> list[HiddenPremiseEvalCase]:
    cases: list[HiddenPremiseEvalCase] = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(
            HiddenPremiseEvalCase(
                query=payload['query'],
                expected_hidden_goals=list(payload.get('expected_hidden_goals', [])),
                expected_required_premises=list(payload.get('expected_required_premises', [])),
                expected_risky_actions=list(payload.get('expected_risky_actions', [])),
            )
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate hidden premise extraction and goal preservation.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--mode', default='heuristic', choices=['heuristic', 'llm'])
    parser.add_argument('--model-id', default='Qwen/Qwen2.5-3B-Instruct')
    args = parser.parse_args()

    evaluator = HiddenPremiseEvaluator(StructuredMeaningPipeline(mode=args.mode, model_id=args.model_id))
    summary = evaluator.evaluate(load_cases(Path(args.input)))
    print(json.dumps({
        'num_cases': summary.num_cases,
        'critical_premise_recall': summary.critical_premise_recall,
        'hidden_goal_recall': summary.hidden_goal_recall,
        'goal_preservation_accuracy': summary.goal_preservation_accuracy,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
