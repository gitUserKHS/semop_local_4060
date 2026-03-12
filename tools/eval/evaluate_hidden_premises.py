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
                expected_satisfied_premises=list(payload.get('expected_satisfied_premises', [])),
                expected_missing_premises=list(payload.get('expected_missing_premises', [])),
                expected_risky_actions=list(payload.get('expected_risky_actions', [])),
                forbidden_premises=list(payload.get('forbidden_premises', [])),
                expected_clarification_needed=bool(payload.get('expected_clarification_needed', False)),
                expected_clarification_score=payload.get('expected_clarification_score'),
                expected_support_operators=list(payload.get('expected_support_operators', [])) if 'expected_support_operators' in payload else None,
            )
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate hidden premise extraction and goal preservation.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--mode', default='heuristic', choices=['heuristic', 'llm'])
    parser.add_argument('--model-id', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--memory-store')
    parser.add_argument('--memory-source')
    parser.add_argument('--script-compatibility-model')
    args = parser.parse_args()

    evaluator = HiddenPremiseEvaluator(StructuredMeaningPipeline(mode=args.mode, model_id=args.model_id, memory_store_path=args.memory_store, memory_source=args.memory_source, script_compatibility_model_path=args.script_compatibility_model))
    summary = evaluator.evaluate(load_cases(Path(args.input)))
    print(json.dumps({
        'num_cases': summary.num_cases,
        'critical_premise_recall': summary.critical_premise_recall,
        'hidden_goal_recall': summary.hidden_goal_recall,
        'goal_preservation_accuracy': summary.goal_preservation_accuracy,
        'unsupported_premise_precision': summary.unsupported_premise_precision,
        'clarification_accuracy': summary.clarification_accuracy,
        'requirement_state_accuracy': summary.requirement_state_accuracy,
        'clarification_score_mae': summary.clarification_score_mae,
        'operator_supported_premise_recall': summary.operator_supported_premise_recall,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
