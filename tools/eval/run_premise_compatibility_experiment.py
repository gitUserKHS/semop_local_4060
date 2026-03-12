from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import CorpusMemoryStore, HiddenPremiseEvaluator, ScriptCompatibilityTrainer, StructuredMeaningPipeline
from evaluate_hidden_premises import load_cases


def _summary_dict(summary):
    return {
        'num_cases': summary.num_cases,
        'critical_premise_recall': summary.critical_premise_recall,
        'hidden_goal_recall': summary.hidden_goal_recall,
        'goal_preservation_accuracy': summary.goal_preservation_accuracy,
        'unsupported_premise_precision': summary.unsupported_premise_precision,
        'clarification_accuracy': summary.clarification_accuracy,
        'requirement_state_accuracy': summary.requirement_state_accuracy,
        'clarification_score_mae': summary.clarification_score_mae,
        'operator_supported_premise_recall': summary.operator_supported_premise_recall,
    }


def _delta(after: dict, before: dict) -> dict:
    keys = [
        'critical_premise_recall',
        'hidden_goal_recall',
        'goal_preservation_accuracy',
        'unsupported_premise_precision',
        'clarification_accuracy',
        'requirement_state_accuracy',
        'operator_supported_premise_recall',
    ]
    return {f'{key}_delta': round(float(after.get(key, 0.0)) - float(before.get(key, 0.0)), 4) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a script compatibility scorer and compare hidden-premise metrics before/after.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--memory-store', required=True)
    parser.add_argument('--memory-source')
    parser.add_argument('--output-model', required=True)
    parser.add_argument('--mode', choices=['heuristic', 'llm'], default='heuristic')
    parser.add_argument('--model-id', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--learning-rate', type=float, default=0.18)
    args = parser.parse_args()

    cases = load_cases(Path(args.input))
    baseline = HiddenPremiseEvaluator(
        StructuredMeaningPipeline(
            mode=args.mode,
            model_id=args.model_id,
            memory_store_path=args.memory_store,
            memory_source=args.memory_source,
        )
    ).evaluate(cases)
    trainer = ScriptCompatibilityTrainer()
    training_summary = trainer.train_from_memory(
        CorpusMemoryStore(args.memory_store),
        output_path=args.output_model,
        source=args.memory_source,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    trained = HiddenPremiseEvaluator(
        StructuredMeaningPipeline(
            mode=args.mode,
            model_id=args.model_id,
            memory_store_path=args.memory_store,
            memory_source=args.memory_source,
            script_compatibility_model_path=args.output_model,
        )
    ).evaluate(cases)
    baseline_payload = _summary_dict(baseline)
    trained_payload = _summary_dict(trained)
    payload = {
        'training_summary': training_summary.model_dump(),
        'baseline': baseline_payload,
        'trained': trained_payload,
        'deltas': _delta(trained_payload, baseline_payload),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
