from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def _reexec_into_preferred_python() -> None:
    preferred = Path('.venv312') / 'Scripts' / 'python.exe'
    try:
        current = Path(sys.executable).resolve()
    except Exception:
        current = Path(sys.executable)
    if preferred.exists():
        try:
            preferred_resolved = preferred.resolve()
        except Exception:
            preferred_resolved = preferred
        if str(current).lower() != str(preferred_resolved).lower():
            os.execv(str(preferred_resolved), [str(preferred_resolved), __file__, *sys.argv[1:]])

from semop import RTX4060ReasoningCoach


def main() -> None:
    _reexec_into_preferred_python()
    parser = argparse.ArgumentParser(description='Assess or improve the local RTX 4060 reasoning stack')
    parser.add_argument('--workspace', default='.')
    parser.add_argument('--bootstrap-missing', action='store_true')
    parser.add_argument('--output', default='data/unified_semop_gui_run/rtx4060_reasoning_coach.json')
    parser.add_argument('--unified-output-dir', default='data/unified_semop_gui_run')
    parser.add_argument('--unified-store-path', default='data/semop_memory.db')
    parser.add_argument('--unified-review-queue-path', default='data/ops_review_queue.db')
    parser.add_argument('--unified-benchmark-corpus-path', default='data/unified_semop_gui_run/persistent_benchmark_corpus.json')
    parser.add_argument('--math-output-dir', default='data/math_world_model_gui_run')
    parser.add_argument('--math-cases-path', default='examples/math_world_model_starter.jsonl')
    parser.add_argument('--visual-output-dir', default='data/math_world_model_gui_run/visual_3d')
    parser.add_argument('--hidden-input', default='examples/hidden_premise_eval.jsonl')
    parser.add_argument('--transfer-input', default='examples/operator_transfer_eval.jsonl')
    parser.add_argument('--vlso-input', default='examples/vlso_eval.jsonl')
    parser.add_argument('--vlso-real-input', default='examples/vlso_real_image_eval_gold.jsonl')
    parser.add_argument('--vision-image', default='data/scene.png')
    parser.add_argument('--improve', action='store_true')
    args = parser.parse_args()

    coach = RTX4060ReasoningCoach()
    if args.improve:
        summary = coach.improve(
            workspace=args.workspace,
            bootstrap_missing=bool(args.bootstrap_missing),
            output_path=args.output,
            unified_output_dir=args.unified_output_dir,
            unified_store_path=args.unified_store_path,
            unified_review_queue_path=args.unified_review_queue_path,
            unified_benchmark_corpus_path=args.unified_benchmark_corpus_path,
            math_output_dir=args.math_output_dir,
            math_cases_path=args.math_cases_path,
            visual_output_dir=args.visual_output_dir,
            hidden_input=args.hidden_input,
            transfer_input=args.transfer_input,
            vlso_input=args.vlso_input,
            vlso_real_input=args.vlso_real_input,
            vision_image=args.vision_image,
        )
    else:
        summary = coach.assess(
            workspace=args.workspace,
            bootstrap_missing=bool(args.bootstrap_missing),
            output_path=args.output,
            unified_output_dir=args.unified_output_dir,
            unified_store_path=args.unified_store_path,
            unified_review_queue_path=args.unified_review_queue_path,
            unified_benchmark_corpus_path=args.unified_benchmark_corpus_path,
            math_output_dir=args.math_output_dir,
            math_cases_path=args.math_cases_path,
            visual_output_dir=args.visual_output_dir,
        )
    print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
