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


from semop import UltimateAGIReadinessRunner


def main() -> None:
    _reexec_into_preferred_python()
    parser = argparse.ArgumentParser(description='Run the ultimate AGI and commercialization readiness audit')
    parser.add_argument('--workspace', default='.')
    parser.add_argument('--bootstrap-missing', action='store_true')
    parser.add_argument('--output', default='data/unified_semop_gui_run/ultimate_agi_readiness.json')
    parser.add_argument('--unified-output-dir', default='data/unified_semop_gui_run')
    parser.add_argument('--unified-store-path', default='data/semop_memory.db')
    parser.add_argument('--unified-review-queue-path', default='data/ops_review_queue.db')
    parser.add_argument('--unified-benchmark-corpus-path', default='data/unified_semop_gui_run/persistent_benchmark_corpus.json')
    parser.add_argument('--math-output-dir', default='data/math_world_model_gui_run')
    parser.add_argument('--math-cases-path', default='examples/math_world_model_starter.jsonl')
    parser.add_argument('--visual-output-dir', default='data/math_world_model_gui_run/visual_3d')
    args = parser.parse_args()

    summary = UltimateAGIReadinessRunner().run(
        workspace=args.workspace,
        output_path=args.output,
        bootstrap_missing=bool(args.bootstrap_missing),
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
