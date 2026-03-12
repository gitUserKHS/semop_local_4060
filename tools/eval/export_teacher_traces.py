from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from semop.distillation import TeacherTraceExporter
from semop import CompetitiveProgrammingReasoner, StructuredMeaningPipeline
from semop.vlso.reasoner import VLSOReasoner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export structured teacher traces for QLoRA/distillation experiments.')
    parser.add_argument('--hidden-premises', help='JSONL hidden-premise evaluation file.')
    parser.add_argument('--cp-input', help='JSONL CP parser evaluation file.')
    parser.add_argument('--cp-hidden-input', help='JSONL CP hidden-constraint evaluation file.')
    parser.add_argument('--vlso-input', help='JSONL VLSO grounded QA evaluation file.')
    parser.add_argument('--vlso-real-image-input', help='JSONL reviewed real-image VLSO evaluation file.')
    parser.add_argument('--operator-transfer-input', help='JSONL operator transfer evaluation file.')
    parser.add_argument('--examples-root', default='examples', help='Base directory for relative VLSO visual_json paths.')
    parser.add_argument('--output', required=True, help='Output JSONL path for teacher traces.')
    parser.add_argument('--sft-output', help='Optional output JSONL path for prompt/completion records.')
    parser.add_argument('--mode', default='heuristic', choices=['heuristic', 'llm'], help='Pipeline mode for hidden-premise export.')
    parser.add_argument('--answer-mode', default='structured', choices=['structured', 'llm'], help='Answer mode for VLSO exports.')
    parser.add_argument('--answer-model-id', default='Qwen/Qwen2.5-3B-Instruct', help='Model id used when --answer-mode llm.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exporter = TeacherTraceExporter(
        premise_pipeline=StructuredMeaningPipeline(mode=args.mode),
        cp_reasoner=CompetitiveProgrammingReasoner(),
        vlso_reasoner=VLSOReasoner(mode='hybrid', language_mode=args.mode, answer_mode=args.answer_mode, answer_model_id=args.answer_model_id),
    )

    traces = []
    if args.hidden_premises:
        traces.extend(exporter.export_hidden_premise_eval(args.hidden_premises))
    if args.cp_input:
        traces.extend(exporter.export_cp_parser_eval(args.cp_input))
    if args.cp_hidden_input:
        traces.extend(exporter.export_cp_parser_eval(args.cp_hidden_input))
    if args.vlso_input:
        traces.extend(exporter.export_vlso_eval(args.vlso_input, base_dir=args.examples_root))
    if args.vlso_real_image_input:
        traces.extend(exporter.export_vlso_eval(args.vlso_real_image_input, base_dir='.'))
    if args.operator_transfer_input:
        traces.extend(exporter.export_operator_transfer_eval(args.operator_transfer_input))

    TeacherTraceExporter.save_jsonl(args.output, traces)
    summary = {
        'output': str(Path(args.output)),
        'num_traces': len(traces),
        'task_counts': {},
    }
    for trace in traces:
        summary['task_counts'][trace.task] = summary['task_counts'].get(trace.task, 0) + 1

    if args.sft_output:
        sft_records = exporter.to_sft_records(traces)
        TeacherTraceExporter.save_jsonl(args.sft_output, sft_records)
        summary['sft_output'] = str(Path(args.sft_output))
        summary['num_sft_records'] = len(sft_records)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
