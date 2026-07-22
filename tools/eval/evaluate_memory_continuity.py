from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semop.chat_cli import create_local_assistant
from semop.conversation_memory import ConversationStore
from semop.memory_evaluation import (
    MemoryContinuityConfig,
    evaluate_correction_consolidation,
    evaluate_episodic_memory_effect,
    evaluate_long_session_memory_effect,
    evaluate_long_history_episodic_recall,
    evaluate_memory_continuity,
    evaluate_semantic_memory_effect,
    evaluate_semantic_paraphrase_memory,
    evaluate_task_auto_resume,
    evaluate_task_context_effect,
)
from semop.task_memory import TaskCheckpointStore
from semop.semantic_models import E5OperatorRetriever


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure SemOp episodic memory, explicit recall, correction "
            "consolidation, task revisions, and an optional local-model memory A/B."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="new SQLite path to retain; an isolated temporary database is the default",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--exchanges", type=int, default=50)
    parser.add_argument("--working-messages", type=int, default=8)
    parser.add_argument("--task-updates", type=int, default=10)
    parser.add_argument("--distractor-memories", type=int, default=24)
    parser.add_argument(
        "--long-history-exchanges",
        type=int,
        default=0,
        help=(
            "also build and query a separate whole-history episode index with "
            "at least 401 exchanges"
        ),
    )
    parser.add_argument(
        "--semantic-tier",
        choices=("balanced", "economy"),
        help=(
            "also run real local-model explicit- and episodic-memory A/Bs; no "
            "model is loaded when omitted"
        ),
    )
    parser.add_argument(
        "--semantic-paraphrase-ab",
        action="store_true",
        help=(
            "load local multilingual-E5-small and compare lexical versus "
            "typed-facet semantic recall"
        ),
    )
    parser.add_argument(
        "--model-ab",
        action="append",
        choices=("semantic", "episodic", "long-session", "task"),
        help=(
            "optional local-model A/B subset; repeat for multiple cases. "
            "All cases run when omitted"
        ),
    )
    parser.add_argument(
        "--promotion-root",
        type=Path,
        default=Path("artifacts/promotion/semantic-model"),
    )
    return parser


def _evaluate(database: Path, args: argparse.Namespace) -> tuple[dict[str, object], bool]:
    continuity = evaluate_memory_continuity(
        database,
        config=MemoryContinuityConfig(
            exchanges=args.exchanges,
            working_messages=args.working_messages,
            task_updates=args.task_updates,
            distractor_memories=args.distractor_memories,
        ),
    )
    payload: dict[str, object] = {
        "evaluation": "semop_memory_continuity_v1",
        "continuity": continuity.to_dict(),
        "correction_consolidation": None,
        "task_auto_resume": None,
        "long_history": None,
        "semantic_paraphrase_ab": None,
        "semantic_ab": None,
        "episodic_ab": None,
        "long_session_ab": None,
        "task_ab": None,
    }
    correction = evaluate_correction_consolidation(ConversationStore(database))
    payload["correction_consolidation"] = correction.to_dict()
    task_resume = evaluate_task_auto_resume(TaskCheckpointStore(database))
    payload["task_auto_resume"] = task_resume.to_dict()
    passed = continuity.passed and correction.passed and task_resume.passed
    if args.long_history_exchanges:
        suffix = database.suffix or ".db"
        long_history_path = database.with_name(
            f"{database.stem}-long-history{suffix}"
        )
        long_history = evaluate_long_history_episodic_recall(
            long_history_path,
            exchanges=args.long_history_exchanges,
        )
        payload["long_history"] = long_history.to_dict()
        passed = passed and long_history.passed
    if args.semantic_paraphrase_ab:
        suffix = database.suffix or ".db"
        semantic_paraphrase_path = database.with_name(
            f"{database.stem}-semantic-paraphrase{suffix}"
        )
        if semantic_paraphrase_path.exists():
            raise ValueError(
                "semantic paraphrase evaluation database must not already exist"
            )
        semantic_paraphrase = evaluate_semantic_paraphrase_memory(
            ConversationStore(semantic_paraphrase_path),
            E5OperatorRetriever(local_files_only=True, device="cpu"),
        )
        payload["semantic_paraphrase_ab"] = semantic_paraphrase.to_dict()
        passed = passed and semantic_paraphrase.passed
    if args.semantic_tier:
        selected = set(args.model_ab or ("semantic", "episodic", "long-session", "task"))
        assistant = create_local_assistant(
            args.semantic_tier,
            promotion_root=args.promotion_root,
        )
        model_results = []
        if "semantic" in selected:
            semantic = evaluate_semantic_memory_effect(
                assistant,
                ConversationStore(database),
                resource_tier=args.semantic_tier,
            )
            payload["semantic_ab"] = semantic.to_dict()
            model_results.append(semantic.passed)
        if "episodic" in selected:
            episodic = evaluate_episodic_memory_effect(
                assistant,
                ConversationStore(database),
                resource_tier=args.semantic_tier,
            )
            payload["episodic_ab"] = episodic.to_dict()
            model_results.append(episodic.passed)
        if "long-session" in selected:
            long_session = evaluate_long_session_memory_effect(
                assistant,
                ConversationStore(database),
                resource_tier=args.semantic_tier,
            )
            payload["long_session_ab"] = long_session.to_dict()
            model_results.append(long_session.passed)
        if "task" in selected:
            task = evaluate_task_context_effect(
                assistant,
                TaskCheckpointStore(database),
                resource_tier=args.semantic_tier,
            )
            payload["task_ab"] = task.to_dict()
            model_results.append(task.passed)
        passed = passed and all(model_results)
    payload["passed"] = passed
    return payload, passed


def _emit(payload: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.database is not None:
            payload, passed = _evaluate(args.database, args)
        else:
            with TemporaryDirectory(prefix="semop-memory-eval-") as temporary:
                payload, passed = _evaluate(
                    Path(temporary) / "memory-continuity.db",
                    args,
                )
        _emit(payload, args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"SemOp memory evaluation error: {exc}", file=sys.stderr)
        return 1
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
