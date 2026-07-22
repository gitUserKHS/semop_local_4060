from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .assistant import SemOpAssistant
from .artifact_promotion import ArtifactPromotionStore
from .prompt_api import AnswerStatus, PromptRequest, ResourceTier
from .prompt_compiler import PromptCompiler
from .semantic_models import (
    E5OperatorRetriever,
    FlorenceVisionSidecar,
    LexicalOperatorRetriever,
    MODEL_SPECS,
    QwenSemanticBackend,
    QwenSemanticConfig,
    resolve_model_profile,
)
from .semantic_experience import SemanticTraceStore
from .semantic_memory import ReviewedSemanticMemory


DEFAULT_SEMANTIC_PROMOTION_ROOT = Path("artifacts/promotion/semantic-model")


def create_local_assistant(
    tier: ResourceTier | str = ResourceTier.AUTO,
    *,
    promotion_root: str | Path = DEFAULT_SEMANTIC_PROMOTION_ROOT,
    semantic_trace_store: SemanticTraceStore | None = None,
) -> SemOpAssistant:
    resolved = ResourceTier(tier)
    if resolved is ResourceTier.SYMBOLIC:
        return SemOpAssistant(
            compiler=PromptCompiler(retriever=LexicalOperatorRetriever()),
            semantic_trace_store=semantic_trace_store,
        )
    profile = resolve_model_profile(resolved)
    adapter_path = ""
    adapter_sha256 = ""
    if resolved is ResourceTier.ECONOMY:
        try:
            candidate, artifact = ArtifactPromotionStore(
                promotion_root
            ).active_candidate(kind="semantic_model")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "economy tier has no valid approved student adapter; use "
                "balanced or symbolic until a sealed candidate is activated"
            ) from exc
        expected_spec = MODEL_SPECS[profile.semantic_model_id]
        if (
            candidate.metadata.get("base_model_id")
            != profile.semantic_model_id
            or candidate.metadata.get("base_model_revision")
            != expected_spec.revision
        ):
            raise RuntimeError(
                "approved economy adapter is not bound to the reviewed student "
                "base model revision"
            )
        adapter_path = str(artifact)
        adapter_sha256 = candidate.artifact_sha256
    backend = QwenSemanticBackend(
        QwenSemanticConfig(
            model_id=profile.semantic_model_id,
            max_context=profile.max_context,
            max_new_tokens=profile.max_new_tokens,
            local_files_only=True,
            adapter_path=adapter_path,
            adapter_sha256=adapter_sha256,
        )
    )
    retriever = E5OperatorRetriever(
        profile.retriever_model_id,
        local_files_only=True,
        device="cpu",
    )
    vision_sidecar = (
        FlorenceVisionSidecar(profile.vision_sidecar_model_id)
        if profile.vision_sidecar_model_id
        else None
    )
    return SemOpAssistant(
        compiler=PromptCompiler(
            backend=backend,
            retriever=retriever,
            memory_retriever=(
                ReviewedSemanticMemory(semantic_trace_store)
                if semantic_trace_store is not None
                else None
            ),
            vision_sidecar=vision_sidecar,
        ),
        semantic_trace_store=semantic_trace_store,
    )


def semantic_activation_identity(
    promotion_root: str | Path = DEFAULT_SEMANTIC_PROMOTION_ROOT,
) -> str:
    """Return the active candidate id for cache invalidation without loading it."""

    try:
        active = ArtifactPromotionStore(promotion_root).active()
        candidate = active.get("candidate", {})
        if not isinstance(candidate, dict):
            return ""
        return str(candidate.get("candidate_id", ""))
    except (KeyError, OSError, TypeError, ValueError):
        return ""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SemOp's prompt-first verifier-guided local assistant."
    )
    parser.add_argument("prompt", nargs="?", help="prompt text; stdin is used when omitted")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        type=Path,
        help="optional image path; may be repeated",
    )
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--tier",
        choices=("auto", "symbolic", "economy", "balanced"),
        default="auto",
    )
    parser.add_argument("--judge-opt-in", action="store_true")
    parser.add_argument(
        "--promotion-root",
        type=Path,
        default=DEFAULT_SEMANTIC_PROMOTION_ROOT,
        help="digest-bound promotion store used by the economy tier",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--show-proof", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    text = args.prompt if args.prompt is not None else sys.stdin.read()
    try:
        request = PromptRequest(
            text=text,
            images=tuple(args.image),
            workspace=args.workspace,
            resource_tier=args.tier,
            judge_opt_in=args.judge_opt_in,
        )
        answer = create_local_assistant(
            args.tier,
            promotion_root=args.promotion_root,
        ).solve(request)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"SemOp 실행 오류: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"[{answer.status.value}] {answer.answer}")
        if args.show_proof and answer.proof:
            print("\n검증 과정\n" + answer.proof)
        if answer.unresolved:
            print("\n확인할 점\n" + "\n".join(f"- {item}" for item in answer.unresolved))
    return 2 if answer.status is AnswerStatus.UNSUPPORTED else 0


if __name__ == "__main__":
    raise SystemExit(main())
