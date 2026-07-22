from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

from semop.chat_cli import create_local_assistant
from semop.prompt_api import PromptRequest, ResourceTier


_TEXT_CASES = {
    "chat": "안녕! 너는 누구고 무엇을 할 수 있어?",
    "math": "사과가 12개 있었는데 5개를 먹고 3개를 더 샀어. 몇 개야?",
    "coding": (
        "Python에서 리스트 컴프리헨션과 for 루프의 차이를 초보자에게 "
        "짧은 예시와 함께 설명해줘."
    ),
}


def _vision_image() -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("the vision probe requires Pillow") from exc

    image = Image.new("RGB", (160, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 18, 76, 82), fill="red")
    draw.text((88, 40), "SEMOP 42", fill="black")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _request(case: str, tier: ResourceTier) -> PromptRequest:
    if case == "vision":
        return PromptRequest(
            "이 이미지에 보이는 도형과 글자를 자연스럽게 설명해줘.",
            images=(_vision_image(),),
            resource_tier=tier,
        )
    return PromptRequest(_TEXT_CASES[case], resource_tier=tier)


def _record(case: str, answer, elapsed: float) -> dict[str, object]:
    proposal = answer.proposals[0]
    return {
        "case": case,
        "status": answer.status.value,
        "answer": answer.answer,
        "answer_source": answer.provenance.get("answer_source", ""),
        "proposal_domain": proposal.domain,
        "proposal_deterministic": proposal.deterministic,
        "typed_result_count": len(answer.results),
        "logical_replay_verified": answer.provenance.get(
            "logical_replay_verified", False
        ),
        "repair_attempts": answer.provenance.get("repair_attempts", 0),
        "semantic_generation": answer.provenance.get("semantic_generation", {}),
        "image_region_count": len(proposal.image_regions),
        "diagnostics": list(proposal.diagnostics),
        "model_id": answer.metrics.model_id,
        "elapsed_seconds": round(elapsed, 3),
        "peak_vram_bytes": answer.metrics.peak_vram_bytes,
    }


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe the real cached SemOp semantic runtime once per domain."
    )
    parser.add_argument(
        "--tier",
        choices=("balanced", "economy"),
        default="balanced",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=(*_TEXT_CASES, "vision"),
        dest="cases",
        help="case to run; repeat this option or omit it to run all cases",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="optional append-only result path; each case is flushed immediately",
    )
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        help="optional local-only log of raw model generations for parser diagnosis",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    tier = ResourceTier(args.tier)
    cases = tuple(args.cases or (*_TEXT_CASES, "vision"))
    assistant = create_local_assistant(tier)
    if args.raw_jsonl is not None:
        backend = assistant.compiler.backend
        if backend is None:
            raise RuntimeError("raw generation capture requires a semantic backend")
        original_generate = backend.generate

        def capture_generate(*generate_args, **generate_kwargs):
            raw = original_generate(*generate_args, **generate_kwargs)
            _append_jsonl(
                args.raw_jsonl,
                {"raw": raw, "generation": capture_generate.calls},
            )
            capture_generate.calls += 1
            return raw

        capture_generate.calls = 1
        backend.generate = capture_generate
    started = perf_counter()

    for case in cases:
        case_started = perf_counter()
        try:
            answer = assistant.solve(_request(case, tier))
            payload = _record(case, answer, perf_counter() - case_started)
        except Exception as exc:  # The probe must preserve unexpected runtime failures.
            payload = {
                "case": case,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": round(perf_counter() - case_started, 3),
            }
        if args.jsonl is not None:
            _append_jsonl(args.jsonl, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "summary": {
                    "cases": len(cases),
                    "elapsed_seconds": round(perf_counter() - started, 3),
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
