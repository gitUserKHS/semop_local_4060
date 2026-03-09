from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable, List


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def guess_domain(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["forklift", "지게차", "blocked aisle", "통로", "incident report", "작업 중지"]):
        return "warehouse_exception"
    return "warehouse_onboarding"


def guess_scenario(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["예외", "blocked", "incident", "긴급", "stop work"]):
        return "exception_response"
    if any(token in lowered for token in ["라벨", "barcode", "scan", "품질"]):
        return "quality_gate"
    return "onboarding"


def split_sections(text: str) -> List[str]:
    sections = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(sections) <= 1:
        sections = [line.strip() for line in text.splitlines() if len(line.strip()) > 10]
    return sections[:50]


def build_case(doc_path: Path, section: str, index: int) -> dict:
    domain = guess_domain(section)
    scenario = guess_scenario(section)
    expected_relations = ["REQUIRES"]
    lowered = section.lower()
    if any(token in lowered for token in ["불일치", "mismatch", "blocked", "막힌", "위험"]):
        expected_relations.append("BLOCKED_BY")
    if any(token in lowered for token in ["대체", "staging", "보류", "hold", "우회"]):
        expected_relations.append("ALTERNATIVE")
    answer_terms = [token for token in ["승인", "보류", "안전", "재스캔", "스테이징", "보고"] if token in section][:4]
    forbidden = [token for token in ["그냥 진행", "승인 없이", "바로 포장", "바로 출고"] if token in section]
    if not forbidden:
        forbidden = ["그냥 진행"]
    query = f"{doc_path.stem} 문서의 이 절차를 따를 때 어떤 점을 먼저 확인해야 하나요?"
    return {
        "id": f"{doc_path.stem}_{index:03d}",
        "domain": domain,
        "scenario": scenario,
        "query": query,
        "context": section,
        "expected_relations": list(dict.fromkeys(expected_relations)),
        "expected_answer_terms": answer_terms or ["승인", "보류"],
        "forbidden_phrases": forbidden,
        "expected_clarification": False,
        "metadata": {"source_doc": str(doc_path), "section_index": index},
    }


def iter_input_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in {".txt", ".md"}:
                    files.append(child)
        elif path.suffix.lower() in {".txt", ".md"}:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build customer-labeled eval stubs from local SOP documents")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=50)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cases: List[dict] = []
    for file_path in iter_input_files(args.inputs):
        text = read_text(file_path)
        for index, section in enumerate(split_sections(text), start=1):
            cases.append(build_case(file_path, section, index))
            if len(cases) >= args.max_cases:
                break
        if len(cases) >= args.max_cases:
            break

    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "case_count": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
