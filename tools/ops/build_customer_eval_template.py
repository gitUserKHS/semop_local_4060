from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CASES = [
    {
        "id": "replace_me_1",
        "domain": "warehouse_exception",
        "scenario": "exception_response",
        "query": "여기에 고객 질문을 넣으세요.",
        "context": "여기에 SOP나 매뉴얼 문맥을 넣으세요.",
        "expected_relations": ["REQUIRES", "BLOCKED_BY", "ALTERNATIVE"],
        "expected_answer_terms": ["승인", "보류", "대체 경로"],
        "forbidden_phrases": ["그냥 진행", "승인 없이"],
        "expected_clarification": False,
        "metadata": {"owner": "customer_team", "severity": "high"},
    },
    {
        "id": "replace_me_2",
        "domain": "warehouse_onboarding",
        "scenario": "onboarding",
        "query": "여기에 온보딩 질문을 넣으세요.",
        "context": "교육용 SOP 문맥을 넣으세요.",
        "expected_relations": ["REQUIRES", "ALTERNATIVE"],
        "expected_answer_terms": ["재스캔", "관리자 승인"],
        "forbidden_phrases": ["바로 포장"],
        "expected_clarification": True,
        "metadata": {"owner": "customer_team", "severity": "medium"},
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a customer-labeled evaluation template for ops copilot testing")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in DEFAULT_CASES:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(str(output))


if __name__ == "__main__":
    main()
