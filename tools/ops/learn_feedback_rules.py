from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop.review_queue import ReviewQueueStore


def infer_rule(detail: Dict[str, object]) -> Dict[str, object]:
    text = f"{detail['query']} {detail['answer_text']} {' '.join(detail['reasons'])} {detail.get('resolution_note', '')}"
    lowered = text.lower()
    trigger_terms: List[str] = []
    for token in ["지게차", "통로", "팔레트", "승인", "라벨", "바코드", "출고", "forklift", "blocked", "approval", "barcode"]:
        if token.lower() in lowered and token not in trigger_terms:
            trigger_terms.append(token)
    require_terms: List[str] = []
    for token in ["승인", "안전", "보류", "재스캔", "incident report", "staging"]:
        if token.lower() in lowered and token not in require_terms:
            require_terms.append(token)
    avoid_phrases: List[str] = []
    for token in ["그냥 진행", "승인 없이", "바로 출고", "바로 포장", "force through"]:
        if token.lower() in lowered and token not in avoid_phrases:
            avoid_phrases.append(token)
    recommended_actions: List[str] = []
    for token in ["출고를 보류한다", "관리자 승인을 확인한다", "스테이징 구역으로 우회한다", "재스캔한다", "incident report를 남긴다"]:
        if token.lower() in lowered and token not in recommended_actions:
            recommended_actions.append(token)
    if not recommended_actions:
        recommended_actions.append("관리자 승인과 안전 조건을 먼저 확인한다.")
    return {
        "id": f"review_rule_{detail['id']}",
        "domain": detail["domain"],
        "scenario": detail["scenario"],
        "trigger_terms": trigger_terms,
        "require_terms": require_terms,
        "avoid_phrases": avoid_phrases,
        "recommended_actions": recommended_actions,
        "explanation": detail.get("resolution_note") or "Derived from review queue feedback.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn feedback rules from resolved review queue items")
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--statuses", nargs="+", default=["approved", "needs_followup"])
    args = parser.parse_args()

    store = ReviewQueueStore(args.review_queue)
    items = store.fetch_items(status=None, limit=500)
    allowed = set(args.statuses)
    rules = []
    for item in items:
        if item.status not in allowed:
            continue
        detail = store.fetch_item_detail(item.id)
        if detail is None:
            continue
        rules.append(infer_rule(detail))

    payload = {"rule_count": len(rules), "rules": rules}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(output), "rule_count": len(rules)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

