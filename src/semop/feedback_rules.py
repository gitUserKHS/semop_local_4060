from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List


@dataclass
class FeedbackRule:
    id: str
    domain: str
    scenario: str
    trigger_terms: List[str] = field(default_factory=list)
    require_terms: List[str] = field(default_factory=list)
    avoid_phrases: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    explanation: str = ""


class FeedbackRuleSet:
    def __init__(self, rules: List[FeedbackRule] | None = None):
        self.rules = rules or []

    @classmethod
    def from_path(cls, path: str | Path | None) -> "FeedbackRuleSet":
        if path is None:
            return cls([])
        rule_path = Path(path)
        if not rule_path.exists():
            return cls([])
        with rule_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        rules = [FeedbackRule(**item) for item in payload.get("rules", [])]
        return cls(rules)

    def matching_rules(self, *, domain: str, scenario: str, text: str) -> List[FeedbackRule]:
        lowered = text.lower()
        matches: List[FeedbackRule] = []
        for rule in self.rules:
            if rule.domain not in {"*", domain}:
                continue
            if rule.scenario not in {"*", scenario}:
                continue
            if rule.trigger_terms and not any(term.lower() in lowered for term in rule.trigger_terms):
                continue
            matches.append(rule)
        return matches

    def is_empty(self) -> bool:
        return not self.rules
