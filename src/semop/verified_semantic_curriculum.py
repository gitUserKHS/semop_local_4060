from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Mapping

from .kernel import OperatorKernel
from .kernel.domains.arithmetic import ArithmeticExpressionAdapter
from .semantic_distillation import (
    DistillationDecision,
    SemanticDistillationCorpus,
    SemanticDistillationRecord,
)


GENERATOR_VERSION = "semop.verified-arithmetic-semantics.v1"


@dataclass(frozen=True)
class ArithmeticWordTemplate:
    template_id: str
    split: str
    operation: str
    text: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "sealed"}:
            raise ValueError("arithmetic word template has an invalid split")
        if self.operation not in {"add", "subtract", "multiply"}:
            raise ValueError("arithmetic word template has an invalid operation")
        if any(token not in self.text for token in ("{a}", "{b}")):
            raise ValueError("arithmetic word template must bind a and b")


@dataclass(frozen=True)
class VerifiedSemanticCurriculum:
    seed: int
    records_by_split: Mapping[str, tuple[SemanticDistillationRecord, ...]]
    template_ids_by_split: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        records = {
            split: tuple(items) for split, items in self.records_by_split.items()
        }
        templates = {
            split: tuple(items) for split, items in self.template_ids_by_split.items()
        }
        expected = {"train", "validation", "sealed"}
        if set(records) != expected or set(templates) != expected:
            raise ValueError("verified semantic curriculum requires three splits")
        template_sets = [set(templates[split]) for split in sorted(expected)]
        if any(left.intersection(right) for index, left in enumerate(template_sets) for right in template_sets[index + 1 :]):
            raise ValueError("semantic curriculum template families overlap across splits")
        for split, split_records in records.items():
            corpus = SemanticDistillationCorpus(split_records)
            corpus.validate_budget()
            corpus.require_split(split)
            if not corpus.positives:
                raise ValueError(f"semantic curriculum {split} split has no positives")
        object.__setattr__(self, "records_by_split", records)
        object.__setattr__(self, "template_ids_by_split", templates)

    def corpus(self, split: str) -> SemanticDistillationCorpus:
        try:
            return SemanticDistillationCorpus(self.records_by_split[split])
        except KeyError as exc:
            raise ValueError(f"unknown semantic curriculum split: {split}") from exc

    def save(self, root: str | Path) -> dict[str, Any]:
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=True)
        files: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation", "sealed"):
            path = self.corpus(split).save_jsonl(destination / f"{split}.jsonl")
            files[split] = {
                "path": path.name,
                "sha256": _file_digest(path),
                "records": len(self.records_by_split[split]),
                "positives": len(self.corpus(split).positives),
            }
        manifest = {
            "schema_version": GENERATOR_VERSION,
            "seed": self.seed,
            "authority": "generator_known_semantics_plus_exact_executor",
            "promotion_eligible": False,
            "purpose": "development-only repair of observed Korean arithmetic grounding failures",
            "template_ids_by_split": {
                split: list(values)
                for split, values in self.template_ids_by_split.items()
            },
            "files": files,
        }
        _atomic_json(destination / "manifest.json", manifest)
        return manifest


_TEMPLATES = (
    ArithmeticWordTemplate(
        "train_stock_eaten",
        "train",
        "subtract",
        "{item}가 {a}개 있었는데 {b}개 먹었어. 몇 개 남았어?",
    ),
    ArithmeticWordTemplate(
        "train_stock_given",
        "train",
        "subtract",
        "{item} {a}개 중 {b}개를 친구에게 줬어. 이제 몇 개야?",
    ),
    ArithmeticWordTemplate(
        "train_stock_received",
        "train",
        "add",
        "{item}가 {a}개 있었고 {b}개를 더 받았어. 모두 몇 개야?",
    ),
    ArithmeticWordTemplate(
        "train_people_joined",
        "train",
        "add",
        "방에 {a}명이 있었는데 {b}명이 더 들어왔어. 지금 몇 명이야?",
    ),
    ArithmeticWordTemplate(
        "train_equal_boxes",
        "train",
        "multiply",
        "상자 {a}개에 {item}가 각각 {b}개씩 있어. 모두 몇 개야?",
    ),
    ArithmeticWordTemplate(
        "train_equal_rows",
        "train",
        "multiply",
        "{a}줄에 학생이 {b}명씩 서 있어. 학생은 모두 몇 명이야?",
    ),
    ArithmeticWordTemplate(
        "validation_stock_sold",
        "validation",
        "subtract",
        "가게에 {item} {a}개가 있었고 {b}개를 팔았어. 남은 것은 몇 개야?",
    ),
    ArithmeticWordTemplate(
        "validation_stock_found",
        "validation",
        "add",
        "{item} {a}개를 모았고 나중에 {b}개를 더 찾았어. 합하면 몇 개야?",
    ),
    ArithmeticWordTemplate(
        "validation_equal_packs",
        "validation",
        "multiply",
        "{item}를 한 묶음에 {b}개씩 {a}묶음 준비했어. 총 몇 개야?",
    ),
    ArithmeticWordTemplate(
        "sealed_stock_used",
        "sealed",
        "subtract",
        "{item}가 {a}개 있었는데 작업에 {b}개를 사용했어. 몇 개가 남아?",
    ),
    ArithmeticWordTemplate(
        "sealed_stock_delivered",
        "sealed",
        "add",
        "창고에 {item} {a}개가 있고 새로 {b}개가 도착했어. 현재 수량은 몇 개야?",
    ),
    ArithmeticWordTemplate(
        "sealed_equal_tables",
        "sealed",
        "multiply",
        "탁자 {a}개마다 의자가 {b}개씩 있어. 의자는 전부 몇 개야?",
    ),
)

_ITEMS = {
    "train": ("사과", "쿠키", "공", "스티커"),
    "validation": ("귤", "연필", "컵"),
    "sealed": ("나사", "전구", "배터리"),
}


def build_verified_arithmetic_curriculum(
    *,
    seed: int = 4060,
    train_examples_per_template: int = 8,
    evaluation_examples_per_template: int = 4,
) -> VerifiedSemanticCurriculum:
    if train_examples_per_template <= 0 or evaluation_examples_per_template <= 0:
        raise ValueError("semantic curriculum example counts must be positive")
    counts = {
        "train": train_examples_per_template,
        "validation": evaluation_examples_per_template,
        "sealed": evaluation_examples_per_template,
    }
    records: dict[str, list[SemanticDistillationRecord]] = {
        split: [] for split in counts
    }
    template_ids: dict[str, list[str]] = {split: [] for split in counts}
    split_offsets = {"train": 11, "validation": 23, "sealed": 37}
    for template_index, template in enumerate(_TEMPLATES):
        template_ids[template.split].append(template.template_id)
        randomizer = random.Random(seed + split_offsets[template.split] + template_index * 101)
        pairs = _numeric_pairs(
            template.operation,
            counts[template.split],
            randomizer,
        )
        for case_index, (a, b) in enumerate(pairs):
            item = _ITEMS[template.split][case_index % len(_ITEMS[template.split])]
            prompt = template.text.format(item=item, a=a, b=b)
            expression, answer = _expression_and_answer(template.operation, a, b)
            verified_answer = _verify_expression(expression)
            if verified_answer != str(answer):
                raise RuntimeError("synthetic arithmetic answer disagrees with executor")
            source = f"{GENERATOR_VERSION}:{template.template_id}:{case_index:03d}"
            positive_completion = _completion(expression, answer)
            negative_expression = f"({expression}) + 1"
            negative_answer = _verify_expression(negative_expression)
            negative_completion = _completion(negative_expression, int(negative_answer))
            records[template.split].append(
                _record(
                    prompt,
                    positive_completion,
                    template.split,
                    source,
                    DistillationDecision.ACCEPTED,
                    "",
                )
            )
            records[template.split].append(
                _record(
                    prompt,
                    negative_completion,
                    template.split,
                    source,
                    DistillationDecision.HARD_NEGATIVE,
                    (
                        "generator-known semantic mismatch: the expression adds one "
                        "to the exact answer"
                    ),
                )
            )
    return VerifiedSemanticCurriculum(
        seed=seed,
        records_by_split={split: tuple(values) for split, values in records.items()},
        template_ids_by_split={
            split: tuple(values) for split, values in template_ids.items()
        },
    )


def _numeric_pairs(
    operation: str,
    count: int,
    randomizer: random.Random,
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < count:
        if operation == "multiply":
            pair = (randomizer.randint(2, 10), randomizer.randint(2, 9))
        elif operation == "subtract":
            a = randomizer.randint(8, 48)
            pair = (a, randomizer.randint(1, a - 1))
        else:
            pair = (randomizer.randint(2, 40), randomizer.randint(1, 24))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return tuple(pairs)


def _expression_and_answer(operation: str, a: int, b: int) -> tuple[str, int]:
    if operation == "add":
        return f"{a} + {b}", a + b
    if operation == "subtract":
        return f"{a} - {b}", a - b
    return f"{a} * {b}", a * b


def _verify_expression(expression: str) -> str:
    instance = ArithmeticExpressionAdapter().adapt(expression)
    result = OperatorKernel(instance.registry).solve(instance.state, instance.goals)
    if not result.success or not result.verified:
        raise RuntimeError("synthetic arithmetic expression failed proof replay")
    return str(instance.metadata["answer"])


def _completion(expression: str, answer: int) -> str:
    return _canonical_json(
        {
            "domain": "math",
            "confidence": 1.0,
            "operator_program": ["DECOMPOSE", "INFER", "VERIFY", "EXPLAIN"],
            "payload": {"expression": expression},
            "answer": str(answer),
        }
    )


def _record(
    prompt: str,
    completion: str,
    split: str,
    source: str,
    decision: DistillationDecision,
    rejection_reason: str,
) -> SemanticDistillationRecord:
    proposal_fingerprint = _text_digest(
        _canonical_json(
            {
                "generator": GENERATOR_VERSION,
                "prompt": prompt,
                "completion": json.loads(completion),
                "split": split,
                "source": source,
                "decision": decision.value,
            }
        )
    )
    return SemanticDistillationRecord(
        record_id="",
        domain="math",
        prompt=prompt,
        completion=completion,
        decision=decision,
        proposal_fingerprint=proposal_fingerprint,
        replay_verified=True,
        semantic_review_digest="",
        data_split=split,
        rejection_reason=rejection_reason,
        source=source,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _text_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
