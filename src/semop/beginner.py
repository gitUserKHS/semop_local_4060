from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .beginner_learning import run_beginner_reviewed_learning
from .kernel import (
    ExperienceQueueItem,
    LanguageTextProblem,
    RasterImage,
    RasterVisionProblem,
    TypedDomainRequest,
    TypedExperienceCollector,
    TypedExperienceStore,
    UnifiedTypedReasoner,
    UnifiedTypedResult,
    VerifiedRuleLibrary,
    VisionAreaGoal,
    VisionCountGoal,
    VisionPropertyGoal,
    VisionRelationGoal,
)


class BeginnerInputError(ValueError):
    """An input problem that can be explained without exposing a traceback."""


@dataclass(frozen=True)
class VisionPreset:
    key: str
    title: str
    question: str
    description: str
    pattern: tuple[str, ...]
    goal: VisionRelationGoal | VisionPropertyGoal | VisionCountGoal | VisionAreaGoal

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "question": self.question,
            "description": self.description,
            "pattern": list(self.pattern),
        }


@dataclass(frozen=True)
class BeginnerSolveResult:
    domain: str
    success: bool
    verified: bool
    conclusion: str
    title: str
    summary: str
    interpreted_input: tuple[str, ...]
    proof: str
    trust_notice: str
    diagnostics: tuple[str, ...]
    technical: Mapping[str, Any]
    experience_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "success": self.success,
            "verified": self.verified,
            "conclusion": self.conclusion,
            "title": self.title,
            "summary": self.summary,
            "interpreted_input": list(self.interpreted_input),
            "proof": self.proof,
            "trust_notice": self.trust_notice,
            "diagnostics": list(self.diagnostics),
            "technical": dict(self.technical),
            "experience_digest": self.experience_digest,
        }


@dataclass(frozen=True)
class _ExperienceProposal:
    expected_solved: bool
    rationale: str


@dataclass(frozen=True)
class _BeginnerCoreRun:
    result: UnifiedTypedResult
    request_digest: str = ""


WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

_PALETTE = {
    ".": WHITE,
    "R": RED,
    "G": GREEN,
    "B": BLUE,
}


VISION_PRESETS: dict[str, VisionPreset] = {
    "red_square": VisionPreset(
        key="red_square",
        title="빨간 정사각형",
        question="빨간 도형은 정사각형일까?",
        description="픽셀의 채움 상태와 가로·세로 길이를 차례로 확인해.",
        pattern=(
            "......",
            ".RR...",
            ".RR...",
            "......",
        ),
        goal=VisionPropertyGoal("SQUARE", "red", "빨간 도형은 정사각형"),
    ),
    "left_right": VisionPreset(
        key="left_right",
        title="왼쪽과 오른쪽",
        question="빨간 도형은 파란 도형보다 왼쪽일까?",
        description="두 색상 덩어리의 경계 상자를 비교해 좌우 관계를 확인해.",
        pattern=(
            "........",
            ".RR..BB.",
            ".RR..BB.",
            "........",
        ),
        goal=VisionRelationGoal(
            "LEFT_OF", "red", "blue", "빨간 도형은 파란 도형보다 왼쪽"
        ),
    ),
    "three_objects": VisionPreset(
        key="three_objects",
        title="도형 세 개",
        question="서로 떨어진 도형은 모두 세 개일까?",
        description="연결된 같은 색 픽셀을 한 물체로 묶은 뒤 개수를 세어.",
        pattern=(
            "...........",
            ".RR.GG.BB..",
            ".RR.GG.BB..",
            "...........",
        ),
        goal=VisionCountGoal("all", 3, "도형은 모두 세 개"),
    ),
    "red_larger": VisionPreset(
        key="red_larger",
        title="더 큰 빨간 도형",
        question="빨간 도형의 면적이 파란 도형보다 클까?",
        description="각 색상 덩어리를 이루는 정확한 픽셀 수를 비교해.",
        pattern=(
            "..........",
            ".RRR...B..",
            ".RRR...B..",
            ".RRR......",
            "..........",
        ),
        goal=VisionAreaGoal("red", "blue", "빨간 도형의 면적이 더 큼"),
    ),
}


class BeginnerReasoner:
    """Small facade that keeps typed internals out of the first-use workflow."""

    def __init__(
        self,
        reasoner: UnifiedTypedReasoner | None = None,
        *,
        experience_store: TypedExperienceStore | None = None,
        active_rule_library: VerifiedRuleLibrary | None = None,
        rules_artifact_path: str | Path | None = None,
    ) -> None:
        base_reasoner = reasoner or UnifiedTypedReasoner()
        detected_libraries = tuple(
            augmenter
            for augmenter in base_reasoner.augmenters
            if isinstance(augmenter, VerifiedRuleLibrary)
        )
        if len(detected_libraries) > 1:
            raise ValueError("beginner runtime accepts one active rule library")
        if (
            active_rule_library is not None
            and detected_libraries
            and active_rule_library != detected_libraries[0]
        ):
            raise ValueError("beginner runtime active rule libraries disagree")
        self.active_rule_library = (
            active_rule_library
            if active_rule_library is not None
            else (detected_libraries[0] if detected_libraries else VerifiedRuleLibrary())
        )
        if (
            self.active_rule_library.records
            and self.active_rule_library not in base_reasoner.augmenters
        ):
            base_reasoner = UnifiedTypedReasoner(
                catalog=base_reasoner.catalog,
                augmenters=(*base_reasoner.augmenters, self.active_rule_library),
            )
        self.reasoner = base_reasoner
        self.experience_store = experience_store
        self.rules_artifact_path = (
            Path(rules_artifact_path) if rules_artifact_path is not None else None
        )
        self.experience_collector = (
            TypedExperienceCollector(
                experience_store,
                reasoner=self.reasoner,
            )
            if experience_store is not None
            else None
        )

    @property
    def experience_enabled(self) -> bool:
        return self.experience_collector is not None

    def solve(self, domain: str, values: Mapping[str, Any]) -> BeginnerSolveResult:
        return self._solve(domain, values)

    def experience_snapshot(self, *, limit: int = 20) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise BeginnerInputError("학습 후보 표시 개수는 1부터 100 사이여야 해.")
        if self.experience_store is None:
            return {
                "enabled": False,
                "stats": None,
                "reviewed_split_counts": {},
                "learning": {
                    "active_rules": len(self.active_rule_library.records),
                    "artifact_configured": self.rules_artifact_path is not None,
                },
                "items": [],
            }
        stats = self.experience_store.stats()
        corpus = self.experience_store.export_reviewed()
        items = self.experience_store.list_items(limit=limit)
        return {
            "enabled": True,
            "stats": {
                "requests": stats.requests,
                "observations": stats.observations,
                "pending": stats.pending,
                "conflicted": stats.conflicted,
                "approved": stats.approved,
                "rejected": stats.rejected,
                "by_domain": dict(stats.by_domain),
            },
            "reviewed_split_counts": dict(corpus.role_counts),
            "learning": {
                "active_rules": len(self.active_rule_library.records),
                "artifact_configured": self.rules_artifact_path is not None,
            },
            "items": [_experience_item_dict(item) for item in items],
        }

    def review_experience(
        self,
        request_digest: Any,
        *,
        expected_solved: Any,
        include_for_learning: Any = True,
        attest_human_review: Any,
        note: Any = "",
    ) -> dict[str, Any]:
        self._require_experience_store()
        expected = _require_bool(expected_solved, "기대 결과")
        include = _require_bool(include_for_learning, "학습 사용 여부")
        _require_human_attestation(attest_human_review)
        digest = _single_text(request_digest, "학습 후보 ID", maximum=64).lower()
        note_text = _optional_text(note, "검토 메모", maximum=500)
        assert self.experience_store is not None
        try:
            item = self.experience_store.get_item(digest)
            record = self.experience_store.review(
                digest,
                expected_solved=expected,
                phenomenon=_review_phenomenon(item.domain, item.triggers),
                rationale=note_text or _default_review_rationale(expected),
                reviewer="human:local-beginner-user",
                decision="approved" if include else "rejected",
                notes="explicit_local_human_attestation",
                tags=("beginner_ui", item.domain),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BeginnerInputError(f"학습 후보를 검토하지 못했어: {exc}") from exc
        return {
            "review": record.to_dict(),
            "queue": self.experience_snapshot(),
        }

    def review_example(
        self,
        domain: str,
        values: Mapping[str, Any],
        *,
        expected_solved: Any,
        attest_human_review: Any,
        note: Any = "",
    ) -> dict[str, Any]:
        self._require_experience_store()
        expected = _require_bool(expected_solved, "기대 결과")
        _require_human_attestation(attest_human_review)
        note_text = _optional_text(note, "검토 메모", maximum=500)
        rationale = note_text or _default_review_rationale(expected)
        result = self._solve(
            domain,
            values,
            proposal=_ExperienceProposal(expected, rationale),
        )
        if not result.experience_digest:
            raise BeginnerInputError("학습 후보 ID를 만들지 못했어.")
        reviewed = self.review_experience(
            result.experience_digest,
            expected_solved=expected,
            include_for_learning=True,
            attest_human_review=True,
            note=rationale,
        )
        return {"result": result.to_dict(), **reviewed}

    def learn_from_reviewed_experience(self) -> dict[str, Any]:
        self._require_experience_store()
        if self.rules_artifact_path is None:
            raise BeginnerInputError("검증된 규칙을 저장할 파일이 설정되지 않았어.")
        assert self.experience_store is not None
        try:
            run = run_beginner_reviewed_learning(
                self.experience_store,
                self.reasoner,
                self.rules_artifact_path,
                incumbent_library=self.active_rule_library,
            )
        except (TypeError, ValueError) as exc:
            raise BeginnerInputError(
                f"아직 안전한 규칙 학습을 시작할 준비가 안 됐어: {exc}"
            ) from exc
        learning = run.result.learning
        if run.promoted:
            self._activate_rule_library(learning.active_library)
        return {
            "promoted": run.promoted,
            "reviewed_records": run.reviewed_records,
            "active_rules": len(self.active_rule_library.records),
            "rejection_reasons": list(learning.rejection_reasons),
            "improved_domains": list(learning.improved_domains),
            "checkpoint": (
                run.checkpoint.to_dict() if run.checkpoint is not None else None
            ),
            "evaluation": learning.to_dict(),
            "queue": self.experience_snapshot(),
        }

    def _solve(
        self,
        domain: str,
        values: Mapping[str, Any],
        *,
        proposal: _ExperienceProposal | None = None,
    ) -> BeginnerSolveResult:
        normalized = str(domain).strip().lower()
        if normalized == "language":
            return self.solve_language(
                goal=values.get("goal", ""),
                required=values.get("required", ""),
                satisfied=values.get("satisfied", ""),
                blocked=values.get("blocked", ""),
                _proposal=proposal,
            )
        if normalized == "math":
            return self.solve_math(values.get("expression", ""), _proposal=proposal)
        if normalized == "vision":
            return self.solve_vision(
                values.get("preset", "red_square"),
                _proposal=proposal,
            )
        raise BeginnerInputError("언어, 수학, 비전 중 하나를 선택해 줘.")

    def solve_language(
        self,
        *,
        goal: Any,
        required: Any,
        satisfied: Any = "",
        blocked: Any = "",
        _proposal: _ExperienceProposal | None = None,
    ) -> BeginnerSolveResult:
        goal_text = _single_text(goal, "목표", maximum=80)
        required_items = _items(required, "필요 조건")
        satisfied_items = _items(satisfied, "충족한 조건", allow_empty=True)
        blocked_items = _items(blocked, "막힌 조건", allow_empty=True)

        overlap = _ordered_intersection(satisfied_items, blocked_items)
        if overlap:
            joined = ", ".join(overlap)
            raise BeginnerInputError(
                f"{joined} 항목이 '충족'과 '막힘'에 동시에 있어. 한쪽만 선택해 줘."
            )

        known_requirements = {item.casefold() for item in required_items}
        unrelated = tuple(
            item
            for item in (*satisfied_items, *blocked_items)
            if item.casefold() not in known_requirements
        )
        if unrelated:
            joined = ", ".join(dict.fromkeys(unrelated))
            raise BeginnerInputError(
                f"{joined} 항목은 필요 조건 목록에 없어. 먼저 필요 조건에 넣어 줘."
            )

        lines = [f"Goal: {goal_text}", f"Requires: {', '.join(required_items)}"]
        lines.extend(f"Satisfied: {item}" for item in satisfied_items)
        lines.extend(f"Blocked: {item}" for item in blocked_items)
        controlled_text = "\n".join(lines)
        core_run = self._run_core(
            TypedDomainRequest(
                "language",
                LanguageTextProblem(
                    controlled_text,
                    source_context="semop_beginner",
                    use_legacy_heuristics=False,
                ),
                mode="shadow",
            ),
            proposal=_proposal,
        )
        core = core_run.result

        satisfied_keys = {item.casefold() for item in satisfied_items}
        blocked_keys = {item.casefold() for item in blocked_items}
        missing = tuple(
            item
            for item in required_items
            if item.casefold() not in satisfied_keys and item.casefold() not in blocked_keys
        )
        predicate = _goal_predicate(core)
        verified = core.success and core.verified
        if verified and predicate == "READY":
            conclusion = "ready"
            title = "목표를 진행할 조건이 갖춰졌어"
            summary = f"{goal_text}에 필요한 조건이 모두 충족됐음을 검증했어."
        elif verified and predicate == "NOT_READY":
            conclusion = "not_ready"
            title = "목표가 아직 준비되지 않았어"
            summary = f"막힌 조건({', '.join(blocked_items)}) 때문에 {goal_text}을 진행할 수 없어."
        else:
            conclusion = "not_proven"
            title = "아직 결론을 증명하지 못했어"
            if missing:
                summary = f"상태를 알려 주지 않은 조건이 있어: {', '.join(missing)}"
            else:
                summary = "입력을 읽었지만 현재 연산자만으로 결론까지 이어지지 않았어."

        interpreted = (
            f"목표: {goal_text}",
            f"필요 조건: {', '.join(required_items)}",
            f"충족: {_joined_or_none(satisfied_items)}",
            f"막힘: {_joined_or_none(blocked_items)}",
        )
        return _result(
            domain="language",
            core=core,
            conclusion=conclusion,
            title=title,
            summary=summary,
            interpreted=interpreted,
            trust_notice=(
                "입력 문장에 적힌 조건을 논리적으로 재생 검증한 결과야. "
                "실제 승인이나 완료 여부 자체는 외부 증거로 확인하지 않았어."
            ),
            experience_digest=core_run.request_digest,
        )

    def solve_math(
        self,
        expression: Any,
        *,
        _proposal: _ExperienceProposal | None = None,
    ) -> BeginnerSolveResult:
        expression_text = _single_text(expression, "수학식", maximum=240)
        try:
            core_run = self._run_core(
                TypedDomainRequest("math", expression_text, mode="shadow"),
                proposal=_proposal,
            )
        except (TypeError, ValueError) as exc:
            raise BeginnerInputError(f"수학식을 읽지 못했어: {exc}") from exc

        core = core_run.result
        verified = core.success and core.verified
        return _result(
            domain="math",
            core=core,
            conclusion="proven" if verified else "not_proven",
            title="계산과 결론을 검증했어" if verified else "이 식은 증명되지 않았어",
            summary=(
                f"{expression_text}의 계산 과정을 연산자로 실행하고 처음부터 다시 확인했어."
                if verified
                else "등식이나 부등식의 양변, 숫자, 기호가 맞는지 확인해 봐."
            ),
            interpreted=(f"검증할 식: {expression_text}",),
            trust_notice=(
                "지원되는 사칙연산, 비교식, 일차방정식을 정확한 수 규칙으로 계산해. "
                "계산기 추측이나 언어 모델의 답을 그대로 믿지 않아."
            ),
            experience_digest=core_run.request_digest,
        )

    def solve_vision(
        self,
        preset_key: Any,
        *,
        _proposal: _ExperienceProposal | None = None,
    ) -> BeginnerSolveResult:
        key = _single_text(preset_key, "비전 예제", maximum=40)
        preset = VISION_PRESETS.get(key)
        if preset is None:
            raise BeginnerInputError("목록에 있는 비전 예제를 선택해 줘.")

        image = _image_from_pattern(preset.pattern, source=f"beginner:{preset.key}")
        core_run = self._run_core(
            TypedDomainRequest(
                "vision",
                RasterVisionProblem(
                    image=image,
                    goals=(preset.goal,),
                    query=preset.question,
                ),
                mode="shadow",
            ),
            proposal=_proposal,
        )
        core = core_run.result
        verified = core.success and core.verified
        return _result(
            domain="vision",
            core=core,
            conclusion="proven" if verified else "not_proven",
            title="색상 격자에서 관계를 확인했어" if verified else "관계를 확인하지 못했어",
            summary=(
                f"'{preset.question}'라는 질문을 픽셀 측정과 typed 연산자로 검증했어."
                if verified
                else "선택한 장면에서 질문의 결론을 증명하지 못했어."
            ),
            interpreted=(
                f"장면: {preset.title}",
                f"질문: {preset.question}",
                f"확인 방법: {preset.description}",
            ),
            trust_notice=(
                "현재 비전은 일반 사진 인식이 아니라 작은 색상 격자의 연결 요소, "
                "경계 상자, 픽셀 수만 정확히 다루는 MVP야."
            ),
            experience_digest=core_run.request_digest,
        )

    def _run_core(
        self,
        request: TypedDomainRequest,
        *,
        proposal: _ExperienceProposal | None = None,
    ) -> _BeginnerCoreRun:
        if self.experience_collector is None:
            if proposal is not None:
                raise BeginnerInputError("로컬 학습 후보 수집이 꺼져 있어.")
            return _BeginnerCoreRun(self.reasoner.run(request))
        observed = self.experience_collector.run(
            request,
            proposed_expected_solved=(
                proposal.expected_solved if proposal is not None else None
            ),
            proposal_authority=(
                "user_proposal" if proposal is not None else "unknown"
            ),
            rationale=proposal.rationale if proposal is not None else "",
            source=(
                "semop-beginner-human-review"
                if proposal is not None
                else "semop-beginner-runtime"
            ),
        )
        if observed.result is None:
            detail = observed.error_message or "typed runtime execution failed"
            raise ValueError(f"{observed.error_type}: {detail}")
        return _BeginnerCoreRun(observed.result, observed.request_digest)

    def _require_experience_store(self) -> None:
        if self.experience_store is None:
            raise BeginnerInputError("로컬 학습 후보 수집이 꺼져 있어.")

    def _activate_rule_library(self, library: VerifiedRuleLibrary) -> None:
        preserved = tuple(
            augmenter
            for augmenter in self.reasoner.augmenters
            if not isinstance(augmenter, VerifiedRuleLibrary)
        )
        self.active_rule_library = library
        self.reasoner = UnifiedTypedReasoner(
            catalog=self.reasoner.catalog,
            augmenters=(*preserved, library),
        )
        if self.experience_store is not None:
            self.experience_collector = TypedExperienceCollector(
                self.experience_store,
                reasoner=self.reasoner,
            )


def vision_presets_for_ui() -> tuple[dict[str, Any], ...]:
    return tuple(preset.to_dict() for preset in VISION_PRESETS.values())


def _experience_item_dict(item: ExperienceQueueItem) -> dict[str, Any]:
    payload = json.loads(item.payload_json)
    return {
        "request_digest": item.request_digest,
        "domain": item.domain,
        "summary": _experience_payload_summary(item.domain, payload),
        "payload": payload,
        "occurrences": item.occurrences,
        "observed_failures": item.observed_failures,
        "unverified_results": item.unverified_results,
        "triggers": list(item.triggers),
        "status": item.status.value,
        "priority": item.priority,
        "grounding_uncertainties": item.grounding_uncertainties,
        "latest_rationale": item.latest_rationale,
        "latest_review": (
            item.latest_review.to_dict()
            if item.latest_review is not None
            else None
        ),
    }


def _experience_payload_summary(domain: str, payload: Mapping[str, Any]) -> str:
    if domain == "language":
        return str(payload.get("text", "언어 입력"))
    if domain == "math":
        return str(payload.get("expression", "수학 입력"))
    if domain == "vision":
        query = str(payload.get("query", "비전 입력"))
        rows = payload.get("rows", ())
        height = len(rows) if isinstance(rows, list) else 0
        width = len(rows[0]) if height and isinstance(rows[0], list) else 0
        return f"{query} ({width}x{height} 격자)"
    return f"{domain} 입력"


def _result(
    *,
    domain: str,
    core: UnifiedTypedResult,
    conclusion: str,
    title: str,
    summary: str,
    interpreted: Sequence[str],
    trust_notice: str,
    experience_digest: str = "",
) -> BeginnerSolveResult:
    return BeginnerSolveResult(
        domain=domain,
        success=core.success,
        verified=core.verified,
        conclusion=conclusion,
        title=title,
        summary=summary,
        interpreted_input=tuple(interpreted),
        proof=core.proof_ko or "아직 표시할 증명 단계가 없어.",
        trust_notice=trust_notice,
        diagnostics=core.diagnostics,
        technical=core.to_dict(),
        experience_digest=experience_digest,
    )


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise BeginnerInputError(f"{label}를 다시 선택해 줘.")
    return value


def _require_human_attestation(value: Any) -> None:
    if value is not True:
        raise BeginnerInputError(
            "정확한 입력과 기대 결과를 직접 확인했다는 체크가 필요해."
        )


def _optional_text(value: Any, label: str, *, maximum: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise BeginnerInputError(f"{label}는 글자로 입력해 줘.")
    cleaned = " ".join(value.strip().split())
    if len(cleaned) > maximum:
        raise BeginnerInputError(f"{label}는 {maximum}자보다 짧게 적어 줘.")
    return cleaned


def _review_phenomenon(domain: str, triggers: Sequence[str]) -> str:
    suffix = "_".join(triggers[:3]) or "manual"
    return f"beginner_{domain}_{suffix}"


def _default_review_rationale(expected_solved: bool) -> str:
    outcome = "풀려야 한다" if expected_solved else "풀리지 않아야 한다"
    return f"로컬 사용자가 정확한 입력을 보고 이 문제는 {outcome}고 확인했다."


def _single_text(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise BeginnerInputError(f"{label}을(를) 글자로 입력해 줘.")
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise BeginnerInputError(f"{label}을(를) 입력해 줘.")
    if len(cleaned) > maximum:
        raise BeginnerInputError(f"{label}은(는) {maximum}자 이하로 입력해 줘.")
    return cleaned


def _items(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = value.replace("，", ",").replace(";", ",").replace("\n", ",").split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = list(value)
    else:
        raise BeginnerInputError(f"{label}을(를) 쉼표로 나눠 입력해 줘.")

    items: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, str):
            raise BeginnerInputError(f"{label}에는 글자 항목만 넣을 수 있어.")
        item = " ".join(raw.strip().split())
        if not item:
            continue
        if len(item) > 80:
            raise BeginnerInputError(f"{label}의 각 항목은 80자 이하로 입력해 줘.")
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            items.append(item)

    if not items and not allow_empty:
        raise BeginnerInputError(f"{label}을(를) 하나 이상 입력해 줘.")
    if len(items) > 20:
        raise BeginnerInputError(f"{label}은(는) 한 번에 20개까지만 넣을 수 있어.")
    return tuple(items)


def _ordered_intersection(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    right_keys = {item.casefold() for item in right}
    return tuple(item for item in left if item.casefold() in right_keys)


def _joined_or_none(items: Sequence[str]) -> str:
    return ", ".join(items) if items else "없음"


def _goal_predicate(core: UnifiedTypedResult) -> str:
    if core.instance is None or not core.instance.goals:
        return ""
    return core.instance.goals[0].atom.predicate.name


def _image_from_pattern(pattern: Sequence[str], *, source: str) -> RasterImage:
    try:
        rows = tuple(tuple(_PALETTE[cell] for cell in row) for row in pattern)
    except KeyError as exc:
        raise ValueError(f"unknown beginner vision color: {exc.args[0]}") from exc
    return RasterImage.from_rows(rows, background=WHITE, source=source)
