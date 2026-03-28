from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from .vlso.types import SharedWorldModel
from .vlso.world_model_narrator import WorldModelNarrator


@dataclass
class CommonWorldTaskResult:
    task_type: str
    answer_text: str
    rationale: list[str] = field(default_factory=list)
    recommended_steps: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class CommonWorldTaskEngine:
    def __init__(self, narrator: WorldModelNarrator | None = None) -> None:
        self.narrator = narrator or WorldModelNarrator()

    def run(
        self,
        prompt: str,
        world: SharedWorldModel,
        reasoning: dict[str, Any] | None,
        *,
        route: str,
        prompt_understanding: dict[str, Any] | None = None,
        fallback_answer_text: str = '',
        detail_payload: dict[str, Any] | None = None,
    ) -> CommonWorldTaskResult:
        prompt_understanding = prompt_understanding or {}
        reasoning = reasoning or {}
        detail_payload = detail_payload or {}
        task_type = self.infer_task_type(prompt, route, prompt_understanding)
        highlights = self._highlights(world)
        rationale = self._rationale(reasoning)
        steps = self._recommended_steps(reasoning, world)
        checks = self._checks(reasoning, world, route, detail_payload)

        if task_type == 'decide':
            answer_text = self._decide_answer(reasoning, highlights, rationale, steps, checks)
        elif task_type == 'plan':
            answer_text = self._plan_answer(highlights, rationale, steps, checks)
        elif task_type == 'verify':
            answer_text = self._verify_answer(rationale, checks)
        elif task_type == 'solve':
            answer_text = self._solve_answer(world, reasoning, route, rationale, steps, checks, detail_payload)
        elif task_type == 'explain':
            answer_text = self._explain_answer(prompt, world, route, reasoning, highlights, rationale, checks)
        else:
            answer_text = self._describe_answer(prompt, world, route, reasoning, highlights, rationale)

        if not str(answer_text or '').strip():
            answer_text = str(fallback_answer_text or '').strip()
        if not str(answer_text or '').strip():
            answer_text = self._fallback_from_reasoning(reasoning, highlights)
        return CommonWorldTaskResult(
            task_type=task_type,
            answer_text=answer_text.strip(),
            rationale=rationale,
            recommended_steps=steps,
            checks=checks,
            highlights=highlights,
        )

    @staticmethod
    def infer_task_type(prompt: str, route: str, prompt_understanding: dict[str, Any] | None = None) -> str:
        normalized = str(prompt or '').lower()
        if any(term in normalized for term in ('검증', '검사', '확인', '맞나요', 'verify', 'validate', 'check')):
            return 'verify'
        if any(term in normalized for term in ('왜', '이유', '설명', 'why', 'explain', 'because')):
            return 'explain'
        if route in {'math', 'cp'}:
            return 'solve'
        if route == 'ops':
            return 'decide'
        if any(term in normalized for term in ('어떻게', '순서', '계획', 'plan', 'steps', 'how should', 'what should be done')):
            return 'plan'
        if any(term in normalized for term in ('해도', '가능', '괜찮', 'should i', 'can i', 'allowed', '승인')):
            return 'decide'
        if route in {'vision', 'video', 'visual_3d'}:
            return 'describe'
        return 'explain'

    def _describe_answer(
        self,
        prompt: str,
        world: SharedWorldModel,
        route: str,
        reasoning: dict[str, Any],
        highlights: list[str],
        rationale: list[str],
    ) -> str:
        if route in {'vision', 'video', 'visual_3d'}:
            described = self.narrator.describe(prompt, world)
            if described.strip():
                return described.strip()
        lines: list[str] = []
        if highlights:
            lines.append('\uacf5\ud1b5 \uc6d4\ub4dc \ubaa8\ub378\uc5d0\uc11c \uc7a1\ud78c \ud575\uc2ec \uc694\uc18c\ub294 ' + ', '.join(highlights[:4]) + '\uc785\ub2c8\ub2e4.')
        if reasoning.get('primary_goal'):
            lines.append('\uc9c0\uae08 \ucd94\ub860\uc758 \uc911\uc2ec \ubaa9\ud45c\ub294 ' + str(reasoning.get('primary_goal')) + '\uc785\ub2c8\ub2e4.')
        if rationale:
            lines.append('\uadfc\uac70\ub294 ' + ' / '.join(rationale[:3]) + '\uc785\ub2c8\ub2e4.')
        return ' '.join(lines).strip()

    def _explain_answer(
        self,
        prompt: str,
        world: SharedWorldModel,
        route: str,
        reasoning: dict[str, Any],
        highlights: list[str],
        rationale: list[str],
        checks: list[str],
    ) -> str:
        lines: list[str] = []
        if route in {'vision', 'video', 'visual_3d'}:
            described = self.narrator.describe(prompt, world).strip()
            if described:
                lines.append(described)
        elif reasoning.get('primary_goal'):
            lines.append('\ud575\uc2ec \uc124\uba85\uc740 ' + str(reasoning.get('primary_goal')) + '\uc785\ub2c8\ub2e4.')
        elif highlights:
            lines.append('\ud575\uc2ec \ub300\uc0c1\uc740 ' + ', '.join(highlights[:3]) + '\uc785\ub2c8\ub2e4.')
        if rationale:
            lines.append('\uc774\ub807\uac8c \ubcf8 \uc774\uc720:')
            lines.extend(f'- {item}' for item in rationale[:4])
        if checks:
            lines.append('\uac80\uc99d \ub610\ub294 \uc8fc\uc758:')
            lines.extend(f'- {item}' for item in checks[:3])
        return '\n'.join(lines).strip()

    @staticmethod
    def _decide_answer(
        reasoning: dict[str, Any],
        highlights: list[str],
        rationale: list[str],
        steps: list[str],
        checks: list[str],
    ) -> str:
        blockers = [str(item) for item in reasoning.get('blockers', []) if str(item).strip()]
        prerequisites = [str(item) for item in reasoning.get('prerequisites', []) if str(item).strip()]
        if blockers or prerequisites:
            conclusion = '\uc9c0\uae08 \ud310\ub2e8\uc740 \ubc14\ub85c \uc9c4\ud589\ud558\uc9c0 \ub9d0\uace0 \uc120\ud589 \uc870\uac74\uacfc \ucc28\ub2e8 \uc694\uc778\uc744 \uba3c\uc800 \uc815\ub9ac\ud558\ub294 \ucabd\uc785\ub2c8\ub2e4.'
        else:
            conclusion = '\uc9c0\uae08 \ud310\ub2e8\uc740 \uc989\uc2dc \uc2e4\ud589\ubcf4\ub2e4\ub294 \ud604\uc7ac \uc0c1\ud0dc\ub97c \ud55c \ubc88 \ub354 \ud655\uc778\ud55c \ub4a4 \uc6c0\uc9c1\uc774\ub294 \ucabd\uc774 \uc548\uc804\ud569\ub2c8\ub2e4.'
        lines = [conclusion]
        if rationale:
            lines.extend(['', '\ud310\ub2e8 \uc774\uc720:'])
            lines.extend(f'- {item}' for item in rationale[:4])
        if steps:
            lines.extend(['', '\uad8c\uc7a5 \uc21c\uc11c:'])
            lines.extend(f'{index}. {item}' for index, item in enumerate(steps[:4], start=1))
        if checks:
            lines.extend(['', '\uc8fc\uc758:'])
            lines.extend(f'- {item}' for item in checks[:3])
        elif highlights:
            lines.extend(['', '\uad00\ucc30\ub41c \ub300\uc0c1:'])
            lines.extend(f'- {item}' for item in highlights[:3])
        return '\n'.join(lines).strip()

    @staticmethod
    def _plan_answer(
        highlights: list[str],
        rationale: list[str],
        steps: list[str],
        checks: list[str],
    ) -> str:
        lines = ['\uacf5\ud1b5 \uc6d4\ub4dc \ubaa8\ub378 \uae30\uc900 \uacc4\ud68d\uc740 \ub2e4\uc74c \uc21c\uc11c\uac00 \ub9de\uc2b5\ub2c8\ub2e4.']
        if steps:
            lines.extend(f'{index}. {item}' for index, item in enumerate(steps[:5], start=1))
        elif highlights:
            lines.extend(f'- {item}' for item in highlights[:3])
        if rationale:
            lines.extend(['', '\uadfc\uac70:'])
            lines.extend(f'- {item}' for item in rationale[:4])
        if checks:
            lines.extend(['', '\uac80\uc99d \ud3ec\uc778\ud2b8:'])
            lines.extend(f'- {item}' for item in checks[:3])
        return '\n'.join(lines).strip()

    @staticmethod
    def _verify_answer(rationale: list[str], checks: list[str]) -> str:
        lines = ['\uacf5\ud1b5 \uc6d4\ub4dc \ubaa8\ub378 \uae30\uc900 \uac80\uc99d \uacb0\uacfc\uc785\ub2c8\ub2e4.']
        if checks:
            lines.extend(f'- {item}' for item in checks[:4])
        if rationale:
            lines.extend(['', '\ud310\uc815 \uadfc\uac70:'])
            lines.extend(f'- {item}' for item in rationale[:4])
        return '\n'.join(lines).strip()

    @staticmethod
    def _solve_answer(
        world: SharedWorldModel,
        reasoning: dict[str, Any],
        route: str,
        rationale: list[str],
        steps: list[str],
        checks: list[str],
        detail_payload: dict[str, Any],
    ) -> str:
        metadata = world.metadata if isinstance(world.metadata, dict) else {}
        if route == 'cp':
            cp_meta = metadata.get('cp', {}) if isinstance(metadata.get('cp'), dict) else {}
            approach = str(detail_payload.get('approach') or cp_meta.get('approach') or '').strip()
            time_complexity = str(detail_payload.get('time_complexity') or cp_meta.get('time_complexity') or '').strip()
            memory_complexity = str(detail_payload.get('memory_complexity') or cp_meta.get('memory_complexity') or '').strip()
            lines: list[str] = []
            if world.goals:
                lines.append('\ubb38\uc81c \uad6c\uc870\ub294 ' + ', '.join(str(item) for item in world.goals[:2]) + ' \ucabd\uc785\ub2c8\ub2e4.')
            elif reasoning.get('primary_goal'):
                lines.append('\ubb38\uc81c \ubaa9\ud45c\ub294 ' + str(reasoning.get('primary_goal')) + '\uc785\ub2c8\ub2e4.')
            if approach:
                lines.append('\uc54c\uace0\ub9ac\uc998 \ucd08\uc548: ' + approach)
            if time_complexity or memory_complexity:
                lines.append('\ubcf5\uc7a1\ub3c4: \uc2dc\uac04 ' + (time_complexity or '-') + ', \uba54\ubaa8\ub9ac ' + (memory_complexity or '-'))
            if rationale:
                lines.append('\uadfc\uac70: ' + ' / '.join(rationale[:3]))
            if checks:
                lines.append('\uac80\uc99d: ' + ' / '.join(checks[:3]))
            return '\n'.join(lines).strip()
        math_meta = metadata.get('math', {}) if isinstance(metadata.get('math'), dict) else {}
        chosen_answer = str(math_meta.get('chosen_answer') or detail_payload.get('safe_answer') or '').strip()
        lines = []
        if chosen_answer:
            lines.append('\ud604\uc7ac \ud574\ub2f5 \ud6c4\ubcf4: ' + chosen_answer)
        if rationale:
            lines.append('\ud575\uc2ec \uadfc\uac70: ' + ' / '.join(rationale[:3]))
        if steps:
            lines.append('\ub2e4\uc74c \uac80\uc99d \ub2e8\uacc4: ' + ' / '.join(steps[:3]))
        if checks:
            lines.append('\uac80\uc99d \uc0c1\ud0dc: ' + ' / '.join(checks[:3]))
        return '\n'.join(lines).strip()

    @staticmethod
    def _fallback_from_reasoning(reasoning: dict[str, Any], highlights: list[str]) -> str:
        if reasoning.get('summary'):
            return str(reasoning.get('summary'))
        if highlights:
            return '\ud575\uc2ec \uc694\uc18c: ' + ', '.join(highlights[:4])
        return '\uacf5\ud1b5 \uc6d4\ub4dc \ubaa8\ub378\uc740 \ub9cc\ub4e4\uc5b4\uc84c\uc9c0\ub9cc, \uc544\uc9c1 \uc0ac\ub78c\uc774 \uc77d\uae30 \uc88b\uc740 \uacb0\uacfc\ub97c \ucda9\ubd84\ud788 \ub9cc\ub4e4\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.'

    @staticmethod
    def _highlights(world: SharedWorldModel) -> list[str]:
        items: list[str] = []
        for entity in getattr(world, 'entities', [])[:8]:
            label = str(getattr(entity, 'label', '') or getattr(entity, 'id', '') or '').strip()
            if label:
                items.append(CommonWorldTaskEngine._humanize(label))
        for event in getattr(world, 'events', [])[:4]:
            label = str(getattr(event, 'label', '') or getattr(event, 'id', '') or '').strip()
            if label:
                items.append(CommonWorldTaskEngine._humanize(label))
        return CommonWorldTaskEngine._dedupe(items)[:6]

    @staticmethod
    def _rationale(reasoning: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        lines.extend(str(item) for item in reasoning.get('blockers', [])[:3] if str(item).strip())
        lines.extend(str(item) for item in reasoning.get('prerequisites', [])[:3] if str(item).strip())
        lines.extend(str(item) for item in reasoning.get('evidence', [])[:4] if str(item).strip())
        return CommonWorldTaskEngine._dedupe([CommonWorldTaskEngine._humanize(item) for item in lines if str(item).strip()])

    @staticmethod
    def _recommended_steps(reasoning: dict[str, Any], world: SharedWorldModel) -> list[str]:
        items = [str(item) for item in reasoning.get('next_steps', []) if str(item).strip()]
        if not items:
            items = [str(item) for item in getattr(world, 'inferred_steps', []) if str(item).strip()]
        return CommonWorldTaskEngine._dedupe([CommonWorldTaskEngine._humanize(item) for item in items])[:6]

    @staticmethod
    def _checks(reasoning: dict[str, Any], world: SharedWorldModel, route: str, detail_payload: dict[str, Any]) -> list[str]:
        checks: list[str] = [str(item) for item in reasoning.get('warnings', []) if str(item).strip()]
        metadata = world.metadata if isinstance(world.metadata, dict) else {}
        if route == 'cp':
            compile_ok = detail_payload.get('compile_ok')
            if compile_ok is not None:
                checks.append('\ucef4\ud30c\uc77c \ud1b5\uacfc' if bool(compile_ok) else '\ucef4\ud30c\uc77c \uc2e4\ud328')
            validation_report = detail_payload.get('validation_report')
            if isinstance(validation_report, dict):
                checks.append('\uac80\uc99d \ud1b5\uacfc' if bool(validation_report.get('overall_ok')) else '\uac80\uc99d \uc2e4\ud328')
        if route == 'math':
            math_meta = metadata.get('math', {}) if isinstance(metadata.get('math'), dict) else {}
            if math_meta.get('verification_score') not in (None, ''):
                checks.append('verification score ' + str(math_meta.get('verification_score')))
            if math_meta.get('solved') is not None:
                checks.append('\ucd5c\uc885 \ud574\ub2f5 \ud655\uc815' if bool(math_meta.get('solved')) else '\ucd5c\uc885 \ud574\ub2f5 \ubbf8\ud655\uc815')
        return CommonWorldTaskEngine._dedupe([CommonWorldTaskEngine._humanize(item) for item in checks if str(item).strip()])[:5]

    @staticmethod
    def _humanize(value: str) -> str:
        return re.sub(r'\s+', ' ', str(value or '').replace('_', ' ')).strip()

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            deduped.append(normalized)
            seen.add(normalized)
        return deduped
