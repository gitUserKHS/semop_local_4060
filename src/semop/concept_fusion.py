from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any
import re


@dataclass
class ConceptFusionHypothesis:
    label: str
    fusion_type: str
    source_concepts: list[str] = field(default_factory=list)
    source_domains: list[str] = field(default_factory=list)
    rationale: str = ''
    novelty_score: float = 0.0
    grounded_constraints: list[str] = field(default_factory=list)
    candidate_applications: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConceptFusionSummary:
    prompt: str
    route: str
    headline: str
    hypotheses: list[ConceptFusionHypothesis] = field(default_factory=list)
    extracted_concepts: list[str] = field(default_factory=list)
    recommended_focus: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'prompt': self.prompt,
            'route': self.route,
            'headline': self.headline,
            'hypotheses': [item.model_dump() for item in self.hypotheses],
            'extracted_concepts': list(self.extracted_concepts),
            'recommended_focus': list(self.recommended_focus),
        }


class ConceptFusionEngine:
    _STOPWORDS = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'what', 'when', 'then', 'have', 'will',
        'should', 'could', 'would', 'about', 'there', 'their', 'your', 'after', 'before', 'through', 'while',
        'prompt', 'answer', 'route', 'input', 'output', 'state', 'using', 'user', 'need', 'needs', 'more',
        'hidden', 'likely', 'context', 'constraints', 'reasoning', 'question', 'video', 'image', 'math',
    }

    _ROUTE_PRIORS = {
        'ops': ['context inference', 'safety gating', 'action planning'],
        'vision': ['scene grounding', 'object relations', 'world-model perception'],
        'video': ['temporal grounding', 'event tracking', 'world-model perception'],
        'visual_3d': ['topology reconstruction', 'scene assembly', 'embodied geometry'],
        'math': ['proof planning', 'symbolic verification', 'diagram grounding'],
    }

    def build_summary(
        self,
        prompt: str,
        route: str,
        prompt_understanding: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ConceptFusionSummary:
        prompt_understanding = prompt_understanding if isinstance(prompt_understanding, dict) else {}
        payload = payload if isinstance(payload, dict) else {}
        concepts = self._collect_concepts(prompt, route, prompt_understanding, payload)
        hypotheses = self._build_hypotheses(route, concepts, prompt_understanding)
        headline = self._headline(route, hypotheses)
        recommended_focus = self._recommended_focus(route, hypotheses, prompt_understanding)
        return ConceptFusionSummary(
            prompt=str(prompt or ''),
            route=str(route or 'general'),
            headline=headline,
            hypotheses=hypotheses,
            extracted_concepts=concepts,
            recommended_focus=recommended_focus,
        )

    def _collect_concepts(
        self,
        prompt: str,
        route: str,
        prompt_understanding: dict[str, Any],
        payload: dict[str, Any],
    ) -> list[str]:
        concepts: list[str] = []
        concepts.extend(self._ROUTE_PRIORS.get(str(route or ''), []))
        concepts.extend(self._extract_phrases(str(prompt or '')))
        for key in ('hidden_context', 'hidden_constraints', 'required_inputs'):
            values = prompt_understanding.get(key, [])
            if isinstance(values, list):
                for value in values:
                    concepts.extend(self._extract_phrases(str(value)))
        for key in ('notes', 'warnings', 'recommended_actions'):
            values = payload.get(key, [])
            if isinstance(values, list):
                for value in values[:4]:
                    concepts.extend(self._extract_phrases(str(value)))
        likely_domain = str(prompt_understanding.get('likely_domain') or '').strip()
        likely_scenario = str(prompt_understanding.get('likely_scenario') or '').strip()
        if likely_domain:
            concepts.append(likely_domain.replace('_', ' '))
        if likely_scenario:
            concepts.append(likely_scenario.replace('_', ' '))
        unique: list[str] = []
        seen: set[str] = set()
        for concept in concepts:
            normalized = self._normalize(concept)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique[:12]

    def _build_hypotheses(
        self,
        route: str,
        concepts: list[str],
        prompt_understanding: dict[str, Any],
    ) -> list[ConceptFusionHypothesis]:
        hypotheses: list[ConceptFusionHypothesis] = []
        constraints = [str(item) for item in (prompt_understanding.get('hidden_constraints') or [])[:3]]
        domain = str(prompt_understanding.get('likely_domain') or route or 'general').replace('_', ' ')
        scenario = str(prompt_understanding.get('likely_scenario') or 'general reasoning').replace('_', ' ')
        pairs = list(combinations(concepts[:8], 2))
        if not pairs and concepts:
            pairs = [(concepts[0], domain)]
        for left, right in pairs[:3]:
            label = self._label_for_pair(route, left, right)
            fusion_type = self._fusion_type(route, left, right)
            novelty = self._novelty_score(left, right, domain, scenario)
            rationale = (
                f'Combine {left} with {right} so the world model can transfer structure from {domain} into '
                f'{scenario} while staying grounded by explicit operator checks.'
            )
            applications = self._applications(route, left, right)
            hypotheses.append(
                ConceptFusionHypothesis(
                    label=label,
                    fusion_type=fusion_type,
                    source_concepts=[left, right],
                    source_domains=[domain, scenario],
                    rationale=rationale,
                    novelty_score=novelty,
                    grounded_constraints=constraints,
                    candidate_applications=applications,
                )
            )
        return hypotheses

    def _extract_phrases(self, text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        phrases: list[str] = []
        for token in tokens:
            if len(token) < 4 or token in self._STOPWORDS:
                continue
            phrases.append(token.replace('_', ' '))
        return phrases[:8]

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r'\s+', ' ', str(value or '').strip().lower())

    @staticmethod
    def _fusion_type(route: str, left: str, right: str) -> str:
        route = str(route or 'general')
        if route == 'math':
            return 'proof-world-model fusion'
        if route in {'vision', 'video', 'visual_3d'}:
            return 'perception-planning fusion'
        if route == 'ops':
            return 'context-action fusion'
        return 'operator-concept fusion'

    @staticmethod
    def _label_for_pair(route: str, left: str, right: str) -> str:
        route_prefix = {
            'math': 'Creative proof prior',
            'vision': 'Scene reasoning prior',
            'video': 'Temporal world-model prior',
            'visual_3d': '3D topology prior',
            'ops': 'Embodied action prior',
        }.get(str(route or ''), 'Concept fusion prior')
        return f'{route_prefix}: {left} + {right}'

    @staticmethod
    def _novelty_score(left: str, right: str, domain: str, scenario: str) -> float:
        diversity = len({left, right, domain, scenario})
        overlap_penalty = 0.0 if left != right else 0.2
        return round(max(0.1, min(0.98, 0.22 * diversity - overlap_penalty)), 3)

    @staticmethod
    def _applications(route: str, left: str, right: str) -> list[str]:
        route = str(route or 'general')
        if route == 'math':
            return [
                f'Use {left} to bias proof search before verifying {right}.',
                'Generate subgoals that can be audited by symbolic checks.',
            ]
        if route in {'vision', 'video', 'visual_3d'}:
            return [
                f'Fuse {left} with {right} to propose richer scene hypotheses.',
                'Reuse the fused hypothesis for embodied planning or reconstruction order.',
            ]
        if route == 'ops':
            return [
                f'Use {left} to refine action safety around {right}.',
                'Turn the fused prior into a checklist before execution.',
            ]
        return [
            f'Fuse {left} with {right} to propose a new operator family.',
            'Test the fused hypothesis with compiler-style validation before trusting it.',
        ]

    @staticmethod
    def _headline(route: str, hypotheses: list[ConceptFusionHypothesis]) -> str:
        if not hypotheses:
            return 'No strong concept-fusion hypothesis was generated yet.'
        if route == 'math':
            return 'Creative proof hints were generated by fusing world-model and symbolic concepts.'
        if route in {'vision', 'video', 'visual_3d'}:
            return 'Cross-modal concept fusion is proposing richer world-model hypotheses.'
        if route == 'ops':
            return 'Context, action, and safety concepts were fused into reusable embodied priors.'
        return 'Operator-level concept fusion generated reusable creative hypotheses.'

    @staticmethod
    def _recommended_focus(
        route: str,
        hypotheses: list[ConceptFusionHypothesis],
        prompt_understanding: dict[str, Any],
    ) -> list[str]:
        if not hypotheses:
            return ['Collect more reviewed traces before trusting creative recombination.']
        route = str(route or 'general')
        focus = [
            'Keep fused hypotheses grounded by explicit evidence or operator traces.',
            'Promote only creative hypotheses that improve verification or planning quality.',
        ]
        if route in {'vision', 'video', 'visual_3d'}:
            focus.append('Validate fused scene ideas against visible entities before using them for control.')
        if route == 'math':
            focus.append('Use fused proof ideas only when symbolic checks can still verify each step.')
        likely_domain = str(prompt_understanding.get('likely_domain') or '').strip()
        if likely_domain:
            focus.append(f'Reuse the best fused prior in nearby {likely_domain.replace("_", " ")} tasks.')
        return focus[:4]
