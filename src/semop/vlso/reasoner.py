from __future__ import annotations

import json

from .aligner import VLSOAligner
from .embedding_store import VisualEmbeddingRecord, VisualEmbeddingStore
from ..llm_client import LocalLLMConfig
from .language_parser import VLSOLanguageParser
from .qa import LocalTextGenerator, VLSOAnswer, VLSOQuestionAnswerer
from .types import SharedWorldModel, VisualObservation
from .vision_backbones import VisionEmbeddingExtractor
from .visual_parser import VLSOVisualParser


class VLSOReasoner:
    def __init__(
        self,
        mode: str = 'hybrid',
        language_mode: str = 'heuristic',
        visual_store_path: str | None = None,
        vision_model_id: str | None = None,
        vision_model_path: str | None = None,
        affordance_weights_path: str | None = None,
        concept_store_path: str | None = None,
        operator_store_path: str | None = None,
        answer_mode: str = 'structured',
        answer_model_id: str = 'Qwen/Qwen2.5-3B-Instruct',
    ) -> None:
        self.mode = mode
        self.language_parser = VLSOLanguageParser(mode=language_mode)
        self.visual_parser = VLSOVisualParser(
            affordance_weights_path=affordance_weights_path,
            concept_store_path=concept_store_path,
            operator_store_path=operator_store_path,
        )
        self.aligner = VLSOAligner()
        resolved_model_id = self._resolve_model_id(mode, vision_model_id)
        auto_resolve = mode in {'deep', 'hybrid'}
        self.embedding_extractor = VisionEmbeddingExtractor(
            model_id=resolved_model_id,
            local_model_path=vision_model_path,
            auto_resolve=auto_resolve,
        )
        self.visual_store = VisualEmbeddingStore(visual_store_path) if visual_store_path else None
        self.answer_mode = answer_mode
        self.answerer = VLSOQuestionAnswerer(
            generator=LocalTextGenerator(LocalLLMConfig(model_id=answer_model_id)) if answer_mode == 'llm' else None
        )

    def _resolve_model_id(self, mode: str, vision_model_id: str | None) -> str:
        if vision_model_id:
            return vision_model_id
        if mode == 'deep':
            return 'dinov2_adapter'
        if mode == 'hybrid':
            return 'dinov2_adapter'
        return 'token_geometry_v1'

    def run(self, query: str, visual_input: str | dict | None = None, remember_visual: bool = False, visual_key: str = '') -> SharedWorldModel:
        language_model, _ = self.language_parser.parse(query)
        language_model.metadata['reasoning_mode'] = self.mode
        if visual_input is None:
            return self.aligner.align(language_model, None)
        visual_model, observation = self.visual_parser.parse(visual_input)
        visual_model.metadata['reasoning_mode'] = self.mode
        world = self.aligner.align(language_model, visual_model)
        world.metadata['reasoning_mode'] = self.mode
        self._attach_visual_memory(world, observation, remember_visual=remember_visual, visual_key=visual_key)
        return world

    def _attach_visual_memory(self, world: SharedWorldModel, observation: VisualObservation, remember_visual: bool, visual_key: str) -> None:
        vector = self.embedding_extractor.embed_observation(observation)
        summary = self.embedding_extractor.backend_summary()
        world.metadata['vision_backend'] = summary
        world.audit_trace.append(f'reasoning mode: {self.mode}')
        world.audit_trace.append(f"vision backend: {summary['active_backend']}")
        if summary.get('auto_resolved'):
            world.audit_trace.append(f"vision model auto-resolved: {summary['local_model_path']}")
        if summary.get('load_error'):
            world.audit_trace.append(f"vision backend note: {summary['load_error']}")
        if self.visual_store is not None:
            matches = self.visual_store.search(vector, limit=3)
            if matches:
                world.warnings.append('visual memory matches were found in the embedding store')
                world.audit_trace.append('visual embedding retrieval executed')
                world.metadata['visual_memory_matches'] = [item.model_dump() for item in matches]
                for match in matches:
                    world.audit_trace.append(f'visual memory match: {match.label} ({match.score:.3f})')
        if remember_visual and self.visual_store is not None:
            key = visual_key or f'visual:{len(world.entities)}:{len(world.relations)}'
            metadata = {
                'query': world.query,
                'constraints': world.constraints,
                'entity_ids': [item.id for item in world.entities],
            }
            metadata.update(observation.metadata)
            self.visual_store.upsert(
                VisualEmbeddingRecord(
                    key=key,
                    label=world.query,
                    vector=vector,
                    metadata=metadata,
                )
            )
            world.audit_trace.append('visual observation stored in embedding memory')

    @staticmethod
    def to_text(model: SharedWorldModel) -> str:
        lines = [
            f'Query: {model.query}',
            f"Entities: {', '.join(item.id for item in model.entities) or 'none'}",
            f"Relations: {', '.join(f'{item.source}-{item.relation}-{item.target}' for item in model.relations) or 'none'}",
            f"Operators: {', '.join(item.name for item in model.operators) or 'none'}",
            f"Constraints: {', '.join(model.constraints) or 'none'}",
            f"Inferred steps: {' | '.join(model.inferred_steps) or 'none'}",
        ]
        if model.warnings:
            lines.append(f"Warnings: {' | '.join(model.warnings)}")
        if model.audit_trace:
            lines.append(f"Audit: {' | '.join(model.audit_trace)}")
        return "\n".join(lines)

    @staticmethod
    def to_json(model: SharedWorldModel) -> str:
        return json.dumps(model.model_dump(), ensure_ascii=False, indent=2)

    def answer(self, query: str, visual_input: str | dict | None = None, remember_visual: bool = False, visual_key: str = '') -> tuple[SharedWorldModel, VLSOAnswer]:
        world = self.run(query, visual_input=visual_input, remember_visual=remember_visual, visual_key=visual_key)
        answer = self.answerer.answer(query, world, answer_mode=self.answer_mode)
        world.metadata['answer'] = answer.model_dump()
        return world, answer
