from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .vlso import VLSOReasoner
from .vlso.types import SharedWorldModel, VLSOEvent


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.ppm', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


@dataclass
class FrameSituationSummary:
    frame_index: int
    label: str
    answer_text: str
    entity_ids: list[str] = field(default_factory=list)
    relation_tuples: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalSituationSummary:
    input_path: str
    frame_count: int
    sampled_frames: list[FrameSituationSummary] = field(default_factory=list)
    stable_entities: list[str] = field(default_factory=list)
    stable_relations: list[str] = field(default_factory=list)
    changed_entities: list[str] = field(default_factory=list)
    temporal_events: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    situation_summary: str = ''
    answer_text: str = ''
    input_kind: str = 'frame_sequence'
    extraction_backend: str = ''
    backend_support: dict[str, bool] = field(default_factory=dict)
    fallback_hint: str = ''
    world: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            'input_path': self.input_path,
            'frame_count': self.frame_count,
            'sampled_frames': [item.model_dump() for item in self.sampled_frames],
            'stable_entities': list(self.stable_entities),
            'stable_relations': list(self.stable_relations),
            'changed_entities': list(self.changed_entities),
            'temporal_events': list(self.temporal_events),
            'warnings': list(self.warnings),
            'situation_summary': self.situation_summary,
            'answer_text': self.answer_text,
            'input_kind': self.input_kind,
            'extraction_backend': self.extraction_backend,
            'backend_support': dict(self.backend_support),
            'fallback_hint': self.fallback_hint,
            'world': dict(self.world),
        }


class TemporalSceneReasoner:
    def __init__(
        self,
        *,
        mode: str = 'deep',
        answer_mode: str = 'structured',
        concept_store_path: str | None = None,
        operator_store_path: str | None = None,
        affordance_weights_path: str | None = None,
    ) -> None:
        self.reasoner = VLSOReasoner(
            mode=mode,
            answer_mode=answer_mode,
            concept_store_path=concept_store_path,
            operator_store_path=operator_store_path,
            affordance_weights_path=affordance_weights_path,
        )

    @staticmethod
    def backend_support() -> dict[str, bool]:
        support = {
            'pillow': False,
            'imageio': False,
            'opencv': False,
            'ffmpeg': bool(shutil.which('ffmpeg')),
        }
        try:
            import PIL  # noqa: F401
            support['pillow'] = True
        except ImportError:
            pass
        try:
            import imageio.v3  # noqa: F401
            support['imageio'] = True
        except ImportError:
            pass
        try:
            import cv2  # noqa: F401
            support['opencv'] = True
        except ImportError:
            pass
        return support

    def summarize(self, query: str, input_path: str, max_frames: int = 6) -> TemporalSituationSummary:
        support = self.backend_support()
        input_kind, frames, cleanup_files, cleanup_dirs, extraction_backend = self._collect_frame_inputs(
            input_path,
            max_frames=max_frames,
            support=support,
        )
        if not frames:
            raise FileNotFoundError(f'No usable frames were found for {input_path}.')
        sampled: list[FrameSituationSummary] = []
        frame_worlds: list[SharedWorldModel] = []
        entity_presence: dict[str, int] = {}
        relation_presence: dict[str, int] = {}
        warnings: list[str] = []
        frame_entity_sets: list[set[str]] = []
        for index, frame in enumerate(frames):
            payload = frame.get('visual_input')
            if payload is None and frame.get('path'):
                payload = {'image_path': frame['path'], 'metadata': {'image_path': frame['path'], 'frame_index': index}}
            world, answer = self.reasoner.answer(query, visual_input=payload)
            frame_worlds.append(world)
            entity_ids = [str(item.id) for item in world.entities if str(item.id).strip()]
            relation_tuples = [f'{item.source}:{item.relation}:{item.target}' for item in world.relations]
            frame_entity_set = set(entity_ids)
            frame_entity_sets.append(frame_entity_set)
            for entity in frame_entity_set:
                entity_presence[entity] = entity_presence.get(entity, 0) + 1
            for relation in set(relation_tuples):
                relation_presence[relation] = relation_presence.get(relation, 0) + 1
            warnings.extend(str(item) for item in (world.warnings or [])[:3])
            sampled.append(
                FrameSituationSummary(
                    frame_index=index,
                    label=str(frame.get('label') or frame.get('path') or f'frame_{index + 1}'),
                    answer_text=answer.answer_text,
                    entity_ids=entity_ids[:12],
                    relation_tuples=relation_tuples[:12],
                    warnings=list(world.warnings[:4]),
                )
            )
        stable_entities = sorted(
            entity
            for entity, count in entity_presence.items()
            if count >= max(2, int(round(len(sampled) * 0.6)))
        )
        stable_relations = sorted(
            relation
            for relation, count in relation_presence.items()
            if count >= max(2, int(round(len(sampled) * 0.6)))
        )
        changed_entities = self._changed_entities(frame_entity_sets, stable_entities)
        temporal_world = self._aggregate_world(
            query,
            frame_worlds,
            sampled,
            stable_entities,
            stable_relations,
            changed_entities,
        )
        temporal_events = [self._event_to_text(item) for item in temporal_world.events[:8]]
        warnings.extend(str(item) for item in temporal_world.warnings[:4])
        warnings = self._unique(warnings)
        fallback_hint = ''
        if input_kind in {'video_file', 'gif'} and extraction_backend in {'manifest_only', ''}:
            fallback_hint = 'Install Pillow, imageio, OpenCV, or ffmpeg for direct video extraction. Frame folders and JSON manifests always work.'
        if len(sampled) < 2:
            warnings.append('Only one usable frame was found, so temporal change evidence is limited.')
        fallback_summary = self._fallback_situation_summary(sampled, stable_entities, changed_entities, temporal_events, warnings)
        situation_summary = self.reasoner.answerer.narrate_world_model(query, temporal_world) or fallback_summary
        for file_path in cleanup_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError:
                pass
        for dir_path in cleanup_dirs:
            shutil.rmtree(dir_path, ignore_errors=True)
        return TemporalSituationSummary(
            input_path=input_path,
            frame_count=len(sampled),
            sampled_frames=sampled,
            stable_entities=stable_entities[:10],
            stable_relations=stable_relations[:10],
            changed_entities=changed_entities[:10],
            temporal_events=temporal_events[:8],
            warnings=warnings[:8],
            situation_summary=situation_summary,
            answer_text=situation_summary,
            input_kind=input_kind,
            extraction_backend=extraction_backend,
            backend_support=support,
            fallback_hint=fallback_hint,
            world=temporal_world.model_dump(),
        )

    def _aggregate_world(
        self,
        query: str,
        frame_worlds: list[SharedWorldModel],
        sampled: list[FrameSituationSummary],
        stable_entities: list[str],
        stable_relations: list[str],
        changed_entities: list[str],
    ) -> SharedWorldModel:
        world = SharedWorldModel(query=query)
        for frame_world in frame_worlds:
            self._merge_world(world, frame_world)
        for event in self._temporal_events(frame_worlds, sampled):
            world.add_event(event)
        world.metadata['temporal_scene_summary'] = {
            'frame_count': len(sampled),
            'stable_entities': list(stable_entities[:10]),
            'stable_relations': list(stable_relations[:10]),
            'changed_entities': list(changed_entities[:10]),
            'sampled_frame_labels': [item.label for item in sampled[:10]],
        }
        world.audit_trace.append(f'temporal aggregation merged {len(frame_worlds)} frame world(s)')
        self.reasoner._attach_scene_operator_reasoning(world, query)
        return world

    @staticmethod
    def _merge_world(destination: SharedWorldModel, source: SharedWorldModel) -> None:
        for entity in source.entities:
            destination.add_entity(entity)
        for relation in source.relations:
            destination.add_relation(relation)
        for operator in source.operators:
            destination.add_operator(operator)
        for event in source.events:
            destination.add_event(event)
        destination.goals.extend(item for item in source.goals if item not in destination.goals)
        destination.constraints.extend(item for item in source.constraints if item not in destination.constraints)
        destination.warnings.extend(item for item in source.warnings if item not in destination.warnings)
        destination.audit_trace.extend(item for item in source.audit_trace if item not in destination.audit_trace)
        destination.inferred_steps.extend(item for item in source.inferred_steps if item not in destination.inferred_steps)
        for key, value in source.metadata.items():
            if key not in destination.metadata:
                destination.metadata[key] = value
            elif isinstance(destination.metadata[key], list) and isinstance(value, list):
                for item in value:
                    if item not in destination.metadata[key]:
                        destination.metadata[key].append(item)
            elif isinstance(destination.metadata[key], dict) and isinstance(value, dict):
                merged = dict(destination.metadata[key])
                merged.update(value)
                destination.metadata[key] = merged

    def _temporal_events(self, frame_worlds: list[SharedWorldModel], sampled: list[FrameSituationSummary]) -> list[VLSOEvent]:
        events: list[VLSOEvent] = []
        for frame_index, (previous, current) in enumerate(zip(frame_worlds, frame_worlds[1:]), start=1):
            current_label = sampled[frame_index].label if frame_index < len(sampled) else f'frame_{frame_index + 1}'
            previous_entities = {item.id for item in previous.entities if item.modality == 'vision'}
            current_entities = {item.id for item in current.entities if item.modality == 'vision'}
            for entity_id in sorted(current_entities - previous_entities):
                events.append(
                    VLSOEvent(
                        id=f'event:appearance:{frame_index}:{entity_id}',
                        label=f'{entity_id} appeared',
                        event_type='appearance',
                        frame_index=frame_index,
                        confidence=0.8,
                        participants={'entity': entity_id},
                        attributes={'frame_label': current_label},
                    )
                )
            for entity_id in sorted(previous_entities - current_entities):
                events.append(
                    VLSOEvent(
                        id=f'event:disappearance:{frame_index}:{entity_id}',
                        label=f'{entity_id} disappeared',
                        event_type='disappearance',
                        frame_index=frame_index,
                        confidence=0.8,
                        participants={'entity': entity_id},
                        attributes={'frame_label': current_label},
                    )
                )
            previous_states = self._entity_state_map(previous)
            current_states = self._entity_state_map(current)
            for entity_id in sorted(set(previous_states) & set(current_states)):
                if previous_states[entity_id] == current_states[entity_id]:
                    continue
                events.append(
                    VLSOEvent(
                        id=f'event:state_change:{frame_index}:{entity_id}',
                        label=f'{entity_id} changed state',
                        event_type='state_change',
                        frame_index=frame_index,
                        confidence=0.78,
                        participants={
                            'entity': entity_id,
                            'previous_state': previous_states[entity_id],
                            'current_state': current_states[entity_id],
                        },
                        attributes={'frame_label': current_label},
                    )
                )
            if len(events) >= 12:
                break
        return events

    @staticmethod
    def _entity_state_map(world: SharedWorldModel) -> dict[str, str]:
        states: dict[str, str] = {}
        for relation in world.relations:
            if relation.relation != 'STATE':
                continue
            states[relation.source] = relation.target
        return states

    @staticmethod
    def _event_to_text(event: VLSOEvent) -> str:
        event_type = str(event.event_type or '').strip()
        entity = str(event.participants.get('entity') or '').strip()
        frame_label = str(event.attributes.get('frame_label') or '').strip()
        previous_state = str(event.participants.get('previous_state') or '').split(':')[-1]
        current_state = str(event.participants.get('current_state') or '').split(':')[-1]
        prefix = (frame_label + ': ') if frame_label else ''
        if event_type == 'appearance' and entity:
            return prefix + 'appeared -> ' + entity
        if event_type == 'disappearance' and entity:
            return prefix + 'disappeared -> ' + entity
        if event_type == 'state_change' and entity:
            return prefix + f'state_change -> {entity} ({previous_state} -> {current_state})'
        return str(event.label or '').strip()

    def _collect_frame_inputs(
        self,
        input_path: str,
        *,
        max_frames: int,
        support: dict[str, bool],
    ) -> tuple[str, list[dict[str, Any]], list[str], list[str], str]:
        path = Path(str(input_path or '').strip())
        cleanup_files: list[str] = []
        cleanup_dirs: list[str] = []
        if path.is_dir():
            frames = [
                {'path': str(candidate), 'label': candidate.name}
                for candidate in sorted(path.iterdir())
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
            ]
            return 'frame_directory', self._sample_frames(frames, max_frames=max_frames), cleanup_files, cleanup_dirs, 'directory'
        if path.is_file() and path.suffix.lower() == '.json':
            payload = json.loads(path.read_text(encoding='utf-8'))
            frames = self._frames_from_manifest(payload)
            return 'json_manifest', self._sample_frames(frames, max_frames=max_frames), cleanup_files, cleanup_dirs, 'manifest'
        if path.is_file() and path.suffix.lower() == '.gif':
            frames, extracted, temp_dir, backend = self._extract_gif_frames(path, max_frames=max_frames, support=support)
            cleanup_files.extend(extracted)
            if temp_dir:
                cleanup_dirs.append(temp_dir)
            return 'gif', frames, cleanup_files, cleanup_dirs, backend
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            frames, extracted, temp_dir, backend = self._extract_video_frames(path, max_frames=max_frames, support=support)
            cleanup_files.extend(extracted)
            if temp_dir:
                cleanup_dirs.append(temp_dir)
            return 'video_file', frames, cleanup_files, cleanup_dirs, backend
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return 'single_image', [{'path': str(path), 'label': path.name}], cleanup_files, cleanup_dirs, 'single_image'
        raise FileNotFoundError(f'Unsupported visual input: {input_path}')

    @staticmethod
    def _frames_from_manifest(payload: Any) -> list[dict[str, Any]]:
        rows = payload.get('frames', []) if isinstance(payload, dict) else payload
        frames: list[dict[str, Any]] = []
        if not isinstance(rows, list):
            return frames
        for index, row in enumerate(rows):
            if isinstance(row, str):
                frames.append({'path': row, 'label': Path(row).name or f'frame_{index + 1}'})
                continue
            if not isinstance(row, dict):
                continue
            label = str(row.get('label') or row.get('image_path') or row.get('path') or f'frame_{index + 1}')
            if row.get('visual_input') is not None:
                frames.append({'visual_input': row.get('visual_input'), 'label': label})
                continue
            candidate_path = str(row.get('image_path') or row.get('path') or '').strip()
            if candidate_path:
                frames.append({'path': candidate_path, 'label': label})
        return frames

    @staticmethod
    def _sample_frames(frames: list[dict[str, Any]], *, max_frames: int) -> list[dict[str, Any]]:
        if len(frames) <= max_frames:
            return frames
        indices = sorted({round(index * (len(frames) - 1) / max(1, max_frames - 1)) for index in range(max_frames)})
        return [frames[index] for index in indices]

    def _extract_gif_frames(
        self,
        path: Path,
        *,
        max_frames: int,
        support: dict[str, bool],
    ) -> tuple[list[dict[str, Any]], list[str], str, str]:
        if support.get('pillow'):
            from PIL import Image, ImageSequence
            image = Image.open(path)
            temp_dir = Path(tempfile.mkdtemp(prefix='semop_gif_frames_'))
            extracted: list[str] = []
            frames: list[dict[str, Any]] = []
            for index, frame in enumerate(ImageSequence.Iterator(image)):
                frame_path = temp_dir / f'frame_{index:03d}.png'
                frame.convert('RGB').save(frame_path)
                extracted.append(str(frame_path))
                frames.append({'path': str(frame_path), 'label': frame_path.name})
            return self._sample_frames(frames, max_frames=max_frames), extracted, str(temp_dir), 'pillow'
        if support.get('imageio'):
            return self._extract_video_with_imageio(path, max_frames=max_frames, backend='imageio_gif')
        if support.get('ffmpeg'):
            return self._extract_video_with_ffmpeg(path, max_frames=max_frames, backend='ffmpeg_gif')
        raise RuntimeError('GIF extraction requires Pillow, imageio, or ffmpeg.')

    def _extract_video_frames(
        self,
        path: Path,
        *,
        max_frames: int,
        support: dict[str, bool],
    ) -> tuple[list[dict[str, Any]], list[str], str, str]:
        if support.get('imageio'):
            frames, extracted, temp_dir, backend = self._extract_video_with_imageio(path, max_frames=max_frames, backend='imageio')
            if frames:
                return frames, extracted, temp_dir, backend
        if support.get('opencv'):
            frames, extracted, temp_dir, backend = self._extract_video_with_opencv(path, max_frames=max_frames)
            if frames:
                return frames, extracted, temp_dir, backend
        if support.get('ffmpeg'):
            frames, extracted, temp_dir, backend = self._extract_video_with_ffmpeg(path, max_frames=max_frames, backend='ffmpeg')
            if frames:
                return frames, extracted, temp_dir, backend
        raise RuntimeError('Video extraction requires imageio, OpenCV, or ffmpeg. Frame folders and JSON manifests remain supported without extra dependencies.')

    @staticmethod
    def _extract_video_with_imageio(path: Path, *, max_frames: int, backend: str) -> tuple[list[dict[str, Any]], list[str], str, str]:
        import imageio.v3 as iio
        temp_dir = Path(tempfile.mkdtemp(prefix='semop_video_frames_'))
        extracted: list[str] = []
        frames: list[dict[str, Any]] = []
        total = 0
        try:
            meta = iio.immeta(path)
            total = int(meta.get('nframes') or 0)
        except Exception:
            total = 0
        if total > 0:
            indices = sorted({round(index * (total - 1) / max(1, max_frames - 1)) for index in range(max_frames)})
            raw_frames = [iio.imread(path, index=index) for index in indices]
        else:
            raw_frames = []
            for _, frame in enumerate(iio.imiter(path)):
                raw_frames.append(frame)
                if len(raw_frames) >= max_frames:
                    break
        for idx, array in enumerate(raw_frames):
            frame_path = temp_dir / f'frame_{idx:03d}.png'
            iio.imwrite(frame_path, array)
            extracted.append(str(frame_path))
            frames.append({'path': str(frame_path), 'label': frame_path.name})
        return frames, extracted, str(temp_dir), backend

    @staticmethod
    def _extract_video_with_opencv(path: Path, *, max_frames: int) -> tuple[list[dict[str, Any]], list[str], str, str]:
        import cv2
        temp_dir = Path(tempfile.mkdtemp(prefix='semop_video_frames_'))
        extracted: list[str] = []
        frames: list[dict[str, Any]] = []
        capture = cv2.VideoCapture(str(path))
        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = sorted({round(index * max(0, frame_total - 1) / max(1, max_frames - 1)) for index in range(max_frames)}) if frame_total > 0 else []
        frame_lookup = set(indices)
        current_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_total <= 0 or current_index in frame_lookup:
                frame_path = temp_dir / f'frame_{len(frames):03d}.png'
                cv2.imwrite(str(frame_path), frame)
                extracted.append(str(frame_path))
                frames.append({'path': str(frame_path), 'label': frame_path.name})
                if frame_total <= 0 and len(frames) >= max_frames:
                    break
            current_index += 1
        capture.release()
        return frames, extracted, str(temp_dir), 'opencv'

    @staticmethod
    def _extract_video_with_ffmpeg(path: Path, *, max_frames: int, backend: str) -> tuple[list[dict[str, Any]], list[str], str, str]:
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            return [], [], '', backend
        temp_dir = Path(tempfile.mkdtemp(prefix='semop_video_frames_'))
        pattern = temp_dir / 'frame_%03d.png'
        command = [
            ffmpeg,
            '-hide_banner',
            '-loglevel', 'error',
            '-i', str(path),
            '-vf', "select='not(mod(n,10))'",
            '-vsync', 'vfr',
            '-frames:v', str(max_frames),
            str(pattern),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return [], [], '', backend
        extracted = [str(candidate) for candidate in sorted(temp_dir.glob('frame_*.png'))]
        frames = [{'path': item, 'label': Path(item).name} for item in extracted]
        return frames, extracted, str(temp_dir), backend

    @staticmethod
    def _changed_entities(frame_entity_sets: list[set[str]], stable_entities: list[str]) -> list[str]:
        changing: set[str] = set()
        stable_set = set(stable_entities)
        previous: set[str] | None = None
        for entities in frame_entity_sets:
            if previous is not None:
                changing.update(item for item in entities.symmetric_difference(previous) if item not in stable_set)
            previous = entities
        return sorted(changing)

    @staticmethod
    def _fallback_situation_summary(
        frames: list[FrameSituationSummary],
        stable_entities: list[str],
        changed_entities: list[str],
        temporal_events: list[str],
        warnings: list[str],
    ) -> str:
        parts = [f'Observed {len(frames)} frame(s) and aggregated the shared scene structure.']
        if stable_entities:
            parts.append('Stable entities: ' + ', '.join(stable_entities[:5]) + '.')
        if changed_entities:
            parts.append('Changing entities: ' + ', '.join(changed_entities[:5]) + '.')
        if temporal_events:
            parts.append('Temporal events: ' + ' | '.join(temporal_events[:3]))
        if warnings:
            parts.append('Warnings: ' + ' | '.join(warnings[:2]))
        return ' '.join(parts)

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            normalized = str(item or '').strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered
