from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .leworldmodel_adapter import LeWorldModelAdapter
from .leworldmodel_planner import LeWorldModelTrajectoryPlanner
from .vlso import GeometryPipelineSummary, SharedWorldModel, SyntheticGeometrySceneBuilder, VLSOReasoner, VisualGeometryBootstrapPipeline


@dataclass
class ScenePrimitive3D:
    primitive_id: str
    label: str
    shape: str
    x: float
    y: float
    z: float
    width: float
    height: float
    depth: float
    color: str
    source_entity: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneRelation3D:
    source: str
    relation: str
    target: str
    confidence: float = 1.0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scene3DReconstruction:
    query: str
    visual_input: str
    answer_text: str
    primitives: list[ScenePrimitive3D] = field(default_factory=list)
    relations: list[SceneRelation3D] = field(default_factory=list)
    inferred_steps: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audit_trace: list[str] = field(default_factory=list)
    topology_summary: dict[str, Any] = field(default_factory=dict)
    mesh_stats: dict[str, Any] = field(default_factory=dict)
    export_paths: dict[str, str] = field(default_factory=dict)
    leworldmodel_alignment: dict[str, Any] = field(default_factory=dict)
    leworldmodel_plan: dict[str, Any] = field(default_factory=dict)
    raw_world_model: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            'query': self.query,
            'visual_input': self.visual_input,
            'answer_text': self.answer_text,
            'primitives': [item.model_dump() for item in self.primitives],
            'relations': [item.model_dump() for item in self.relations],
            'inferred_steps': list(self.inferred_steps),
            'constraints': list(self.constraints),
            'warnings': list(self.warnings),
            'audit_trace': list(self.audit_trace),
            'topology_summary': dict(self.topology_summary),
            'mesh_stats': dict(self.mesh_stats),
            'export_paths': dict(self.export_paths),
            'leworldmodel_alignment': dict(self.leworldmodel_alignment),
            'leworldmodel_plan': dict(self.leworldmodel_plan),
            'raw_world_model': dict(self.raw_world_model),
        }


@dataclass
class VisualGeometryCollectionSummary:
    output_dir: str
    scene_dir: str
    eval_path: str
    scene_count: int
    scene_files: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualGeometryDatasetSummary:
    output_dir: str
    manifest_path: str
    input_count: int
    input_paths: list[str] = field(default_factory=list)
    source_kind: str = 'custom'

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualGeometryBatchReconstructionSummary:
    query: str
    output_dir: str
    input_count: int
    reconstructed_count: int
    result_paths: list[str] = field(default_factory=list)
    obj_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class VisualGeometry3DWorkbench:
    def __init__(
        self,
        *,
        concept_store_path: str | None = None,
        operator_store_path: str | None = None,
        affordance_weights_path: str | None = None,
        mode: str = 'heuristic',
        answer_mode: str = 'structured',
    ) -> None:
        self.concept_store_path = concept_store_path
        self.operator_store_path = operator_store_path
        self.affordance_weights_path = affordance_weights_path
        self.mode = mode
        self.answer_mode = answer_mode
        self.lewm = LeWorldModelAdapter()
        self.lewm_planner = LeWorldModelTrajectoryPlanner(adapter=self.lewm)

    def collect_starter_scenes(self, output_dir: str) -> VisualGeometryCollectionSummary:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        scene_dir = root / 'starter_scenes'
        eval_path = root / 'starter_geometry_eval.jsonl'
        builder = SyntheticGeometrySceneBuilder()
        scenes = builder.build(scene_dir)
        builder.write_eval_jsonl(scenes, eval_path)
        scene_files = [scene.visual_json_path for scene in scenes] + [scene.image_path for scene in scenes]
        summary = VisualGeometryCollectionSummary(
            output_dir=str(root),
            scene_dir=str(scene_dir),
            eval_path=str(eval_path),
            scene_count=len(scenes),
            scene_files=scene_files,
        )
        (root / 'visual_geometry_collection_summary.json').write_text(
            json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return summary

    def collect_input_manifest(self, inputs: str | Path | Iterable[str | Path], output_dir: str) -> VisualGeometryDatasetSummary:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        pipeline = VisualGeometryBootstrapPipeline(weights_path=self.affordance_weights_path)
        normalized = pipeline.collect_input_paths(self._iter_inputs(inputs))
        manifest_path = root / 'visual_geometry_input_manifest.json'
        source_kind = self._source_kind(inputs)
        summary = VisualGeometryDatasetSummary(
            output_dir=str(root),
            manifest_path=str(manifest_path),
            input_count=len(normalized),
            input_paths=normalized,
            source_kind=source_kind,
        )
        manifest_path.write_text(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        return summary

    def train_starter_bundle(self, output_dir: str, *, limit_scenes: int | None = None) -> GeometryPipelineSummary:
        collection = self.collect_starter_scenes(output_dir)
        return self.train_bundle_from_inputs(collection.scene_dir, output_dir, limit_scenes=limit_scenes, eval_input=collection.eval_path)

    def train_bundle_from_inputs(
        self,
        inputs: str | Path | Iterable[str | Path],
        output_dir: str,
        *,
        limit_scenes: int | None = None,
        eval_input: str | None = None,
    ) -> GeometryPipelineSummary:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        pipeline = VisualGeometryBootstrapPipeline(weights_path=self.affordance_weights_path)
        normalized = pipeline.collect_input_paths(self._iter_inputs(inputs))
        if limit_scenes is not None:
            normalized = normalized[:max(1, int(limit_scenes))]
        if not normalized:
            raise ValueError('No visual inputs were found for visual geometry training.')
        manifest_path = root / 'visual_geometry_input_manifest.json'
        candidates_path = root / 'geometry_candidates.jsonl'
        pseudo_labels_path = root / 'geometry_pseudo_labels.jsonl'
        manifest_path.write_text(
            json.dumps(
                {
                    'output_dir': str(root),
                    'manifest_path': str(manifest_path),
                    'input_count': len(normalized),
                    'input_paths': normalized,
                    'source_kind': self._source_kind(inputs),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        summary = pipeline.run(
            inputs=normalized,
            candidates_path=candidates_path,
            pseudo_labels_path=pseudo_labels_path,
            concept_store_path=root / 'visual_geometry_concepts.db',
            operator_store_path=root / 'visual_geometry_operators.db',
            eval_input=eval_input,
            eval_mode=self.mode,
            answer_mode=self.answer_mode,
            concept_summary_output=root / 'geometry_concept_summary.json',
        )
        self._fit_leworldmodel_visual_prior(
            candidates_path=candidates_path,
            pseudo_labels_path=pseudo_labels_path,
            output_path=self._lewm_artifact_path(root),
            input_count=len(normalized),
        )
        return summary

    def reconstruct_scene(self, query: str, visual_input: str, output_dir: str) -> Scene3DReconstruction:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        reasoner = VLSOReasoner(
            mode='deep',
            answer_mode='structured',
            concept_store_path=self._resolve_concept_store(output_dir),
            operator_store_path=self._resolve_operator_store(output_dir),
            affordance_weights_path=self.affordance_weights_path,
        )
        world, answer = reasoner.answer(query, visual_input=visual_input)
        source_payload = self._load_source_payload(visual_input)
        primitives = self._build_primitives(world, source_payload)
        relations = [
            SceneRelation3D(source=item.source, relation=item.relation, target=item.target, confidence=float(item.confidence))
            for item in world.relations
        ]
        topology_summary = self._topology_summary(primitives, relations, world, source_payload)
        lewm_alignment = self._score_reconstruction_alignment(root, query, primitives, relations, world)
        lewm_plan = self._plan_reconstruction_trajectory(root, query, primitives, relations, world)
        if lewm_alignment:
            topology_summary['leworldmodel_alignment_score'] = lewm_alignment.get('alignment_score', 0.0)
            if lewm_alignment.get('alignment_score', 0.0) < 0.42:
                world.warnings.append('lewm latent alignment is low for this reconstruction')
                world.audit_trace.append('lewm latent prior suggests the scene is weakly aligned with trained geometry trajectories')
            else:
                world.audit_trace.append(f"lewm latent alignment: {lewm_alignment.get('alignment_score', 0.0):.3f}")
        if lewm_plan:
            topology_summary['leworldmodel_plan_score'] = lewm_plan.get('plan_score', 0.0)
            world.audit_trace.append('lewm latent planner proposed a reconstruction order from the learned geometry prior')
        export_paths, mesh_stats = self._export_scene_bundle(primitives, root)
        reconstruction = Scene3DReconstruction(
            query=query,
            visual_input=visual_input,
            answer_text=answer.answer_text,
            primitives=primitives,
            relations=relations,
            inferred_steps=list(world.inferred_steps),
            constraints=list(world.constraints),
            warnings=list(world.warnings),
            audit_trace=list(world.audit_trace),
            topology_summary=topology_summary,
            mesh_stats=mesh_stats,
            export_paths=export_paths,
            leworldmodel_alignment=lewm_alignment,
            leworldmodel_plan=lewm_plan,
            raw_world_model=world.model_dump(),
        )
        (root / 'scene_3d_reconstruction.json').write_text(
            json.dumps(reconstruction.model_dump(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return reconstruction

    def reconstruct_batch(
        self,
        query: str,
        inputs: str | Path | Iterable[str | Path],
        output_dir: str,
        *,
        limit: int | None = None,
    ) -> VisualGeometryBatchReconstructionSummary:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        pipeline = VisualGeometryBootstrapPipeline(weights_path=self.affordance_weights_path)
        normalized = pipeline.collect_input_paths(self._iter_inputs(inputs))
        if limit is not None:
            normalized = normalized[:max(1, int(limit))]
        if not normalized:
            raise ValueError('No visual inputs were found for batch 3D reconstruction.')
        result_paths: list[str] = []
        obj_paths: list[str] = []
        warnings: list[str] = []
        batch_dir = root / 'batch_reconstructions'
        batch_dir.mkdir(parents=True, exist_ok=True)
        for index, visual_input in enumerate(normalized):
            item_dir = batch_dir / f'item_{index:03d}'
            reconstruction = self.reconstruct_scene(query, visual_input, str(item_dir))
            result_paths.append(str(item_dir / 'scene_3d_reconstruction.json'))
            obj_path = reconstruction.export_paths.get('obj_path')
            if obj_path:
                obj_paths.append(obj_path)
            if reconstruction.warnings:
                warnings.append(f"{visual_input}: {' | '.join(reconstruction.warnings[:3])}")
        summary = VisualGeometryBatchReconstructionSummary(
            query=query,
            output_dir=str(root),
            input_count=len(normalized),
            reconstructed_count=len(result_paths),
            result_paths=result_paths,
            obj_paths=obj_paths,
            warnings=warnings,
        )
        (root / 'batch_reconstruction_summary.json').write_text(
            json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return summary


    @staticmethod
    def _lewm_artifact_path(output_dir: Path | str) -> str:
        return str(Path(output_dir) / 'lewm_visual_prior.json')

    @staticmethod
    def _lewm_report_path(output_dir: Path | str) -> str:
        return str(Path(output_dir) / 'lewm_visual_training_report.json')

    def _fit_leworldmodel_visual_prior(
        self,
        *,
        candidates_path: Path,
        pseudo_labels_path: Path,
        output_path: str,
        input_count: int,
    ) -> None:
        pseudo_by_image: dict[str, list[dict[str, Any]]] = {}
        if pseudo_labels_path.exists():
            for raw in pseudo_labels_path.read_text(encoding='utf-8').splitlines():
                line = raw.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                pseudo_by_image[str(payload.get('image_path') or '')] = [item for item in payload.get('targets', []) if isinstance(item, dict)]
        records = []
        if candidates_path.exists():
            for index, raw in enumerate(candidates_path.read_text(encoding='utf-8').splitlines()):
                line = raw.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                image_path = str(payload.get('image_path') or f'image_{index:03d}')
                targets = [item for item in payload.get('targets', []) if isinstance(item, dict)]
                pseudo_targets = pseudo_by_image.get(image_path, [])
                step_items = [
                    [Path(image_path).suffix.lower().lstrip('.'), Path(image_path).stem, str(payload.get('source_kind') or '')],
                    [
                        str(target.get('shape_hint') or '')
                        for target in targets
                    ] + [
                        str(target.get('parent') or '')
                        for target in targets
                    ],
                    [
                        str(label)
                        for target in targets
                        for label in ((target.get('suggested_labels') or [])[:4])
                    ],
                    [
                        str(label)
                        for target in pseudo_targets
                        for label in (target.get('positive_labels') or [])
                    ],
                ]
                record = self.lewm.make_record(
                    sequence_id=f'visual_training_{index:03d}',
                    domain='visual_geometry_3d',
                    step_items=step_items,
                    metadata={'image_path': image_path},
                )
                if record.steps:
                    records.append(record)
        if not records:
            return
        artifact = self.lewm.fit(
            records,
            output_path=output_path,
            domain='visual_geometry_3d',
            source_summary={'input_count': input_count, 'record_count': len(records)},
        )
        Path(self._lewm_report_path(Path(output_path).parent)).write_text(
            json.dumps(
                {
                    'artifact_path': artifact.artifact_path,
                    'example_count': artifact.example_count,
                    'step_count': artifact.step_count,
                    'sigreg_score': artifact.sigreg_score,
                    'predictive_consistency': artifact.predictive_consistency,
                    'straightness_score': artifact.straightness_score,
                    'top_tokens': list(artifact.top_tokens[:12]),
                    'source_summary': dict(artifact.source_summary),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )

    def _score_reconstruction_alignment(
        self,
        output_root: Path,
        query: str,
        primitives: list[ScenePrimitive3D],
        relations: list[SceneRelation3D],
        world: SharedWorldModel,
    ) -> dict[str, Any]:
        artifact = self.lewm.load(self._lewm_artifact_path(output_root))
        if artifact is None:
            return {}
        step_items = [
            [query],
            [item.label for item in primitives] + [item.shape for item in primitives],
            [relation.relation for relation in relations] + list(world.constraints) + list(world.goals),
        ]
        record = self.lewm.make_record('reconstruction_probe', 'visual_geometry_3d', step_items)
        alignment = self.lewm.score_sequence(artifact, record.steps)
        return alignment.model_dump()

    def _plan_reconstruction_trajectory(
        self,
        output_root: Path,
        query: str,
        primitives: list[ScenePrimitive3D],
        relations: list[SceneRelation3D],
        world: SharedWorldModel,
    ) -> dict[str, Any]:
        artifact = self.lewm.load(self._lewm_artifact_path(output_root))
        if artifact is None:
            return {}
        candidate_actions = list(dict.fromkeys(
            [item.shape for item in primitives[:6]]
            + [item.label for item in primitives[:6]]
            + [relation.relation for relation in relations[:6]]
            + list(world.constraints[:4])
        ))
        initial_steps = [
            [query],
            [item.label for item in primitives[:4]],
            [relation.relation for relation in relations[:4]] + list(world.constraints[:4]),
        ]
        goal_tokens = [item.shape for item in primitives[:4]] + list(world.goals[:4])
        plan = self.lewm_planner.plan(
            artifact,
            domain='visual_geometry_3d',
            initial_steps=initial_steps,
            candidate_actions=candidate_actions,
            goal_tokens=goal_tokens,
            horizon=min(4, max(2, len(candidate_actions) // 2 or 2)),
            samples=20,
            elites=5,
            iterations=4,
            seed=len(primitives) + len(relations) + len(query),
        )
        return plan.model_dump()

    def _resolve_concept_store(self, output_dir: str) -> str | None:
        trained = Path(output_dir) / 'visual_geometry_concepts.db'
        if trained.exists():
            return str(trained)
        return self.concept_store_path

    def _resolve_operator_store(self, output_dir: str) -> str | None:
        trained = Path(output_dir) / 'visual_geometry_operators.db'
        if trained.exists():
            return str(trained)
        return self.operator_store_path

    @staticmethod
    def _iter_inputs(inputs: str | Path | Iterable[str | Path]) -> list[str | Path]:
        if isinstance(inputs, (str, Path)):
            return [inputs]
        return list(inputs)

    @staticmethod
    def _source_kind(inputs: str | Path | Iterable[str | Path]) -> str:
        if isinstance(inputs, (str, Path)):
            path = Path(inputs)
            if path.is_dir():
                return 'directory'
            if path.is_file():
                return 'file'
            return 'path'
        return 'list'

    @staticmethod
    def _load_source_payload(visual_input: str) -> dict[str, Any]:
        if not visual_input or not visual_input.lower().endswith('.json') or not os.path.exists(visual_input):
            return {}
        try:
            with open(visual_input, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _build_primitives(self, world: SharedWorldModel, source_payload: dict[str, Any]) -> list[ScenePrimitive3D]:
        source_objects = {str(item.get('id')): item for item in source_payload.get('objects', []) if isinstance(item, dict)}
        primitives: list[ScenePrimitive3D] = []
        for index, entity in enumerate(world.entities):
            attributes = dict(entity.attributes)
            if entity.id in source_objects:
                for key, value in source_objects[entity.id].items():
                    attributes.setdefault(key, value)
            bbox = attributes.get('bbox')
            if not (isinstance(bbox, list) and len(bbox) == 4):
                bbox = [index * 3.0, index * 2.0, index * 3.0 + 2.0, index * 2.0 + 2.0]
            x1, y1, x2, y2 = [float(value) for value in bbox]
            width = max(0.8, x2 - x1)
            height = max(0.8, y2 - y1)
            depth = max(0.8, min(width, height) * 0.7)
            z_layer = float(index % 5) * 1.8
            shape = self._shape_name(entity, attributes)
            primitives.append(
                ScenePrimitive3D(
                    primitive_id=f'primitive_{index:03d}_{entity.id}',
                    label=entity.label or entity.id,
                    shape=shape,
                    x=round(x1 + width / 2.0, 3),
                    y=round(-(y1 + height / 2.0), 3),
                    z=round(z_layer, 3),
                    width=round(width, 3),
                    height=round(height, 3),
                    depth=round(depth, 3),
                    color=self._color_for(entity.id or entity.label or f'item_{index}'),
                    source_entity=entity.id,
                    metadata={
                        'entity_type': entity.entity_type,
                        'bbox': bbox,
                        'modality': entity.modality,
                        'shape_hint': attributes.get('shape_hint'),
                        'polygon': attributes.get('polygon'),
                    },
                )
            )
        return primitives

    def _topology_summary(
        self,
        primitives: list[ScenePrimitive3D],
        relations: list[SceneRelation3D],
        world: SharedWorldModel,
        source_payload: dict[str, Any],
    ) -> dict[str, Any]:
        shape_counts = Counter(item.shape for item in primitives)
        relation_counts = Counter(item.relation for item in relations)
        adjacency: dict[str, set[str]] = {item.source_entity: set() for item in primitives}
        for relation in relations:
            adjacency.setdefault(relation.source, set()).add(relation.target)
            adjacency.setdefault(relation.target, set()).add(relation.source)
        components = 0
        seen: set[str] = set()
        for node in adjacency:
            if node in seen:
                continue
            components += 1
            stack = [node]
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(neighbor for neighbor in adjacency.get(current, set()) if neighbor not in seen)
        relation_density = round(len(relations) / max(1, len(primitives)), 3)
        geometry_items = source_payload.get('geometry', []) if isinstance(source_payload.get('geometry'), list) else []
        return {
            'primitive_count': len(primitives),
            'relation_count': len(relations),
            'shape_counts': dict(shape_counts),
            'relation_families': dict(relation_counts),
            'connected_components': components,
            'relation_density': relation_density,
            'source_object_count': len(source_payload.get('objects', [])) if isinstance(source_payload.get('objects'), list) else 0,
            'source_geometry_count': len(geometry_items),
            'constraint_count': len(world.constraints),
            'operator_count': len(world.operators),
        }

    def _export_scene_bundle(self, primitives: list[ScenePrimitive3D], output_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
        bundle_dir = output_root / 'scene_3d_bundle'
        bundle_dir.mkdir(parents=True, exist_ok=True)
        obj_path = bundle_dir / 'scene_3d_reconstruction.obj'
        mtl_path = bundle_dir / 'scene_3d_reconstruction.mtl'
        manifest_path = bundle_dir / 'scene_3d_bundle_manifest.json'
        material_map = {item.color: f'mat_{index:03d}' for index, item in enumerate({primitive.color: primitive for primitive in primitives}.values())}
        mtl_lines: list[str] = []
        for color, material_name in material_map.items():
            red, green, blue = self._rgb_tuple(color)
            mtl_lines.extend([
                f'newmtl {material_name}',
                f'Kd {red:.4f} {green:.4f} {blue:.4f}',
                'Ka 0.2000 0.2000 0.2000',
                'Ks 0.0500 0.0500 0.0500',
                'Ns 8.0000',
                '',
            ])
        mtl_path.write_text('\n'.join(mtl_lines), encoding='utf-8')

        obj_lines = [f'mtllib {mtl_path.name}']
        vertex_count = 0
        face_count = 0
        current_index = 1
        for primitive in primitives:
            vertices, faces = self._mesh_for_primitive(primitive)
            obj_lines.append(f'g {primitive.primitive_id}')
            obj_lines.append(f'usemtl {material_map[primitive.color]}')
            for x, y, z in vertices:
                obj_lines.append(f'v {x:.5f} {y:.5f} {z:.5f}')
            for face in faces:
                shifted = [str(current_index + offset) for offset in face]
                obj_lines.append('f ' + ' '.join(shifted))
            current_index += len(vertices)
            vertex_count += len(vertices)
            face_count += len(faces)
            obj_lines.append('')
        obj_path.write_text('\n'.join(obj_lines), encoding='utf-8')

        mesh_stats = {
            'primitive_count': len(primitives),
            'vertex_count': vertex_count,
            'face_count': face_count,
        }
        export_paths = {
            'bundle_dir': str(bundle_dir),
            'obj_path': str(obj_path),
            'mtl_path': str(mtl_path),
            'manifest_path': str(manifest_path),
        }
        manifest_path.write_text(
            json.dumps({'export_paths': export_paths, 'mesh_stats': mesh_stats}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return export_paths, mesh_stats

    def _mesh_for_primitive(self, primitive: ScenePrimitive3D) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
        if primitive.shape == 'triangle_prism':
            return self._triangle_prism_mesh(primitive)
        if primitive.shape == 'cylinder':
            return self._cylinder_mesh(primitive)
        return self._box_mesh(primitive)

    @staticmethod
    def _box_mesh(primitive: ScenePrimitive3D) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
        x1 = primitive.x - primitive.width / 2.0
        x2 = primitive.x + primitive.width / 2.0
        y1 = primitive.y - primitive.height / 2.0
        y2 = primitive.y + primitive.height / 2.0
        z1 = primitive.z
        z2 = primitive.z + primitive.depth
        vertices = [
            (x1, y1, z1), (x2, y1, z1), (x2, y2, z1), (x1, y2, z1),
            (x1, y1, z2), (x2, y1, z2), (x2, y2, z2), (x1, y2, z2),
        ]
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
        ]
        return vertices, faces

    @staticmethod
    def _triangle_prism_mesh(primitive: ScenePrimitive3D) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
        x_left = primitive.x - primitive.width / 2.0
        x_right = primitive.x + primitive.width / 2.0
        y_base = primitive.y + primitive.height / 2.0
        y_peak = primitive.y - primitive.height / 2.0
        z1 = primitive.z
        z2 = primitive.z + primitive.depth
        vertices = [
            (x_left, y_base, z1), (x_right, y_base, z1), (primitive.x, y_peak, z1),
            (x_left, y_base, z2), (x_right, y_base, z2), (primitive.x, y_peak, z2),
        ]
        faces = [
            [0, 1, 2], [3, 5, 4], [0, 3, 4, 1], [1, 4, 5, 2], [2, 5, 3, 0],
        ]
        return vertices, faces

    @staticmethod
    def _cylinder_mesh(primitive: ScenePrimitive3D, segments: int = 10) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
        radius_x = primitive.width / 2.0
        radius_y = primitive.height / 2.0
        bottom_z = primitive.z
        top_z = primitive.z + primitive.depth
        vertices: list[tuple[float, float, float]] = []
        for layer_z in (bottom_z, top_z):
            for index in range(segments):
                angle = (2.0 * math.pi * index) / segments
                vertices.append(
                    (
                        primitive.x + math.cos(angle) * radius_x,
                        primitive.y + math.sin(angle) * radius_y,
                        layer_z,
                    )
                )
        faces: list[list[int]] = []
        bottom = list(range(segments))
        top = list(range(segments, segments * 2))
        faces.append(bottom)
        faces.append(list(reversed(top)))
        for index in range(segments):
            next_index = (index + 1) % segments
            faces.append([index, next_index, segments + next_index, segments + index])
        return vertices, faces

    @staticmethod
    def _shape_name(entity: Any, attributes: dict[str, Any]) -> str:
        shape_hint = str(attributes.get('shape_hint') or '').lower()
        label = f"{entity.id} {entity.label} {entity.entity_type}".lower()
        polygon = attributes.get('polygon')
        if isinstance(polygon, list) and len(polygon) == 3:
            return 'triangle_prism'
        if 'triangle' in shape_hint or 'triangle' in label:
            return 'triangle_prism'
        if 'circle' in shape_hint or 'circle' in label:
            return 'cylinder'
        if 'line' in label:
            return 'beam'
        return 'box'

    @staticmethod
    def _color_for(seed: str) -> str:
        total = sum(ord(ch) for ch in seed)
        palette = ['#cc5a43', '#3f7f88', '#b38728', '#5d6eb3', '#4b8c57', '#9c5e9d']
        return palette[total % len(palette)]

    @staticmethod
    def _rgb_tuple(color: str) -> tuple[float, float, float]:
        raw = color.replace('#', '').strip()
        if len(raw) == 3:
            raw = ''.join(ch * 2 for ch in raw)
        raw = raw.ljust(6, '0')[:6]
        return tuple(int(raw[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
