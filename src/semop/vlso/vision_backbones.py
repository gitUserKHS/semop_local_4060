from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, List

from .types import VisualObservation

DEFAULT_VISION_MODEL_ROOT = Path(__file__).resolve().parents[3] / "models" / "vision"


@dataclass(frozen=True)
class VisionBackboneSpec:
    model_id: str
    family: str
    description: str
    requires_local_weights: bool = False


VISION_BACKBONE_SPECS = {
    "token_geometry_v1": VisionBackboneSpec(
        model_id="token_geometry_v1",
        family="heuristic",
        description="Deterministic local embedding from object, relation, geometry, and topology tokens.",
        requires_local_weights=False,
    ),
    "dinov2_adapter": VisionBackboneSpec(
        model_id="dinov2_adapter",
        family="dinov2",
        description="Local DINOv2 adapter through transformers when a local checkpoint path is available.",
        requires_local_weights=True,
    ),
    "openclip_adapter": VisionBackboneSpec(
        model_id="openclip_adapter",
        family="open_clip",
        description="Local OpenCLIP or CLIP-like adapter through transformers when a local checkpoint path is available.",
        requires_local_weights=True,
    ),
}


def resolve_local_vision_model_path(model_id: str, search_root: str | Path | None = None) -> str | None:
    root = Path(search_root) if search_root else DEFAULT_VISION_MODEL_ROOT
    if not root.exists():
        return None
    if model_id == "dinov2_adapter":
        family_root = root / "dinov2"
    elif model_id == "openclip_adapter":
        family_root = root / "openclip"
    else:
        return None
    if not family_root.exists():
        return None
    candidates = []
    for path in family_root.rglob("config.json"):
        candidates.append(path.parent)
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda item: (len(item.parts), str(item)))
    return str(candidates[0])


class VisionEmbeddingExtractor:
    def __init__(
        self,
        model_id: str = "token_geometry_v1",
        local_model_path: str | None = None,
        dimension: int = 64,
        auto_resolve: bool = True,
        search_root: str | Path | None = None,
    ) -> None:
        self.spec = VISION_BACKBONE_SPECS.get(model_id, VISION_BACKBONE_SPECS["token_geometry_v1"])
        self.search_root = Path(search_root) if search_root else DEFAULT_VISION_MODEL_ROOT
        self.local_model_path = local_model_path or (resolve_local_vision_model_path(self.spec.model_id, self.search_root) if auto_resolve else None)
        self.dimension = dimension
        self._runtime_backend = "token_geometry_v1"
        self._model = None
        self._processor = None
        self._load_error = ""
        self._auto_resolved = bool(not local_model_path and self.local_model_path)

    def backend_summary(self) -> dict:
        return {
            "model_id": self.spec.model_id,
            "family": self.spec.family,
            "requires_local_weights": self.spec.requires_local_weights,
            "local_model_path": self.local_model_path or "",
            "search_root": str(self.search_root),
            "auto_resolved": self._auto_resolved,
            "backend_ready": (not self.spec.requires_local_weights) or bool(self.local_model_path),
            "active_backend": self._runtime_backend,
            "load_error": self._load_error,
        }

    def embed_observation(self, observation: VisualObservation | dict[str, Any] | str) -> List[float]:
        if isinstance(observation, str):
            observation = json.loads(observation) if observation.strip().startswith("{") else {"constraints": [observation]}
        if isinstance(observation, dict):
            maybe_image = observation.get("image_path") or observation.get("metadata", {}).get("image_path")
            if isinstance(maybe_image, str) and maybe_image.strip():
                vector = self._embed_image_path(maybe_image)
                if vector is not None:
                    return vector
            observation = VisualObservation(
                objects=list(observation.get("objects", [])),
                relations=list(observation.get("relations", [])),
                affordances=list(observation.get("affordances", [])),
                states=list(observation.get("states", [])),
                geometry=list(observation.get("geometry", [])),
                constraints=list(observation.get("constraints", [])),
                metadata=dict(observation.get("metadata", {})),
            )
        maybe_image = observation.metadata.get("image_path") if isinstance(observation, VisualObservation) else None
        if isinstance(maybe_image, str) and maybe_image.strip():
            vector = self._embed_image_path(maybe_image)
            if vector is not None:
                return vector
        self._runtime_backend = "token_geometry_v1"
        return self._embed_tokens(observation)

    def _embed_image_path(self, image_path: str) -> List[float] | None:
        if not self.local_model_path or not Path(self.local_model_path).exists():
            self._load_error = "local vision model path is missing"
            return None
        try:
            from PIL import Image
            import torch
            from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        except Exception as exc:
            self._load_error = f"vision dependencies unavailable: {exc}"
            return None
        try:
            if self._model is None or self._processor is None:
                if self.spec.model_id == "dinov2_adapter":
                    self._processor = AutoImageProcessor.from_pretrained(self.local_model_path, local_files_only=True)
                    self._model = AutoModel.from_pretrained(self.local_model_path, local_files_only=True)
                else:
                    self._processor = AutoProcessor.from_pretrained(self.local_model_path, local_files_only=True)
                    self._model = AutoModel.from_pretrained(self.local_model_path, local_files_only=True)
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
            vector = self._extract_vector(outputs)
            if vector is None:
                self._load_error = "model outputs did not contain a usable embedding"
                return None
            self._runtime_backend = self.spec.model_id
            return vector
        except Exception as exc:
            self._load_error = f"vision embedding failed: {exc}"
            return None

    def _extract_vector(self, outputs: object) -> List[float] | None:
        tensor = None
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            tensor = outputs.image_embeds[0]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            tensor = outputs.pooler_output[0]
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            tensor = outputs.last_hidden_state.mean(dim=1)[0]
        if tensor is None:
            return None
        values = tensor.detach().cpu().float().tolist()
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [round(value / norm, 6) for value in values]

    def _embed_tokens(self, observation: VisualObservation) -> List[float]:
        tokens = list(self._tokens(observation))
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 6) for value in vector]

    def _tokens(self, observation: VisualObservation) -> Iterable[str]:
        for item in observation.objects:
            yield f"obj:{item.get('id') or item.get('label') or item.get('name') or 'unknown'}"
            yield f"kind:{item.get('kind') or item.get('type') or 'object'}"
            bbox = item.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                yield "has_bbox"
            polygon = item.get("polygon")
            if isinstance(polygon, list):
                yield f"polygon_vertices:{len(polygon)}"
        for item in observation.relations:
            if item.get("relation"):
                yield f"rel:{item['relation']}"
        for item in observation.affordances:
            if item.get("value") or item.get("affordance"):
                yield f"aff:{item.get('value') or item.get('affordance')}"
        for item in observation.states:
            if item.get("value") or item.get("state"):
                yield f"state:{item.get('value') or item.get('state')}"
        for item in observation.geometry:
            if item.get("relation"):
                yield f"geo:{item['relation']}"
            if item.get("points"):
                yield f"segment_points:{len(item['points'])}"
        for item in observation.constraints:
            yield f"constraint:{item}"
