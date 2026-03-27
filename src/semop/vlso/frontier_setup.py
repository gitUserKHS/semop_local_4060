from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from ..hardware_profiles import detect_local_hardware
from .frontier_vlm import FRONTIER_VISION_SPECS, FrontierVisionSpec
from .vision_backbones import DEFAULT_VISION_MODEL_ROOT


@dataclass
class FrontierBundleItem:
    family: str
    repo_id: str
    target_dir: str
    required: bool
    size_note: str
    installed: bool = False

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrontierSetupSummary:
    target_root: str
    detected_profile: str
    recommended_bundle: list[FrontierBundleItem] = field(default_factory=list)
    installed_count: int = 0
    ready_count: int = 0
    recommended_count: int = 0
    required_ready: bool = False
    missing_required: list[str] = field(default_factory=list)
    installed_families: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'target_root': self.target_root,
            'detected_profile': self.detected_profile,
            'recommended_bundle': [item.model_dump() for item in self.recommended_bundle],
            'installed_count': self.installed_count,
            'ready_count': self.ready_count,
            'recommended_count': self.recommended_count,
            'required_ready': self.required_ready,
            'missing_required': list(self.missing_required),
            'installed_families': list(self.installed_families),
            'notes': list(self.notes),
        }


@dataclass
class FrontierInstallSummary:
    target_root: str
    completed: list[FrontierBundleItem] = field(default_factory=list)
    skipped: list[FrontierBundleItem] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            'target_root': self.target_root,
            'completed': [item.model_dump() for item in self.completed],
            'skipped': [item.model_dump() for item in self.skipped],
            'failed': list(self.failed),
            'notes': list(self.notes),
        }


class FrontierVisionInstaller:
    def __init__(self, target_root: str | Path | None = None, hardware_profile: str = 'auto') -> None:
        self.target_root = Path(target_root) if target_root else DEFAULT_VISION_MODEL_ROOT / 'frontier'
        self.target_root.mkdir(parents=True, exist_ok=True)
        self.hardware = detect_local_hardware(hardware_profile)

    def recommended_bundle(self) -> FrontierSetupSummary:
        items = [
            self._bundle_item('qwen2_5_vl', required=True, size_note='primary high-quality scene reasoning lane for 4060 8GB'),
            self._bundle_item('florence2', required=True, size_note='fast promptable caption/grounding fallback and verifier lane'),
            self._bundle_item('molmo2', required=False, size_note='optional heavier open multimodal lane for richer scene description'),
        ]
        installed_count = sum(1 for item in items if item.installed)
        missing_required = [item.family for item in items if item.required and not item.installed]
        installed_families = [item.family for item in items if item.installed]
        notes = [
            'For RTX 4060 8GB, Qwen2.5-VL-3B and Florence-2 are the main recommended bundle.',
            'Molmo is optional because it is heavier and may be slower locally.',
        ]
        return FrontierSetupSummary(
            target_root=str(self.target_root),
            detected_profile=self.hardware.detected_profile,
            recommended_bundle=items,
            installed_count=installed_count,
            ready_count=installed_count,
            recommended_count=len(items),
            required_ready=not missing_required,
            missing_required=missing_required,
            installed_families=installed_families,
            notes=notes,
        )

    def install_recommended_bundle(self, include_optional: bool = False, force: bool = False) -> FrontierInstallSummary:
        plan = self.recommended_bundle()
        completed: list[FrontierBundleItem] = []
        skipped: list[FrontierBundleItem] = []
        failed: list[str] = []
        for item in plan.recommended_bundle:
            if not include_optional and not item.required:
                skipped.append(item)
                continue
            target = Path(item.target_dir)
            if not force and self._is_installed(target):
                existing = FrontierBundleItem(**item.model_dump())
                existing.installed = True
                skipped.append(existing)
                continue
            try:
                snapshot_download(
                    repo_id=item.repo_id,
                    local_dir=str(target),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                finished = FrontierBundleItem(**item.model_dump())
                finished.installed = True
                completed.append(finished)
            except Exception as exc:
                failed.append(f'{item.repo_id}: {exc}')
        notes = [
            'Installed checkpoints are discovered automatically by the frontier VLM lane.',
            'Restart the GUI after installation if it is already running.',
        ]
        return FrontierInstallSummary(
            target_root=str(self.target_root),
            completed=completed,
            skipped=skipped,
            failed=failed,
            notes=notes,
        )

    def installed_bundle_status(self) -> FrontierSetupSummary:
        return self.recommended_bundle()

    def _bundle_item(self, family: str, *, required: bool, size_note: str) -> FrontierBundleItem:
        spec = FRONTIER_VISION_SPECS[family]
        target = self.target_root / self._safe_name(spec.model_id)
        installed = self._is_installed(target)
        return FrontierBundleItem(
            family=spec.family,
            repo_id=spec.model_id,
            target_dir=str(target),
            required=required,
            size_note=size_note,
            installed=installed,
        )

    @staticmethod
    def _safe_name(model_id: str) -> str:
        return model_id.replace('/', '--').replace(':', '-').replace(' ', '_')

    @staticmethod
    def _is_installed(path: Path) -> bool:
        return path.exists() and any(path.glob('*.json'))
