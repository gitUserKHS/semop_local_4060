from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

from .prompt_api import ResourceTier
from .semantic_models import (
    FLORENCE_SIDECAR,
    MODEL_SPECS,
    MULTILINGUAL_E5,
    ModelSpec,
    resolve_model_profile,
)


@dataclass(frozen=True)
class DependencyCheck:
    name: str
    available: bool
    version: str = ""
    detail: str = ""


@dataclass(frozen=True)
class ModelCheck:
    model_id: str
    role: str
    cached: bool
    revision: str = ""


@dataclass(frozen=True)
class ModelDoctorReport:
    python_version: str
    python_supported: bool
    python_recommended: bool
    platform: str
    cuda_available: bool
    cuda_device: str
    free_disk_bytes: int
    dependencies: tuple[DependencyCheck, ...]
    models: tuple[ModelCheck, ...]

    @property
    def symbolic_ready(self) -> bool:
        return self.python_supported

    @property
    def semantic_runtime_ready(self) -> bool:
        required = {
            "torch",
            "torchvision",
            "transformers",
            "accelerate",
            "PIL",
        }
        available = {
            item.name for item in self.dependencies if item.available
        }
        return self.python_supported and required.issubset(available)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "symbolic_ready": self.symbolic_ready,
            "semantic_runtime_ready": self.semantic_runtime_ready,
        }


def build_model_doctor_report(
    *,
    workspace: str | Path = ".",
) -> ModelDoctorReport:
    dependencies = tuple(
        _dependency_check(name)
        for name in (
            "numpy",
            "torch",
            "torchvision",
            "transformers",
            "accelerate",
            "sentence_transformers",
            "PIL",
            "timm",
            "einops",
            "safetensors",
        )
    )
    cuda_available = False
    cuda_device = ""
    try:
        import torch
    except ImportError:
        pass
    else:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device = str(torch.cuda.get_device_name(0))
    cached_revisions = _cached_model_revisions()
    models = tuple(
        ModelCheck(
            spec.model_id,
            spec.role,
            spec.model_id in cached_revisions,
            cached_revisions.get(spec.model_id, ""),
        )
        for spec in MODEL_SPECS.values()
    )
    root = Path(workspace).resolve()
    free_disk = _free_disk_bytes(root)
    version = sys.version_info
    return ModelDoctorReport(
        python_version=platform.python_version(),
        python_supported=(version.major, version.minor) >= (3, 11),
        python_recommended=(version.major, version.minor) == (3, 12),
        platform=platform.platform(),
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        free_disk_bytes=free_disk,
        dependencies=dependencies,
        models=models,
    )


def download_profile(
    tier: ResourceTier | str,
    *,
    include_vision_sidecar: bool = False,
    receipt_root: str | Path = "artifacts/models/receipts",
) -> tuple[Path, ...]:
    profile = resolve_model_profile(tier)
    model_ids = [
        item
        for item in (profile.semantic_model_id, profile.retriever_model_id)
        if item
    ]
    if include_vision_sidecar and profile.vision_sidecar_model_id:
        model_ids.append(profile.vision_sidecar_model_id)
    receipts = [
        download_model(model_id, receipt_root=receipt_root)
        for model_id in dict.fromkeys(model_ids)
    ]
    return tuple(receipts)


def download_model(
    model_id: str,
    *,
    receipt_root: str | Path = "artifacts/models/receipts",
) -> Path:
    spec = MODEL_SPECS.get(model_id)
    if spec is None:
        raise ValueError("model download is restricted to the reviewed SemOp manifest")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("model download requires huggingface-hub") from exc
    snapshot = Path(
        snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            allow_patterns=_allow_patterns(spec),
            ignore_patterns=("onnx/**", "openvino/**", "*.onnx"),
        )
    )
    files = tuple(
        sorted(
            path for path in snapshot.rglob("*")
            if path.is_file() and ".cache" not in path.parts
        )
    )
    receipt = {
        "schema_version": "semop.model-download-receipt.v1",
        "model_id": spec.model_id,
        "role": spec.role,
        "license_spdx": spec.license_spdx,
        "requested_revision": spec.revision,
        "resolved_revision": _resolved_revision(snapshot),
        "snapshot_path": str(snapshot),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": [
            {
                "path": str(path.relative_to(snapshot)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in files
        ],
    }
    root = Path(receipt_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{_slug(spec.model_id)}.json"
    destination.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and download SemOp's reviewed local model profiles."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="inspect local runtime and caches")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.add_argument("--workspace", default=".")
    download = subparsers.add_parser(
        "download",
        help="download only models from the reviewed SemOp manifest",
    )
    download.add_argument(
        "--profile",
        choices=("economy", "balanced"),
        default="balanced",
    )
    download.add_argument("--include-vision-sidecar", action="store_true")
    download.add_argument(
        "--receipt-root",
        default="artifacts/models/receipts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.command == "doctor":
        report = build_model_doctor_report(workspace=args.workspace)
        if args.as_json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_doctor(report)
        return 0 if report.symbolic_ready else 1
    try:
        receipts = download_profile(
            args.profile,
            include_vision_sidecar=args.include_vision_sidecar,
            receipt_root=args.receipt_root,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"model download failed: {exc}", file=sys.stderr)
        return 1
    for receipt in receipts:
        print(receipt.resolve())
    return 0


def _dependency_check(name: str) -> DependencyCheck:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return DependencyCheck(name, False)
    package_name = {
        "PIL": "Pillow",
        "sentence_transformers": "sentence-transformers",
    }.get(name, name)
    try:
        from importlib.metadata import version

        package_version = version(package_name)
    except Exception:
        package_version = "unknown"
    return DependencyCheck(name, True, package_version)


def _cached_model_revisions() -> dict[str, str]:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return {}
    try:
        cache = scan_cache_dir()
    except Exception:
        return {}
    found: dict[str, str] = {}
    for repository in cache.repos:
        if repository.repo_type != "model" or not repository.revisions:
            continue
        revision = max(
            repository.revisions,
            key=lambda item: item.last_modified,
        )
        found[repository.repo_id] = revision.commit_hash
    return found


def _free_disk_bytes(path: Path) -> int:
    import shutil

    return int(shutil.disk_usage(path).free)


def _allow_patterns(spec: ModelSpec) -> tuple[str, ...]:
    if spec is MULTILINGUAL_E5:
        # SentenceTransformers only needs the root Transformer files and its
        # pooling module.  Excluding exported runtimes also avoids duplicate
        # cache links that require Windows Developer Mode.
        return (
            "config.json",
            "config_sentence_transformers.json",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "1_Pooling/config.json",
        )
    common = (
        "*.json",
        "*.safetensors",
        "tokenizer*",
        "vocab*",
        "merges*",
        "*.model",
        "*.tiktoken",
        "*.jinja",
        "*.txt",
    )
    if spec is FLORENCE_SIDECAR:
        return (*common, "*.py")
    return common


def _resolved_revision(snapshot: Path) -> str:
    if snapshot.parent.name == "snapshots":
        return snapshot.name
    return "unknown"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(model_id: str) -> str:
    return model_id.replace("/", "--").replace("\\", "--")


def _print_doctor(report: ModelDoctorReport) -> None:
    python_status = (
        "recommended"
        if report.python_recommended
        else ("supported; Python 3.12 is recommended" if report.python_supported else "unsupported")
    )
    print(f"Python {report.python_version}: {python_status}")
    print(
        "CUDA: "
        + (report.cuda_device if report.cuda_available else "not available")
    )
    print(f"Free disk: {report.free_disk_bytes / (1024 ** 3):.1f} GiB")
    for dependency in report.dependencies:
        status = dependency.version if dependency.available else "missing"
        print(f"Dependency {dependency.name}: {status}")
    for model in report.models:
        status = f"cached ({model.revision[:12]})" if model.cached else "not cached"
        print(f"Model {model.model_id}: {status}")


if __name__ == "__main__":
    raise SystemExit(main())
