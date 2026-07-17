from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "semop"


@dataclass(frozen=True)
class ModuleAudit:
    module: str
    path: str
    imported_by_modules: tuple[str, ...]
    external_importers: tuple[str, ...]
    exported_symbols: tuple[str, ...]
    externally_used_exports: tuple[str, ...]
    classification: str
    deletion_ready: bool = False


def audit() -> dict:
    module_paths = {
        _module_name(path): path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    inbound: dict[str, set[str]] = defaultdict(set)
    exported: dict[str, set[str]] = defaultdict(set)
    export_owner: dict[str, str] = {}
    for module, path in module_paths.items():
        tree = _parse(path)
        for target in _imports_from_tree(tree, module, path.name == "__init__.py"):
            if target in module_paths:
                inbound[target].add(module)
        if module == "semop":
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
                    continue
                owner = "semop." + node.module
                for alias in node.names:
                    public_name = alias.asname or alias.name
                    exported[owner].add(public_name)
                    export_owner[public_name] = owner

    external_importers: dict[str, set[str]] = defaultdict(set)
    externally_used_exports: dict[str, set[str]] = defaultdict(set)
    external_roots = [ROOT / "tests", ROOT / "tools", ROOT / "examples"]
    external_files = [
        path
        for root in external_roots
        if root.exists()
        for path in root.rglob("*.py")
    ] + [path for path in ROOT.glob("*.py") if path.is_file()]
    for path in external_files:
        tree = _parse(path)
        relative = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in module_paths:
                    external_importers[node.module].add(relative)
                if node.module == "semop":
                    for alias in node.names:
                        owner = export_owner.get(alias.name)
                        if owner:
                            external_importers[owner].add(relative)
                            externally_used_exports[owner].add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in module_paths:
                        external_importers[alias.name].add(relative)

    records: list[ModuleAudit] = []
    for module, path in sorted(module_paths.items()):
        internal = inbound.get(module, set()) - {"semop"}
        external = external_importers.get(module, set())
        public = exported.get(module, set())
        used_public = externally_used_exports.get(module, set())
        protected = (
            module == "semop"
            or module.startswith("semop.kernel")
            or module.startswith("semop.tiny_controller")
        )
        if protected or internal or external or used_public:
            classification = "active"
        elif public:
            classification = "export_only_review_candidate"
        else:
            classification = "static_orphan_review_candidate"
        records.append(
            ModuleAudit(
                module=module,
                path=str(path.relative_to(ROOT)),
                imported_by_modules=tuple(sorted(inbound.get(module, set()))),
                external_importers=tuple(sorted(external)),
                exported_symbols=tuple(sorted(public)),
                externally_used_exports=tuple(sorted(used_public)),
                classification=classification,
            )
        )
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.classification] += 1
    return {
        "schema_version": 1,
        "warning": (
            "Static evidence is insufficient for deletion. Dynamic imports, GUI use, "
            "and persisted artifact compatibility require shadow telemetry and tests."
        ),
        "counts": dict(sorted(counts.items())),
        "modules": [asdict(record) for record in records],
    }


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("semop", *parts)) if parts else "semop"


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _imports_from_tree(
    tree: ast.AST, current_module: str, is_package: bool
) -> set[str]:
    targets: set[str] = set()
    current_parts = current_module.split(".")
    package_parts = current_parts if is_package else current_parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                base = package_parts[: max(0, keep)]
                suffix = node.module.split(".") if node.module else []
                targets.add(".".join((*base, *suffix)))
            elif node.module:
                targets.add(node.module)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit SemOp module usage before legacy code removal"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("json", "summary"), default="summary")
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
        print(report["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
