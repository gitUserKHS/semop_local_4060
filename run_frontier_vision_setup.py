from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from semop import FrontierVisionInstaller


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Inspect or install the recommended local frontier vision bundle.')
    parser.add_argument('command', choices=('status', 'install'))
    parser.add_argument('--target-root', default='models/vision/frontier', help='Local directory where frontier vision checkpoints live.')
    parser.add_argument('--include-optional', action='store_true', help='Also install the optional heavier Molmo bundle.')
    parser.add_argument('--force', action='store_true', help='Re-download even if a checkpoint already appears installed.')
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    installer = FrontierVisionInstaller(target_root=args.target_root)
    if args.command == 'status':
        print(json.dumps(installer.installed_bundle_status().model_dump(), ensure_ascii=False, indent=2))
        return 0
    install = installer.install_recommended_bundle(include_optional=args.include_optional, force=args.force).model_dump()
    status = installer.installed_bundle_status().model_dump()
    print(json.dumps({'install': install, 'status': status}, ensure_ascii=False, indent=2))
    return 0 if not install.get('failed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
