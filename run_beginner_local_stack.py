from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))



def _reexec_into_preferred_python() -> None:
    preferred = Path('.venv312') / 'Scripts' / 'python.exe'
    try:
        current = Path(sys.executable).resolve()
    except Exception:
        current = Path(sys.executable)
    if preferred.exists():
        try:
            preferred_resolved = preferred.resolve()
        except Exception:
            preferred_resolved = preferred
        if str(current).lower() != str(preferred_resolved).lower():
            os.execv(str(preferred_resolved), [str(preferred_resolved), __file__, *sys.argv[1:]])


from semop import build_runtime_doctor_report, launch_beginner_one_click, launch_runtime_stack


def main() -> None:
    _reexec_into_preferred_python()
    parser = argparse.ArgumentParser(description='Beginner-friendly local operations runner for SemOp')
    subparsers = parser.add_subparsers(dest='command')

    doctor = subparsers.add_parser('doctor', help='Inspect the local runtime environment and recommend the safest launch path')
    doctor.add_argument('--host', default='127.0.0.1')
    doctor.add_argument('--python', dest='python_executable', default='')

    launch = subparsers.add_parser('launch', help='Launch the beginner local stack in the background and write a manifest')
    launch.add_argument('--host', default='127.0.0.1')
    launch.add_argument('--python', dest='python_executable', default='')
    launch.add_argument('--no-math-service', action='store_true')
    launch.add_argument('--dry-run', action='store_true')

    one_click = subparsers.add_parser('one_click', help='Diagnose, launch the beginner stack, and open the browser automatically')
    one_click.add_argument('--host', default='127.0.0.1')
    one_click.add_argument('--python', dest='python_executable', default='')
    one_click.add_argument('--no-math-service', action='store_true')
    one_click.add_argument('--dry-run', action='store_true')
    one_click.add_argument('--no-browser', action='store_true')
    one_click.add_argument('--open-all', action='store_true')
    one_click.add_argument('--wait-seconds', type=float, default=15.0)

    argv = sys.argv[1:] or ['one_click']
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        report = build_runtime_doctor_report(
            workspace='.',
            python_executable=args.python_executable or None,
            host=args.host,
        )
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
        return
    if args.command == 'launch':
        summary = launch_runtime_stack(
            workspace='.',
            python_executable=args.python_executable or None,
            host=args.host,
            include_math_service=not args.no_math_service,
            dry_run=bool(args.dry_run),
        )
        print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))
        return
    if args.command == 'one_click':
        summary = launch_beginner_one_click(
            workspace='.',
            python_executable=args.python_executable or None,
            host=args.host,
            include_math_service=not args.no_math_service,
            dry_run=bool(args.dry_run),
            open_browser=not bool(args.no_browser),
            open_all=bool(args.open_all),
            wait_timeout_seconds=float(args.wait_seconds),
        )
        print(json.dumps(summary.model_dump(), ensure_ascii=False, indent=2))
        return
    parser.error('Unsupported command.')


if __name__ == '__main__':
    main()
