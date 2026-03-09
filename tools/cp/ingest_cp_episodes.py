from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop import CpIncidentDataset, CpIncidentIngestor


def main() -> None:
    parser = argparse.ArgumentParser(description='Ingest real CP statements, editorials, and WA/TLE incidents into episodic memory')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--store', required=True)
    parser.add_argument('--no-solve', action='store_true', help='store incidents without generating a SemOp solution first')
    args = parser.parse_args()

    dataset = CpIncidentDataset()
    cases = dataset.load_inputs(args.inputs)
    ingestor = CpIncidentIngestor(args.store)
    ids = ingestor.ingest_cases(cases, solve_missing=not args.no_solve)
    print(json.dumps({'loaded_cases': len(cases), 'stored_episodes': len(ids)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

