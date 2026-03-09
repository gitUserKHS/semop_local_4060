from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from semop.latent_abstraction import OperatorAbstractionClustering


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phrases", nargs="+", help="phrases to cluster")
    args = parser.parse_args()

    model = OperatorAbstractionClustering()
    clusters = model.cluster(args.phrases)
    print(json.dumps([c.__dict__ for c in clusters], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

