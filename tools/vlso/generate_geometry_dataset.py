from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from semop.vlso.geometry_dataset import SyntheticGeometrySceneBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate synthetic geometry scenes, paired JSON payloads, and a VLSO eval set')
    parser.add_argument('--output-dir', default='examples/vlso/generated_geometry', help='directory for generated images and visual json payloads')
    parser.add_argument('--eval-output', default='examples/vlso_geometry_eval_generated.jsonl', help='output eval jsonl path')
    args = parser.parse_args()

    builder = SyntheticGeometrySceneBuilder()
    scenes = builder.build(args.output_dir)
    builder.write_eval_jsonl(scenes, args.eval_output)
    print(json.dumps({
        'output_dir': str(Path(args.output_dir)),
        'eval_output': str(Path(args.eval_output)),
        'num_scenes': len(scenes),
        'scene_ids': [scene.scene_id for scene in scenes],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
