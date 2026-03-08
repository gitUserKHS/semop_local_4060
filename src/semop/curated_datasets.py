from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


CURATED_PUBLIC_DATASETS: Dict[str, dict] = {
    "gsm8k_train": {
        "name": "gsm8k_train",
        "source": "gsm8k_train",
        "urls": [
            "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
        ],
        "extract": False,
        "augment": False,
        "train_ratio": 0.9,
        "query_fields": ["question"],
        "answer_fields": ["answer"],
    },
    "gsm8k_test": {
        "name": "gsm8k_test",
        "source": "gsm8k_test",
        "urls": [
            "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        ],
        "extract": False,
        "augment": False,
        "train_ratio": 0.5,
        "query_fields": ["question"],
        "answer_fields": ["answer"],
    },
    "gsm_ic_2step": {
        "name": "gsm_ic_2step",
        "source": "gsm_ic_2step",
        "urls": [
            "https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/GSM-IC_2step.json"
        ],
        "extract": False,
        "augment": False,
        "train_ratio": 0.8,
        "query_fields": ["new_question", "original_question"],
        "answer_fields": ["answer"],
    },
    "gsm_ic_mstep": {
        "name": "gsm_ic_mstep",
        "source": "gsm_ic_mstep",
        "urls": [
            "https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/GSM-IC_mstep.json"
        ],
        "extract": False,
        "augment": False,
        "train_ratio": 0.8,
        "query_fields": ["new_question", "original_question"],
        "answer_fields": ["answer"],
    },
    "officeqa": {
        "name": "officeqa",
        "source": "officeqa",
        "urls": [
            "https://raw.githubusercontent.com/databricks/officeqa/main/officeqa.csv"
        ],
        "extract": False,
        "augment": False,
        "train_ratio": 0.85,
        "query_fields": ["question"],
        "answer_fields": ["answer"],
    },
    "bigbench_mistake_logical": {
        "name": "bigbench_mistake_logical",
        "source": "bigbench_mistake_logical",
        "urls": [
            "https://raw.githubusercontent.com/WHGTyen/BIG-Bench-Mistake/main/logical_deduction.jsonl"
        ],
        "extract": False,
        "augment": False,
        "train_ratio": 0.85,
        "query_fields": ["input"],
        "answer_fields": ["target", "answer"],
    },
    "financebench": {
        "name": "financebench",
        "source": "financebench",
        "urls": [
            "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
        ],
        "extract": False,
        "augment": False,
        "train_ratio": 0.85,
        "query_fields": ["question"],
        "context_fields": ["question_reasoning"],
        "answer_fields": ["answer", "justification"],
    },
}


CURATED_PRESETS: Dict[str, List[str]] = {
    "starter": ["gsm_ic_2step", "officeqa"],
    "math": ["gsm8k_train", "gsm8k_test", "gsm_ic_2step", "gsm_ic_mstep"],
    "reasoning_core": ["gsm8k_train", "gsm_ic_2step", "officeqa", "bigbench_mistake_logical"],
    "all_public_reasoning": list(CURATED_PUBLIC_DATASETS.keys()),
}


def curated_manifest(dataset_names: List[str]) -> List[dict]:
    manifest: List[dict] = []
    for name in dataset_names:
        if name not in CURATED_PUBLIC_DATASETS:
            raise KeyError(f"Unknown curated dataset: {name}")
        manifest.append(deepcopy(CURATED_PUBLIC_DATASETS[name]))
    return manifest


def preset_manifest(preset_name: str) -> List[dict]:
    if preset_name not in CURATED_PRESETS:
        raise KeyError(f"Unknown curated preset: {preset_name}")
    return curated_manifest(CURATED_PRESETS[preset_name])
