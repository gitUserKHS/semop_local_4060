from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def resolve_embedding_model_id(model_id: str | None = None) -> str:
    env_path = os.environ.get("SEMOP_EMBED_MODEL_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return str(candidate)
    return model_id or os.environ.get("SEMOP_EMBED_MODEL_ID") or DEFAULT_EMBEDDING_MODEL_ID


class EmbeddingModelCache:
    def __init__(self, cache_root: str | Path = "models/embeddings"):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def target_dir(self, model_id: str) -> Path:
        safe_name = model_id.replace("/", "--").replace(":", "-")
        return self.cache_root / safe_name

    def cache(self, model_id: str = DEFAULT_EMBEDDING_MODEL_ID, output_dir: str | Path | None = None, force: bool = False) -> Path:
        target = Path(output_dir) if output_dir else self.target_dir(model_id)
        target.mkdir(parents=True, exist_ok=True)
        if not force and any(target.iterdir()):
            return target

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return target

    @staticmethod
    def export_env(model_path: str | Path) -> str:
        return f"SEMOP_EMBED_MODEL_PATH={Path(model_path).resolve()}"
