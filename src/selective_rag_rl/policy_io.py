from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_checkpoint(path: Path, model: object, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "metadata": metadata}
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "model" not in payload or "metadata" not in payload:
        raise ValueError(f"Invalid checkpoint format: {path}")
    return payload
