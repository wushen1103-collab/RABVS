from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def cpu_worker_cap(requested: int | None = None, reserve: int = 30) -> int:
    total = os.cpu_count() or 1
    usable = max(1, total - reserve)
    if requested is None or requested <= 0:
        return usable
    return max(1, min(requested, usable))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + eps)

