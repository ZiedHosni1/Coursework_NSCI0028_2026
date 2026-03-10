from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_csv_arg(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip())[:140]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def now_seconds() -> float:
    return time.perf_counter()


def finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def normalize_series(values: list[float], reverse: bool = False) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return []
    if np.allclose(arr, arr[0]):
        norm = np.zeros_like(arr)
    else:
        norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
    if reverse:
        norm = 1.0 - norm
    return norm.tolist()


def dataclass_to_dict(instance: Any) -> dict[str, Any]:
    return asdict(instance)
