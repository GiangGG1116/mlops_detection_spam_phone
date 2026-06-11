from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def normalize_phone(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    digits = re.sub(r"\D+", "", text)
    return digits.lstrip("0") or "0"


def infer_ext(path: str | Path) -> str:
    p = str(path).lower()
    if p.endswith(".parquet"): return "parquet"
    if p.endswith(".csv"):     return "csv"
    if p.endswith(".jsonl"):   return "jsonl"
    if p.endswith(".json"):    return "json"
    return "csv"


def load_threshold(metrics_path: Path | str, default: float = 0.5) -> float:
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return default
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    threshold = metrics.get("threshold_best", default)
    if isinstance(threshold, dict):
        threshold = threshold.get("threshold", default)

    try:
        return float(threshold)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")
