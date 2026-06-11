from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Any, List

import pandas as pd

from .logging import get_logger




@dataclass(frozen=True)
class ValidationRule:
    required_columns: tuple[str, ...]
    min_rows: int = 1


class SchemaValidationError(ValueError):
    pass


def validate_dataframe(df: pd.DataFrame, rule: ValidationRule, name: str) -> None:
    missing = [c for c in rule.required_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"{name}: missing required columns {missing}. Available={list(df.columns)}"
        )

    if len(df) < rule.min_rows:
        raise SchemaValidationError(
            f"{name}: expected at least {rule.min_rows} rows but got {len(df)}"
        )


def validate_file_columns(path: str, required_columns: Iterable[str], name: str) -> None:
    df = pd.read_csv(path, nrows=5)
    rule = ValidationRule(tuple(required_columns), min_rows=0)
    validate_dataframe(df, rule, name)
