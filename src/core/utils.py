from __future__ import annotations

import re


def normalize_phone(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    digits = re.sub(r"\D+", "", text)
    return digits.lstrip("0") or "0"
