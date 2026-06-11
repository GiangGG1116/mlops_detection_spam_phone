"""
risk_evaluator.py
-----------------
Evaluate phone number risk level based on:
  1. Model score (XGBoost probability)
    2. Rule-based signals: blacklist, sensitive international/domestic prefixes, call behavior

Output: final_level (0-4) + title + reasons
"""
from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Tuple


# =========================
# VN DOMESTIC PREFIXES
# =========================
VN_MOBILE_PREFIXES = {
    # viettel
    "86", "96", "97", "98", "32", "33", "34", "35", "36", "37", "38", "39",
    # mobi
    "89", "90", "93", "70", "79", "77", "76", "78",
    # vina
    "88", "91", "94", "83", "84", "85", "81", "82",
    # vnmobile
    "92", "58", "56",
    # gmobile
    "99", "59",
    # itelecom
    "87",
}

VN_LANDLINE_PREFIXES = {
    # 2-digit
    "24", "28",
    # 3-digit
    "203", "204", "205", "206", "207", "208", "209", "210", "211", "212",
    "213", "214", "215", "216", "218", "219",
    "220", "221", "222", "225", "226", "227", "228", "229",
    "232", "233", "234", "235", "236", "237", "238", "239",
    "251", "252", "254", "255", "256", "257", "258", "259",
    "260", "261", "262", "263", "269",
    "270", "271", "272", "273", "274", "275", "276", "277",
    "290", "291", "292", "293", "294", "296", "297", "299",
}

VN_SWITCHBOARD_PREFIXES = {
    "2361091", "241098", "241091", "241083", "241077",
    "1094", "1077", "198", "197",
}

# =========================
# Country calling codes (E.164)
# =========================
COUNTRY_CODES = {
    "1", "7",
    "20", "27", "30", "31", "32", "33", "34", "36", "39",
    "40", "41", "43", "44", "45", "46", "47", "48", "49",
    "51", "52", "53", "54", "55", "56", "57", "58",
    "60", "61", "62", "63", "64", "65", "66",
    "81", "82", "84", "86",
    "90", "91", "92", "93", "94", "95", "98",
    "212", "213", "216", "218",
    "351", "352", "353", "354", "355", "356", "357", "358", "359",
    "380", "381", "382", "385", "386", "387", "389",
}

# =========================
# Risk rules config
# =========================
import os
import yaml
from pathlib import Path


def load_risk_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        from ...core.paths import PROJECT_ROOT
        config_path = PROJECT_ROOT / "configs/risk_config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# Load default config for backward compatibility
RISK_CONFIG = load_risk_config()


# =========================
# Helpers
# =========================
def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _normalize_phone(raw: Any) -> str:
    return "" if raw is None else str(raw).strip()


def _match_country_code(digits: str) -> str:
    if not digits:
        return ""
    for k in (3, 2, 1):
        cc = digits[:k]
        if cc in COUNTRY_CODES:
            return cc
    return ""


def _parse_is_international_and_prefix(phone: str) -> Tuple[bool, str]:
    """
    Classify phone number as domestic VN or international.
    Return: (is_international, prefix)
    """
    p = (phone or "").strip()
    if not p:
        return False, ""

    digits = "".join(ch for ch in p if ch.isdigit())
    if not digits:
        return False, ""

    if len(digits) <= 5:
        return False, digits

    vn = None
    if digits.startswith("84") and not digits.startswith("00"):
        vn = "0" + digits[2:]
    elif digits.startswith("0"):
        vn = digits

    if vn:
        for sw in sorted(VN_SWITCHBOARD_PREFIXES, key=len, reverse=True):
            if vn.startswith(sw):
                return False, sw

    if vn and len(vn) >= 3:
        mobile2 = vn[1:3]
        if mobile2 in VN_MOBILE_PREFIXES:
            return False, vn[:3]

    if vn:
        for k in (3, 2):
            if len(vn) >= 1 + k:
                land = vn[1: 1 + k]
                if land in VN_LANDLINE_PREFIXES:
                    return False, vn[: 1 + k]

    if p.startswith("+00") or p.startswith("00") or digits.startswith("00"):
        rest = digits[2:] if digits.startswith("00") else digits
        cc = _match_country_code(rest) or rest[:3]
        return True, cc

    if p.startswith("+"):
        cc = _match_country_code(digits) or digits[:3]
        return True, cc

    cc = _match_country_code(digits)
    if cc:
        return True, cc
    return True, digits[:3]


# =========================
# Feature extraction
# =========================
def row_to_risk_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract rule-based evaluation features from one prediction row."""
    phone = _normalize_phone(row.get("phone"))
    is_international, prefix = _parse_is_international_and_prefix(phone)

    sum_call = _to_int(row.get("sum_call"), 0)
    call_to_miss = _to_int(row.get("call_to_miss"), 0)
    call_in_miss = _to_int(row.get("call_in_miss"), 0)
    miss_call = call_to_miss + call_in_miss

    duration_total = _to_float(row.get("duration"), 0.0)
    avg_duration = (duration_total / sum_call) if sum_call > 0 else 0.0

    frequency_per_day = _to_float(row.get("mean_call_by_day"), 0.0)
    callback_rate = _to_float(row.get("call_back_rate"), 0.0)

    in_hour = _to_float(row.get("in_hour"), 0.0)
    not_in_hour = _to_float(row.get("not_in_hour"), 0.0)
    mostly_out_of_business_hour = bool(not_in_hour > in_hour)

    avg_in_contact = _to_float(row.get("avg_in_contact"), 0.0)
    in_contact = bool(avg_in_contact > 0.0)

    successful_call_count = _to_int(row.get("total_contacted"), 0)
    avg_in_duration = _to_float(row.get("avg_duration_call_in"), 0.0)

    return {
        "is_international": is_international,
        "prefix": prefix,
        "total_call": sum_call,
        "miss_call": miss_call,
        "avg_duration": avg_duration,
        "frequency_per_day": frequency_per_day,
        "callback_rate": callback_rate,
        "mostly_out_of_business_hour": mostly_out_of_business_hour,
        "in_contact": in_contact,
        "successful_call_count": successful_call_count,
        "avg_in_duration": avg_in_duration,
    }


def parse_csv_text_to_rows(csv_text: str) -> List[Dict[str, Any]]:
    buf = io.StringIO(csv_text.strip())
    reader = csv.DictReader(buf)
    return [dict(r) for r in reader]


# =========================
# Risk evaluation logic
# =========================
def get_model_level(score: float, config: dict | None = None) -> int:
    cfg = config or RISK_CONFIG
    thresholds = cfg.get("model_thresholds", [])
    for threshold, level in thresholds:
        if score < threshold:
            return level
    return thresholds[-1][1] if thresholds else 0


def get_rule_level(
    phone: str,
    prefix: str,
    is_international: bool,
    total_call: int,
    miss_call: int,
    avg_duration: float,
    frequency_per_day: float,
    callback_rate: float,
    mostly_out_of_business_hour: bool,
    config: dict | None = None,
) -> Tuple[int, List[str]]:
    cfg = config or RISK_CONFIG
    reasons: List[str] = []
    level = 0
    rules = cfg.get("rule_levels", {})

    if phone in cfg.get("hard_blacklist_numbers", []):
        reasons.append("Số này thuộc danh sách cảnh báo lừa đảo được công bố.")
        return rules.get("hard_blacklist", 4), reasons

    if is_international:
        intl_prefix = prefix
        if intl_prefix in cfg.get("intl_risk_prefixes", []):
            level = max(level, rules.get("intl_prefix_boost_to", 3))
            reasons.append(f"Đầu số quốc tế {intl_prefix} có nhiều cảnh báo lừa đảo.")
        else:
            level = max(level, rules.get("intl_base", 2))
            reasons.append("Cuộc gọi từ số quốc tế, cần cảnh giác.")

    domestic_hit = None
    for dp in cfg.get("domestic_risk_prefixes", []):
        if prefix.startswith(dp) or phone.startswith(dp):
            domestic_hit = dp
            break

    behavior = cfg.get("behavior_rules", {})

    if domestic_hit is not None:
        if total_call >= behavior.get("short_call_min_total", 2) and avg_duration < behavior.get("short_call_seconds", 5):
            level = max(level, rules.get("domestic_short_call_boost_to", 3))
            reasons.append(f"Đầu số {domestic_hit} với nhiều cuộc gọi rất ngắn giống hành vi nháy máy.")
        else:
            level = max(level, rules.get("domestic_prefix_base", 2))
            reasons.append(f"Đầu số {domestic_hit} thường xuất hiện trong các cuộc gọi nghi vấn.")

    if total_call >= behavior.get("miss_ratio_min_total", 5):
        miss_ratio = miss_call / total_call if total_call > 0 else 0.0
        if miss_ratio > behavior.get("miss_ratio_high", 0.6):
            level = max(level, rules.get("behavior_miss_ratio_to", 3))
            reasons.append("Tỷ lệ cuộc gọi nhỡ rất cao, bất thường so với các số thông thường.")

    if frequency_per_day >= behavior.get("high_freq_per_day", 10) and avg_duration < behavior.get("high_freq_short_call_seconds", 10):
        level = max(level, rules.get("behavior_high_freq_to", 3))
        reasons.append("Gọi rất nhiều lần/ngày nhưng thời lượng ngắn, giống pattern spam.")

    if callback_rate < behavior.get("low_callback_rate", 0.1) and total_call >= behavior.get("low_callback_min_total", 5):
        level = max(level, rules.get("behavior_low_callback_to", 2))
        reasons.append("Rất ít người gọi lại số này, cho thấy mức độ tin cậy thấp.")

    if mostly_out_of_business_hour:
        level = max(level, behavior.get("outside_business_hour_level", 2))
        reasons.append("Cuộc gọi chủ yếu diễn ra ngoài giờ hành chính, cần cảnh giác.")

    return level, reasons


def apply_whitelist_adjustments(
    base_level: int,
    in_contact: bool,
    successful_call_count: int,
    avg_in_duration: float,
    is_international: bool,
    config: dict | None = None,
) -> Tuple[int, int]:
    cfg = config or RISK_CONFIG
    adj = cfg.get("whitelist_adjustments", {})
    adjust = 0
    
    if in_contact:
        adjust -= adj.get("in_contact_reduce", 2)
    if successful_call_count >= adj.get("good_history_min_success_calls", 5) and avg_in_duration >= adj.get("good_history_min_avg_in_duration", 60):
        adjust -= adj.get("good_history_reduce", 1)

    raw_level = base_level + adjust
    intl_min = adj.get("intl_min_level_after_reduce", 1)
    final_level = max(intl_min, raw_level) if is_international else max(0, raw_level)
    return final_level, adjust


def evaluate_phone_risk(
    phone: str,
    model_score: float,
    features: Dict[str, Any],
    config: dict | None = None,
) -> Dict[str, Any]:
    """
    Evaluate the overall risk level of a phone number.

    Args:
        phone: Phone number to evaluate
        model_score: Spam probability from the XGBoost model (0.0 - 1.0)
        features: Feature dict from row_to_risk_features()
        config: Risk config dict (defaults to RISK_CONFIG)

    Returns:
        Dict with: final_level, model_level, rule_level, title, subtitle, reasons
    """
    cfg = config or RISK_CONFIG
    model_level = get_model_level(model_score, cfg)

    rule_level, rule_reasons = get_rule_level(
        phone=phone,
        prefix=features["prefix"],
        is_international=features["is_international"],
        total_call=features["total_call"],
        miss_call=features["miss_call"],
        avg_duration=features["avg_duration"],
        frequency_per_day=features["frequency_per_day"],
        callback_rate=features["callback_rate"],
        mostly_out_of_business_hour=features["mostly_out_of_business_hour"],
        config=cfg,
    )

    base_level = max(model_level, rule_level)

    final_level, adjust = apply_whitelist_adjustments(
        base_level=base_level,
        in_contact=features["in_contact"],
        successful_call_count=features["successful_call_count"],
        avg_in_duration=features["avg_in_duration"],
        is_international=features["is_international"],
        config=cfg,
    )

    msg = cfg.get("level_messages", {}).get(str(final_level)) or cfg.get("level_messages", {}).get(final_level, {"title": "Không xác định", "subtitle": ""})
    reasons: List[str] = []
    reasons.append(f"Điểm rủi ro từ mô hình: {model_score:.2f} (Level {model_level}).")
    if rule_level > 0 or rule_reasons:
        reasons.append(f"Level từ rule: {rule_level}.")
        reasons.extend(rule_reasons)
    if adjust < 0:
        reasons.append("Đã giảm mức cảnh báo do số này thuộc danh bạ/lịch sử liên lạc tốt.")

    return {
        "phone": phone,
        "final_level": int(final_level),
        "model_level": int(model_level),
        "rule_level": int(rule_level),
        "title": msg["title"],
        "subtitle": msg["subtitle"],
        "reasons": reasons,
        "features_used": features,
    }
