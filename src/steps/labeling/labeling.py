from __future__ import annotations

import numpy as np
import pandas as pd


def label_3class_user_feedback(
    df_summary: pd.DataFrame,
    spam_u_min: int = 3,
    spam_r_min: int = 5,
    spam_rate_min: float = 0.70,
    ham_u_min: int = 3,
    ham_r_min: int = 5,
    ham_rate_min: float = 0.70,
    min_total_for_strong: int = 6,
) -> pd.DataFrame:
    """Generate labels from user feedback with precision-first rules.

    Output labels:
    - `1`: spam
    - `0`: ham
    - `-1`: unknown / review
    """
    df = df_summary.copy()

    if "ham_reports" not in df.columns and "not_spam_reports" in df.columns:
        df["ham_reports"] = df["not_spam_reports"]

    for col in [
        "total_reports",
        "spam_reports",
        "unique_spam_members",
        "ham_reports",
        "unique_ham_members",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "total_reports" not in df.columns:
        df["total_reports"] = df.get("spam_reports", 0) + df.get("ham_reports", 0)

    df["spam_rate"] = np.where(
        df["total_reports"] > 0,
        df["spam_reports"] / df["total_reports"],
        0.0,
    )
    df["ham_rate"] = np.where(
        df["total_reports"] > 0,
        df["ham_reports"] / df["total_reports"],
        0.0,
    )

    def safe_q_int(series: pd.Series, q: float, fallback: int) -> int:
        num = pd.to_numeric(series, errors="coerce").fillna(0)
        if num.size >= 50:
            return max(fallback, int(round(np.quantile(num, q))))
        return fallback

    spam_r_hi = safe_q_int(df["spam_reports"], 0.95, 8)
    ham_r_hi = safe_q_int(df["ham_reports"], 0.95, 8)

    df["label"] = -1
    df["rule_id"] = "unknown"
    df["confidence"] = ""
    df["rule_note"] = ""

    def set_rule(mask: pd.Series, label: int, rule_id: str, confidence: str, note: str = "") -> None:
        idx = mask.fillna(False) & (df["label"] == -1)
        if not idx.any():
            return
        df.loc[idx, "label"] = label
        df.loc[idx, "rule_id"] = rule_id
        df.loc[idx, "confidence"] = confidence
        if note:
            df.loc[idx, "rule_note"] = note

    spam_strong = (
        (df["total_reports"] >= min_total_for_strong)
        & (df["unique_spam_members"] >= spam_u_min)
        & (df["spam_reports"] >= spam_r_min)
        & (df["spam_rate"] >= spam_rate_min)
    )
    set_rule(
        spam_strong,
        1,
        "SPAM_CONSENSUS_STRONG",
        "high",
        f"total>={min_total_for_strong}, Uspam>={spam_u_min}, Rspam>={spam_r_min}, SR>={spam_rate_min}",
    )

    spam_volume = (
        (df["unique_spam_members"] >= 2)
        & (df["spam_reports"] >= spam_r_hi)
        & (df["spam_rate"] >= 0.60)
    )
    set_rule(
        spam_volume,
        1,
        "SPAM_VOLUME_HI",
        "medium",
        f"Rspam>={spam_r_hi} (p95), Uspam>=2, SR>=0.60",
    )

    ham_strong = (
        (df["total_reports"] >= min_total_for_strong)
        & (df["unique_ham_members"] >= ham_u_min)
        & (df["ham_reports"] >= ham_r_min)
        & (df["ham_rate"] >= ham_rate_min)
        & (df["spam_reports"] <= 1)
    )
    set_rule(
        ham_strong,
        0,
        "HAM_CONSENSUS_STRONG",
        "high",
        f"total>={min_total_for_strong}, Uham>={ham_u_min}, Rham>={ham_r_min}, HR>={ham_rate_min}, spam<=1",
    )

    ham_volume = (
        (df["unique_ham_members"] >= 2)
        & (df["ham_reports"] >= ham_r_hi)
        & (df["ham_rate"] >= 0.60)
        & (df["spam_reports"] == 0)
    )
    set_rule(
        ham_volume,
        0,
        "HAM_VOLUME_HI",
        "medium",
        f"Rham>={ham_r_hi} (p95), Uham>=2, HR>=0.60, spam=0",
    )

    both_sides = (df["spam_reports"] > 0) & (df["ham_reports"] > 0)
    low_evidence = (df["total_reports"] < 3) | (
        (df["unique_spam_members"] + df["unique_ham_members"]) < 2
    )

    review_idx = (df["label"] == -1) & (both_sides | low_evidence)
    df.loc[review_idx, "rule_id"] = "REVIEW_CONFLICT_OR_LOW_EVID"
    df.loc[review_idx, "confidence"] = "low"

    return df
