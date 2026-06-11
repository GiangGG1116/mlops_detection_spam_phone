# import argparse
from __future__ import annotations
import pandas as pd
import numpy as np
import warnings
import joblib
import json
import os

import os

from typing import Optional, List, Dict, Any

from ..features.build_feature_duckdb import build_features_duckdb
from ...core.config import load_config
from ...core.utils import infer_ext





def load_feature_cols(path: str):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "cols" in obj:
        return obj["cols"]
    return obj


def resolve_feature_cols_path(feature_cols_arg: str | None, model_path: str) -> str | None:
    """If --feature_cols is not provided, auto-discover feature_columns.json next to the model."""
    if feature_cols_arg:
        return feature_cols_arg
    cand = os.path.join(os.path.dirname(model_path), "feature_columns.json")
    return cand if os.path.exists(cand) else None


def align_features(X_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    # Add missing columns with zero as a safe fallback
    for c in feature_cols:
        if c not in X_df.columns:
            X_df[c] = 0
    # Drop extra columns and enforce training column order
    return X_df[feature_cols]


def to_numeric_keep_nan(X_df: pd.DataFrame) -> pd.DataFrame:
    """Match training preprocessing: coerce to numeric and keep NaN (XGBoost handles NaN)."""
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    # Keep only numeric columns and cast to float32 for efficiency
    X_df = X_df.select_dtypes(include=["number"]).astype(np.float32)
    return X_df


# ---------- map report->phone then SUM per phone (DuckDB) ----------
def _pick_raw_table_with_report_phone(con):
    rows = con.execute("""
        SELECT table_schema, table_name
        FROM information_schema.columns
        WHERE lower(column_name) IN ('report','phone')
        GROUP BY table_schema, table_name
        HAVING COUNT(DISTINCT lower(column_name)) = 2
    """).fetchall()

    if not rows:
        raise RuntimeError("No raw table/view with both (report, phone) columns was found in the DuckDB connection.")

    priority = {"raw": 0, "raw_view": 1, "history": 2, "call_history": 3, "data": 4}
    rows_sorted = sorted(rows, key=lambda x: priority.get(x[1].lower(), 999))
    schema, name = rows_sorted[0]
    return f'"{schema}"."{name}"'


def aggregate_final_features_by_phone(con, feat_table="final_features") -> pd.DataFrame:
    raw_ref = _pick_raw_table_with_report_phone(con)

    info = con.execute(f"PRAGMA table_info('{feat_table}')").fetchall()
    exclude = {"report", "phone", "Spam", "label", "Score"}

    numeric_types = (
        "INT", "INTEGER", "BIGINT", "HUGEINT",
        "DOUBLE", "FLOAT", "REAL", "DECIMAL",
        "UBIGINT", "UINTEGER", "USMALLINT", "SMALLINT",
        "TINYINT", "UTINYINT"
    )

    sum_cols = []
    for _, col, typ, *_ in info:
        if col in exclude:
            continue
        if typ is None:
            continue
        t = str(typ).upper()
        if any(nt in t for nt in numeric_types):
            sum_cols.append(col)

    if not sum_cols:
        raise RuntimeError("No numeric columns available to SUM in final_features (after exclusions).")

    sum_expr = ",\n        ".join([f'SUM(COALESCE(f."{c}", 0)) AS "{c}"' for c in sum_cols])

    sql = f"""
    WITH map AS (
        SELECT DISTINCT
            report::VARCHAR AS report,
            phone::VARCHAR  AS phone
        FROM {raw_ref}
        WHERE report IS NOT NULL AND phone IS NOT NULL
    )
    SELECT
        m.phone AS phone,
        {sum_expr}
    FROM "{feat_table}" f
    JOIN map m USING (report)
    GROUP BY m.phone
    ORDER BY m.phone
    """
    return con.execute(sql).df()
# ----------------------------------------------------------------------


def predict(
    load_data: str,
    load_model: str,
    return_output: str,
    *,
    feature_cols: Optional[str] = None,
    debug_csv: Optional[str] = None,
    external_yaml: Optional[str] = None,
    ext: Optional[str] = None,
    threshold: float = 0.5,
    agg_by_phone: bool = True,
) -> List[Dict[str, Any]]:
    """
    Predict pipeline:
      - build_features_duckdb
      - (optional) aggregate by phone
      - align feature columns
      - predict_proba -> Score, Spam
      - write JSON output
      - (optional) write debug CSV

    Returns: output list (same JSON schema as written to file)
    """

    in_path = load_data
    _ext = ext or infer_ext(in_path)

    if external_yaml:
        _external_yaml = external_yaml
    else:
        cfg = load_config()
        _external_yaml = str(cfg.data.external_data_yaml)

    # 1) Build features (DuckDB)
    con = build_features_duckdb(in_path, _external_yaml, ext=_ext, out_parquet=None)

    # 2) Get df_pool (phone-level or report-level)
    if agg_by_phone:
        df_pool = aggregate_final_features_by_phone(con, feat_table="final_features")
        key_col = "phone"
    else:
        df_pool = con.execute("SELECT * FROM final_features").df()
        key_col = "report"
        if "report" not in df_pool.columns:
            raise ValueError("final_features does not contain column 'report'.")

    if key_col not in df_pool.columns:
        raise ValueError(f"df_pool does not contain key column '{key_col}'.")

    # 3) Prepare X (match training drops: phone + Score, and drop label/Spam if present)
    drop_cols = [c for c in [key_col, "report", "phone", "Spam", "label", "Score"] if c in df_pool.columns]
    X_df = df_pool.drop(columns=drop_cols, errors="ignore")

    # 4) Align feature columns to training schema
    feature_cols_path = resolve_feature_cols_path(feature_cols, load_model)
    feature_cols_list = load_feature_cols(feature_cols_path) if feature_cols_path else None
    if feature_cols_list:
        X_df = align_features(X_df, feature_cols_list)

    # 5) Convert to numeric (keep NaN) + numpy float32
    X_df = to_numeric_keep_nan(X_df)
    X = X_df.to_numpy(dtype=np.float32, copy=False)

    # 6) Load model + predict (score = proba[:,1], class from threshold)
    model = joblib.load(load_model)
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model does not expose predict_proba(). Make sure you loaded an XGBClassifier.")

    y_score = model.predict_proba(X)[:, 1]
    thr = float(threshold)
    y_pred = (y_score >= thr).astype(int)

    # 7) Debug file
    df_pool["label"] = y_pred
    df_pool["Score"] = y_score
    if debug_csv:
        os.makedirs(os.path.dirname(debug_csv), exist_ok=True)
        df_pool.to_csv(debug_csv, index=False)

    # 8) Output JSON
    id_list = df_pool[key_col].astype(str).tolist()
    id_name = "phone" if key_col == "phone" else "report"
    output = [
        {id_name: _id, "Spam": int(pred), "Score": float(score)}
        for _id, pred, score in zip(id_list, y_pred.tolist(), y_score.tolist())
    ]

    os.makedirs(os.path.dirname(return_output), exist_ok=True)
    with open(return_output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("✅ Wrote:", return_output)
    print("🔎 Sample output:", output[:5])
    return output
