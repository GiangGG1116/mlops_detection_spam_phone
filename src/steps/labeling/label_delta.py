from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...core.utils import normalize_phone


def export_rows_with_changed_label_only(
    file1_path: str,
    file2_path: str,
    out_path: str,
    file1_phone_col: str = "phone",
    file1_label_col: str = "label",
    file2_phone_col: str = "phone",
    file2_label_col: str = "label",
) -> tuple[Path, pd.DataFrame]:
    """Export rows where label from file1 differs from file2.

    - `file1`: label source (feedback labels)
    - `file2`: feature pool to be relabeled
    """
    source = Path(file1_path)
    target = Path(file2_path)
    out = Path(out_path)

    df1 = pd.read_csv(source)
    df1[file1_label_col] = pd.to_numeric(df1[file1_label_col], errors="coerce")
    df1 = df1[df1[file1_label_col].isin([0, 1])].copy()
    df1["_phone_key"] = df1[file1_phone_col].map(normalize_phone)
    label_map = dict(zip(df1["_phone_key"], df1[file1_label_col].astype(int)))

    df2 = pd.read_csv(target)
    if file2_label_col not in df2.columns:
        df2[file2_label_col] = pd.NA

    df2["_phone_key"] = df2[file2_phone_col].map(normalize_phone)
    label_old = pd.to_numeric(df2[file2_label_col], errors="coerce")
    label_new = df2["_phone_key"].map(label_map)

    mask = label_new.notna() & (label_old != label_new)
    changed = df2.loc[mask].copy()

    if file2_label_col in changed.columns:
        changed = changed.drop(columns=[file2_label_col])

    changed.insert(1, file2_label_col, label_new[mask].astype(int).values)
    changed = changed.drop(columns=["_phone_key"], errors="ignore")

    out.parent.mkdir(parents=True, exist_ok=True)
    changed.to_csv(out, index=False)
    return out, changed
