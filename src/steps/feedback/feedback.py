from __future__ import annotations

import numpy as np
import pandas as pd


class ProcessPhoneReport:
    """Aggregate user feedback by phone number."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def clean_data(self) -> pd.DataFrame:
        data = pd.read_parquet(self.data_path).drop_duplicates()
        data = data[pd.to_numeric(data["device_report"], errors="coerce").isin([1, 2])]
        data = data[data["in_contact"].isin(["no", "\\N"])]
        return data

    def aggregate_data(self) -> pd.DataFrame:
        df = self.clean_data().copy()
        df["device_report"] = pd.to_numeric(df["device_report"], errors="coerce")

        if "member_id" in df.columns:
            df["member_id"] = df["member_id"].astype(str)
            df.loc[df["member_id"].isin(["nan", "None", "NaT", ""]), "member_id"] = np.nan

        summary = df["phone"].value_counts().rename_axis("phone").reset_index(name="total_reports")

        spam_counts = df[df["device_report"] == 1]["phone"].value_counts()
        ham_counts = df[df["device_report"] == 2]["phone"].value_counts()

        summary["spam_reports"] = summary["phone"].map(spam_counts).fillna(0).astype(int)
        summary["not_spam_reports"] = summary["phone"].map(ham_counts).fillna(0).astype(int)

        spam_members = (
            df[df["device_report"] == 1]
            .groupby("phone")["member_id"]
            .nunique(dropna=True)
        )
        ham_members = (
            df[df["device_report"] == 2]
            .groupby("phone")["member_id"]
            .nunique(dropna=True)
        )

        summary["unique_spam_members"] = summary["phone"].map(spam_members).fillna(0).astype(int)
        summary["unique_ham_members"] = summary["phone"].map(ham_members).fillna(0).astype(int)
        return summary


# Backward compatible alias
processPhoneReport = ProcessPhoneReport
