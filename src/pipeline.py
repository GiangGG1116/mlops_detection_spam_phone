from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .core.config import PipelineConfig, load_config
from .core.paths import ensure_dir, ensure_parent
from .core.utils import normalize_phone
from .steps.dataset.make_dataset import (
    export_last_n_days_call_histories,
    export_last_n_days_report,
)
from .steps.feedback.feedback import ProcessPhoneReport
from .steps.inference.predict import predict
from .steps.labeling.label_delta import export_rows_with_changed_label_only
from .steps.labeling.labeling import label_3class_user_feedback
from .steps.training.train import (
    promote_model_if_better,
    train_xgb_for_your_schema,
    tune_xgb_params,
)


def _load_threshold(metrics_path: Path, default: float = 0.5) -> float:
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


class SpamDetectionPipeline:
    def __init__(self, config: PipelineConfig | None = None, run_id: str | None = None):
        self.config = config or load_config()
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._prepare_dirs()

    def _prepare_dirs(self) -> None:
        ensure_dir(self.config.model.candidates_dir)
        ensure_dir(self.config.model.archive_dir)
        ensure_dir(self.config.model.production_model.parent)

        ensure_parent(self.config.data.predictions_json)
        ensure_parent(self.config.data.prediction_debug_csv)
        ensure_parent(self.config.data.feedback_label_csv)
        ensure_parent(self.config.data.changed_rows_csv)
        ensure_parent(self.config.data.merged_train_dataset)
        ensure_parent(self.config.model.best_params_json)

    def run_inference(self, history_source: str | None = None) -> dict:
        history_input = history_source or str(self.config.data.history_source)
        exported_history = export_last_n_days_call_histories(
            in_path=history_input,
            out_dir=str(self.config.data.history_source.parent),
            days=self.config.settings.history_days,
            tz=self.config.settings.timezone,
        )

        threshold = _load_threshold(self.config.model.production_metrics)
        predict(
            load_data=str(exported_history),
            load_model=str(self.config.model.production_model),
            return_output=str(self.config.data.predictions_json),
            feature_cols=str(self.config.model.production_feature_columns),
            debug_csv=str(self.config.data.prediction_debug_csv),
            external_yaml=str(self.config.data.external_data_yaml),
            threshold=threshold,
            agg_by_phone=True,
        )

        return {
            "history_export": str(exported_history),
            "predictions_json": str(self.config.data.predictions_json),
            "prediction_debug_csv": str(self.config.data.prediction_debug_csv),
            "threshold_used": threshold,
        }

    def run_feedback_labeling(self, report_source: str | None = None) -> dict:
        report_input = report_source or str(self.config.data.report_source)
        exported_report = export_last_n_days_report(
            in_path=report_input,
            out_dir=str(self.config.data.report_source.parent),
            days=self.config.settings.report_days,
            tz=self.config.settings.timezone,
        )

        summary = ProcessPhoneReport(str(exported_report)).aggregate_data()
        labeled = label_3class_user_feedback(summary)
        labeled.to_csv(self.config.data.feedback_label_csv, index=False)

        return {
            "report_export": str(exported_report),
            "feedback_label_csv": str(self.config.data.feedback_label_csv),
            "rows": int(len(labeled)),
        }

    def build_training_dataset(self) -> dict:
        changed_path, changed_df = export_rows_with_changed_label_only(
            file1_path=str(self.config.data.feedback_label_csv),
            file2_path=str(self.config.data.prediction_debug_csv),
            out_path=str(self.config.data.changed_rows_csv),
        )

        base_path = self.config.data.base_train_dataset
        if base_path.exists():
            base_df = pd.read_csv(base_path, dtype={"phone": str})
        else:
            base_df = pd.DataFrame()

        changed_df = changed_df.copy()
        if not changed_df.empty:
            changed_df["phone"] = changed_df["phone"].astype(str)
            changed_df["_phone_key"] = changed_df["phone"].map(normalize_phone)

        if not base_df.empty and "phone" in base_df.columns:
            base_df["phone"] = base_df["phone"].astype(str)
            base_df["_phone_key"] = base_df["phone"].map(normalize_phone)

        if base_df.empty and changed_df.empty:
            raise ValueError(
                "No training data available. Both base train dataset and changed rows are empty."
            )

        if base_df.empty:
            merged_df = changed_df.copy()
        elif changed_df.empty:
            merged_df = base_df.copy()
        elif "_phone_key" in base_df.columns and "_phone_key" in changed_df.columns:
            merged_df = (
                pd.concat([changed_df, base_df], ignore_index=True)
                .drop_duplicates(subset=["_phone_key"], keep="first")
            )
        else:
            merged_df = pd.concat([changed_df, base_df], ignore_index=True).drop_duplicates()

        merged_df = merged_df.drop(columns=["_phone_key"], errors="ignore")
        merged_df.to_csv(self.config.data.merged_train_dataset, index=False)

        return {
            "changed_rows_csv": str(changed_path),
            "changed_rows": int(len(changed_df)),
            "train_dataset_path": str(self.config.data.merged_train_dataset),
            "train_rows": int(len(merged_df)),
        }

    def train_and_promote(self, train_data_path: str | None = None) -> dict:
        dataset_path = Path(train_data_path) if train_data_path else self.config.data.merged_train_dataset
        if not dataset_path.exists():
            dataset_path = self.config.data.base_train_dataset

        tune_result = tune_xgb_params(
            data_path=str(dataset_path),
            n_trials=self.config.settings.tune_trials,
            timeout=self.config.settings.tune_timeout_seconds,
            save_best_params=str(self.config.model.best_params_json),
        )

        best_params = tune_result["best_params"]
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate_dir = self.config.model.candidates_dir / run_id
        ensure_dir(candidate_dir)

        cand_model = candidate_dir / "xgb.pkl"
        cand_cols = candidate_dir / "feature_columns.json"
        cand_metrics = candidate_dir / "train_metrics.json"

        train_result = train_xgb_for_your_schema(
            data_path=str(dataset_path),
            out_model=str(cand_model),
            out_feature_cols=str(cand_cols),
            out_metrics=str(cand_metrics),
            n_estimators=best_params.get("n_estimators", 3000),
            max_depth=best_params.get("max_depth", 6),
            learning_rate=best_params.get("learning_rate", 0.05),
            subsample=best_params.get("subsample", 0.8),
            colsample_bytree=best_params.get("colsample_bytree", 0.8),
            min_child_weight=best_params.get("min_child_weight", 1.0),
            gamma=best_params.get("gamma", 0.0),
            reg_alpha=best_params.get("reg_alpha", 0.0),
            reg_lambda=best_params.get("reg_lambda", 1.0),
            max_bin=best_params.get("max_bin", 256),
        )

        promote_result = promote_model_if_better(
            candidate_model=str(cand_model),
            candidate_feature_cols=str(cand_cols),
            candidate_metrics=str(cand_metrics),
            prod_model=str(self.config.model.production_model),
            prod_feature_cols=str(self.config.model.production_feature_columns),
            prod_metrics=str(self.config.model.production_metrics),
            archive_dir=str(self.config.model.archive_dir),
            min_delta=self.config.settings.min_delta,
        )

        return {
            "train_data": str(dataset_path),
            "candidate_dir": str(candidate_dir),
            "tune": tune_result,
            "train": train_result,
            "promote": promote_result,
        }

    def run_full(
        self,
        report_source: str | None = None,
        history_source: str | None = None,
    ) -> dict:
        infer_result = self.run_inference(history_source=history_source)
        feedback_result = self.run_feedback_labeling(report_source=report_source)
        dataset_result = self.build_training_dataset()
        train_result = self.train_and_promote(dataset_result["train_dataset_path"])

        return {
            "inference": infer_result,
            "feedback": feedback_result,
            "dataset": dataset_result,
            "training": train_result,
        }

    # ------------------------------------------------------------------
    # Run manifest
    # ------------------------------------------------------------------
    def emit_run_manifest(
        self,
        command: str,
        status: str = "success",
        result: dict | None = None,
        error: Exception | None = None,
    ) -> Path:
        """Write a JSON run-manifest to data/runs/<run_id>.json and return path."""
        manifest_dir = self.config.data.run_manifest_dir
        ensure_dir(manifest_dir)
        manifest_path = manifest_dir / f"{self.run_id}_{command}.json"

        payload: dict = {
            "run_id": self.run_id,
            "command": command,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = {"type": type(error).__name__, "message": str(error)}

        ensure_parent(manifest_path)
        with manifest_path.open("w", encoding="utf-8") as fh:
            import json
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        return manifest_path
