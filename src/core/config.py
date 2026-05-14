from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT, resolve_path

DEFAULT_CONFIG_FILE = PROJECT_ROOT / "configs/pipeline.yaml"


@dataclass(frozen=True)
class DataConfig:
    report_source: Path
    history_source: Path
    external_data_yaml: Path
    predictions_json: Path
    prediction_debug_csv: Path
    feedback_label_csv: Path
    changed_rows_csv: Path
    base_train_dataset: Path
    merged_train_dataset: Path
    run_manifest_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    production_model: Path
    production_feature_columns: Path
    production_metrics: Path
    candidates_dir: Path
    archive_dir: Path
    best_params_json: Path


@dataclass(frozen=True)
class PipelineSettings:
    report_days: int = 100
    history_days: int = 90
    timezone: str = "Asia/Ho_Chi_Minh"
    tune_trials: int = 5
    tune_timeout_seconds: int = 1800
    min_delta: float = 1e-6
    min_train_rows: int = 10
    min_auc_to_promote: float = 0.5


@dataclass(frozen=True)
class MlflowConfig:
    enabled: bool = True
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "spam_phone_detection"
    log_artifacts: bool = False


@dataclass(frozen=True)
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig
    model: ModelConfig
    settings: PipelineSettings
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} must be a YAML object.")
    return payload


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    config_file = resolve_path(
        config_path or os.getenv("PIPELINE_CONFIG", str(DEFAULT_CONFIG_FILE))
    )
    raw = _read_yaml(config_file)

    data_raw = raw.get("data") or {}
    model_raw = raw.get("model") or {}
    settings_raw = raw.get("settings") or {}
    mlflow_raw = raw.get("mlflow") or {}
    api_raw = raw.get("api") or {}

    data = DataConfig(
        report_source=resolve_path(
            data_raw.get(
                "report_source",
                "data/input/phone_reports_processed(1).parquet",
            )
        ),
        history_source=resolve_path(
            data_raw.get(
                "history_source",
                "data/input/call_histories_last_120d_2025-09-14_to_2026-01-11.parquet",
            )
        ),
        external_data_yaml=resolve_path(
            data_raw.get("external_data_yaml", "configs/external_data.yml")
        ),
        predictions_json=resolve_path(
            data_raw.get("predictions_json", "data/predictions/pred_by_phone.json")
        ),
        prediction_debug_csv=resolve_path(
            data_raw.get("prediction_debug_csv", "data/predictions/predict_debug.csv")
        ),
        feedback_label_csv=resolve_path(
            data_raw.get("feedback_label_csv", "data/train/feedback_labels.csv")
        ),
        changed_rows_csv=resolve_path(
            data_raw.get("changed_rows_csv", "data/train/changed_rows.csv")
        ),
        base_train_dataset=resolve_path(
            data_raw.get("base_train_dataset", "data/train/label.csv")
        ),
        merged_train_dataset=resolve_path(
            data_raw.get("merged_train_dataset", "data/train/label_merged.csv")
        ),
        run_manifest_dir=resolve_path(
            data_raw.get("run_manifest_dir", "data/runs")
        ),
    )

    model = ModelConfig(
        production_model=resolve_path(
            model_raw.get("production_model", "models/production/xgb.pkl")
        ),
        production_feature_columns=resolve_path(
            model_raw.get(
                "production_feature_columns",
                "models/production/feature_columns.json",
            )
        ),
        production_metrics=resolve_path(
            model_raw.get("production_metrics", "models/production/train_metrics.json")
        ),
        candidates_dir=resolve_path(
            model_raw.get("candidates_dir", "models/candidates")
        ),
        archive_dir=resolve_path(model_raw.get("archive_dir", "models/archive")),
        best_params_json=resolve_path(
            model_raw.get("best_params_json", "models/best_xgb_params.json")
        ),
    )

    settings = PipelineSettings(
        report_days=_to_int(settings_raw.get("report_days"), 100),
        history_days=_to_int(settings_raw.get("history_days"), 90),
        timezone=str(settings_raw.get("timezone", "Asia/Ho_Chi_Minh")),
        tune_trials=_to_int(settings_raw.get("tune_trials"), 5),
        tune_timeout_seconds=_to_int(settings_raw.get("tune_timeout_seconds"), 1800),
        min_delta=_to_float(settings_raw.get("min_delta"), 1e-6),
        min_train_rows=_to_int(settings_raw.get("min_train_rows"), 10),
        min_auc_to_promote=_to_float(settings_raw.get("min_auc_to_promote"), 0.5),
    )

    mlflow_cfg = MlflowConfig(
        enabled=_to_bool(mlflow_raw.get("enabled"), True),
        tracking_uri=str(mlflow_raw.get("tracking_uri", "http://localhost:5000")),
        experiment_name=str(
            mlflow_raw.get("experiment_name", "spam_phone_detection")
        ),
        log_artifacts=_to_bool(mlflow_raw.get("log_artifacts"), False),
    )

    api_cfg = ApiConfig(
        host=str(api_raw.get("host", "0.0.0.0")),
        port=_to_int(api_raw.get("port"), 8000),
        reload=_to_bool(api_raw.get("reload"), False),
    )

    return PipelineConfig(
        data=data,
        model=model,
        settings=settings,
        mlflow=mlflow_cfg,
        api=api_cfg,
    )
