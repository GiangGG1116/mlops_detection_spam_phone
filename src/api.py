from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core.config import PipelineConfig, load_config
from .core.logging import get_logger
from .core.paths import ensure_parent, resolve_path
from .core.utils import load_threshold


def _new_api_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]


class PredictFileRequest(BaseModel):
    history_path: str = Field(..., description="Path to input call-history file")
    output_path: str | None = Field(
        default=None,
        description="Optional output JSON path. Defaults to config.data.predictions_json",
    )
    debug_csv_path: str | None = Field(
        default=None,
        description="Optional debug CSV path. Defaults to config.data.prediction_debug_csv",
    )
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    agg_by_phone: bool = True


class PipelineRunRequest(BaseModel):
    command: Literal["pipeline", "infer", "label", "dataset", "train"]
    run_id: str | None = None
    history_source: str | None = None
    report_source: str | None = None
    train_data: str | None = None


class PredictFileResponse(BaseModel):
    run_id: str
    output_path: str
    debug_csv_path: str
    threshold_used: float
    rows: int
    sample: list[dict]


def create_app(config: PipelineConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    logger = get_logger("spam_api")

    app = FastAPI(
        title="Spam Phone Detection API",
        version="1.0.0",
        description="Inference and pipeline control API for spam phone MLOps.",
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_exists": cfg.model.production_model.exists(),
            "feature_columns_exists": cfg.model.production_feature_columns.exists(),
            "metrics_exists": cfg.model.production_metrics.exists(),
        }

    class RiskEvaluationRequest(BaseModel):
        phone: str
        model_score: float
        features: dict[str, Any]

    @app.post("/evaluate/risk")
    def evaluate_risk_endpoint(req: RiskEvaluationRequest):
        from src.steps.risk.risk_evaluator import evaluate_phone_risk, load_risk_config
        # We assume cfg is available globally or we reload risk config, let's just use load_risk_config()
        risk_cfg = load_risk_config()
        result = evaluate_phone_risk(req.phone, req.model_score, req.features, risk_cfg)
        return {
            "status": "success",
            "risk_evaluation": result
        }

    @app.post("/predict/file", response_model=PredictFileResponse)
    def predict_file(req: PredictFileRequest) -> PredictFileResponse:
        history_path = resolve_path(req.history_path)
        if not history_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Input history file does not exist: {history_path}",
            )

        if not cfg.model.production_model.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Production model not found: {cfg.model.production_model}",
            )

        if not cfg.model.production_feature_columns.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Production feature columns not found: "
                    f"{cfg.model.production_feature_columns}"
                ),
            )

        run_id = _new_api_run_id()
        output_path = resolve_path(req.output_path or str(cfg.data.predictions_json))
        debug_csv_path = resolve_path(
            req.debug_csv_path or str(cfg.data.prediction_debug_csv)
        )
        ensure_parent(output_path)
        ensure_parent(debug_csv_path)

        threshold = (
            float(req.threshold)
            if req.threshold is not None
            else load_threshold(cfg.model.production_metrics)
        )

        logger.info(
            "api_predict_start",
            extra={
                "extra": {
                    "run_id": run_id,
                    "history_path": str(history_path),
                    "output_path": str(output_path),
                }
            },
        )

        from .steps.inference.predict import predict as run_predict

        run_predict(
            load_data=str(history_path),
            load_model=str(cfg.model.production_model),
            return_output=str(output_path),
            feature_cols=str(cfg.model.production_feature_columns),
            debug_csv=str(debug_csv_path),
            external_yaml=str(cfg.data.external_data_yaml),
            threshold=threshold,
            agg_by_phone=req.agg_by_phone,
        )

        with output_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            raise HTTPException(
                status_code=500,
                detail="Prediction output must be a JSON array.",
            )

        logger.info(
            "api_predict_done",
            extra={"extra": {"run_id": run_id, "rows": len(payload)}},
        )

        return PredictFileResponse(
            run_id=run_id,
            output_path=str(output_path),
            debug_csv_path=str(debug_csv_path),
            threshold_used=threshold,
            rows=len(payload),
            sample=payload[:5],
        )

    @app.post("/pipeline/run")
    def run_pipeline(req: PipelineRunRequest) -> dict:
        from .pipeline import SpamDetectionPipeline

        run_id = req.run_id or _new_api_run_id()
        pipeline = SpamDetectionPipeline(cfg, run_id=run_id)

        try:
            if req.command == "pipeline":
                result = pipeline.run_full(
                    report_source=req.report_source,
                    history_source=req.history_source,
                )
            elif req.command == "infer":
                result = pipeline.run_inference(history_source=req.history_source)
            elif req.command == "label":
                result = pipeline.run_feedback_labeling(report_source=req.report_source)
            elif req.command == "dataset":
                result = pipeline.build_training_dataset()
            elif req.command == "train":
                result = pipeline.train_and_promote(train_data_path=req.train_data)
            else:
                raise HTTPException(status_code=400, detail="Unsupported command")

            manifest = pipeline.emit_run_manifest(
                req.command,
                status="success",
                result=result,
            )
            result["run_manifest"] = str(manifest)
            return result

        except Exception as exc:
            manifest = pipeline.emit_run_manifest(req.command, status="failed", error=exc)
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "failed",
                    "run_id": run_id,
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    "run_manifest": str(manifest),
                },
            )

    return app


app = create_app()
