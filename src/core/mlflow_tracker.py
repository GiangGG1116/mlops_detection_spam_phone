from __future__ import annotations

import numbers
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .config import MlflowConfig


class MLflowTracker:
    def __init__(self, config: MlflowConfig, run_id: str, logger):
        self.config = config
        self.run_id = run_id
        self.logger = logger
        self._mlflow = None
        self._active = False

        if not self.config.enabled:
            return

        try:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
        except Exception as exc:
            self.logger.warning(
                "mlflow_disabled",
                extra={"extra": {"run_id": run_id, "reason": str(exc)}},
            )
            self._mlflow = None

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    def start(self, run_name: str | None = None, tags: dict[str, Any] | None = None) -> None:
        if not self.enabled or self._active:
            return

        self._mlflow.start_run(run_name=run_name or self.run_id)
        self._active = True

        merged_tags = {"run_id": self.run_id}
        if tags:
            merged_tags.update({k: str(v) for k, v in tags.items()})
        self._mlflow.set_tags(merged_tags)

    def end(self, status: str = "FINISHED") -> None:
        if not self.enabled or not self._active:
            return
        self._mlflow.end_run(status=status)
        self._active = False

    def set_tags(self, tags: dict[str, Any]) -> None:
        if not self.enabled or not self._active:
            return
        self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_params(self, params: dict[str, Any], prefix: str = "") -> None:
        if not self.enabled or not self._active:
            return

        flat = _flatten_dict(params, prefix=prefix)
        for key, value in flat.items():
            if value is None:
                continue
            self._mlflow.log_param(key, _param_value(value))

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "", step: int = 0) -> None:
        if not self.enabled or not self._active:
            return

        flat = _flatten_dict(metrics, prefix=prefix)
        numeric_metrics = {
            key: float(value)
            for key, value in flat.items()
            if isinstance(value, numbers.Number)
        }
        if numeric_metrics:
            self._mlflow.log_metrics(numeric_metrics, step=step)

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        if not self.enabled or not self._active:
            return
        self._mlflow.log_dict(payload, artifact_file)

    def log_artifacts(self, path: str | Path) -> None:
        if not self.enabled or not self._active or not self.config.log_artifacts:
            return

        path_obj = Path(path)
        if path_obj.is_dir():
            self._mlflow.log_artifacts(str(path_obj))
        elif path_obj.exists():
            self._mlflow.log_artifact(str(path_obj))


def _param_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Number):
        return str(value)
    return str(value)


def _flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _walk(node: Any, key_prefix: str) -> None:
        if is_dataclass(node):
            node = asdict(node)

        if isinstance(node, dict):
            for key, value in node.items():
                child_key = f"{key_prefix}.{key}" if key_prefix else str(key)
                _walk(value, child_key)
            return

        if isinstance(node, (list, tuple)):
            flat[key_prefix] = ",".join(str(v) for v in node)
            return

        flat[key_prefix] = node

    _walk(payload, prefix)
    return flat
