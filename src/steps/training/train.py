from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ...core.utils import infer_ext, to_float, to_int

import xgboost as xgb
import joblib


# =========================
# IO helpers
# =========================


def _load_table(path: str) -> pd.DataFrame:
    ext = infer_ext(path)
    if ext == "parquet": return pd.read_parquet(path)
    if ext == "csv":     return pd.read_csv(path)
    if ext == "jsonl":   return pd.read_json(path, lines=True)
    if ext == "json":    return pd.read_json(path)
    raise ValueError(f"Unsupported file: {path}")


def _save_json(path: str, obj: Any) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return 1.0
    return max(neg / pos, 1.0)


# =========================
# Threshold selection (best F1)
# =========================
def _find_best_threshold_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    step: float = 0.01,
    min_t: float = 0.01,
    max_t: float = 0.99,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # If validation has only one class, threshold optimization is not possible
    if len(np.unique(y_true)) < 2:
        t = 0.5
        pred = (y_score >= t).astype(int)
        return {
            "threshold": float(t),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
            "step": float(step),
        }

    thresholds = np.arange(min_t, max_t + 1e-12, step, dtype=float)

    best = {
        "threshold": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
        "confusion_matrix": None,
        "step": float(step),
    }

    for t in thresholds:
        pred = (y_score >= t).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best["f1"]:
            best["threshold"] = float(t)
            best["f1"] = f1
            best["precision"] = float(precision_score(y_true, pred, zero_division=0))
            best["recall"] = float(recall_score(y_true, pred, zero_division=0))
            best["accuracy"] = float(accuracy_score(y_true, pred))
            best["confusion_matrix"] = confusion_matrix(y_true, pred).tolist()

    return best


# =========================
# Hyperparameter tuning
# =========================
def tune_xgb_params(
    data_path: str,
    label_col: str = "label",
    drop_cols: List[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_trials: int = 5,
    timeout: int | None = None,
    n_jobs: int = 4,
    save_best_params: str | None = None,
) -> Dict[str, Any]:
    df = _load_table(data_path)

    base_drop = ["phone", "Score"]
    if drop_cols:
        base_drop += [c for c in drop_cols if c not in base_drop]
    base_drop = [c for c in base_drop if c in df.columns]
    if base_drop:
        df = df.drop(columns=base_drop)

    if label_col not in df.columns:
        raise ValueError(f"Không thấy cột '{label_col}'. Columns={list(df.columns)[:30]}...")

    y = pd.to_numeric(df[label_col], errors="coerce")
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int).to_numpy()

    X = df.drop(columns=[label_col]).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.select_dtypes(include=["number"]).astype(np.float32)
    feature_cols = list(X.columns)
    if len(feature_cols) == 0:
        raise ValueError("Không còn feature numeric nào sau khi drop/convert.")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    # Inside tune_xgb_params, we only care about train and val for tuning.
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=random_state, stratify=y_temp
    )

    X_tr_np = X_tr.to_numpy(dtype=np.float32, copy=False)
    X_va_np = X_va.to_numpy(dtype=np.float32, copy=False)
    y_tr_np = y_tr.astype(np.int32, copy=False)
    y_va_np = y_va.astype(np.int32, copy=False)

    spw = _scale_pos_weight(y_tr_np)

    def _build_model(params: Dict[str, Any]) -> xgb.XGBClassifier:
        fixed = dict(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
            scale_pos_weight=spw,
        )
        return xgb.XGBClassifier(**{**fixed, **params})

    def _suggest_params(trial_like) -> Dict[str, Any]:
        return {
            "n_estimators": trial_like.suggest_int("n_estimators", 800, 3500, step=100),
            "max_depth": trial_like.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial_like.suggest_float("min_child_weight", 1.0, 12.0),
            "learning_rate": trial_like.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial_like.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial_like.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial_like.suggest_float("gamma", 0.0, 8.0),
            "reg_alpha": trial_like.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial_like.suggest_float("reg_lambda", 0.5, 5.0),
            "max_bin": trial_like.suggest_int("max_bin", 128, 512, step=64),
        }

    best_params: Dict[str, Any] | None = None
    best_auc: float = -1.0
    used_optuna = False

    try:
        import optuna
        from optuna.pruners import MedianPruner

        def objective(trial: "optuna.Trial") -> float:
            params = _suggest_params(trial)
            model = _build_model(params)
            model.fit(
                X_tr_np, y_tr_np,
                eval_set=[(X_va_np, y_va_np)],
                early_stopping_rounds=80,
                verbose=False,
            )
            proba = model.predict_proba(X_va_np)[:, 1]
            return float(roc_auc_score(y_va_np, proba)) if len(np.unique(y_va_np)) > 1 else 0.0

        study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=3))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1, gc_after_trial=True)

        best_params = dict(study.best_params)
        best_auc = float(study.best_value)
        used_optuna = True

    except Exception:
        class _DummyTrial:
            rng = np.random.default_rng(2025)
            def suggest_int(self, name, low, high, step=1):
                vals = list(range(low, high + 1, step))
                return int(self.rng.choice(vals))
            def suggest_float(self, name, low, high, log=False):
                if log:
                    u = self.rng.random()
                    return float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
                return float(self.rng.uniform(low, high))

        for _ in range(int(n_trials)):
            t = _DummyTrial()
            params = _suggest_params(t)
            try:
                model = _build_model(params)
                model.fit(
                    X_tr_np, y_tr_np,
                    eval_set=[(X_va_np, y_va_np)],
                    early_stopping_rounds=80,
                    verbose=False,
                )
                proba = model.predict_proba(X_va_np)[:, 1]
                auc = float(roc_auc_score(y_va_np, proba)) if len(np.unique(y_va_np)) > 1 else 0.0
            except Exception:
                auc = -1.0

            if auc > best_auc:
                best_auc = auc
                best_params = params

    if best_params is None:
        raise RuntimeError("Tuning thất bại: không tìm được bộ tham số hợp lệ.")

    result = {
        "data_path": data_path,
        "label_col": label_col,
        "dropped_cols": base_drop,
        "n_features": int(len(feature_cols)),
        "scale_pos_weight_used": float(spw),
        "used_optuna": bool(used_optuna),
        "best_auc": float(best_auc),
        "best_params": best_params,
    }

    if save_best_params:
        _save_json(save_best_params, result)

    return result


# =========================
# Train (save best threshold)
# =========================
def train_xgb_for_your_schema(
    data_path: str,
    out_model: str,
    out_feature_cols: str,
    out_metrics: str,
    label_col: str = "label",
    drop_cols: List[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 3000,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    n_jobs: int = 4,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    max_bin: int = 256,
) -> Dict[str, Any]:
    df = _load_table(data_path)

    base_drop = ["phone", "Score"]
    if drop_cols:
        base_drop += [c for c in drop_cols if c not in base_drop]
    base_drop = [c for c in base_drop if c in df.columns]
    if base_drop:
        df = df.drop(columns=base_drop)

    if label_col not in df.columns:
        raise ValueError(f"Không thấy cột '{label_col}'. Columns={list(df.columns)[:30]}...")

    y = pd.to_numeric(df[label_col], errors="coerce")
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int).to_numpy()

    X = df.drop(columns=[label_col]).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.select_dtypes(include=["number"]).astype(np.float32)
    feature_cols = list(X.columns)
    if len(feature_cols) == 0:
        raise ValueError("Không còn feature numeric nào sau khi drop/convert.")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=random_state, stratify=y_temp
    )

    X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
    X_val_np   = X_val.to_numpy(dtype=np.float32, copy=False)
    X_test_np  = X_test.to_numpy(dtype=np.float32, copy=False)
    y_train_np = y_train.astype(np.int32, copy=False)
    y_val_np   = y_val.astype(np.int32, copy=False)
    y_test_np  = y_test.astype(np.int32, copy=False)

    spw = _scale_pos_weight(y_train_np)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        max_bin=max_bin,
        n_jobs=n_jobs,
        scale_pos_weight=spw,
        random_state=random_state,
    )

    model.fit(
        X_train_np, y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        early_stopping_rounds=50,
        verbose=False
    )

    proba_test = model.predict_proba(X_test_np)[:, 1]

    # Best threshold by F1 on validation set, but evaluate on test set
    proba_val = model.predict_proba(X_val_np)[:, 1]
    best_thr = _find_best_threshold_f1(y_val_np, proba_val, step=0.01)
    thr_best = float(best_thr["threshold"])

    pred_05   = (proba_test >= 0.5).astype(int)
    pred_best = (proba_test >= thr_best).astype(int)

    metrics = {
        "data_path": data_path,
        "label_col": label_col,
        "dropped_cols": base_drop,
        "n_rows_used": int(len(X)),
        "n_features": int(len(feature_cols)),
        "scale_pos_weight": float(spw),

        "auc_val": float(roc_auc_score(y_test_np, proba_test)) if len(np.unique(y_test_np)) > 1 else None,
        "best_iteration": int(getattr(model, "best_iteration", -1)),

        # --- metrics @ 0.5 ---
        "threshold@0.5": 0.5,
        "accuracy@0.5": float(accuracy_score(y_test_np, pred_05)),
        "precision@0.5": float(precision_score(y_test_np, pred_05, zero_division=0)),
        "recall@0.5": float(recall_score(y_test_np, pred_05, zero_division=0)),
        "f1@0.5": float(f1_score(y_test_np, pred_05, zero_division=0)),
        "confusion_matrix@0.5": confusion_matrix(y_test_np, pred_05).tolist(),

        # --- metrics @ best threshold (applied on test set) ---
        "threshold_best": thr_best,
        "threshold_search_step": float(best_thr["step"]),
        "accuracy@best": float(accuracy_score(y_test_np, pred_best)),
        "precision@best": float(precision_score(y_test_np, pred_best, zero_division=0)),
        "recall@best": float(recall_score(y_test_np, pred_best, zero_division=0)),
        "f1@best": float(f1_score(y_test_np, pred_best, zero_division=0)),
        "confusion_matrix@best": confusion_matrix(y_test_np, pred_best).tolist(),
    }

    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    joblib.dump(model, out_model)
    _save_json(out_feature_cols, {"cols": feature_cols})
    _save_json(out_metrics, metrics)

    return {
        "model_path": out_model,
        "feature_cols_path": out_feature_cols,
        "metrics_path": out_metrics,
        "n_features": len(feature_cols),
        "threshold_best": thr_best,
        "top_features_hint": feature_cols[:10],
    }


# =========================
# Promote if better
# =========================
def _get_metric(m: dict | None, key: str, default=None):
    if not isinstance(m, dict):
        return default
    v = m.get(key, default)
    if v is None:
        return default
    try:
        if isinstance(v, float) and np.isnan(v):
            return default
    except Exception:
        pass
    return v


def should_promote(
    new_metrics: dict,
    old_metrics: dict | None,
    primary: str = "auc_val",
    secondary: str = "f1@best",   # compare using the best threshold
    min_delta: float = 1e-6,
) -> tuple[bool, dict]:
    if old_metrics is None:
        return True, {"reason": "no_old_metrics"}

    new_p = _get_metric(new_metrics, primary, None)
    old_p = _get_metric(old_metrics, primary, None)

    if new_p is None or old_p is None:
        new_s = _get_metric(new_metrics, secondary, None)
        old_s = _get_metric(old_metrics, secondary, None)
        if old_s is None and new_s is not None:
            return True, {"reason": f"old_{secondary}_missing_new_has"}
        if new_s is None:
            return False, {"reason": f"new_{secondary}_missing"}
        if old_s is None:
            return True, {"reason": f"old_{secondary}_missing"}
        if float(new_s) > float(old_s) + min_delta:
            return True, {"reason": f"better_{secondary}", "new": float(new_s), "old": float(old_s)}
        return False, {"reason": f"not_better_{secondary}", "new": float(new_s), "old": float(old_s)}

    if float(new_p) > float(old_p) + min_delta:
        return True, {"reason": f"better_{primary}", "new": float(new_p), "old": float(old_p)}

    if abs(float(new_p) - float(old_p)) <= min_delta:
        new_s = _get_metric(new_metrics, secondary, None)
        old_s = _get_metric(old_metrics, secondary, None)
        if new_s is not None and old_s is not None and float(new_s) > float(old_s) + min_delta:
            return True, {"reason": f"tie_{primary}_better_{secondary}", "new": float(new_s), "old": float(old_s)}

    return False, {"reason": f"not_better_{primary}", "new": float(new_p), "old": float(old_p)}


def _atomic_copy_replace(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def promote_model_if_better(
    candidate_model: str,
    candidate_feature_cols: str,
    candidate_metrics: str,
    prod_model: str,
    prod_feature_cols: str,
    prod_metrics: str,
    archive_dir: str | None = None,
    min_delta: float = 1e-6,
) -> dict:
    new_m = _read_json(candidate_metrics)
    if new_m is None:
        raise RuntimeError(f"Candidate metrics not found: {candidate_metrics}")

    old_m = _read_json(prod_metrics)

    ok, why = should_promote(new_m, old_m, min_delta=min_delta)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "promoted": bool(ok),
        "why": why,
        "candidate": {"model": candidate_model, "feature_cols": candidate_feature_cols, "metrics": candidate_metrics},
        "current": {"model": prod_model, "feature_cols": prod_feature_cols, "metrics": prod_metrics},
        "timestamp": ts,
    }

    if not ok:
        return result

    # backup old bundle
    if archive_dir:
        os.makedirs(archive_dir, exist_ok=True)
        bdir = os.path.join(archive_dir, ts)
        os.makedirs(bdir, exist_ok=True)

        if os.path.exists(prod_model):
            shutil.copy2(prod_model, os.path.join(bdir, os.path.basename(prod_model)))
        if os.path.exists(prod_feature_cols):
            shutil.copy2(prod_feature_cols, os.path.join(bdir, os.path.basename(prod_feature_cols)))
        if os.path.exists(prod_metrics):
            shutil.copy2(prod_metrics, os.path.join(bdir, os.path.basename(prod_metrics)))

        result["backup_dir"] = bdir

    # atomic copy -> replace (keep candidate artifacts for traceability)
    _atomic_copy_replace(candidate_model, prod_model)
    _atomic_copy_replace(candidate_feature_cols, prod_feature_cols)
    _atomic_copy_replace(candidate_metrics, prod_metrics)

    return result



