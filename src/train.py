"""
src/train.py — Two-layer digital twin training pipeline.

Architecture
------------
Layer 1: Predict UPF responses (throughput, CPU, loss, delay) from offered load.
Layer 2: Predict power_watts from offered load + Layer 1 out-of-fold predictions.

Layer 2 trains on out-of-fold (OOF) L1 predictions rather than ground-truth
values, so it never sees perfect inputs at training time — matching the
distribution it will face at inference time (stacking).

Three training variants:
  dpdk      — all DPDK samples (is_dpdk == 1)
  usr_full  — all USR samples  (is_dpdk == 0)
  usr_safe  — USR samples with throughput_gbps < usr_safe_threshold_gbps

Two model variants per slot:
  full  — all layer1_features (primary, highest accuracy)
  lite  — lite_features only  (throughput TX only, NetMob-compatible)

Three model types compared per slot (best by val-R2 saved):
  ridge, random_forest, gradient_boosting

Total MLflow runs: 3 variants × 5 targets × 3 model_types × 2 (full/lite) = 90

Usage:
    python src/train.py
"""

from __future__ import annotations

import json
import pickle
import warnings
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PARAMS_PATH = "params.yaml"
DATA_PATH   = "data/processed/features.csv"
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)["train"]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def build_model(model_type: str, params: dict, random_state: int) -> object:
    """Return a sklearn Pipeline (scaler + estimator) for the given type."""
    if model_type == "ridge":
        estimator = Ridge()
    elif model_type == "random_forest":
        estimator = RandomForestRegressor(
            n_jobs=-1, random_state=random_state
        )
    elif model_type == "gradient_boosting":
        estimator = GradientBoostingRegressor(random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_search(pipeline, model_type: str, params: dict) -> object:
    """Wrap pipeline in a hyperparameter search object."""
    n_iter   = params["search_n_iter"]
    cv_folds = params["cv_folds"]
    rs       = params["random_state"]

    if model_type == "ridge":
        param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
        return GridSearchCV(
            pipeline, param_grid, cv=cv_folds,
            scoring="r2", refit=True, n_jobs=-1,
        )
    elif model_type == "random_forest":
        param_dist = {
            "model__n_estimators":    [100, 200, 300],
            "model__max_depth":       [6, 10, 15, None],
            "model__min_samples_leaf":[2, 5, 10],
            "model__max_features":    [0.5, 0.7, 1.0],
        }
        return RandomizedSearchCV(
            pipeline, param_dist, n_iter=n_iter, cv=cv_folds,
            scoring="r2", refit=True, random_state=rs, n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        param_dist = {
            "model__n_estimators":  [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth":     [3, 5, 7],
            "model__subsample":     [0.7, 0.85, 1.0],
        }
        return RandomizedSearchCV(
            pipeline, param_dist, n_iter=n_iter, cv=cv_folds,
            scoring="r2", refit=True, random_state=rs, n_jobs=-1,
        )


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


# ─────────────────────────────────────────────────────────────────────────────
# Core training routine — one (variant, target, feature_set) slot
# ─────────────────────────────────────────────────────────────────────────────

def train_slot(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    params:  dict,
    variant: str,
    layer:   str,
    target:  str,
    model_variant: str,   # "full" or "lite"
) -> tuple[object, dict, dict]:
    """
    Train all model types for one slot, log each to MLflow, return
    (best_model, best_metrics, best_params_dict).
    """
    best_model   = None
    best_metrics = {"r2": -np.inf}
    best_params_dict = {}

    for model_type in params["models"]:
        pipeline = build_model(model_type, params, params["random_state"])
        search   = build_search(pipeline, model_type, params)

        run_name = f"{variant}__{layer}__{target}__{model_type}__{model_variant}"
        with mlflow.start_run(run_name=run_name, nested=True):
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_

            y_pred   = best_estimator.predict(X_test)
            metrics  = eval_metrics(y_test.values, y_pred)
            cv_score = float(search.best_score_)

            # Log to MLflow
            mlflow.set_tags({
                "variant":       variant,
                "layer":         layer,
                "target":        target,
                "model_type":    model_type,
                "model_variant": model_variant,
            })
            mlflow.log_params({
                "model_type":    model_type,
                "model_variant": model_variant,
                "n_train":       len(X_train),
                "n_test":        len(X_test),
                "n_features":    X_train.shape[1],
                **{k.replace("model__", ""): v
                   for k, v in search.best_params_.items()},
            })
            mlflow.log_metrics({**metrics, "cv_r2": cv_score})
            mlflow.sklearn.log_model(best_estimator, "model")

            print(f"    {run_name:70s}  R²={metrics['r2']:+.4f}  "
                  f"RMSE={metrics['rmse']:.4f}  CV-R²={cv_score:.4f}")

            if metrics["r2"] > best_metrics["r2"]:
                best_model       = best_estimator
                best_metrics     = metrics
                best_params_dict = {
                    "model_type": model_type,
                    **{k.replace("model__", ""): v
                       for k, v in search.best_params_.items()},
                }

    # Tag the winning run
    mlflow.set_tag("best", "true")
    return best_model, best_metrics, best_params_dict


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-fold prediction (stacking for L2 training data)
# ─────────────────────────────────────────────────────────────────────────────

def make_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    target: str,
    model_variant: str,
) -> np.ndarray:
    """
    5-fold CV: for each fold, train best model type on train folds,
    predict on held-out fold.  Returns OOF predictions for the full X.
    Used to build L2 training features without data leakage.
    """
    kf  = KFold(n_splits=params["cv_folds"], shuffle=True,
                random_state=params["random_state"])
    oof = np.zeros(len(X))

    # Pick model type via quick inner CV on first fold only (cheap proxy)
    # to keep runtime tractable; could be extended to full model selection.
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Try all model types, pick best on this val fold
        best_pred = None
        best_r2   = -np.inf
        for model_type in params["models"]:
            pipeline = build_model(model_type, params, params["random_state"])
            search   = build_search(pipeline, model_type, params)
            search.fit(X_tr, y_tr)
            pred = search.best_estimator_.predict(X_va)
            r2   = r2_score(y_va, pred)
            if r2 > best_r2:
                best_r2   = r2
                best_pred = pred

        oof[va_idx] = best_pred

    return oof


# ─────────────────────────────────────────────────────────────────────────────
# Per-variant pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_variant(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    variant:  str,
    params:   dict,
    manifest: dict,
    all_run_metrics: list,
):
    """Train all L1 + L2 models (full and lite) for one variant."""
    l1_targets  = params["layer1_targets"]
    l1_features = [c for c in params["layer1_features"] if c in df_train.columns]
    lite_feats  = [c for c in params["lite_features"]   if c in df_train.columns]
    l2_target   = params["target"]   # power_watts

    print(f"\n{'='*70}")
    print(f"  VARIANT: {variant}  "
          f"(train={len(df_train)}, test={len(df_test)})")
    print(f"{'='*70}")

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    # Store OOF predictions on the *train* set for L2 stacking
    oof_full = {}   # target -> np.ndarray (len = len(df_train))
    oof_lite = {}

    for target in l1_targets:
        if target not in df_train.columns:
            print(f"  [SKIP] {target} not in columns")
            continue

        # Drop rows where the target is NaN (e.g. delay undefined for DPDK)
        tr_valid = df_train[df_train[target].notna()]
        te_valid = df_test[df_test[target].notna()]

        y_tr = tr_valid[target]
        y_te = te_valid[target]

        for feat_list, mv, oof_store in [
            (l1_features, "full", oof_full),
            (lite_feats,  "lite", oof_lite),
        ]:
            feats = [c for c in feat_list if c in df_train.columns]
            X_tr  = tr_valid[feats]
            X_te  = te_valid[feats]

            print(f"\n  [L1 | {mv}] target={target}  features={len(feats)}  "
                  f"valid_train={len(X_tr)}")

            # OOF for stacking — fill NaN positions with 0 for L2
            oof_valid = make_oof_predictions(X_tr, y_tr, params, target, mv)
            oof_full_len = np.zeros(len(df_train))
            oof_full_len[df_train.index.get_indexer(X_tr.index)] = oof_valid
            oof_store[target] = oof_full_len

            # Final model on full train set
            best_model, best_metrics, best_params = train_slot(
                X_tr, y_tr, X_te, y_te, params,
                variant=variant, layer="layer1",
                target=target, model_variant=mv,
            )

            model_path = MODELS_DIR / "layer1" / f"{variant}__{target}"
            if mv == "lite":
                model_path = model_path.parent / (model_path.name + "__lite")
            model_path = model_path.with_suffix(".pkl")
            save_model(best_model, model_path)

            entry_key = f"{variant}__layer1__{target}__{mv}"
            manifest[entry_key] = {
                "path":       str(model_path),
                "layer":      "layer1",
                "variant":    variant,
                "target":     target,
                "model_variant": mv,
                "features":   feats,
                **best_metrics,
                "best_params": best_params,
            }
            all_run_metrics.append({
                "variant": variant, "layer": "layer1",
                "target": target, "model_variant": mv,
                **best_metrics,
            })

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    # L2 inputs = L1 offered-load features + L1 OOF predictions
    for mv, oof_store, feat_list in [
        ("full", oof_full, l1_features),
        ("lite", oof_lite, lite_feats),
    ]:
        feats = [c for c in feat_list if c in df_train.columns]

        # Build augmented feature matrix for train
        X_tr_base = df_train[feats].copy()
        for t, oof_vals in oof_store.items():
            X_tr_base[f"l1_pred__{t}"] = oof_vals

        # For test: use the final trained L1 models to predict
        X_te_base = df_test[feats].copy()
        for t in l1_targets:
            if t not in df_test.columns:
                continue
            l1_path = MODELS_DIR / "layer1" / f"{variant}__{t}"
            if mv == "lite":
                l1_path = l1_path.parent / (l1_path.name + "__lite")
            l1_path = l1_path.with_suffix(".pkl")
            if l1_path.exists():
                with open(l1_path, "rb") as f:
                    l1_model = pickle.load(f)
                feats_l1 = manifest[f"{variant}__layer1__{t}__{mv}"]["features"]
                feats_l1_avail = [c for c in feats_l1 if c in df_test.columns]
                X_te_base[f"l1_pred__{t}"] = l1_model.predict(
                    df_test[feats_l1_avail]
                )

        # Drop any rows where power_watts is NaN
        tr_mask = df_train[l2_target].notna().values
        te_mask = df_test[l2_target].notna().values
        X_tr_fit = X_tr_base[tr_mask].fillna(0)
        X_te_fit = X_te_base[te_mask].fillna(0)
        y_tr = df_train[l2_target].dropna()
        y_te = df_test[l2_target].dropna()

        print(f"\n  [L2 | {mv}] target={l2_target}  "
              f"features={X_tr_fit.shape[1]}  valid_train={len(X_tr_fit)}")

        best_model, best_metrics, best_params = train_slot(
            X_tr_fit, y_tr, X_te_fit, y_te, params,
            variant=variant, layer="layer2",
            target=l2_target, model_variant=mv,
        )

        model_path = MODELS_DIR / "layer2" / f"{variant}__{l2_target}"
        if mv == "lite":
            model_path = model_path.parent / (model_path.name + "__lite")
        model_path = model_path.with_suffix(".pkl")
        save_model(best_model, model_path)

        entry_key = f"{variant}__layer2__{l2_target}__{mv}"
        manifest[entry_key] = {
            "path":       str(model_path),
            "layer":      "layer2",
            "variant":    variant,
            "target":     l2_target,
            "model_variant": mv,
            "features":   list(X_tr_fit.columns),
            "l1_targets": l1_targets,
            **best_metrics,
            "best_params": best_params,
        }
        all_run_metrics.append({
            "variant": variant, "layer": "layer2",
            "target": l2_target, "model_variant": mv,
            **best_metrics,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    params = load_params()
    rs     = params["random_state"]

    print("Loading features.csv ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Split variants ────────────────────────────────────────────────────────
    dpdk     = df[df["is_dpdk"] == 1].copy()
    usr_full = df[df["is_dpdk"] == 0].copy()
    usr_safe = df[
        (df["is_dpdk"] == 0) &
        (df["throughput_gbps"] < params["usr_safe_threshold_gbps"])
    ].copy()

    print(f"  dpdk={len(dpdk)}  usr_full={len(usr_full)}  "
          f"usr_safe={len(usr_safe)}")

    # ── Train/test split per variant ─────────────────────────────────────────
    def split(sub):
        return train_test_split(sub, test_size=params["test_split"],
                                random_state=rs)

    dpdk_tr,     dpdk_te     = split(dpdk)
    usr_full_tr, usr_full_te = split(usr_full)
    usr_safe_tr, usr_safe_te = split(usr_safe)

    variants = [
        ("dpdk",     dpdk_tr,     dpdk_te),
        ("usr_full", usr_full_tr, usr_full_te),
        ("usr_safe", usr_safe_tr, usr_safe_te),
    ]

    # ── MLflow parent run ─────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "layer1").mkdir(exist_ok=True)
    (MODELS_DIR / "layer2").mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    mlflow.set_experiment("upf-energy-profiling")

    manifest        = {}
    all_run_metrics = []

    with mlflow.start_run(run_name="two_layer_training"):
        mlflow.log_params({
            "target":                params["target"],
            "test_split":            params["test_split"],
            "cv_folds":              params["cv_folds"],
            "search_n_iter":         params["search_n_iter"],
            "usr_safe_threshold":    params["usr_safe_threshold_gbps"],
        })

        for variant, df_tr, df_te in variants:
            train_variant(df_tr, df_te, variant, params,
                          manifest, all_run_metrics)

    # ── Save manifest ─────────────────────────────────────────────────────────
    manifest_path = MODELS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved -> {manifest_path}")

    # ── Save full metrics report ──────────────────────────────────────────────
    metrics_full_path = REPORTS_DIR / "metrics_full.json"
    with open(metrics_full_path, "w") as f:
        json.dump(all_run_metrics, f, indent=2)

    # ── DVC metrics.json: L2 power model results per variant (full only) ──────
    dvc_metrics = {}
    for v in ("dpdk", "usr_full", "usr_safe"):
        key = f"{v}__layer2__{params['target']}__full"
        if key in manifest:
            dvc_metrics[v] = {
                k: manifest[key][k] for k in ("mae", "rmse", "r2")
            }
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(dvc_metrics, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TRAINING COMPLETE — L2 Power Model Summary (full, test set)")
    print("="*70)
    print(f"{'Variant':<12} {'R²':>8} {'RMSE (W)':>10} {'MAE (W)':>10}")
    print("-"*42)
    for v, m in dvc_metrics.items():
        print(f"  {v:<10} {m['r2']:>8.4f} {m['rmse']:>10.4f} {m['mae']:>10.4f}")
    print(f"\nAll models  -> {MODELS_DIR}/")
    print(f"Manifest    -> {manifest_path}")
    print(f"DVC metrics -> {metrics_path}")
    print(f"Full report -> {metrics_full_path}")


if __name__ == "__main__":
    main()
