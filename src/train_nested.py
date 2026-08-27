"""
src/train_nested.py — Two-layer training with *honest* model selection.

Supersedes src/train_grouped.py. That script fixed the train/test SPLIT
(GroupShuffleSplit on run_dir, so no measurement run straddles the split) but
kept two selection defects that let test/fold information reach the choice of
model:

  1. train_slot() chose the winning model FAMILY by R^2 on the held-out test
     set (`if metrics["r2"] > best_metrics["r2"]`). The inner-CV score was
     computed on the following line, logged, and then ignored. The reported
     test R^2 was therefore a maximum over three families evaluated on the very
     set it is meant to estimate generalisation on.

  2. The hyperparameter search used `cv=5`, a bare integer, which sklearn
     expands to plain KFold. Samples 3 s apart inside one sustained load step
     are near-duplicates, so folds cut through runs and hyperparameters were
     selected under leakage even though the outer split was grouped.
     make_oof_predictions() had the same problem, and additionally picked the
     model type by R^2 on the fold it was about to predict.

The protocol here is the one the review asks for:

  * split by measurement run (GroupShuffleSplit, unchanged);
  * select family AND hyperparameters by GroupKFold CV *inside the training
    set only*, groups = run_dir;
  * evaluate the selected model once on the untouched test set, and report
    that number without further selection.

Every candidate's CV score is recorded in the report next to the winner, so
the effect of the correction is auditable: `cv_r2_all` lists what each family
scored, `selected_by` names the criterion, and `test_r2_all` records what the
old criterion would have chosen.

Outputs go to models_nested/ and reports_nested/ so the previous run remains
on disk for comparison.

Usage:
    python src/train_nested.py
"""

from __future__ import annotations

import json
import pickle
import warnings
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PARAMS_PATH = "params.yaml"
DATA_PATH   = "data/processed/features_with_run.csv"
MODELS_DIR  = Path("models_nested")
REPORTS_DIR = Path("reports_nested")


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
    if model_type == "ridge":
        model = Ridge(random_state=random_state)
    elif model_type == "random_forest":
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=random_state)
    else:
        raise ValueError(f"unknown model_type {model_type!r}")
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def build_search(pipeline, model_type: str, params: dict, cv) -> object:
    """
    `cv` is a splitter object (GroupKFold), never an int: the caller is
    required to decide how folds are drawn, because the honest choice here
    depends on the group structure and the default is wrong.
    """
    n_iter = params["search_n_iter"]
    rs     = params["random_state"]

    if model_type == "ridge":
        grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
        return GridSearchCV(pipeline, grid, cv=cv, scoring="r2",
                            refit=True, n_jobs=-1)
    if model_type == "random_forest":
        dist = {
            "model__n_estimators":     [100, 200, 300],
            "model__max_depth":        [6, 10, 15, None],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features":     [0.5, 0.7, 1.0],
        }
    elif model_type == "gradient_boosting":
        dist = {
            "model__n_estimators":  [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth":     [3, 5, 7],
            "model__subsample":     [0.7, 0.85, 1.0],
        }
    else:
        raise ValueError(f"unknown model_type {model_type!r}")
    return RandomizedSearchCV(pipeline, dist, n_iter=n_iter, cv=cv,
                              scoring="r2", refit=True,
                              random_state=rs, n_jobs=-1)


def n_group_folds(groups: pd.Series, requested: int) -> int:
    """GroupKFold cannot make more folds than there are distinct groups."""
    return int(min(requested, groups.nunique()))


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


# ─────────────────────────────────────────────────────────────────────────────
# Core training routine — one (variant, target, feature_set) slot
# ─────────────────────────────────────────────────────────────────────────────

def train_slot(
    X_train, y_train, g_train,
    X_test,  y_test,
    params, variant, layer, target, model_variant,
):
    """
    Fit every candidate family under GroupKFold CV on the training set,
    select on the CV score, then score the winner ONCE on the test set.
    """
    cv = GroupKFold(n_splits=n_group_folds(g_train, params["cv_folds"]))

    cv_r2_all, test_r2_all, fitted = {}, {}, {}

    for model_type in params["models"]:
        pipeline = build_model(model_type, params, params["random_state"])
        search   = build_search(pipeline, model_type, params, cv)
        search.fit(X_train, y_train, groups=g_train)

        est      = search.best_estimator_
        cv_score = float(search.best_score_)
        cv_r2_all[model_type] = cv_score
        fitted[model_type] = (est, search.best_params_)

        # Recorded for the audit trail only — NEVER used to select.
        test_r2_all[model_type] = float(
            r2_score(y_test.values, est.predict(X_test))
        )

        print(f"    {variant}__{layer}__{target}__{model_type}__{model_variant:4s}"
              f"  CV-R²={cv_score:+.4f}   (test R²={test_r2_all[model_type]:+.4f})")

    winner = max(cv_r2_all, key=cv_r2_all.get)
    est, best_params = fitted[winner]

    metrics = eval_metrics(y_test.values, est.predict(X_test))
    metrics.update({
        "cv_r2":       cv_r2_all[winner],
        "cv_r2_all":   cv_r2_all,
        "test_r2_all": test_r2_all,
        "selected_by": "cv_r2 (GroupKFold on training set)",
        "selected_model": winner,
        "would_have_selected_on_test": max(test_r2_all, key=test_r2_all.get),
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "n_train_groups": int(g_train.nunique()),
    })

    flag = "" if winner == metrics["would_have_selected_on_test"] else "   <-- SELECTION CHANGED"
    print(f"    -> selected {winner} by CV; test R²={metrics['r2']:+.4f}{flag}")

    return est, metrics, {"model_type": winner,
                          **{k.replace("model__", ""): v
                             for k, v in best_params.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-fold predictions for L2 stacking
# ─────────────────────────────────────────────────────────────────────────────

def make_oof_predictions(X, y, g, params) -> np.ndarray:
    """
    Grouped OOF: folds are cut on run_dir, and within each fold the model type
    is chosen by a further GroupKFold CV on that fold's *training* part -- not
    by score on the held-out part, which is what the previous version did.
    """
    outer = GroupKFold(n_splits=n_group_folds(g, params["cv_folds"]))
    oof   = np.zeros(len(X))

    for tr_idx, va_idx in outer.split(X, y, groups=g):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr       = y.iloc[tr_idx]
        g_tr       = g.iloc[tr_idx]

        inner = GroupKFold(n_splits=n_group_folds(g_tr, params["cv_folds"]))

        best_est, best_cv = None, -np.inf
        for model_type in params["models"]:
            pipeline = build_model(model_type, params, params["random_state"])
            search   = build_search(pipeline, model_type, params, inner)
            search.fit(X_tr, y_tr, groups=g_tr)
            if search.best_score_ > best_cv:
                best_cv  = search.best_score_
                best_est = search.best_estimator_

        oof[va_idx] = best_est.predict(X_va)

    return oof


# ─────────────────────────────────────────────────────────────────────────────
# Per-variant pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_variant(df_train, df_test, variant, params, manifest, all_run_metrics):
    l1_targets  = params["layer1_targets"]
    l1_features = [c for c in params["layer1_features"] if c in df_train.columns]
    lite_feats  = [c for c in params["lite_features"]   if c in df_train.columns]
    l2_target   = params["target"]

    print(f"\n{'='*78}")
    print(f"  VARIANT: {variant}  (train={len(df_train)} rows / "
          f"{df_train['run_dir'].nunique()} runs, test={len(df_test)} rows / "
          f"{df_test['run_dir'].nunique()} runs)")
    print(f"{'='*78}")

    oof_full, oof_lite = {}, {}

    for target in l1_targets:
        if target not in df_train.columns:
            print(f"  [SKIP] {target} not in columns")
            continue

        tr_valid = df_train[df_train[target].notna()]
        te_valid = df_test[df_test[target].notna()]
        if len(tr_valid) == 0 or len(te_valid) == 0:
            print(f"  [SKIP] {target}: empty after NaN drop")
            continue

        y_tr, y_te = tr_valid[target], te_valid[target]
        g_tr       = tr_valid["run_dir"]

        for feat_list, mv, oof_store in [
            (l1_features, "full", oof_full),
            (lite_feats,  "lite", oof_lite),
        ]:
            feats = [c for c in feat_list if c in df_train.columns]
            X_tr, X_te = tr_valid[feats], te_valid[feats]

            print(f"\n  [L1 | {mv}] {target}  features={len(feats)}  "
                  f"train={len(X_tr)}")

            oof_valid = make_oof_predictions(X_tr, y_tr, g_tr, params)
            oof_col = np.zeros(len(df_train))
            oof_col[df_train.index.get_indexer(X_tr.index)] = oof_valid
            oof_store[target] = oof_col

            model, metrics, best_params = train_slot(
                X_tr, y_tr, g_tr, X_te, y_te, params,
                variant, "layer1", target, mv,
            )

            path = MODELS_DIR / "layer1" / f"{variant}__{target}"
            if mv == "lite":
                path = path.parent / (path.name + "__lite")
            path = path.with_suffix(".pkl")
            save_model(model, path)

            manifest[f"{variant}__layer1__{target}__{mv}"] = {
                "path": str(path), "layer": "layer1", "variant": variant,
                "target": target, "model_variant": mv, "features": feats,
                **metrics, "best_params": best_params,
            }
            all_run_metrics.append({
                "variant": variant, "layer": "layer1",
                "target": target, "model_variant": mv, **metrics,
            })

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    for mv, oof_store, feat_list in [("full", oof_full, l1_features),
                                     ("lite", oof_lite, lite_feats)]:
        feats = [c for c in feat_list if c in df_train.columns]

        X_tr_base = df_train[feats].copy()
        for t, vals in oof_store.items():
            X_tr_base[f"l1_pred__{t}"] = vals

        X_te_base = df_test[feats].copy()
        for t in l1_targets:
            key = f"{variant}__layer1__{t}__{mv}"
            if t not in df_test.columns or key not in manifest:
                continue
            p = MODELS_DIR / "layer1" / f"{variant}__{t}"
            if mv == "lite":
                p = p.parent / (p.name + "__lite")
            p = p.with_suffix(".pkl")
            if p.exists():
                with open(p, "rb") as f:
                    l1_model = pickle.load(f)
                fl = [c for c in manifest[key]["features"] if c in df_test.columns]
                X_te_base[f"l1_pred__{t}"] = l1_model.predict(df_test[fl])

        tr_mask = df_train[l2_target].notna().values
        te_mask = df_test[l2_target].notna().values
        X_tr_fit = X_tr_base[tr_mask].fillna(0)
        X_te_fit = X_te_base[te_mask].fillna(0)
        y_tr = df_train[l2_target].dropna()
        y_te = df_test[l2_target].dropna()
        g_tr = df_train.loc[tr_mask, "run_dir"]

        print(f"\n  [L2 | {mv}] {l2_target}  features={X_tr_fit.shape[1]}  "
              f"train={len(X_tr_fit)}")

        model, metrics, best_params = train_slot(
            X_tr_fit, y_tr, g_tr, X_te_fit, y_te, params,
            variant, "layer2", l2_target, mv,
        )

        path = MODELS_DIR / "layer2" / f"{variant}__{l2_target}"
        if mv == "lite":
            path = path.parent / (path.name + "__lite")
        path = path.with_suffix(".pkl")
        save_model(model, path)

        manifest[f"{variant}__layer2__{l2_target}__{mv}"] = {
            "path": str(path), "layer": "layer2", "variant": variant,
            "target": l2_target, "model_variant": mv,
            "features": list(X_tr_fit.columns), "l1_targets": l1_targets,
            **metrics, "best_params": best_params,
        }
        all_run_metrics.append({
            "variant": variant, "layer": "layer2",
            "target": l2_target, "model_variant": mv, **metrics,
        })


def main():
    params = load_params()
    rs     = params["random_state"]

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {df.shape[0]} rows × {df.shape[1]} cols, "
          f"{df['run_dir'].nunique()} runs")

    dpdk     = df[df["is_dpdk"] == 1].copy()
    usr_full = df[df["is_dpdk"] == 0].copy()
    usr_safe = df[(df["is_dpdk"] == 0) &
                  (df["throughput_gbps"] < params["usr_safe_threshold_gbps"])].copy()

    def split(sub):
        gss = GroupShuffleSplit(n_splits=1, test_size=params["test_split"],
                                random_state=rs)
        tr, te = next(gss.split(sub, groups=sub["run_dir"]))
        return sub.iloc[tr].copy(), sub.iloc[te].copy()

    variants = [("dpdk", *split(dpdk)),
                ("usr_full", *split(usr_full)),
                ("usr_safe", *split(usr_safe))]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "layer1").mkdir(exist_ok=True)
    (MODELS_DIR / "layer2").mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    manifest, all_run_metrics = {}, []
    for variant, df_tr, df_te in variants:
        train_variant(df_tr, df_te, variant, params, manifest, all_run_metrics)

    with open(MODELS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(REPORTS_DIR / "metrics_full.json", "w") as f:
        json.dump(all_run_metrics, f, indent=2)

    dvc_metrics = {}
    for v in ("dpdk", "usr_full", "usr_safe"):
        key = f"{v}__layer2__{params['target']}__full"
        if key in manifest:
            dvc_metrics[v] = {k: manifest[key][k] for k in ("mae", "rmse", "r2", "cv_r2")}
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(dvc_metrics, f, indent=2)

    print("\n" + "="*78)
    print("TRAINING COMPLETE — L2 power model (full features, untouched test set)")
    print("="*78)
    print(f"{'Variant':<12} {'test R²':>9} {'CV R²':>9} {'RMSE (W)':>10} {'MAE (W)':>10}")
    for v, m in dvc_metrics.items():
        print(f"  {v:<10} {m['r2']:>9.4f} {m['cv_r2']:>9.4f} "
              f"{m['rmse']:>10.4f} {m['mae']:>10.4f}")

    changed = [r for r in all_run_metrics
               if r["selected_model"] != r["would_have_selected_on_test"]]
    print(f"\nSlots where honest selection picked a DIFFERENT family than "
          f"test-set selection would have: {len(changed)} / {len(all_run_metrics)}")
    for r in changed:
        print(f"  {r['variant']:<9} {r['layer']:<7} {r['target'][:44]:<44} "
              f"{r['model_variant']:<5} cv->{r['selected_model']:<18} "
              f"test->{r['would_have_selected_on_test']}")


if __name__ == "__main__":
    main()
