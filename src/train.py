"""
src/train.py — Train a model on the processed feature matrix.

Logs parameters, metrics, and the model artifact to MLflow.
Also saves locally for the DVC pipeline.
"""
import json
import pickle
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["train"]


MODELS = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def main():
    params = load_params()

    # ── Load data ──
    df = pd.read_csv("data/processed/features.csv")
    target_col = "sec"

    # Exclude non-feature columns
    exclude = {target_col, "timestamp", "source_file", "source_file_loadcore", "source_file_scaphandre"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64")]

    X = df[feature_cols]
    y = df[target_col]

    print(f"Features: {len(feature_cols)} columns, {len(X)} samples")
    print(f"Target '{target_col}': mean={y.mean():.4f}, std={y.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_split"],
        random_state=params["random_state"],
    )

    # ── MLflow tracking ──
    mlflow.set_experiment("upf-energy-profiling")

    with mlflow.start_run():
        # Log all parameters
        mlflow.log_params({
            "model_type": params["model_type"],
            "n_estimators": params.get("n_estimators"),
            "max_depth": params.get("max_depth"),
            "min_samples_leaf": params.get("min_samples_leaf"),
            "test_split": params["test_split"],
            "n_features": len(feature_cols),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        })

        # Train
        ModelClass = MODELS[params["model_type"]]
        model_kwargs = {
            k: v for k, v in params.items()
            if k in ("n_estimators", "max_depth", "min_samples_leaf") and v is not None
        }
        model = ModelClass(**model_kwargs, random_state=params["random_state"])
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
            "r2": float(r2_score(y_test, y_pred)),
        }

        # Log to MLflow
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # Log feature names as an artifact
        feature_info = {"feature_names": feature_cols, "n_features": len(feature_cols)}
        Path("reports").mkdir(exist_ok=True)
        with open("reports/feature_info.json", "w") as f:
            json.dump(feature_info, f, indent=2)
        mlflow.log_artifact("reports/feature_info.json")

        # Save locally for DVC pipeline
        Path("models").mkdir(exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Save metrics for DVC
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print(f"Metrics: {metrics}")
        print(f"Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
