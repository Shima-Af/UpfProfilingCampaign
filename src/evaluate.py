"""
src/evaluate.py — Evaluate the trained model and generate reports.

Produces feature importance plots and summary metrics.
"""
import json
import pickle
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["evaluate"]


def plot_feature_importance(model, feature_names: list[str], output_path: str):
    """Generate and save a feature importance bar chart."""
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1][:15]  # Top 15

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        align="center",
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Features for SEC Prediction")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved feature importance plot to {output_path}")


def main():
    params = load_params()

    # Load model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load feature info
    df = pd.read_csv("data/processed/features.csv")
    exclude = {"sec", "timestamp", "source_file", "source_file_loadcore", "source_file_scaphandre"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64")]

    # Load metrics
    with open("reports/metrics.json") as f:
        metrics = json.load(f)

    print("=== Model Evaluation Summary ===")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        print("\n=== Generating Feature Importance Plot ===")
        plot_feature_importance(model, feature_cols, "reports/figures/feature_importance.png")
    else:
        print("\n  Model does not support feature_importances_, skipping plot.")


if __name__ == "__main__":
    main()
