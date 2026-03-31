"""
src/evaluate.py — Post-training evaluation: figures + extended metrics.

Generates (saved to reports/figures/):
  fig8_predicted_vs_actual.{png,pdf}   — Predicted vs measured power per variant
  fig9_r2_heatmap.{png,pdf}            — R² across all slots (variant x target)
  fig10_algo_comparison_l2.{png,pdf}   — All algorithms compared on L2 power model
  fig11_algo_comparison_l1.{png,pdf}   — All algorithms compared on L1 models (heatmap)
  fig12_interpretability.{png,pdf}     — Ridge coefficients vs tree importances side-by-side
  fig13_variant_summary.{png,pdf}      — KPI distributions per variant (violin plots)
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import mlflow
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Style constants ───────────────────────────────────────────────────────────

DPDK_COLOR = "#2171b5"
USR_COLOR  = "#cb181d"
VARIANT_COLORS = {
    "dpdk":     DPDK_COLOR,
    "usr_full": USR_COLOR,
    "usr_safe": "#fc8d59",
}
VARIANT_LABELS = {
    "dpdk":     "DPDK",
    "usr_full": "USR (full range)",
    "usr_safe": "USR (safe regime)",
}
ALGO_COLORS = {
    "ridge":             "#4292c6",
    "random_forest":     "#41ab5d",
    "gradient_boosting": "#fe9929",
}
ALGO_LABELS = {
    "ridge":             "Ridge",
    "random_forest":     "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SHORT_TARGETS = {
    "throughput_gbps": "Throughput",
    "cpu_pct": "CPU %",
    "gtpu_packets_dn__packets_lost_delta": "Pkt Loss",
    "downlink_one_way_delay_distribution__weighted_mean_delay_us": "DL Delay",
    "power_watts": "Power",
}
SHORT_VARIANTS = {
    "dpdk":     "DPDK",
    "usr_full": "USR-full",
    "usr_safe": "USR-safe",
}
SHORT_FEATURES = {
    "gtpu_kbitss_dn__kbits_tx_s":                               "DL kbits/s",
    "gtpu_kbitss_ngran__gtpu_kbits_tx_s":                       "UL kbits/s",
    "gtpu_packets_dn__packets_tx_delta":                         "DL pkts/\u0394t",
    "gtpu_packets_ngran__gtpu_packets_tx_delta":                 "UL pkts/\u0394t",
    "user_plane_throughput__l2_3_device_tx_traffic":             "L2/3 TX traffic",
    "avg_packet_size_bytes":                                     "Avg pkt size",
    "l2l3_overhead_ratio":                                       "L2/3 ratio",
    "l1_pred__throughput_gbps":                                  "L1: Throughput",
    "l1_pred__cpu_pct":                                          "L1: CPU %",
    "l1_pred__gtpu_packets_dn__packets_lost_delta":              "L1: Pkt Loss",
    "l1_pred__downlink_one_way_delay_distribution__weighted_mean_delay_us": "L1: DL Delay",
}

ALGOS = ["ridge", "random_forest", "gradient_boosting"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def load_manifest() -> dict:
    with open("models/manifest.json") as f:
        return json.load(f)

def load_pkl(path: str) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)

def savefig(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.{{png,pdf}}")


def load_model_from_mlflow(
    client: "mlflow.tracking.MlflowClient",
    exp_id: str,
    variant: str,
    layer: str,
    target: str,
    model_variant: str,
    model_type: str,
) -> object | None:
    """
    Load a fitted sklearn Pipeline from an MLflow artifact.
    Returns the Pipeline, or None if the run/artifact is not found.
    """
    runs = client.search_runs(
        exp_id,
        filter_string=(
            f"tags.variant = '{variant}' AND "
            f"tags.layer = '{layer}' AND "
            f"tags.target = '{target}' AND "
            f"tags.model_variant = '{model_variant}' AND "
            f"tags.model_type = '{model_type}'"
        ),
        max_results=1,
    )
    if not runs:
        return None
    art_uri = runs[0].info.artifact_uri
    # Normalise file URI to a plain path (works on Windows and POSIX)
    pkl_path = Path(art_uri.replace("file:///", "").replace("file://", "")) / "model" / "model.pkl"
    if not pkl_path.exists():
        return None
    return load_pkl(pkl_path)


def unwrap_estimator(pipeline) -> object:
    """Extract the underlying estimator from a sklearn Pipeline (or return as-is)."""
    try:
        return pipeline["model"]
    except (KeyError, TypeError):
        return pipeline


def load_mlflow_runs() -> pd.DataFrame:
    """Query all child runs from MLflow and return as a tidy DataFrame."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("upf-energy-profiling")
    if exp is None:
        raise RuntimeError("MLflow experiment 'upf-energy-profiling' not found.")
    runs = client.search_runs(exp.experiment_id, max_results=500)
    rows = []
    for r in runs:
        t = r.data.tags
        m = r.data.metrics
        if "model_type" not in t:
            continue   # skip parent runs
        rows.append({
            "variant":       t.get("variant", ""),
            "layer":         t.get("layer", ""),
            "target":        t.get("target", ""),
            "model_type":    t.get("model_type", ""),
            "model_variant": t.get("model_variant", ""),
            "r2":            m.get("r2", np.nan),
            "mae":           m.get("mae", np.nan),
            "rmse":          m.get("rmse", np.nan),
            "cv_r2":         m.get("cv_r2", np.nan),
        })
    df = pd.DataFrame(rows)
    # Keep best r2 per (variant, layer, target, model_type, model_variant)
    df = (df.sort_values("r2", ascending=False)
            .drop_duplicates(["variant", "layer", "target", "model_type", "model_variant"])
            .reset_index(drop=True))
    return df


def get_test_split(df: pd.DataFrame, variant: str, train_params: dict) -> pd.DataFrame:
    thr = train_params["usr_safe_threshold_gbps"]
    if variant == "dpdk":
        sub = df[df["is_dpdk"] == 1].copy()
    elif variant == "usr_full":
        sub = df[df["is_dpdk"] == 0].copy()
    else:
        sub = df[(df["is_dpdk"] == 0) & (df["throughput_gbps"] < thr)].copy()
    _, df_test = train_test_split(
        sub,
        test_size=train_params["test_split"],
        random_state=train_params["random_state"],
    )
    return df_test


def build_l2_predictions(df_test, manifest, variant, mode, train_params):
    l1_targets = train_params["layer1_targets"]
    features = (train_params["layer1_features"] if mode == "full"
                else train_params["lite_features"])
    l1_preds = {}
    for target in l1_targets:
        key = f"{variant}__layer1__{target}__{mode}"
        if key not in manifest:
            continue
        model = load_pkl(manifest[key]["path"])
        X = df_test[features].fillna(0)
        l1_preds[target] = model.predict(X)

    l2_key    = f"{variant}__layer2__power_watts__{mode}"
    l2_entry  = manifest[l2_key]
    l2_model  = load_pkl(l2_entry["path"])
    l2_feats  = l2_entry["features"]

    X_l2 = pd.DataFrame(index=df_test.index)
    for feat in l2_feats:
        if feat.startswith("l1_pred__"):
            tname = feat[len("l1_pred__"):]
            arr = l1_preds.get(tname)
            X_l2[feat] = arr if arr is not None else np.zeros(len(df_test))
        else:
            X_l2[feat] = df_test[feat].fillna(0).values

    y_pred = np.maximum(l2_model.predict(X_l2[l2_feats]), 0.0)
    y_true = df_test["power_watts"].values
    return y_true, y_pred


def _winner_per_slot(mlflow_df: pd.DataFrame, layer: str, target: str,
                     variant: str, model_variant: str) -> str:
    """Return the model_type with highest R² for a given slot."""
    sub = mlflow_df[
        (mlflow_df["variant"]       == variant) &
        (mlflow_df["layer"]         == layer) &
        (mlflow_df["target"]        == target) &
        (mlflow_df["model_variant"] == model_variant)
    ]
    if sub.empty:
        return "ridge"
    return sub.loc[sub["r2"].idxmax(), "model_type"]


# ── Figure 8: Predicted vs Actual power ──────────────────────────────────────

def fig8_predicted_vs_actual(df, manifest, train_params):
    variants = ["dpdk", "usr_full", "usr_safe"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, variant in zip(axes, variants):
        df_test = get_test_split(df, variant, train_params)
        color   = VARIANT_COLORS[variant]
        all_true, all_pred = [], []

        for mode, marker in [("full", "o"), ("lite", "s")]:
            y_true, y_pred = build_l2_predictions(
                df_test, manifest, variant, mode, train_params)
            r2_val = manifest[f"{variant}__layer2__power_watts__{mode}"]["r2"]
            ax.scatter(y_true, y_pred, s=14, alpha=0.6, marker=marker,
                       facecolors=color if mode == "full" else "none",
                       edgecolors=color, linewidths=0.8,
                       label=f"{mode.capitalize()}  R\u00b2={r2_val:.3f}")
            all_true.extend(y_true.tolist())
            all_pred.extend(y_pred.tolist())

        lo = min(min(all_true), min(all_pred)) * 0.95
        hi = max(max(all_true), max(all_pred)) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.55, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Measured power (W)")
        ax.set_ylabel("Predicted power (W)")
        ax.set_title(VARIANT_LABELS[variant], fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Two-layer digital twin: predicted vs measured power",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig8_predicted_vs_actual")


# ── Figure 9: R² heatmap ─────────────────────────────────────────────────────

def fig9_r2_heatmap(manifest, train_params):
    l1_targets  = train_params["layer1_targets"]
    all_targets = l1_targets + ["power_watts"]
    all_layers  = ["layer1"] * len(l1_targets) + ["layer2"]
    variants    = ["dpdk", "usr_full", "usr_safe"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, mode in zip(axes, ["full", "lite"]):
        data = np.full((len(all_targets), len(variants)), np.nan)
        for i, (target, layer) in enumerate(zip(all_targets, all_layers)):
            for j, variant in enumerate(variants):
                key = f"{variant}__{layer}__{target}__{mode}"
                if key in manifest:
                    data[i, j] = manifest[key]["r2"]

        im = ax.imshow(data, vmin=-0.1, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([SHORT_VARIANTS[v] for v in variants],
                           rotation=30, ha="right")
        ax.set_yticks(range(len(all_targets)))
        ax.set_yticklabels([SHORT_TARGETS[t] for t in all_targets])
        ax.set_title(f"R\u00b2 \u2014 {mode.capitalize()} model", fontweight="bold")

        for i in range(len(all_targets)):
            for j in range(len(variants)):
                val = data[i, j]
                if not np.isnan(val):
                    txt = f"{val:.2f}" if val >= -0.09 else f"{val:.1f}"
                    txt_color = "white" if (val < 0.25 or val > 0.88) else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=9, color=txt_color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="R\u00b2")

    fig.suptitle("Model R\u00b2 \u2014 all prediction slots (full vs lite features)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig9_r2_heatmap")


# ── Figure 10: Algorithm comparison — L2 power model ─────────────────────────

def fig10_algo_comparison_l2(mlflow_df: pd.DataFrame, manifest: dict):
    """
    For the L2 power target, show all 3 algorithms' R² for each variant × mode.
    The winning algorithm is highlighted with a gold star and thicker edge.
    This directly justifies why a specific algorithm was chosen.
    """
    variants = ["dpdk", "usr_full", "usr_safe"]
    modes    = ["full", "lite"]
    n_groups = len(modes)          # 2 mode groups per variant subplot
    n_bars   = len(ALGOS)          # 3 bars per group

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for ax, variant in zip(axes, variants):
        group_width = 0.75
        bar_w = group_width / n_bars
        group_centers = np.arange(n_groups)

        for gi, mode in enumerate(modes):
            sub = mlflow_df[
                (mlflow_df["variant"] == variant) &
                (mlflow_df["layer"]   == "layer2") &
                (mlflow_df["target"]  == "power_watts") &
                (mlflow_df["model_variant"] == mode)
            ]
            if sub.empty:
                continue

            winner = sub.loc[sub["r2"].idxmax(), "model_type"]
            r2_by_algo = sub.set_index("model_type")["r2"]

            for bi, algo in enumerate(ALGOS):
                x = group_centers[gi] - group_width / 2 + bar_w * (bi + 0.5)
                r2_val = r2_by_algo.get(algo, np.nan)
                is_winner = (algo == winner)

                bar = ax.bar(
                    x, r2_val if not np.isnan(r2_val) else 0,
                    width=bar_w * 0.88,
                    color=ALGO_COLORS[algo],
                    alpha=0.85 if is_winner else 0.40,
                    edgecolor=ALGO_COLORS[algo],
                    linewidth=2.5 if is_winner else 0.5,
                )
                # Annotate value
                if not np.isnan(r2_val):
                    ax.text(x, r2_val + 0.005,
                            f"{r2_val:.3f}",
                            ha="center", va="bottom", fontsize=6.5,
                            fontweight="bold" if is_winner else "normal")
                # Star for winner
                if is_winner:
                    ax.text(x, r2_val + 0.025, "\u2605",
                            ha="center", va="bottom", fontsize=9,
                            color="goldenrod")

        ax.set_xticks(group_centers)
        ax.set_xticklabels(["Full model", "Lite model"], fontsize=9)
        ax.set_ylabel("Test R\u00b2")
        ax.set_title(VARIANT_LABELS[variant], fontweight="bold")
        ax.set_ylim(-0.1, 1.15)
        ax.axhline(0, color="k", lw=0.5)

        # Draw separator between groups
        ax.axvline(0.5, color="lightgrey", lw=0.8, ls="--", zorder=0)

    # Shared legend for algorithms
    legend_patches = [
        mpatches.Patch(color=ALGO_COLORS[a], alpha=0.85, label=ALGO_LABELS[a])
        for a in ALGOS
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=True)

    fig.suptitle(
        "L2 power model \u2014 algorithm comparison\n"
        "(filled + \u2605 = selected winner; transparent = runner-up)",
        fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "fig10_algo_comparison_l2")


# ── Figure 11: Algorithm comparison — L1 models (heatmap per variant) ────────

def fig11_algo_comparison_l1(mlflow_df: pd.DataFrame, train_params: dict):
    """
    For each variant × mode, display a heatmap of R² values:
      rows  = L1 prediction targets
      cols  = algorithm types (Ridge, RF, GBM)
    Every cell is always populated from MLflow metrics (no model loading needed).
    NaN cells (e.g. DPDK delay — structurally undefined) are shown in grey with
    an explicit "N/A" label rather than left blank.
    Gold border marks the winning algorithm per target row.
    """
    l1_targets = train_params["layer1_targets"]
    variants   = ["dpdk", "usr_full", "usr_safe"]
    modes      = ["full", "lite"]

    fig, axes = plt.subplots(len(variants), len(modes), figsize=(12, 9))

    # Use a masked colormap so NaN cells render in a neutral grey
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#cccccc")

    for row_i, variant in enumerate(variants):
        for col_j, mode in enumerate(modes):
            ax = axes[row_i][col_j]

            # Build matrix: rows=targets, cols=algos
            data = np.full((len(l1_targets), len(ALGOS)), np.nan)
            for ti, target in enumerate(l1_targets):
                sub = mlflow_df[
                    (mlflow_df["variant"] == variant) &
                    (mlflow_df["layer"]   == "layer1") &
                    (mlflow_df["target"]  == target) &
                    (mlflow_df["model_variant"] == mode)
                ]
                for ai, algo in enumerate(ALGOS):
                    match = sub[sub["model_type"] == algo]
                    if not match.empty:
                        data[ti, ai] = match["r2"].values[0]

            masked = np.ma.masked_invalid(data)
            im = ax.imshow(masked, vmin=-0.1, vmax=1.0, cmap=cmap, aspect="auto")

            ax.set_xticks(range(len(ALGOS)))
            ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS],
                               rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(l1_targets)))
            ax.set_yticklabels(
                [SHORT_TARGETS[t] for t in l1_targets] if col_j == 0 else [],
                fontsize=8)

            if col_j == 0:
                ax.set_ylabel(SHORT_VARIANTS[variant], fontweight="bold", fontsize=9)
            if row_i == 0:
                ax.set_title(f"{mode.capitalize()} model", fontweight="bold", fontsize=9)

            # Annotate every cell — value or "N/A"; gold border for row winner
            for ti in range(len(l1_targets)):
                row_vals = data[ti, :]
                valid    = ~np.isnan(row_vals)
                best_ai  = int(np.nanargmax(row_vals)) if valid.any() else None
                for ai in range(len(ALGOS)):
                    val = data[ti, ai]
                    if np.isnan(val):
                        ax.text(ai, ti, "N/A", ha="center", va="center",
                                fontsize=7, color="#888888", style="italic")
                    else:
                        txt_color = "white" if (val < 0.25 or val > 0.88) else "black"
                        weight    = "bold" if ai == best_ai else "normal"
                        ax.text(ai, ti, f"{val:.2f}", ha="center", va="center",
                                fontsize=7.5, color=txt_color, fontweight=weight)
                    if ai == best_ai and not np.isnan(val):
                        ax.add_patch(plt.Rectangle(
                            (ai - 0.5, ti - 0.5), 1, 1,
                            fill=False, edgecolor="gold", linewidth=2.5, zorder=5,
                        ))

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "L1 model R\u00b2 \u2014 all algorithms compared per target\n"
        "(gold border = winner per row;  grey = structurally undefined)",
        fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    savefig(fig, "fig11_algo_comparison_l1")


# ── Figure 12: Interpretability — Ridge coef vs Tree importance ───────────────

def fig12_interpretability(manifest: dict, mlflow_df: pd.DataFrame):
    """
    For each variant's L2 full power model, show side-by-side:
      Left  — Ridge regression normalised |coefficients|
      Right — Best tree (RF or GBM, whichever has higher R²) feature importances

    Both panels are ALWAYS populated by loading the relevant model directly from
    its MLflow artifact — regardless of which algorithm won overall.
    Both use the same feature order (sorted by tree importance descending) so
    the two interpretations can be read in parallel.
    Panel titles show the algorithm name, its test R², and whether it was selected.
    """
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name("upf-energy-profiling")
    exp_id = exp.experiment_id

    variants = ["dpdk", "usr_full", "usr_safe"]
    fig, axes = plt.subplots(len(variants), 2, figsize=(13, 10))

    for row_i, variant in enumerate(variants):
        key    = f"{variant}__layer2__power_watts__full"
        entry  = manifest[key]
        feats  = entry["features"]
        winner = entry["best_params"]["model_type"]
        feat_labels = [SHORT_FEATURES.get(f, f) for f in feats]

        # ── Load Ridge from MLflow artifact ───────────────────────────────────
        ridge_pipeline = load_model_from_mlflow(
            client, exp_id, variant, "layer2", "power_watts", "full", "ridge")
        ridge_est  = unwrap_estimator(ridge_pipeline) if ridge_pipeline else None
        ridge_coef = (np.abs(ridge_est.coef_)
                      if ridge_est is not None and hasattr(ridge_est, "coef_")
                      else None)
        ridge_r2   = mlflow_df.loc[
            (mlflow_df["variant"] == variant) &
            (mlflow_df["layer"] == "layer2") &
            (mlflow_df["target"] == "power_watts") &
            (mlflow_df["model_variant"] == "full") &
            (mlflow_df["model_type"] == "ridge"), "r2"
        ].values
        ridge_r2 = float(ridge_r2[0]) if len(ridge_r2) else np.nan

        # ── Load best tree (RF or GBM) from MLflow artifact ──────────────────
        sub_trees = mlflow_df[
            (mlflow_df["variant"] == variant) &
            (mlflow_df["layer"]   == "layer2") &
            (mlflow_df["target"]  == "power_watts") &
            (mlflow_df["model_variant"] == "full") &
            (mlflow_df["model_type"].isin(["random_forest", "gradient_boosting"]))
        ]
        best_tree_type = sub_trees.loc[sub_trees["r2"].idxmax(), "model_type"] \
            if not sub_trees.empty else "random_forest"
        best_tree_r2 = float(sub_trees["r2"].max()) if not sub_trees.empty else np.nan

        tree_pipeline = load_model_from_mlflow(
            client, exp_id, variant, "layer2", "power_watts", "full", best_tree_type)
        tree_est = unwrap_estimator(tree_pipeline) if tree_pipeline else None
        tree_imp = (tree_est.feature_importances_
                    if tree_est is not None and hasattr(tree_est, "feature_importances_")
                    else None)

        # ── Shared sort order: by tree importance descending (ascending index) ─
        if tree_imp is not None:
            sort_idx = np.argsort(tree_imp)          # ascending → barh bottom = least
        elif ridge_coef is not None:
            norm = ridge_coef / (ridge_coef.sum() + 1e-12)
            sort_idx = np.argsort(norm)
        else:
            sort_idx = np.arange(len(feats))

        sorted_labels = [feat_labels[i] for i in sort_idx]
        ridge_is_winner = (winner == "ridge")
        tree_is_winner  = (winner == best_tree_type)

        # ── Left panel: Ridge |coefficients| ─────────────────────────────────
        ax_left = axes[row_i][0]
        if ridge_coef is not None:
            ridge_norm   = ridge_coef / (ridge_coef.sum() + 1e-12)
            ridge_sorted = ridge_norm[sort_idx]
            alpha  = 0.85 if ridge_is_winner else 0.42
            lw     = 1.8  if ridge_is_winner else 0.5
            suffix = " \u2605" if ridge_is_winner else ""
            ax_left.barh(range(len(sort_idx)), ridge_sorted,
                         color="#4292c6", alpha=alpha,
                         edgecolor="#4292c6", linewidth=lw)
            ax_left.set_yticks(range(len(sort_idx)))
            ax_left.set_yticklabels(sorted_labels, fontsize=8)
            ax_left.set_xlabel("Normalised |coefficient|")
            ax_left.set_title(
                f"Ridge{suffix}  (R\u00b2={ridge_r2:.3f})",
                fontsize=9, fontweight="bold",
                color="#4292c6" if ridge_is_winner else "#666666")
        ax_left.set_ylabel(SHORT_VARIANTS[variant] + "\n", fontweight="bold",
                           fontsize=9)

        # ── Right panel: tree feature importances ─────────────────────────────
        ax_right = axes[row_i][1]
        if tree_imp is not None:
            tree_sorted = tree_imp[sort_idx]
            algo_color  = ALGO_COLORS[best_tree_type]
            alpha  = 0.85 if tree_is_winner else 0.42
            lw     = 1.8  if tree_is_winner else 0.5
            suffix = " \u2605" if tree_is_winner else ""
            algo_name = ALGO_LABELS[best_tree_type]
            ax_right.barh(range(len(sort_idx)), tree_sorted,
                          color=algo_color, alpha=alpha,
                          edgecolor=algo_color, linewidth=lw)
            ax_right.set_yticks(range(len(sort_idx)))
            ax_right.set_yticklabels(sorted_labels, fontsize=8)
            ax_right.set_xlabel("Feature importance (Gini)")
            ax_right.set_title(
                f"{algo_name}{suffix}  (R\u00b2={best_tree_r2:.3f})",
                fontsize=9, fontweight="bold",
                color=algo_color if tree_is_winner else "#666666")

    fig.suptitle(
        "L2 power model interpretability \u2014 Ridge |coefficients| vs best-tree importances\n"
        "(\u2605 = overall selected algorithm;  same feature order for direct comparison)",
        fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    savefig(fig, "fig12_interpretability")


# ── Figure 13: Variant performance summary — distributions per metric ─────────

def fig13_variant_summary(df: pd.DataFrame, train_params: dict):
    """
    Violin plots for each of the 5 KPIs (throughput, CPU, packet loss, delay,
    power) split by deployment variant (DPDK / USR-full / USR-safe).

    Rows are derived from is_dpdk flag and usr_safe_threshold_gbps.
    This figure bridges the EDA and digital-twin sections by showing WHY some
    variants are harder to model (e.g. near-constant DPDK power, near-zero USR
    safe loss).
    """
    threshold = train_params.get("usr_safe_threshold_gbps", 0.5)

    dpdk     = df[df["is_dpdk"] == 1].copy()
    usr_full = df[df["is_dpdk"] == 0].copy()
    usr_safe = df[(df["is_dpdk"] == 0) & (df["throughput_gbps"] < threshold)].copy()

    subsets = [
        ("DPDK",          dpdk,     VARIANT_COLORS["dpdk"]),
        ("USR\n(full)",   usr_full, VARIANT_COLORS["usr_full"]),
        ("USR\n(safe)",   usr_safe, VARIANT_COLORS["usr_safe"]),
    ]

    metrics = [
        ("throughput_gbps",    "Throughput (Gbps)",  None),
        ("cpu_pct",            "CPU (%)",            None),
        ("gtpu_packets_dn__packets_lost_delta",
                               "Packet Loss (Δpkts)", None),
        ("downlink_one_way_delay_distribution__weighted_mean_delay_us",
                               "DL Delay (µs)",       None),
        ("power_watts",        "Power (W)",           None),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle(
        "UPF Deployment Variant Profiles — Measured KPI Distributions",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, (col, ylabel, _) in zip(axes, metrics):
        data   = [s[col].dropna().values for _, s, _ in subsets]
        colors = [c for _, _, c in subsets]
        labels = [lbl for lbl, _, _ in subsets]

        parts = ax.violinplot(data, positions=range(len(subsets)),
                              showmedians=True, showextrema=True)

        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.8)
        for key in ("cmins", "cmaxes", "cbars"):
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

        # Overlay scatter (jittered, semi-transparent)
        for i, (values, color) in enumerate(zip(data, colors)):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(values))
            ax.scatter(i + jitter, values, s=2, color=color, alpha=0.18, zorder=2)

        # Annotate n per group
        for i, values in enumerate(data):
            ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.02 * max(v.max() for v in data if len(v)),
                    f"n={len(values)}", ha="center", va="top", fontsize=7, color="#555555")

        ax.set_xticks(range(len(subsets)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel.split(" (")[0], fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    savefig(fig, "fig13_variant_summary")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    params       = load_params()
    train_params = params["train"]
    manifest     = load_manifest()

    print("Loading MLflow run data ...")
    mlflow_df = load_mlflow_runs()
    print(f"  {len(mlflow_df)} child runs loaded")

    df = pd.read_csv("data/processed/features.csv")
    print(f"Loaded features.csv: {len(df)} rows, {len(df.columns)} columns")

    print("\n[fig8]  Predicted vs actual power ...")
    fig8_predicted_vs_actual(df, manifest, train_params)

    print("\n[fig9]  R2 heatmap ...")
    fig9_r2_heatmap(manifest, train_params)

    print("\n[fig10] Algorithm comparison — L2 power model ...")
    fig10_algo_comparison_l2(mlflow_df, manifest)

    print("\n[fig11] Algorithm comparison — L1 models ...")
    fig11_algo_comparison_l1(mlflow_df, train_params)

    print("\n[fig12] Interpretability: Ridge coef vs tree importances ...")
    fig12_interpretability(manifest, mlflow_df)

    print("\n[fig13] Variant performance summary ...")
    fig13_variant_summary(df, train_params)

    print("\nAll evaluation figures saved to reports/figures/")


if __name__ == "__main__":
    main()
