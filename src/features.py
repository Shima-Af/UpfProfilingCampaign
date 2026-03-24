"""
src/features.py — Feature engineering for UPF energy profiling.

Reads the merged LoadCore + Scaphandre dataset and computes features
relevant to predicting Specific Energy Consumption (SEC).

TODO: Adapt column names to match your actual merged.csv columns.
"""
import pandas as pd
import yaml
from pathlib import Path


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["features"]


def compute_sec(df: pd.DataFrame, normalization: str = "per_gbps") -> pd.Series:
    """Compute Specific Energy Consumption.

    SEC = energy consumed / data transferred

    TODO: Replace column names with your actual Scaphandre/LoadCore columns.
    """
    # --- ADAPT THESE TO YOUR ACTUAL COLUMN NAMES ---
    # Scaphandre typically exports: host_power_microwatts or power_watts
    # LoadCore typically exports: throughput_bps or throughput_gbps

    if normalization == "per_gbps":
        # SEC = power (watts) / throughput (gbps)
        # Units: Joules per Gigabit = Watts / (Gbps)
        sec = df["power_watts"] / df["throughput_gbps"].replace(0, float("nan"))
    elif normalization == "per_packet":
        sec = df["power_watts"] / df["packet_rate_pps"].replace(0, float("nan"))
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return sec


def add_rolling_features(df: pd.DataFrame, window_sec: int) -> pd.DataFrame:
    """Add rolling window statistics for power and throughput.

    These capture temporal patterns — e.g., power spikes lag behind
    throughput changes due to DPDK polling behavior.
    """
    # --- ADAPT COLUMN NAMES ---
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    window = f"{window_sec}s"

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

        for col in ["power_watts", "throughput_gbps"]:
            if col in df.columns:
                rolling = df[col].rolling(window, min_periods=1)
                df[f"{col}_mean_{window_sec}s"] = rolling.mean()
                df[f"{col}_std_{window_sec}s"] = rolling.std().fillna(0)
                df[f"{col}_max_{window_sec}s"] = rolling.max()

        df = df.reset_index()

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific derived features.

    These encode knowledge about how UPF energy relates to traffic patterns.
    """
    # Power-throughput ratio (instantaneous efficiency)
    if "power_watts" in df.columns and "throughput_gbps" in df.columns:
        df["power_throughput_ratio"] = (
            df["power_watts"] / df["throughput_gbps"].replace(0, float("nan"))
        )

    # CPU-power efficiency
    if all(c in df.columns for c in ["throughput_gbps", "cpu_usage_percent", "power_watts"]):
        df["cpu_power_efficiency"] = (
            df["throughput_gbps"]
            / (df["cpu_usage_percent"].replace(0, float("nan")) * df["power_watts"])
        )

    # Packet size (affects DPDK batching efficiency)
    if "throughput_gbps" in df.columns and "packet_rate_pps" in df.columns:
        # bits / packets = bits per packet → bytes per packet
        df["avg_packet_size_bytes"] = (
            (df["throughput_gbps"] * 1e9) / df["packet_rate_pps"].replace(0, float("nan")) / 8
        )

    return df


def main():
    params = load_params()

    print("=== Loading merged dataset ===")
    df = pd.read_csv("data/interim/merged.csv", parse_dates=["timestamp"])
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print("\n=== Computing SEC (target) ===")
    df["sec"] = compute_sec(df, params["sec_normalization"])
    print(f"  SEC: mean={df['sec'].mean():.4f}, std={df['sec'].std():.4f}")

    print("\n=== Adding rolling features ===")
    df = add_rolling_features(df, params["rolling_window_sec"])

    print("\n=== Adding derived features ===")
    df = add_derived_features(df)

    # Drop rows where SEC couldn't be computed (zero throughput)
    before = len(df)
    df = df.dropna(subset=["sec"])
    print(f"\n  Dropped {before - len(df)} rows with NaN SEC")

    output_path = params["features_output"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved feature matrix to {output_path} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
