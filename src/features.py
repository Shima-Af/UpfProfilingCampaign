"""
src/features.py — Feature engineering for UPF energy profiling.

Reads data/interim/merged.csv (LoadCore + Scaphandre aligned by timestamp) and
produces data/processed/features.csv with:
  - Cleaned canonical traffic columns (zero-variance and duplicates removed)
  - Unit conversions (power_watts, throughput_gbps)
  - Multiple prediction targets (power_watts, net_power_watts, sec_total, sec_net)
  - Rolling temporal features (mean/std/max over configurable window)
  - Derived cross-metric features (packet size, bidirectional ratio, etc.)
  - Categorical encoding of UPF variant
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


# ── Column selection ─────────────────────────────────────────────────────────
# These 44 columns are always zero in our test campaign (no dedicated bearers,
# no ethernet passthrough, no RTP/video, no QoS enforcement).
_ZERO_VARIANCE_COLS = {
    "gtpu_kbitss_ngran__gtpu_kbits_ethernet_rx_s",
    "gtpu_kbitss_ngran__gtpu_kbits_ethernet_tx_s",
    "gtpu_kbitss_ngran__gtpu_kbits_rx_on_dedicated_bearers_s",
    "gtpu_kbitss_ngran__gtpu_kbits_tx_on_dedicated_bearers_s",
    "gtpu_kbitss_ngran__ng_ran_passthrough_kbits_rx_s",
    "gtpu_kbitss_ngran__ng_ran_passthrough_kbits_tx_s",
    "gtpu_packets_dn__packets_rx_exceeding_qos_settings_delta",
    "gtpu_packets_dn__packets_tx_dropped_due_to_qos_enforcement_delta",
    "gtpu_packets_ngran__gtpu_packets_ethernet_rx_delta",
    "gtpu_packets_ngran__gtpu_packets_ethernet_tx_delta",
    "gtpu_packets_ngran__gtpu_packets_rx_on_dedicated_bearers_delta",
    "gtpu_packets_ngran__gtpu_packets_tx_on_dedicated_bearers_delta",
    "gtpu_packets_ngran__ng_ran_passthrough_packets_rx_delta",
    "gtpu_packets_ngran__ng_ran_passthrough_packets_tx_delta",
    "gtpu_packets_ngran__packets_rx_exceeding_qos_settings_delta",
    "gtpu_packets_ngran__packets_tx_dropped_due_to_qos_enforcement_delta",
    "gtpu_packetss_ngran__gtpu_packets_ethernet_rx_s",
    "gtpu_packetss_ngran__gtpu_packets_ethernet_tx_s",
    "gtpu_packetss_ngran__gtpu_packets_rx_on_dedicated_bearers_s",
    "gtpu_packetss_ngran__gtpu_packets_tx_on_dedicated_bearers_s",
    "gtpu_packetss_ngran__ng_ran_passthrough_packets_rx_s",
    "gtpu_packetss_ngran__ng_ran_passthrough_packets_tx_s",
    "gtpu_traffic_ngran__gtpu_bytes_ethernet_rx_delta",
    "gtpu_traffic_ngran__gtpu_bytes_ethernet_tx_delta",
    "gtpu_traffic_ngran__gtpu_bytes_rx_on_dedicated_bearers_delta",
    "gtpu_traffic_ngran__gtpu_bytes_tx_on_dedicated_bearers_delta",
    "gtpu_traffic_ngran__ng_ran_passthrough_bytes_rx_delta",
    "gtpu_traffic_ngran__ng_ran_passthrough_bytes_tx_delta",
    "user_plane_throughput_dnrx_traffic__data_kbits_rx_s",
    "user_plane_throughput_dnrx_traffic__kbits_received_s",
    "user_plane_throughput_dnrx_traffic__rtp_rx_kbits_s",
    "user_plane_throughput_dnrx_traffic__video_rtp_rx_kbits_s",
    "user_plane_throughput_dntx_traffic__data_kbits_tx_s",
    "user_plane_throughput_dntx_traffic__kbits_sent_s",
    "user_plane_throughput_dntx_traffic__rtp_tx_kbits_s",
    "user_plane_throughput_dntx_traffic__video_rtp_tx_kbits_s",
    "user_plane_throughput_ranrx_traffic__data_kbits_rx_s",
    "user_plane_throughput_ranrx_traffic__kbits_received_s",
    "user_plane_throughput_ranrx_traffic__rtp_rx_kbits_s",
    "user_plane_throughput_ranrx_traffic__video_rtp_rx_kbits_s",
    "user_plane_throughput_rantx_traffic__data_kbits_tx_s",
    "user_plane_throughput_rantx_traffic__kbits_sent_s",
    "user_plane_throughput_rantx_traffic__rtp_tx_kbits_s",
    "user_plane_throughput_rantx_traffic__video_rtp_tx_kbits_s",
}

# Near-duplicate columns (r > 0.999).  For each cluster we keep the most
# interpretable canonical column and drop the rest.
_DUPLICATE_COLS = {
    # Cluster 2: DN Rx — keep gtpu_kbitss_dn__kbits_rx_s
    "gtpu_packetss_dn__packets_rx_s",
    "user_plane_throughput_dnrx_traffic__kbits_rx_s",
    # Cluster 3: DN Tx — keep gtpu_kbitss_dn__kbits_tx_s
    "gtpu_packetss_dn__packets_tx_s",
    "user_plane_throughput_dntx_traffic__kbits_tx_s",
    # Cluster 4: NGRAN Rx — keep gtpu_kbitss_ngran__gtpu_kbits_rx_s
    "gtpu_kbitss_ngran__gtpu_kbits_rx_on_default_bearers_s",
    "gtpu_packetss_ngran__gtpu_packets_rx_on_default_bearers_s",
    "gtpu_packetss_ngran__gtpu_packets_rx_s",
    "user_plane_throughput_ranrx_traffic__gtpu_kbits_rx_s",
    # Cluster 5: NGRAN Tx — keep gtpu_kbitss_ngran__gtpu_kbits_tx_s
    "gtpu_kbitss_ngran__gtpu_kbits_tx_on_default_bearers_s",
    "gtpu_packetss_ngran__gtpu_packets_tx_on_default_bearers_s",
    "gtpu_packetss_ngran__gtpu_packets_tx_s",
    "user_plane_throughput_rantx_traffic__gtpu_kbits_tx_s",
    # Cluster 6: DN Rx delta — keep gtpu_packets_dn__packets_rx_delta
    "gtpu_traffic_dn__bytes_rx_delta",
    # Cluster 7: DN Tx delta — keep gtpu_packets_dn__packets_tx_delta
    "gtpu_traffic_dn__bytes_tx_delta",
    # Cluster 8: NGRAN Rx delta — keep gtpu_packets_ngran__gtpu_packets_rx_delta
    "gtpu_packets_ngran__gtpu_packets_rx_on_default_bearers_delta",
    "gtpu_traffic_ngran__gtpu_bytes_rx_delta",
    "gtpu_traffic_ngran__gtpu_bytes_rx_on_default_bearers_delta",
    # Cluster 9: NGRAN Tx delta — keep gtpu_packets_ngran__gtpu_packets_tx_delta
    "gtpu_packets_ngran__gtpu_packets_tx_on_default_bearers_delta",
    "gtpu_traffic_ngran__gtpu_bytes_tx_delta",
    "gtpu_traffic_ngran__gtpu_bytes_tx_on_default_bearers_delta",
}

# Metadata columns — used during processing but excluded from final features
_META_COLS = {"Timestamp epoch ms", "timestamp", "variant", "dataplane", "run_dir"}


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["features"]


# ── Column cleaning ──────────────────────────────────────────────────────────

def drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove zero-variance and near-duplicate columns."""
    to_drop = (_ZERO_VARIANCE_COLS | _DUPLICATE_COLS) & set(df.columns)
    return df.drop(columns=list(to_drop))


# ── Unit conversions & targets ───────────────────────────────────────────────

def add_unit_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """Add power_watts and throughput_gbps from raw columns."""
    df["power_watts"] = df["power_microwatts"] / 1e6
    df["idle_power_watts"] = df["idle_power_microwatts"] / 1e6
    df["net_power_watts"] = df["power_watts"] - df["idle_power_watts"]

    # Primary throughput: DN Rx Kbits/s → Gbps  (downlink to DN = main load direction)
    df["throughput_gbps"] = df["gtpu_kbitss_dn__kbits_rx_s"] / 1e6

    # Packet rate: DN Rx packets per interval (from delta column)
    # Delta is per-interval (3s), convert to per-second
    if "gtpu_packets_dn__packets_rx_delta" in df.columns:
        df["packet_rate_pps"] = df["gtpu_packets_dn__packets_rx_delta"] / 3.0

    return df


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all four prediction targets."""
    safe_tp = df["throughput_gbps"].replace(0, np.nan)

    # SEC total: true energy cost per bit (W / Gbps = J/Gb)
    df["sec_total"] = df["power_watts"] / safe_tp

    # SEC net: marginal energy cost (rewards efficient data path)
    df["sec_net"] = df["net_power_watts"] / safe_tp

    return df


# ── Rolling temporal features ────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, window_sec: int) -> pd.DataFrame:
    """Add rolling mean/std/max within each run for power and throughput.

    These capture temporal dynamics — e.g., DPDK poll-mode driver has power
    that lags behind throughput changes.
    """
    roll_cols = ["power_watts", "throughput_gbps"]
    window = window_sec // 3  # 3s sample interval → number of rows

    frames = []
    for _, group in df.groupby("run_dir", sort=False):
        g = group.sort_values("Timestamp epoch ms").copy()
        for col in roll_cols:
            if col not in g.columns:
                continue
            rolling = g[col].rolling(window, min_periods=1)
            g[f"{col}_roll_mean_{window_sec}s"] = rolling.mean()
            g[f"{col}_roll_std_{window_sec}s"] = rolling.std().fillna(0)
            g[f"{col}_roll_max_{window_sec}s"] = rolling.max()

        # Rate of change of throughput (first difference)
        if "throughput_gbps" in g.columns:
            g["throughput_gbps_diff"] = g["throughput_gbps"].diff().fillna(0)

        frames.append(g)

    return pd.concat(frames, ignore_index=True)


# ── Derived cross-metric features ───────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-specific features encoding UPF energy-traffic relationships."""

    # Average packet size (bytes).  Fixed at ~1250 in this campaign, but
    # still useful: it captures measurement noise and edge effects.
    if "throughput_gbps" in df.columns and "packet_rate_pps" in df.columns:
        safe_pps = df["packet_rate_pps"].replace(0, np.nan)
        df["avg_packet_size_bytes"] = (df["throughput_gbps"] * 1e9 / 8) / safe_pps

    # Bidirectional throughput ratio: DN Rx / NGRAN Rx
    # Deviation from 1.0 suggests encap overhead or loss
    dn_rx = df.get("gtpu_kbitss_dn__kbits_rx_s")
    ngran_rx = df.get("gtpu_kbitss_ngran__gtpu_kbits_rx_s")
    if dn_rx is not None and ngran_rx is not None:
        safe_ngran = ngran_rx.replace(0, np.nan)
        df["dn_ngran_throughput_ratio"] = dn_rx / safe_ngran

    # CPU-power coupling: how linearly CPU tracks power
    if "cpu_pct" in df.columns and "power_watts" in df.columns:
        safe_power = df["power_watts"].replace(0, np.nan)
        df["cpu_per_watt"] = df["cpu_pct"] / safe_power

    # L2/L3 overhead ratio: L2/L3 device throughput vs GTP-u throughput
    # If >1, header/framing overhead is significant
    l2l3_rx = df.get("user_plane_throughput__l2_3_device_rx_traffic")
    if l2l3_rx is not None and dn_rx is not None:
        safe_dn = dn_rx.replace(0, np.nan)
        df["l2l3_overhead_ratio"] = l2l3_rx / safe_dn

    # Delay-related: high delay fraction × power  (congestion-energy coupling)
    for direction in ("uplink_data_one_way_delay_distribution",
                      "downlink_one_way_delay_distribution"):
        frac_col = f"{direction}__high_delay_frac"
        if frac_col in df.columns:
            df[f"{direction}__delay_power_product"] = (
                df[frac_col].fillna(0) * df["power_watts"]
            )

    return df


# ── Categorical encoding ────────────────────────────────────────────────────

def encode_variant(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the UPF variant (only 2 levels: dpdk vs usr)."""
    df["is_dpdk"] = (df["dataplane"] == "dpdk").astype(int)
    return df


# ── Cleanup ──────────────────────────────────────────────────────────────────

def cleanup(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Drop first row per run (NaN deltas), rows with NaN target, and metadata."""
    # First row of each run has NaN deltas from .diff()
    mask = df.groupby("run_dir").cumcount() > 0
    df = df[mask].copy()

    # Drop rows where primary target is NaN (zero throughput / idle)
    before = len(df)
    df = df.dropna(subset=[target])
    dropped = before - len(df)

    # Drop metadata columns
    drop_meta = list(_META_COLS & set(df.columns))
    df = df.drop(columns=drop_meta)

    # Drop raw columns that were only needed for derived features
    intermediate = ["power_microwatts", "idle_power_microwatts", "idle_power_watts"]
    df = df.drop(columns=[c for c in intermediate if c in df.columns])

    return df, dropped


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    params = load_params()
    target = params["target"]
    window = params["rolling_window_sec"]

    print("=== Loading merged dataset ===")
    df = pd.read_csv("data/interim/merged.csv")
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print("\n=== Dropping uninformative columns ===")
    df = drop_uninformative_columns(df)
    print(f"  {len(df.columns)} columns after cleanup")

    print("\n=== Adding unit conversions ===")
    df = add_unit_conversions(df)

    print("\n=== Computing prediction targets ===")
    df = compute_targets(df)
    for t in ("sec_total", "sec_net", "power_watts", "net_power_watts"):
        valid = df[t].dropna()
        print(f"  {t}: mean={valid.mean():.4f}, std={valid.std():.4f}")

    print(f"\n=== Adding rolling features (window={window}s) ===")
    df = add_rolling_features(df, window)

    print("\n=== Adding derived features ===")
    df = add_derived_features(df)

    print("\n=== Encoding categorical (variant) ===")
    df = encode_variant(df)

    print(f"\n=== Cleanup (target={target}) ===")
    df, dropped = cleanup(df, target)
    print(f"  Dropped {dropped} rows with NaN {target}")
    print(f"  Final: {len(df)} rows, {len(df.columns)} columns")

    output_path = params["features_output"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
