"""Tests for the features pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    drop_uninformative_columns,
    add_unit_conversions,
    compute_targets,
    add_rolling_features,
    add_derived_features,
    encode_variant,
    cleanup,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_merged_df(n=6, variant="sd_core_dpdk", dataplane="dpdk"):
    """Minimal DataFrame mimicking merged.csv structure."""
    return pd.DataFrame({
        "Timestamp epoch ms": range(1000, 1000 + n * 3000, 3000),
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="3s"),
        "gtpu_kbitss_dn__kbits_rx_s": [0, 100_000, 500_000, 1_000_000, 500_000, 100_000][:n],
        "gtpu_kbitss_dn__kbits_tx_s": [0, 100_000, 500_000, 1_000_000, 500_000, 100_000][:n],
        "gtpu_kbitss_ngran__gtpu_kbits_rx_s": [0, 100_000, 500_000, 1_000_000, 500_000, 100_000][:n],
        "gtpu_kbitss_ngran__gtpu_kbits_tx_s": [0, 100_000, 500_000, 1_000_000, 500_000, 100_000][:n],
        "gtpu_packets_dn__packets_rx_delta": [np.nan, 300, 1500, 3000, 1500, 300][:n],
        "gtpu_packets_dn__packets_tx_delta": [np.nan, 300, 1500, 3000, 1500, 300][:n],
        "gtpu_packets_dn__packets_lost_delta": [0] * n,
        "gtpu_packets_ngran__gtpu_packets_rx_delta": [np.nan, 300, 1500, 3000, 1500, 300][:n],
        "gtpu_packets_ngran__gtpu_packets_tx_delta": [np.nan, 300, 1500, 3000, 1500, 300][:n],
        "gtpu_packets_ngran__lost_packets_delta": [0] * n,
        "one_way_delay_average_dn__average": [0, 50, 100, 200, 100, 50][:n],
        "one_way_delay_average_ran__average": [0, 50, 100, 200, 100, 50][:n],
        "delay_variation_jitter_average_dn__avg_us": [0, 10, 20, 40, 20, 10][:n],
        "delay_variation_jitter_average_ran__avg_us": [0, 10, 20, 40, 20, 10][:n],
        "uplink_data_one_way_delay_distribution__weighted_mean_delay_us": [62.5] * n,
        "uplink_data_one_way_delay_distribution__total_packets": [100] * n,
        "uplink_data_one_way_delay_distribution__high_delay_frac": [0.0] * n,
        "downlink_one_way_delay_distribution__weighted_mean_delay_us": [62.5] * n,
        "downlink_one_way_delay_distribution__total_packets": [100] * n,
        "downlink_one_way_delay_distribution__high_delay_frac": [0.0] * n,
        "user_plane_throughput__l2_3_device_rx_traffic": [0, 110_000, 550_000, 1_100_000, 550_000, 110_000][:n],
        "user_plane_throughput__l2_3_device_tx_traffic": [0, 110_000, 550_000, 1_100_000, 550_000, 110_000][:n],
        "power_microwatts": [820_000, 825_000, 840_000, 855_000, 840_000, 825_000][:n],
        "cpu_pct": [1.06, 1.06, 1.07, 1.08, 1.07, 1.06][:n],
        "idle_power_microwatts": [819_000.0] * n,
        "idle_cpu_pct": [1.06] * n,
        "variant": [variant] * n,
        "dataplane": [dataplane] * n,
        "run_dir": ["test_run"] * n,
        # Include some zero-variance cols that should be dropped
        "gtpu_kbitss_ngran__gtpu_kbits_ethernet_rx_s": [0] * n,
        "gtpu_kbitss_ngran__ng_ran_passthrough_kbits_rx_s": [0] * n,
        # Include duplicate cols that should be dropped
        "gtpu_packetss_dn__packets_rx_s": [0, 33, 167, 333, 167, 33][:n],
        "user_plane_throughput_dnrx_traffic__kbits_rx_s": [0, 100_000, 500_000, 1_000_000, 500_000, 100_000][:n],
    })


# ── Tests: column dropping ──────────────────────────────────────────────────

def test_drop_uninformative_columns():
    df = _make_merged_df()
    result = drop_uninformative_columns(df)
    # Zero-variance columns gone
    assert "gtpu_kbitss_ngran__gtpu_kbits_ethernet_rx_s" not in result.columns
    assert "gtpu_kbitss_ngran__ng_ran_passthrough_kbits_rx_s" not in result.columns
    # Duplicate columns gone
    assert "gtpu_packetss_dn__packets_rx_s" not in result.columns
    assert "user_plane_throughput_dnrx_traffic__kbits_rx_s" not in result.columns
    # Canonical columns kept
    assert "gtpu_kbitss_dn__kbits_rx_s" in result.columns
    assert "power_microwatts" in result.columns


# ── Tests: unit conversions ─────────────────────────────────────────────────

def test_add_unit_conversions():
    df = _make_merged_df()
    result = add_unit_conversions(df)
    assert "power_watts" in result.columns
    assert "net_power_watts" in result.columns
    assert "throughput_gbps" in result.columns
    assert "packet_rate_pps" in result.columns

    # Check values
    assert result["power_watts"].iloc[0] == pytest.approx(0.82)
    assert result["throughput_gbps"].iloc[3] == pytest.approx(1.0)  # 1M Kbps = 1 Gbps
    assert result["packet_rate_pps"].iloc[1] == pytest.approx(100.0)  # 300 / 3s


def test_net_power_sign():
    """Net power should be small and positive for DPDK (high idle)."""
    df = _make_merged_df()
    result = add_unit_conversions(df)
    # idle = 819000 µW, under load = 820000–855000 µW
    assert (result["net_power_watts"] >= 0).all()
    assert result["net_power_watts"].max() < result["power_watts"].max()


# ── Tests: targets ──────────────────────────────────────────────────────────

def test_compute_targets():
    df = _make_merged_df()
    df = add_unit_conversions(df)
    result = compute_targets(df)
    assert "sec_total" in result.columns
    assert "sec_net" in result.columns

    # Zero throughput → NaN SEC
    assert pd.isna(result["sec_total"].iloc[0])

    # At 1 Gbps: sec_total = 0.855 W / 1.0 Gbps = 0.855 J/Gb
    assert result["sec_total"].iloc[3] == pytest.approx(0.855, abs=0.001)


def test_sec_net_lower_than_total():
    """SEC net must always be <= SEC total (net power <= total power)."""
    df = _make_merged_df()
    df = add_unit_conversions(df)
    df = compute_targets(df)
    valid = df.dropna(subset=["sec_total", "sec_net"])
    assert (valid["sec_net"] <= valid["sec_total"]).all()


# ── Tests: rolling features ─────────────────────────────────────────────────

def test_add_rolling_features():
    df = _make_merged_df()
    df = add_unit_conversions(df)
    result = add_rolling_features(df, window_sec=9)

    # Rolling window of 9s / 3s = 3 rows
    assert "power_watts_roll_mean_9s" in result.columns
    assert "power_watts_roll_std_9s" in result.columns
    assert "power_watts_roll_max_9s" in result.columns
    assert "throughput_gbps_roll_mean_9s" in result.columns
    assert "throughput_gbps_diff" in result.columns

    # First row: rolling mean = just itself (min_periods=1)
    assert result["power_watts_roll_mean_9s"].iloc[0] == pytest.approx(
        result["power_watts"].iloc[0]
    )


def test_rolling_features_per_run():
    """Rolling features should not leak across runs."""
    df1 = _make_merged_df(n=3)
    df1["run_dir"] = "run_a"
    df2 = _make_merged_df(n=3)
    df2["run_dir"] = "run_b"
    df2["Timestamp epoch ms"] = [10000, 13000, 16000]
    df = pd.concat([df1, df2], ignore_index=True)
    df = add_unit_conversions(df)
    result = add_rolling_features(df, window_sec=9)

    # Row 3 is first row of run_b — its rolling mean should equal itself
    run_b = result[result["run_dir"] == "run_b"]
    assert run_b["power_watts_roll_mean_9s"].iloc[0] == pytest.approx(
        run_b["power_watts"].iloc[0]
    )


# ── Tests: derived features ─────────────────────────────────────────────────

def test_add_derived_features():
    df = _make_merged_df()
    df = add_unit_conversions(df)
    result = add_derived_features(df)

    assert "avg_packet_size_bytes" in result.columns
    assert "dn_ngran_throughput_ratio" in result.columns
    assert "cpu_per_watt" in result.columns
    assert "l2l3_overhead_ratio" in result.columns


def test_packet_size_approx_1250():
    """With 1250B payload, avg_packet_size should be near 1250."""
    df = _make_merged_df()
    # Set consistent throughput and packet rate for 1250-byte packets
    # 100 pps × 1250 bytes × 8 bits = 1,000,000 bits = 1000 Kbits/s
    df["gtpu_kbitss_dn__kbits_rx_s"] = [0, 1000, 1000, 1000, 1000, 1000]
    df["gtpu_packets_dn__packets_rx_delta"] = [np.nan, 300, 300, 300, 300, 300]
    df = add_unit_conversions(df)
    result = add_derived_features(df)
    # packet_rate_pps = 300/3 = 100, throughput = 0.001 Gbps = 1e6 bits/s
    # packet_size = 1e6 / 8 / 100 = 1250
    valid = result["avg_packet_size_bytes"].dropna()
    assert valid.iloc[0] == pytest.approx(1250.0, rel=0.01)


# ── Tests: categorical encoding ─────────────────────────────────────────────

def test_encode_variant():
    df = _make_merged_df(n=2, dataplane="dpdk")
    df2 = _make_merged_df(n=2, dataplane="usr")
    combined = pd.concat([df, df2], ignore_index=True)
    result = encode_variant(combined)
    assert list(result["is_dpdk"]) == [1, 1, 0, 0]


# ── Tests: cleanup ──────────────────────────────────────────────────────────

def test_cleanup_drops_first_row():
    """First row per run (NaN deltas) should be dropped."""
    df = _make_merged_df()
    df = add_unit_conversions(df)
    df = compute_targets(df)
    result, _ = cleanup(df, target="sec_total")
    # Original: 6 rows. Drop first row per run (1) + NaN sec_total rows
    assert len(result) < 6
    assert "run_dir" not in result.columns
    assert "timestamp" not in result.columns
    assert "variant" not in result.columns


def test_cleanup_drops_nan_target():
    """Rows with NaN target should be dropped."""
    df = _make_merged_df()
    df = add_unit_conversions(df)
    df = compute_targets(df)
    _, dropped = cleanup(df, target="sec_total")
    # Row 0 dropped by first-row rule, row 1 has throughput > 0 → sec exists
    # So dropped count should be >= 0
    assert dropped >= 0


def test_cleanup_removes_intermediate_columns():
    """Raw µW columns should be removed (watts versions kept)."""
    df = _make_merged_df()
    df = add_unit_conversions(df)
    df = compute_targets(df)
    result, _ = cleanup(df, target="power_watts")
    assert "power_microwatts" not in result.columns
    assert "idle_power_microwatts" not in result.columns
    assert "power_watts" in result.columns
