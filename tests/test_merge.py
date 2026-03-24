"""Tests for the merge pipeline."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.merge import (
    build_csv_prefix,
    classify_csv,
    summarize_distribution,
    merge_loadcore_run,
    load_upf_pids,
    compute_idle_baseline,
    align_timestamps,
)


# ── Unit tests: helpers ──────────────────────────────────────────────────────

def test_build_csv_prefix_rate():
    assert build_csv_prefix("Fullcoreapplicationtraffic_GTPuKbitss_DN.csv") == "gtpu_kbitss_dn"


def test_build_csv_prefix_distribution():
    assert build_csv_prefix(
        "Fullcoreapplicationtraffic_UplinkDataOneWayDelayDistribution.csv"
    ) == "uplink_data_one_way_delay_distribution"


def test_classify_csv_rate():
    assert classify_csv("Fullcoreapplicationtraffic_GTPuKbitss_DN.csv") == "rate"
    # GTPuPacketss (double s) = rate, not cumulative
    assert classify_csv("Fullcoreapplicationtraffic_GTPuPacketss_DN.csv") == "rate"


def test_classify_csv_cumulative():
    assert classify_csv("Fullcoreapplicationtraffic_GTPuTraffic_DN.csv") == "cumulative"
    assert classify_csv("Fullcoreapplicationtraffic_GTPuPackets_DN.csv") == "cumulative"


def test_classify_csv_distribution():
    assert classify_csv(
        "Fullcoreapplicationtraffic_UplinkDataOneWayDelayDistribution.csv"
    ) == "distribution"


# ── Unit tests: distribution summarization ───────────────────────────────────

def test_summarize_distribution_basic():
    """Weighted mean, total, and high-delay fraction from histogram bins."""
    df = pd.DataFrame({
        "Timestamp epoch ms": [1000, 2000],
        "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:03Z"],
        "0us - 125us": [10, 0],
        "125us - 250us": [0, 0],
        "250us - 500us": [0, 0],
        "500us - 1ms": [0, 0],
        "1ms - 5ms": [0, 0],
        "5ms - 10ms": [0, 5],
        "10ms - 15ms": [0, 5],
        "15ms - 20ms": [0, 0],
        "20ms - inf": [0, 0],
    })
    result = summarize_distribution(df, "test")

    assert len(result) == 2
    # Row 0: all 10 packets in 0-125us bin → mean = 62.5
    assert result["test__weighted_mean_delay_us"].iloc[0] == pytest.approx(62.5)
    assert result["test__total_packets"].iloc[0] == 10
    assert result["test__high_delay_frac"].iloc[0] == 0.0

    # Row 1: 5 in 5-10ms, 5 in 10-15ms → mean = 10000, high_frac = 1.0
    assert result["test__weighted_mean_delay_us"].iloc[1] == pytest.approx(10000.0)
    assert result["test__high_delay_frac"].iloc[1] == pytest.approx(1.0)


def test_summarize_distribution_zero_packets():
    """Zero packets should produce NaN for mean and fraction."""
    df = pd.DataFrame({
        "Timestamp epoch ms": [1000],
        "0us - 125us": [0],
        "5ms - 10ms": [0],
    })
    result = summarize_distribution(df, "test")
    assert pd.isna(result["test__weighted_mean_delay_us"].iloc[0])
    assert pd.isna(result["test__high_delay_frac"].iloc[0])
    assert result["test__total_packets"].iloc[0] == 0


# ── Unit tests: PID loading ──────────────────────────────────────────────────

def test_load_upf_pids():
    """Parse PIDs from the text file format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        variant_dir = Path(tmpdir) / "dpdk"
        variant_dir.mkdir()
        pid_file = variant_dir / "upf_PIDs.txt"
        pid_file.write_text(
            "Container: abc123, Image: upf-epc-bess:1.5.0\n"
            "1043174\n"
            "Container: def456, Image: upf-epc-pfcpiface:1.5.0\n"
            "1043099\n"
        )
        pids = load_upf_pids(Path(tmpdir), "dpdk")
        assert pids == [1043174, 1043099]


def test_load_upf_pids_missing_file():
    """Return None when no PID file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "usr").mkdir()
        assert load_upf_pids(Path(tmpdir), "usr") is None


# ── Unit tests: idle baseline ────────────────────────────────────────────────

def test_compute_idle_baseline():
    """Rows outside run intervals should form the baseline."""
    ts = pd.date_range("2025-01-01 00:00", periods=10, freq="3s")
    scaph = pd.DataFrame({
        "timestamp": ts,
        "power_microwatts": [100] * 4 + [500] * 3 + [100] * 3,
        "cpu_pct": [1.0] * 4 + [5.0] * 3 + [1.0] * 3,
    })
    # Run covers indices 4-6 (the high-power period)
    intervals = [(ts[4], ts[6])]
    baseline = compute_idle_baseline(scaph, intervals)

    assert baseline["idle_power_microwatts"] == pytest.approx(100.0)
    assert baseline["idle_cpu_pct"] == pytest.approx(1.0)


# ── Unit tests: timestamp alignment ─────────────────────────────────────────

def test_align_timestamps_basic():
    """merge_asof should join on nearest timestamp within tolerance."""
    lc = pd.DataFrame({
        "Timestamp epoch ms": [1735921536000, 1735921539000, 1735921542000],
        "timestamp": [
            "2025-01-03T16:25:36.000Z",
            "2025-01-03T16:25:39.000Z",
            "2025-01-03T16:25:42.000Z",
        ],
        "kbits": [10, 20, 30],
    })
    scaph = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-01-03 16:25:36.246",
            "2025-01-03 16:25:39.244",
            "2025-01-03 16:25:42.244",
        ]),
        "power_microwatts": [100, 200, 300],
    })
    result = align_timestamps(lc, scaph, tolerance_sec=3)

    assert len(result) == 3
    assert "power_microwatts" in result.columns
    assert result["power_microwatts"].notna().all()
    assert list(result["kbits"]) == [10, 20, 30]


def test_align_timestamps_no_match():
    """Scaphandre rows too far away should produce NaN."""
    lc = pd.DataFrame({
        "Timestamp epoch ms": [1735921536000],
        "timestamp": ["2025-01-03T16:25:36.000Z"],
        "kbits": [10],
    })
    scaph = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-03 17:00:00"]),
        "power_microwatts": [999],
    })
    result = align_timestamps(lc, scaph, tolerance_sec=3)
    assert pd.isna(result["power_microwatts"].iloc[0])


# ── Integration test: merge_loadcore_run ─────────────────────────────────────

def test_merge_loadcore_run():
    """Merge a minimal set of CSVs in a temp run directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Rate CSV
        pd.DataFrame({
            "Timestamp epoch ms": [1000, 2000, 3000],
            "timestamp": ["t1", "t2", "t3"],
            "Kbits Rx/s": [10, 20, 30],
            "Kbits Tx/s": [10, 20, 30],
        }).to_csv(run_dir / "Fullcoreapplicationtraffic_GTPuKbitss_DN.csv", index=False)

        # Cumulative CSV
        pd.DataFrame({
            "Timestamp epoch ms": [1000, 2000, 3000],
            "timestamp": ["t1", "t2", "t3"],
            "Bytes Rx": [100, 300, 600],
            "Bytes Tx": [100, 300, 600],
        }).to_csv(run_dir / "Fullcoreapplicationtraffic_GTPuTraffic_DN.csv", index=False)

        # Distribution CSV
        pd.DataFrame({
            "Timestamp epoch ms": [1000, 2000, 3000],
            "timestamp": ["t1", "t2", "t3"],
            "0us - 125us": [5, 10, 15],
            "5ms - 10ms": [0, 1, 2],
        }).to_csv(
            run_dir / "Fullcoreapplicationtraffic_UplinkDataOneWayDelayDistribution.csv",
            index=False,
        )

        result = merge_loadcore_run(run_dir)

        assert len(result) == 3
        # Rate columns present and prefixed
        assert "gtpu_kbitss_dn__kbits_rx_s" in result.columns
        # Delta columns present
        assert "gtpu_traffic_dn__bytes_rx_delta" in result.columns
        # First delta row is NaN
        assert pd.isna(result["gtpu_traffic_dn__bytes_rx_delta"].iloc[0])
        # Second delta = 300 - 100 = 200
        assert result["gtpu_traffic_dn__bytes_rx_delta"].iloc[1] == 200
        # Distribution summary present
        assert "uplink_data_one_way_delay_distribution__weighted_mean_delay_us" in result.columns
