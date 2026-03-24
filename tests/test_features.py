"""Tests for feature engineering functions."""
import pandas as pd
from src.features import compute_sec


def test_compute_sec_per_gbps():
    """SEC (per_gbps) = power_watts / throughput_gbps."""
    df = pd.DataFrame({
        "power_watts": [100.0, 200.0, 50.0],
        "throughput_gbps": [1.0, 2.0, 0.5],
    })
    result = compute_sec(df, normalization="per_gbps")

    assert abs(result.iloc[0] - 100.0) < 1e-6   # 100W / 1Gbps
    assert abs(result.iloc[1] - 100.0) < 1e-6   # 200W / 2Gbps
    assert abs(result.iloc[2] - 100.0) < 1e-6   # 50W / 0.5Gbps


def test_compute_sec_zero_throughput():
    """Zero throughput should produce NaN, not crash."""
    df = pd.DataFrame({
        "power_watts": [100.0],
        "throughput_gbps": [0.0],
    })
    result = compute_sec(df, normalization="per_gbps")

    assert pd.isna(result.iloc[0])
