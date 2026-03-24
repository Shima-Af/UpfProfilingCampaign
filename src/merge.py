"""
src/merge.py — Align LoadCore traffic CSVs with Scaphandre power/CPU measurements.

For each of the 220 test runs:
  1. Merge the 19 LoadCore CSVs on timestamp (wide join)
  2. Convert cumulative metrics to deltas, summarize delay distributions
  3. Filter Scaphandre to UPF PIDs, aggregate per timestamp
  4. Align LoadCore ↔ Scaphandre via merge_asof (3s tolerance)
  5. Attach idle-baseline power/CPU and metadata columns

Output: data/interim/merged.csv
"""
import json
import re
import warnings

import pandas as pd
import yaml
from pathlib import Path

# ── Column classification ────────────────────────────────────────────────────
# Cumulative CSVs: running totals → convert to per-interval deltas
_CUMULATIVE_STEMS = {"GTPuTraffic_DN", "GTPuTraffic_NGRAN",
                     "GTPuPackets_DN", "GTPuPackets_NGRAN"}
# Distribution CSVs: histogram bins → summarize
_DISTRIBUTION_STEMS = {"UplinkDataOneWayDelayDistribution",
                       "DownlinkOneWayDelayDistribution"}

# Bin midpoints (µs) for delay-distribution summarization
_BIN_MIDPOINTS_US = {
    "0us - 125us":   62.5,
    "125us - 250us": 187.5,
    "250us - 500us": 375.0,
    "500us - 1ms":   750.0,
    "1ms - 5ms":     3000.0,
    "5ms - 10ms":    7500.0,
    "10ms - 15ms":   12500.0,
    "15ms - 20ms":   17500.0,
    "20ms - inf":    25000.0,
}
_HIGH_DELAY_BINS = {"5ms - 10ms", "10ms - 15ms", "15ms - 20ms", "20ms - inf"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["merge"]


def _stem(csv_filename: str) -> str:
    """Strip the common prefix and .csv suffix.

    'Fullcoreapplicationtraffic_GTPuKbitss_DN.csv' → 'GTPuKbitss_DN'
    """
    name = csv_filename.removesuffix(".csv")
    name = name.removeprefix("Fullcoreapplicationtraffic_")
    return name


def build_csv_prefix(csv_filename: str) -> str:
    """Slugified prefix for column namespacing.

    'GTPuKbitss_DN' → 'gtpu_kbitss_dn'
    """
    stem = _stem(csv_filename)
    slug = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", stem)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", slug).strip("_").lower()
    return slug


def classify_csv(csv_filename: str) -> str:
    """Return 'cumulative', 'distribution', or 'rate'."""
    stem = _stem(csv_filename)
    if stem in _CUMULATIVE_STEMS:
        return "cumulative"
    if stem in _DISTRIBUTION_STEMS:
        return "distribution"
    return "rate"


def _metric_cols(df: pd.DataFrame) -> list[str]:
    """All columns except the two timestamp columns."""
    return [c for c in df.columns if c not in ("Timestamp epoch ms", "timestamp")]


# ── LoadCore processing ──────────────────────────────────────────────────────

def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rename metric columns with a prefix; keep timestamp columns as-is."""
    rename = {}
    for col in _metric_cols(df):
        slug = re.sub(r"[^A-Za-z0-9]+", "_", col).strip("_").lower()
        rename[col] = f"{prefix}__{slug}"
    return df.rename(columns=rename)


def _apply_deltas(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Convert cumulative metric columns to row-to-row deltas."""
    df = _prefix_columns(df, prefix)
    for col in _metric_cols(df):
        df[col] = df[col].diff()
    # Rename to indicate delta
    rename = {c: f"{c}_delta" for c in _metric_cols(df)}
    return df.rename(columns=rename)


def summarize_distribution(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Reduce a histogram-bin DataFrame to 3 summary columns per row."""
    bin_cols = [c for c in df.columns if c in _BIN_MIDPOINTS_US]
    if not bin_cols:
        warnings.warn(f"No recognized bin columns for {prefix}")
        return df[["Timestamp epoch ms"]].copy()

    counts = df[bin_cols].fillna(0)
    midpoints = pd.Series({c: _BIN_MIDPOINTS_US[c] for c in bin_cols})

    total = counts.sum(axis=1)
    weighted_sum = counts.mul(midpoints).sum(axis=1)

    high_delay_cols = [c for c in bin_cols if c in _HIGH_DELAY_BINS]
    high_count = counts[high_delay_cols].sum(axis=1) if high_delay_cols else 0

    out = df[["Timestamp epoch ms"]].copy()
    out[f"{prefix}__weighted_mean_delay_us"] = (weighted_sum / total.replace(0, float("nan")))
    out[f"{prefix}__total_packets"] = total
    out[f"{prefix}__high_delay_frac"] = (high_count / total.replace(0, float("nan")))
    return out


def merge_loadcore_run(run_dir: Path) -> pd.DataFrame:
    """Read all 19 CSVs in a run directory, merge into one wide DataFrame."""
    csvs = sorted(run_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {run_dir}")

    merged = None
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        prefix = build_csv_prefix(csv_path.name)
        category = classify_csv(csv_path.name)

        if category == "distribution":
            piece = summarize_distribution(df, prefix)
        elif category == "cumulative":
            piece = _apply_deltas(df, prefix)
            # Drop the human-readable timestamp from cumulative pieces
            piece = piece.drop(columns=["timestamp"], errors="ignore")
        else:
            piece = _prefix_columns(df, prefix)
            piece = piece.drop(columns=["timestamp"], errors="ignore")

        if merged is None:
            # Keep timestamp from the first CSV
            if category == "distribution":
                # Distribution pieces don't have 'timestamp'; get it from df
                merged = df[["Timestamp epoch ms", "timestamp"]].copy()
                merged = merged.merge(piece, on="Timestamp epoch ms", how="outer")
            else:
                merged = piece.copy()
                # Ensure human timestamp is present
                if "timestamp" not in merged.columns:
                    ts_df = pd.read_csv(csv_path, usecols=["Timestamp epoch ms", "timestamp"])
                    merged = ts_df.merge(merged, on="Timestamp epoch ms", how="outer")
        else:
            merged = merged.merge(
                piece.drop(columns=["timestamp"], errors="ignore"),
                on="Timestamp epoch ms",
                how="outer",
            )

    return merged


# ── Scaphandre processing ────────────────────────────────────────────────────

def load_upf_pids(raw_scaph_dir: Path, variant: str) -> list[int] | None:
    """Parse upf_PIDs.txt if it exists for this variant."""
    pid_file = raw_scaph_dir / variant / "upf_PIDs.txt"
    if not pid_file.exists():
        return None
    pids = []
    for line in pid_file.read_text().splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids if pids else None


def _load_scaphandre_csv(path: Path, pids: list[int] | None) -> pd.DataFrame:
    """Load a scaphandre CSV, filter to PIDs, aggregate by timestamp."""
    df = pd.read_csv(path, usecols=["timestamp", "pid", "value"])
    df["pid"] = pd.to_numeric(df["pid"], errors="coerce")

    if pids is not None:
        df = df[df["pid"].isin(pids)]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    agg = df.groupby("timestamp", as_index=False)["value"].sum()
    return agg.sort_values("timestamp").reset_index(drop=True)


def load_and_aggregate_scaphandre(
    interim_scaph: Path,
    raw_scaph: Path,
    mapping: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Return {variant_key: DataFrame[timestamp, power_microwatts, cpu_pct]}."""
    result = {}
    for variant_prefix, scaph_folder in mapping.items():
        folder = interim_scaph / scaph_folder
        if not folder.exists():
            warnings.warn(f"Scaphandre folder {folder} not found, skipping")
            continue

        pids = load_upf_pids(raw_scaph, scaph_folder)
        pid_info = f"filtered to PIDs {pids}" if pids else "all PIDs"
        print(f"  [{scaph_folder}] Loading scaphandre ({pid_info})")

        power_file = folder / "scaph_process_power_consumption_microwatts.csv"
        cpu_file = folder / "scaph_process_cpu_usage_percentage.csv"

        power = _load_scaphandre_csv(power_file, pids)
        power = power.rename(columns={"value": "power_microwatts"})

        cpu = _load_scaphandre_csv(cpu_file, pids)
        cpu = cpu.rename(columns={"value": "cpu_pct"})

        merged = power.merge(cpu, on="timestamp", how="outer").sort_values("timestamp")
        print(f"    {len(merged)} unique timestamps")
        result[scaph_folder] = merged

    return result


# ── Idle baseline ────────────────────────────────────────────────────────────

def compute_idle_baseline(
    scaph_df: pd.DataFrame,
    run_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, float]:
    """Compute mean power/CPU from scaphandre rows outside any run interval."""
    mask = pd.Series(False, index=scaph_df.index)
    for start, end in run_intervals:
        mask = mask | ((scaph_df["timestamp"] >= start) & (scaph_df["timestamp"] <= end))

    idle = scaph_df[~mask]
    if idle.empty:
        warnings.warn("No idle scaphandre rows found; baseline set to 0")
        return {"idle_power_microwatts": 0.0, "idle_cpu_pct": 0.0}

    return {
        "idle_power_microwatts": idle["power_microwatts"].mean(),
        "idle_cpu_pct": idle["cpu_pct"].mean(),
    }


# ── Timestamp alignment ─────────────────────────────────────────────────────

def align_timestamps(
    lc: pd.DataFrame,
    scaph: pd.DataFrame,
    tolerance_sec: int,
) -> pd.DataFrame:
    """Join LoadCore and Scaphandre on nearest timestamp within tolerance."""
    lc = lc.copy()
    lc["_ts"] = pd.to_datetime(lc["Timestamp epoch ms"], unit="ms", utc=True).dt.tz_localize(None)
    lc = lc.sort_values("_ts")

    scaph = scaph.sort_values("timestamp")

    merged = pd.merge_asof(
        lc,
        scaph,
        left_on="_ts",
        right_on="timestamp",
        tolerance=pd.Timedelta(seconds=tolerance_sec),
        direction="nearest",
    )
    merged = merged.drop(columns=["_ts", "timestamp"], errors="ignore")
    # merge_asof may create timestamp_x/timestamp_y; clean up
    if "timestamp_x" in merged.columns:
        merged = merged.rename(columns={"timestamp_x": "timestamp"})
    merged = merged.drop(columns=["timestamp_y"], errors="ignore")
    return merged


# ── Metadata extraction ──────────────────────────────────────────────────────

def _parse_run_dir(run_dir_name: str) -> dict[str, str]:
    """Extract variant and dataplane from run directory name.

    'sd_core_dpdk__2025-01-03-16-25-35' → variant='sd_core_dpdk', dataplane='dpdk'
    'oai_userspaceapplication__2025-01-13-13-08-54' → variant='oai_userspaceapplication', ...
    """
    parts = run_dir_name.split("__", 1)
    variant_key = parts[0]

    # Derive dataplane from the last segment of the variant key
    segments = variant_key.split("_")
    dataplane = segments[-1] if segments else "unknown"

    return {"variant": variant_key, "dataplane": dataplane}


# ── Main orchestration ───────────────────────────────────────────────────────

def main():
    params = load_params()
    tolerance = params["time_alignment_tolerance_sec"]
    interim_dir = Path(params["interim_dir"])
    raw_scaph_dir = Path(params["raw_scaphandre_dir"])
    mapping = params["scaphandre_mapping"]

    # Load discovery metadata
    with open(interim_dir / "discovery.json") as f:
        discovery = json.load(f)

    # ── Pre-aggregate Scaphandre (once) ──
    print("=== Loading Scaphandre data ===")
    scaph_data = load_and_aggregate_scaphandre(
        interim_dir / "scaphandre", raw_scaph_dir, mapping,
    )

    # ── Compute idle baselines per variant ──
    print("\n=== Computing idle baselines ===")
    loadcore_dir = interim_dir / "loadcore"
    run_dirs = sorted(loadcore_dir.iterdir())

    # Group runs by variant prefix
    variant_runs: dict[str, list[Path]] = {}
    for rd in run_dirs:
        if not rd.is_dir():
            continue
        meta = _parse_run_dir(rd.name)
        variant_runs.setdefault(meta["variant"], []).append(rd)

    baselines: dict[str, dict] = {}
    for variant_prefix, scaph_folder in mapping.items():
        if variant_prefix not in variant_runs:
            print(f"  WARNING: no runs found for {variant_prefix}")
            continue
        if scaph_folder not in scaph_data:
            print(f"  WARNING: no scaphandre data for {scaph_folder}")
            continue

        scaph_df = scaph_data[scaph_folder]
        runs = variant_runs[variant_prefix]

        # Build run intervals from LoadCore timestamps
        intervals = []
        for rd in runs:
            sample_csv = next(rd.glob("*.csv"), None)
            if sample_csv is None:
                continue
            ts_col = pd.read_csv(sample_csv, usecols=["Timestamp epoch ms"])
            epoch_ms = ts_col["Timestamp epoch ms"]
            start = pd.to_datetime(epoch_ms.min(), unit="ms")
            end = pd.to_datetime(epoch_ms.max(), unit="ms")
            intervals.append((start, end))

        baseline = compute_idle_baseline(scaph_df, intervals)
        baselines[scaph_folder] = baseline
        print(f"  [{scaph_folder}] idle power={baseline['idle_power_microwatts']:.0f} µW, "
              f"idle CPU={baseline['idle_cpu_pct']:.2f}%")

    # ── Process each run ──
    print(f"\n=== Merging {len(discovery['zips'])} runs (tolerance={tolerance}s) ===")
    all_frames = []
    for i, zip_info in enumerate(discovery["zips"]):
        run_dir_name = zip_info["run_dir"]
        rd = loadcore_dir / run_dir_name
        if not rd.exists():
            print(f"  WARNING: {run_dir_name} not found, skipping")
            continue

        meta = _parse_run_dir(run_dir_name)
        variant_prefix = meta["variant"]
        scaph_folder = mapping.get(variant_prefix)

        if scaph_folder is None or scaph_folder not in scaph_data:
            print(f"  WARNING: No scaphandre mapping for {variant_prefix}, skipping")
            continue

        # 1. Merge LoadCore CSVs for this run
        lc = merge_loadcore_run(rd)

        # 2. Slice scaphandre to this run's time window (with buffer)
        epoch_ms = lc["Timestamp epoch ms"]
        run_start = pd.to_datetime(epoch_ms.min(), unit="ms") - pd.Timedelta(seconds=tolerance)
        run_end = pd.to_datetime(epoch_ms.max(), unit="ms") + pd.Timedelta(seconds=tolerance)

        scaph_full = scaph_data[scaph_folder]
        scaph_slice = scaph_full[
            (scaph_full["timestamp"] >= run_start) &
            (scaph_full["timestamp"] <= run_end)
        ].copy()

        # 3. Align timestamps
        merged = align_timestamps(lc, scaph_slice, tolerance)

        # 4. Add metadata and baseline
        merged["variant"] = meta["variant"]
        merged["dataplane"] = meta["dataplane"]
        merged["run_dir"] = run_dir_name

        baseline = baselines.get(scaph_folder, {})
        merged["idle_power_microwatts"] = baseline.get("idle_power_microwatts", 0.0)
        merged["idle_cpu_pct"] = baseline.get("idle_cpu_pct", 0.0)

        all_frames.append(merged)

        if (i + 1) % 50 == 0 or (i + 1) == len(discovery["zips"]):
            print(f"  Processed {i + 1}/{len(discovery['zips'])} runs")

    # ── Concatenate and save ──
    if not all_frames:
        raise RuntimeError("No runs were successfully merged")

    result = pd.concat(all_frames, ignore_index=True)

    # Check alignment quality
    n_power_nan = result["power_microwatts"].isna().sum()
    if n_power_nan > 0:
        pct = n_power_nan / len(result) * 100
        print(f"\n  WARNING: {n_power_nan} rows ({pct:.1f}%) missing power data "
              "(timestamp misalignment)")

    output_path = interim_dir / "merged.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}: {len(result)} rows × {len(result.columns)} columns")


if __name__ == "__main__":
    main()
