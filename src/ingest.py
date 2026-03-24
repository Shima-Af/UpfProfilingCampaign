"""
src/ingest.py — Extract CSVs from LoadCore zip files and pair with Scaphandre data.

Each zip file corresponds to one UPF deployment test run and contains the same
19 LoadCore result CSVs. The manifest defines which CSVs to extract (shared across
all zips) and maps each zip to a deployment configuration.

Zip filename convention:
  CSVs-{id} - baremetal _{variant}_{dataplane} (copy from {date})-{timestamp}-UTC.zip
"""
import json
import re
import zipfile
import shutil
import yaml
from pathlib import Path
from datetime import datetime


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["ingest"]


def parse_zip_filename(filename: str) -> dict:
    """Extract metadata from a LoadCore zip filename.

    Example: 'CSVs-248 - baremetal _SD-Core_DPDK (copy from Jan  3 23-35-44)-2025-01-03-23-35-50-UTC.zip'
    Returns: {
        'csv_id': '248',
        'variant_raw': 'SD-Core_DPDK',
        'timestamp_str': '2025-01-03-23-35-50',
        'timestamp': datetime(2025, 1, 3, 23, 35, 50)
    }
    """
    info = {"filename": filename}

    # Extract CSV ID (e.g., 248, 320)
    id_match = re.search(r"CSVs-(\d+)", filename)
    if id_match:
        info["csv_id"] = id_match.group(1)

    # Extract variant info between 'baremetal _' and ' (copy'
    variant_match = re.search(r"baremetal\s*_(.+?)\s*\(copy", filename)
    if variant_match:
        info["variant_raw"] = variant_match.group(1).strip()

    # Extract ISO-style timestamp before -UTC.zip
    ts_match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-UTC\.zip", filename)
    if ts_match:
        info["timestamp_str"] = ts_match.group(1)
        try:
            info["timestamp"] = datetime.strptime(
                ts_match.group(1), "%Y-%m-%d-%H-%M-%S"
            )
        except ValueError:
            pass

    return info


def slugify(value: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    cleaned = cleaned.strip("_").lower()
    return cleaned or "unknown"


def build_run_dir_name(zip_info: dict) -> str:
    """Build a folder name using variant and timestamp."""
    variant = slugify(zip_info.get("variant_raw", "unknown"))
    timestamp_str = zip_info.get("timestamp_str") or "no_ts"
    return f"{variant}__{timestamp_str}"


def extract_from_zip(
    zip_path: Path,
    files_to_extract: list[str],
    output_dir: Path,
) -> list[str]:
    """Extract specific CSVs from a single zip file.

    Returns list of extracted file paths.
    """
    extracted = []

    if not zip_path.exists():
        print(f"  WARNING: {zip_path.name} not found, skipping")
        return extracted

    with zipfile.ZipFile(zip_path, "r") as zf:
        available = zf.namelist()

        for csv_name in files_to_extract:
            # LoadCore zips may have CSVs at root or in subdirectories
            matches = [f for f in available if f.endswith(csv_name)]

            if matches:
                target = matches[0]
                zf.extract(target, output_dir)
                extracted.append(str(output_dir / target))
            else:
                print(f"  WARNING: {csv_name} not found in {zip_path.name}")

    return extracted


def discover_zips(zip_dir: str) -> list[dict]:
    """Find all LoadCore zip files and parse their metadata.

    Returns list of dicts sorted by timestamp.
    """
    zip_path = Path(zip_dir)
    zips = []

    for zp in sorted(zip_path.rglob("*.zip")):
        info = parse_zip_filename(zp.name)
        info["path"] = zp
        zips.append(info)

    zips.sort(key=lambda x: x.get("timestamp", datetime.min))
    return zips


def main():
    params = load_params()

    # Load manifest
    with open(params["manifest_path"]) as f:
        manifest = json.load(f)

    files_to_extract = manifest["files_to_extract"]
    output_base = Path(params["output_dir"])

    # ── Discover and process zip files ──
    print("=== Discovering LoadCore zip files ===")
    zips = discover_zips(params["loadcore_zip_dir"])
    print(f"  Found {len(zips)} zip file(s)")

    for zip_info in zips:
        zip_path = zip_info["path"]
        dir_name = build_run_dir_name(zip_info)

        variant = zip_info.get("variant_raw", "unknown")
        output_dir = output_base / "loadcore" / dir_name
        zip_info["run_dir"] = dir_name

        print(f"\n  [{dir_name}] {variant}")
        print(f"    Zip: {zip_path.name}")

        extracted = extract_from_zip(zip_path, files_to_extract, output_dir)
        print(f"    Extracted: {len(extracted)}/{len(files_to_extract)} files")

    # ── Copy Scaphandre CSVs ──
    print("\n=== Copying Scaphandre CSVs ===")
    scaphandre_dir = Path(params["scaphandre_csv_dir"])
    scaphandre_out = output_base / "scaphandre"
    scaphandre_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for csv_file in sorted(scaphandre_dir.rglob("*.csv")):
        relative = csv_file.relative_to(scaphandre_dir)
        destination = scaphandre_out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_file, destination)
        print(f"  Copied: {relative}")
        copied += 1

    if not copied:
        print("  WARNING: No Scaphandre CSV files found")

    # ── Write discovery metadata ──
    discovery = {
        "n_zips": len(zips),
        "n_scaphandre": copied,
        "zips": [
            {
                "csv_id": z.get("csv_id"),
                "variant_raw": z.get("variant_raw"),
                "timestamp_str": z.get("timestamp_str"),
                "filename": z["filename"],
                "run_dir": z.get("run_dir"),
            }
            for z in zips
        ],
    }
    discovery_path = output_base / "discovery.json"
    with open(discovery_path, "w") as f:
        json.dump(discovery, f, indent=2)
    print(f"\nDiscovery metadata saved to {discovery_path}")

    print(f"\nDone. Interim data in: {output_base}")


if __name__ == "__main__":
    main()
