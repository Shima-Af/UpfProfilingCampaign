"""Tests for the ingestion pipeline."""
import zipfile
import tempfile
from pathlib import Path
from src.ingest import extract_from_zip, parse_zip_filename


def test_parse_zip_filename_sdcore():
    """Verify parsing of SD-Core DPDK zip filename."""
    name = "CSVs-248 - baremetal _SD-Core_DPDK (copy from Jan  3 23-35-44)-2025-01-03-23-35-50-UTC.zip"
    info = parse_zip_filename(name)

    assert info["csv_id"] == "248"
    assert info["variant_raw"] == "SD-Core_DPDK"
    assert info["timestamp_str"] == "2025-01-03-23-35-50"
    assert info["timestamp"].year == 2025
    assert info["timestamp"].month == 1
    assert info["timestamp"].day == 3


def test_parse_zip_filename_oai():
    """Verify parsing of OAI userspace zip filename."""
    name = "CSVs-320 - baremetal _OAI_UserspaceApplication (copy from Jan 13 15-34-27)-2025-01-13-15-34-33-UTC.zip"
    info = parse_zip_filename(name)

    assert info["csv_id"] == "320"
    assert info["variant_raw"] == "OAI_UserspaceApplication"
    assert info["timestamp"].month == 1
    assert info["timestamp"].day == 13


def test_extract_from_zip():
    """Verify that only listed CSVs are extracted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a zip mimicking LoadCore output
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("Fullcoreapplicationtraffic_GTPuKbitss_DN.csv", "ts,val\n1,100\n")
            zf.writestr("Fullcoreapplicationtraffic_GTPuKbitss_NGRAN.csv", "ts,val\n1,200\n")
            zf.writestr("SomeOtherFile.csv", "should,not,extract\n")

        output_dir = Path(tmpdir) / "output"
        files_to_extract = [
            "Fullcoreapplicationtraffic_GTPuKbitss_DN.csv",
            "Fullcoreapplicationtraffic_GTPuKbitss_NGRAN.csv",
        ]

        extracted = extract_from_zip(zip_path, files_to_extract, output_dir)

        assert len(extracted) == 2
        assert not (output_dir / "SomeOtherFile.csv").exists()


def test_extract_missing_file_graceful():
    """Missing CSVs in zip should warn, not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("Fullcoreapplicationtraffic_GTPuKbitss_DN.csv", "ts,val\n1,100\n")

        output_dir = Path(tmpdir) / "output"
        extracted = extract_from_zip(
            zip_path,
            ["Fullcoreapplicationtraffic_GTPuKbitss_DN.csv", "NonExistent.csv"],
            output_dir,
        )

        assert len(extracted) == 1
