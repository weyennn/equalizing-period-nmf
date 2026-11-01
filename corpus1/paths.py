"""Shared paths for the pipeline and corpus scripts, resolved relative to the repository root.

Data live in data/ (raw/ for the downloaded datasets, processed/ for the derived CSV files) and all run outputs in
outputs/. Both folders are created on demand and are not part of the repository.
"""
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROJECT_MANAGEMENT_DIR = OUTPUT_DIR / "reports"
VISUALIZATION_DIR = OUTPUT_DIR / "figures"

DATASET_MAIN = PROCESSED_DATA_DIR / "politik_full_year_2024.csv"
DATASET_CLEAN = PROCESSED_DATA_DIR / "politik_full_year_2024_clean.csv"
CLEANING_REPORT_MD = PROJECT_MANAGEMENT_DIR / "DATASET_CLEANING_REPORT.md"
CLEANING_REPORT_JSON = OUTPUT_DIR / "dataset_cleaning_report.json"


def ensure_project_dirs() -> None:
    """Create output directories used by active scripts."""
    for path in [PROJECT_MANAGEMENT_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, VISUALIZATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)
