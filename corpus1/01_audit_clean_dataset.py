"""Audit and clean the thesis dataset for defensible experiments.

Input:
    03_dataset/data/processed/politik_full_year_2024.csv

Outputs:
    03_dataset/data/processed/politik_full_year_2024_clean.csv
    00_project_management/DATASET_CLEANING_REPORT.md
    06_outputs/dataset_cleaning_report.json
"""
from __future__ import annotations

import json
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

from paths import (
    CLEANING_REPORT_JSON,
    CLEANING_REPORT_MD,
    DATASET_CLEAN,
    DATASET_MAIN,
    ensure_project_dirs,
)

SOURCE_DOMAINS = {
    "detik": "detik",
    "tempo": "tempo",
    "kompas": "kompas",
}


def infer_source_from_link(link: object) -> str | None:
    """Infer source from URL domain when the source column is missing."""
    if pd.isna(link):
        return None
    text = str(link).strip().lower()
    if not text:
        return None

    try:
        domain = urlparse(text).netloc.lower()
    except Exception:
        domain = text

    combined = f"{domain} {text}"
    for key, value in SOURCE_DOMAINS.items():
        if key in combined:
            return value
    return None


def normalize_source(value: object, link: object) -> str | None:
    """Normalize source labels and infer missing values from links."""
    if pd.notna(value):
        source = str(value).strip().lower()
        if source and source not in {"nan", "none", "null"}:
            for key, normalized in SOURCE_DOMAINS.items():
                if key in source:
                    return normalized
            return source
    return infer_source_from_link(link)


def summarize(df: pd.DataFrame) -> dict:
    """Return summary metrics used in reports."""
    parsed_date = pd.to_datetime(df.get("parsed_date"), errors="coerce") if "parsed_date" in df else pd.Series(dtype="datetime64[ns]")
    content = df.get("Content", pd.Series([], dtype=object))
    link = df.get("Link", pd.Series([], dtype=object))
    source = df.get("source", pd.Series([], dtype=object))

    out = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "valid_parsed_date": int(parsed_date.notna().sum()) if len(parsed_date) else 0,
        "min_date": str(parsed_date.min().date()) if len(parsed_date) and parsed_date.notna().any() else None,
        "max_date": str(parsed_date.max().date()) if len(parsed_date) and parsed_date.notna().any() else None,
        "missing_content": int(content.isna().sum()) if len(content) else None,
        "blank_content": int(content.fillna("").astype(str).str.strip().eq("").sum()) if len(content) else None,
        "duplicate_link_rows": int(link.duplicated().sum()) if len(link) else None,
        "missing_source": int(source.isna().sum()) if len(source) else None,
        "source_counts": source.value_counts(dropna=False).to_dict() if len(source) else {},
    }

    if len(parsed_date) and parsed_date.notna().any():
        months = parsed_date.dt.month.value_counts().sort_index()
        out["month_counts"] = {int(k): int(v) for k, v in months.items() if pd.notna(k)}
    else:
        out["month_counts"] = {}

    if len(content):
        lengths = content.fillna("").astype(str).str.len()
        out["content_length"] = {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "mean": float(round(lengths.mean(), 2)),
            "max": int(lengths.max()),
        }
    return out


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean dataset without mutating the original dataframe."""
    cleaned = df.copy()
    initial_rows = len(cleaned)

    # Ensure date helper columns are consistent.
    cleaned["parsed_date"] = pd.to_datetime(cleaned["parsed_date"], errors="coerce")
    cleaned = cleaned[cleaned["parsed_date"].notna()].copy()
    cleaned = cleaned[cleaned["parsed_date"].dt.year == 2024].copy()
    cleaned["year"] = cleaned["parsed_date"].dt.year
    cleaned["month"] = cleaned["parsed_date"].dt.month
    cleaned["year_month"] = cleaned["parsed_date"].dt.to_period("M").astype(str)

    # Normalize source before dropping source-missing rows.
    cleaned["source"] = [normalize_source(s, l) for s, l in zip(cleaned.get("source"), cleaned.get("Link"))]

    before_content = len(cleaned)
    content_text = cleaned["Content"].fillna("").astype(str).str.strip()
    cleaned = cleaned[content_text.ne("")].copy()
    removed_blank_content = before_content - len(cleaned)

    before_duplicate = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["Link"], keep="first").copy()
    removed_duplicate_link = before_duplicate - len(cleaned)

    before_source = len(cleaned)
    cleaned = cleaned[cleaned["source"].notna()].copy()
    removed_missing_source = before_source - len(cleaned)

    cleaned = cleaned.sort_values(["parsed_date", "source", "Judul"], kind="stable").reset_index(drop=True)
    cleaned["parsed_date"] = cleaned["parsed_date"].dt.strftime("%Y-%m-%d")

    stats = {
        "initial_rows": int(initial_rows),
        "removed_invalid_or_non_2024_date": int(initial_rows - before_content),
        "removed_blank_content": int(removed_blank_content),
        "removed_duplicate_link": int(removed_duplicate_link),
        "removed_missing_source_after_inference": int(removed_missing_source),
        "final_rows": int(len(cleaned)),
    }
    return cleaned, stats


def write_reports(before: dict, after: dict, cleaning_stats: dict) -> None:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(DATASET_MAIN),
        "output_file": str(DATASET_CLEAN),
        "cleaning_stats": cleaning_stats,
        "before": before,
        "after": after,
    }
    CLEANING_REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    month_lines = "\n".join(f"- {month:02d}: {count:,}" for month, count in after["month_counts"].items())
    source_lines = "\n".join(f"- {source}: {count:,}" for source, count in after["source_counts"].items())

    md = f"""# DATASET CLEANING REPORT

Generated: {report['generated_at']}

## Files

Input:
`{DATASET_MAIN}`

Output:
`{DATASET_CLEAN}`

JSON report:
`{CLEANING_REPORT_JSON}`

## Cleaning Actions

- Invalid or non-2024 date rows removed: {cleaning_stats['removed_invalid_or_non_2024_date']:,}
- Blank content rows removed: {cleaning_stats['removed_blank_content']:,}
- Duplicate link rows removed: {cleaning_stats['removed_duplicate_link']:,}
- Missing source rows after URL inference removed: {cleaning_stats['removed_missing_source_after_inference']:,}

## Row Counts

- Initial rows: {cleaning_stats['initial_rows']:,}
- Final rows: {cleaning_stats['final_rows']:,}
- Total removed: {cleaning_stats['initial_rows'] - cleaning_stats['final_rows']:,}

## Final Date Range

- Valid parsed dates: {after['valid_parsed_date']:,}
- Min date: {after['min_date']}
- Max date: {after['max_date']}

## Final Distribution by Source

{source_lines}

## Final Distribution by Month

{month_lines}

## Final Data Quality Notes

- Blank content after cleaning: {after['blank_content']:,}
- Duplicate link rows after cleaning: {after['duplicate_link_rows']:,}
- Missing source after cleaning: {after['missing_source']:,}

## Thesis Note

Dataset bersih ini menjadi kandidat dataset final untuk eksperimen utama. Semua angka pada Bab 4 sebaiknya memakai jumlah final dari file ini agar konsisten dan defensible saat seminar.
"""
    CLEANING_REPORT_MD.write_text(md, encoding="utf-8")


def main() -> None:
    ensure_project_dirs()
    if not DATASET_MAIN.exists():
        raise FileNotFoundError(f"Input dataset not found: {DATASET_MAIN}")

    print(f"Loading: {DATASET_MAIN}")
    df = pd.read_csv(DATASET_MAIN)
    before = summarize(df)
    cleaned, cleaning_stats = clean_dataset(df)
    after = summarize(cleaned)

    cleaned.to_csv(DATASET_CLEAN, index=False)
    write_reports(before, after, cleaning_stats)

    print("Cleaning complete")
    print(f"Initial rows : {cleaning_stats['initial_rows']:,}")
    print(f"Final rows   : {cleaning_stats['final_rows']:,}")
    print(f"Output       : {DATASET_CLEAN}")
    print(f"Report       : {CLEANING_REPORT_MD}")


if __name__ == "__main__":
    main()
