"""Create experiment-ready dataset after additional QC filtering.

This script starts from the clean base dataset and creates a separate
experiment dataset. It never overwrites the clean base dataset.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from paths import DATASET_CLEAN, OUTPUT_DIR, PROCESSED_DATA_DIR, PROJECT_MANAGEMENT_DIR, ensure_project_dirs

EXPERIMENT_DATASET = PROCESSED_DATA_DIR / "politik_full_year_2024_experiment_qc.csv"
EXCLUDED_DATASET = PROCESSED_DATA_DIR / "politik_full_year_2024_experiment_qc_excluded.csv"
REPORT_JSON = OUTPUT_DIR / "dataset_experiment_qc_report.json"
REPORT_MD = PROJECT_MANAGEMENT_DIR / "DATASET_EXPERIMENT_QC_REPORT.md"

POLITICAL_PATTERN = re.compile(
    r"\b(?:"
    r"pemilu|pilpres|pilkada|capres|cawapres|prabowo|gibran|ganjar|anies|cak\s+imin|"
    r"jokowi|ma'ruf|kpu|bawaslu|dkpp|partai|parpol|dpr|mpr|dpd|menteri|kabinet|"
    r"presiden|wakil\s+presiden|politik|pemerintah|pemerintahan|gubernur|bupati|wali\s+kota|"
    r"mahkamah\s+konstitusi|putusan\s+mk|mk|kampanye|koalisi|caleg|tps|debat|"
    r"demokrasi|pemilih|suara|quick\s+count|real\s+count|dprd|kabinet|istana|"
    r"nasdem|gerindra|pdip|golkar|pkb|pks|pan|ppp|demokrat|psi|perindo|hanura|"
    r"hukum|korupsi|kpk|kejaksaan|polri|tni|hakim|sidang|undang\s+undang|uu|perppu"
    r")\b",
    re.IGNORECASE,
)

NONPOLITICAL_PATTERN = re.compile(
    r"\b(?:"
    r"gempa|cuaca|bmkg|mudik|lebaran|tol|stasiun|pelabuhan|wisata|narkoba|"
    r"kecelakaan|kebakaran|banjir|haji|umrah|sepakbola|bola|kuliah|universitas|"
    r"kampus|konser|film|artis|zodiak|resep|restoran|liburan|hotel|pantai"
    r")\b",
    re.IGNORECASE,
)

BOILERPLATE_PATTERNS = [
    re.compile(r"SCROLL TO CONTINUE WITH CONTENT", re.IGNORECASE),
    re.compile(r"ADVERTISEMENT", re.IGNORECASE),
    re.compile(r"\bBaca juga\s*:?\s*[^.\n]*(?:\.|$)", re.IGNORECASE),
    re.compile(r"\bSimak Video\s*:?\s*[^.\n]*(?:\.|$)", re.IGNORECASE),
    re.compile(r"\bLihat juga\s*:?\s*[^.\n]*(?:\.|$)", re.IGNORECASE),
]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\r", " ").replace("\n", " ")).strip()


def clean_content_text(text: str) -> str:
    cleaned = str(text)
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    return normalize_spaces(cleaned)


def normalized_content_key(text: str) -> str:
    text = clean_content_text(text).lower()
    text = re.sub(r"[^a-z0-9À-ž\s]", " ", text)
    return normalize_spaces(text)


def infer_url_category(url: str) -> str:
    parts = urlparse(str(url)).path.strip("/").split("/")
    return parts[0] if parts and parts[0] else ""


def detect_flags(df: pd.DataFrame, min_words: int = 50) -> pd.DataFrame:
    work = df.copy()
    work["Content_clean"] = work["Content"].fillna("").map(clean_content_text)
    work["word_count_clean"] = work["Content_clean"].str.split().map(len)
    work["normalized_content_key"] = work["Content_clean"].map(normalized_content_key)
    combined_text = (work["Judul"].fillna("") + " " + work["Content_clean"].fillna("")).str.lower()
    work["url_category"] = work["Link"].fillna("").map(infer_url_category)

    work["flag_too_short"] = work["word_count_clean"] < min_words
    work["flag_retracted"] = combined_text.str.contains(r"berita\s+ini\s+dicabut|artikel\s+ini\s+dicabut", regex=True)
    work["flag_duplicate_content"] = work["normalized_content_key"].duplicated(keep="first")
    work["flag_empty_after_cleaning"] = work["Content_clean"].str.strip().eq("")

    political_hit = combined_text.str.contains(POLITICAL_PATTERN, regex=True)
    nonpolitical_hit = combined_text.str.contains(NONPOLITICAL_PATTERN, regex=True)
    work["flag_nonpolitical_candidate"] = nonpolitical_hit & ~political_hit
    work["political_keyword_hit"] = political_hit
    work["nonpolitical_keyword_hit"] = nonpolitical_hit

    return work


def primary_reason(row: pd.Series) -> str:
    checks = [
        ("empty_after_cleaning", row.get("flag_empty_after_cleaning", False)),
        ("too_short", row.get("flag_too_short", False)),
        ("retracted", row.get("flag_retracted", False)),
        ("duplicate_content", row.get("flag_duplicate_content", False)),
        ("nonpolitical_candidate", row.get("flag_nonpolitical_candidate", False)),
    ]
    return next((name for name, value in checks if bool(value)), "retained")


def filter_experiment_dataset(df: pd.DataFrame, min_words: int = 50):
    work = detect_flags(df, min_words=min_words)
    flag_cols = [
        "flag_empty_after_cleaning",
        "flag_too_short",
        "flag_retracted",
        "flag_duplicate_content",
        "flag_nonpolitical_candidate",
    ]
    work["exclude"] = work[flag_cols].any(axis=1)
    work["exclude_reason_primary"] = work.apply(primary_reason, axis=1)

    retained = work.loc[~work["exclude"]].copy()
    excluded = work.loc[work["exclude"]].copy()

    summary = build_summary(work, retained, excluded, flag_cols)
    return retained, excluded, summary


def build_summary(work: pd.DataFrame, retained: pd.DataFrame, excluded: pd.DataFrame, flag_cols: list[str]) -> dict:
    dates = pd.to_datetime(retained.get("parsed_date"), errors="coerce") if not retained.empty else pd.Series([], dtype="datetime64[ns]")
    summary = {
        "input_rows": int(len(work)),
        "retained_rows": int(len(retained)),
        "excluded_rows": int(len(excluded)),
        "retained_share": float(len(retained) / len(work)) if len(work) else 0.0,
        "flag_counts": {c: int(work[c].sum()) for c in flag_cols},
        "exclude_reason_counts": excluded["exclude_reason_primary"].value_counts().astype(int).to_dict(),
        "retained_source_counts": retained["source"].value_counts().astype(int).to_dict() if "source" in retained else {},
        "retained_month_counts": dates.dt.month.value_counts().sort_index().astype(int).to_dict(),
        "retained_word_count_summary": retained["word_count_clean"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict() if not retained.empty else {},
    }
    return summary


def write_markdown_report(summary: dict, report_md: Path = REPORT_MD) -> None:
    month_names = {
        1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
        7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember",
    }
    month_rows = "\n".join(
        f"| {month_names.get(int(k), k)} | {v:,} |".replace(",", ".")
        for k, v in summary["retained_month_counts"].items()
    )
    source_rows = "\n".join(
        f"| {k} | {v:,} |".replace(",", ".")
        for k, v in summary["retained_source_counts"].items()
    )
    flag_rows = "\n".join(f"| {k} | {v:,} |".replace(",", ".") for k, v in summary["flag_counts"].items())
    reason_rows = "\n".join(f"| {k} | {v:,} |".replace(",", ".") for k, v in summary["exclude_reason_counts"].items())
    text = f"""# Dataset Experiment QC Report

Tanggal: 2026-06-17

## Output

Dataset eksperimen final:

`{EXPERIMENT_DATASET}`

Dataset excluded/audit trail:

`{EXCLUDED_DATASET}`

JSON report:

`{REPORT_JSON}`

## Ringkasan

- Input rows: {summary['input_rows']:,}
- Retained rows: {summary['retained_rows']:,}
- Excluded rows: {summary['excluded_rows']:,}
- Retained share: {summary['retained_share']:.2%}

## Flag Counts

| Flag | Jumlah |
|---|---:|
{flag_rows}

## Primary Exclusion Reason

| Reason | Jumlah |
|---|---:|
{reason_rows}

## Retained Source Counts

| Source | Jumlah |
|---|---:|
{source_rows}

## Retained Month Counts

| Bulan | Jumlah |
|---|---:|
{month_rows}

## Catatan Metodologis

Dataset ini tidak menggantikan clean base dataset. Dataset ini adalah versi eksperimen setelah QC tambahan:

1. content exact duplicate dihapus berdasarkan normalized content;
2. artikel sangat pendek dihapus dengan threshold 50 kata;
3. artikel dicabut dihapus;
4. kandidat non-politik/general news dihapus secara konservatif berbasis negative keyword tanpa political keyword;
5. boilerplate web dibersihkan dari `Content_clean`.

Untuk tesis, laporkan dataset clean base dan dataset eksperimen final secara terpisah agar provenance jelas.
"""
    report_md.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_project_dirs()
    df = pd.read_csv(DATASET_CLEAN)
    retained, excluded, summary = filter_experiment_dataset(df, min_words=50)

    output_cols = list(df.columns) + [
        "Content_clean", "word_count_clean", "url_category", "political_keyword_hit",
        "nonpolitical_keyword_hit", "exclude_reason_primary",
    ]
    retained[output_cols].to_csv(EXPERIMENT_DATASET, index=False)
    excluded[output_cols].to_csv(EXCLUDED_DATASET, index=False)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_markdown_report(summary)

    print("Dataset experiment QC complete")
    print(f"Input rows: {summary['input_rows']:,}")
    print(f"Retained rows: {summary['retained_rows']:,}")
    print(f"Excluded rows: {summary['excluded_rows']:,}")
    print(f"Experiment dataset: {EXPERIMENT_DATASET}")
    print(f"Excluded dataset: {EXCLUDED_DATASET}")
    print(f"Report: {REPORT_MD}")


if __name__ == "__main__":
    main()
