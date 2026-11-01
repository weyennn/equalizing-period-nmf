"""
SCRIPT MENGGABUNGKAN DATASET
=============================
Menggabungkan:
1. Dataset Politik (Jan-Sep 2024) - sudah lengkap
2. Dataset Awal (filter politik Okt-Des 2024) - untuk melengkapi

Output: Dataset Politik 1 Tahun Penuh (Jan-Des 2024)
"""

import os
import pandas as pd
import numpy as np
import re
from collections import Counter

# ===== KONFIGURASI =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
FILE_POLITIK = os.path.join(PROJECT_ROOT, 'data', 'processed', 'politik_merge_dates_fixed.csv')  # Dataset politik Jan-Sep 2024
FILE_AWAL = os.path.join(PROJECT_ROOT, 'data', 'raw', 'dataset_fixed.csv')                        # Dataset awal (semua kategori)
FILE_OUTPUT = os.path.join(PROJECT_ROOT, 'data', 'processed', 'politik_full_year_2024.csv')       # Output gabungan

# Keywords untuk filter politik
POLITICAL_KEYWORDS = [
    'politik', 'pemilu', 'pilpres', 'pilkada', 'prabowo', 'jokowi', 
    'anies', 'ganjar', 'capres', 'cawapres', 'kpu', 'partai', 
    'dpr', 'pemerintah', 'presiden', 'menteri', 'kabinet', 
    'koalisi', 'oposisi', 'gubernur', 'bupati', 'walikota',
    'pdip', 'pdi-p', 'golkar', 'gerindra', 'demokrat', 'pks',
    'pkb', 'nasdem', 'pan', 'ppp', 'perindo', 'psi',
    'megawati', 'gibran', 'mahfud', 'muhaimin', 'cak imin',
    'amin', 'komisi', 'legislatif', 'eksekutif', 'yudikatif'
]

# Mapping bulan Indonesia
BULAN_ID = {
    'januari': 1, 'februari': 2, 'maret': 3, 'april': 4,
    'mei': 5, 'juni': 6, 'juli': 7, 'agustus': 8,
    'september': 9, 'oktober': 10, 'november': 11, 'desember': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'jun': 6, 'jul': 7, 'agu': 8, 'ags': 8, 'aug': 8,
    'sep': 9, 'sept': 9, 'okt': 10, 'oct': 10, 
    'nov': 11, 'nop': 11, 'des': 12, 'dec': 12
}


def parse_tanggal_indonesia(waktu_str):
    """Parse tanggal format Indonesia"""
    if pd.isna(waktu_str) or not waktu_str:
        return None
    
    waktu_str = str(waktu_str).strip().lower()
    pattern = r'(\d{1,2})\s+([a-z]+)\s+(\d{4})'
    match = re.search(pattern, waktu_str)
    
    if match:
        day = int(match.group(1))
        month_str = match.group(2)
        year = int(match.group(3))
        
        month = None
        for key, val in BULAN_ID.items():
            if month_str.startswith(key) or key.startswith(month_str):
                month = val
                break
        
        if month and 1 <= day <= 31 and 2020 <= year <= 2030:
            try:
                return pd.Timestamp(year=year, month=month, day=day)
            except:
                pass
    
    return None


def is_political_article(row, tag_columns=['tag1', 'tag2', 'tag3', 'tag4', 'tag5']):
    """Cek apakah artikel adalah berita politik berdasarkan tags"""
    text_to_check = []
    
    # Cek tags
    for col in tag_columns:
        if col in row.index and pd.notna(row[col]):
            text_to_check.append(str(row[col]).lower())
    
    # Cek judul juga
    if 'Judul' in row.index and pd.notna(row['Judul']):
        text_to_check.append(str(row['Judul']).lower())
    
    combined = ' '.join(text_to_check)
    
    return any(kw in combined for kw in POLITICAL_KEYWORDS)


def merge_datasets():
    """Gabungkan kedua dataset"""
    
    print("=" * 70)
    print("MENGGABUNGKAN DATASET POLITIK")
    print("=" * 70)
    
    # ===== LOAD DATASET POLITIK =====
    print("\n[1] Loading dataset politik (Jan-Sep 2024)...")
    df_politik = pd.read_csv(FILE_POLITIK)
    print(f"    Artikel: {len(df_politik):,}")
    
    # Parse tanggal — pakai waktu_final kalau ada, lebih lengkap
    if 'waktu_final' in df_politik.columns:
        df_politik['parsed_date'] = pd.to_datetime(df_politik['waktu_final'], errors='coerce')
    elif 'parsed_date' in df_politik.columns:
        df_politik['parsed_date'] = pd.to_datetime(df_politik['parsed_date'], errors='coerce')
    else:
        df_politik['parsed_date'] = df_politik['Waktu'].apply(parse_tanggal_indonesia)
    
    df_politik['year'] = df_politik['parsed_date'].dt.year.astype('Int64')
    df_politik['month'] = df_politik['parsed_date'].dt.month.astype('Int64')

    # Tampilkan distribusi
    print("\n    Distribusi dataset politik:")
    monthly_politik = df_politik.dropna(subset=['year','month']).groupby(['year', 'month']).size()
    for (year, month), count in monthly_politik.items():
        print(f"      {int(year)}-{int(month):02d}: {count:,}")
    
    # ===== LOAD DATASET AWAL =====
    print("\n[2] Loading dataset awal (semua kategori)...")
    df_awal = pd.read_csv(FILE_AWAL)
    print(f"    Artikel: {len(df_awal):,}")
    
    # Parse tanggal
    if 'parsed_date' not in df_awal.columns and 'final_date' in df_awal.columns:
        df_awal['parsed_date'] = pd.to_datetime(df_awal['final_date'])
    elif 'parsed_date' not in df_awal.columns:
        df_awal['parsed_date'] = df_awal['Waktu'].apply(parse_tanggal_indonesia)
    else:
        df_awal['parsed_date'] = pd.to_datetime(df_awal['parsed_date'])
    
    df_awal['year'] = df_awal['parsed_date'].dt.year.astype('Int64')
    df_awal['month'] = df_awal['parsed_date'].dt.month.astype('Int64')
    
    # ===== FILTER OKT-DES 2024 =====
    print("\n[3] Filter Okt-Des 2024 dari dataset awal...")
    df_okt_des = df_awal[
        (df_awal['year'] == 2024) & 
        (df_awal['month'].isin([10, 11, 12]))
    ].copy()
    print(f"    Artikel Okt-Des 2024: {len(df_okt_des):,}")
    
    if len(df_okt_des) == 0:
        print("    ⚠️ TIDAK ADA DATA Okt-Des 2024 di dataset awal!")
        print("    Dataset awal mungkin tidak mencakup periode ini.")
        
        # Cek periode yang tersedia
        print("\n    Periode yang tersedia di dataset awal:")
        available = df_awal.dropna(subset=['year','month']).groupby(['year', 'month']).size()
        for (year, month), count in available.items():
            print(f"      {int(year)}-{int(month):02d}: {count:,}")
        
        return None
    
    # ===== FILTER ARTIKEL POLITIK =====
    print("\n[4] Filter artikel politik dari Okt-Des 2024...")
    df_okt_des_politik = df_okt_des[df_okt_des.apply(is_political_article, axis=1)].copy()
    print(f"    Artikel politik Okt-Des 2024: {len(df_okt_des_politik):,}")
    
    # Distribusi
    monthly_tambahan = df_okt_des_politik.dropna(subset=['year','month']).groupby(['year', 'month']).size()
    for (year, month), count in monthly_tambahan.items():
        print(f"      {int(year)}-{int(month):02d}: {count:,}")
    
    # ===== CEK DUPLIKASI =====
    print("\n[5] Cek duplikasi berdasarkan Link...")
    
    # Ambil links dari kedua dataset
    links_politik = set(df_politik['Link'].dropna().astype(str))
    links_tambahan = set(df_okt_des_politik['Link'].dropna().astype(str))
    
    duplicates = links_politik.intersection(links_tambahan)
    print(f"    Link duplikat: {len(duplicates):,}")
    
    # Hapus duplikat dari data tambahan
    if len(duplicates) > 0:
        df_okt_des_politik = df_okt_des_politik[
            ~df_okt_des_politik['Link'].isin(duplicates)
        ]
        print(f"    Artikel setelah hapus duplikat: {len(df_okt_des_politik):,}")
    
    # ===== GABUNGKAN =====
    print("\n[6] Menggabungkan dataset...")
    
    # Samakan kolom
    common_cols = list(set(df_politik.columns) & set(df_okt_des_politik.columns))
    print(f"    Kolom yang sama: {len(common_cols)}")
    
    df_gabungan = pd.concat([
        df_politik[common_cols],
        df_okt_des_politik[common_cols]
    ], ignore_index=True)
    
    print(f"    Total artikel gabungan: {len(df_gabungan):,}")
    
    # ===== HASIL AKHIR =====
    print("\n" + "=" * 70)
    print("HASIL PENGGABUNGAN")
    print("=" * 70)
    
    # Pastikan parsed_date ada
    if 'parsed_date' not in df_gabungan.columns:
        df_gabungan['parsed_date'] = df_gabungan['Waktu'].apply(parse_tanggal_indonesia)
    
    df_gabungan['year'] = pd.to_datetime(df_gabungan['parsed_date']).dt.year
    df_gabungan['month'] = pd.to_datetime(df_gabungan['parsed_date']).dt.month
    df_gabungan['year_month'] = pd.to_datetime(df_gabungan['parsed_date']).dt.to_period('M')
    
    # Filter hanya 2024
    df_2024 = df_gabungan[df_gabungan['year'] == 2024]
    
    print("\nDistribusi Final (2024):")
    print("-" * 50)
    
    monthly_final = df_2024['year_month'].value_counts().sort_index()
    
    for period, count in monthly_final.items():
        bar = "█" * int(count / monthly_final.max() * 40)
        print(f"    {period}: {count:>6,} {bar}")
    
    # Statistik
    print(f"\n    Total artikel 2024: {len(df_2024):,}")
    print(f"    Max: {monthly_final.max():,} ({monthly_final.idxmax()})")
    print(f"    Min: {monthly_final.min():,} ({monthly_final.idxmin()})")
    print(f"    Mean: {monthly_final.mean():,.0f}")
    
    ratio = monthly_final.max() / monthly_final.min()
    print(f"\n    >>> IMBALANCE RATIO: {ratio:.2f}:1 <<<")
    
    if ratio >= 3:
        print("    ✅ Dataset COCOK untuk penelitian!")
    else:
        print("    ⚠️ Imbalance ratio menurun, pertimbangkan kembali")
    
    # Simpan
    df_2024.to_csv(FILE_OUTPUT, index=False)
    print(f"\n✅ Dataset disimpan ke: {FILE_OUTPUT}")
    
    return df_2024


if __name__ == "__main__":
    result = merge_datasets()
