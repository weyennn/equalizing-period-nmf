"""Analisis tambahan paper (27 Oktober 2026): apa yang berubah pada topik bagi analis.

Untuk setiap konfigurasi, topik setiap varian dicocokkan satu-satu dengan topik S1 (algoritme Hungaria pada kosinus baris H).
Topik tanpa padanan: kosinus < 0,5. Bulan puncak topik: bulan dengan lift tertinggi, lift = prevalensi topik pada bulan itu
dibagi rata-rata prevalensinya atas 12 bulan. Prevalensi mengikuti compute_soft_topic_prevalence (W seluruh korpus dari
X_base_all dengan H tetap, dinormalkan per baris, dirata-ratakan per bulan). Untuk S1–S4 dipakai berkas
soft_topic_prevalence_by_month.csv tersimpan; untuk varian kurva bobot dan undersampling dihitung di sini dari H tersimpan.
Masukan (hanya dibaca): outputs/korpus_bersih_15konfigurasi_2026-10-20/ dan
outputs/paper_kurva_bobot_undersampling_2026-10-26/ dan paper_nmf_per_periode_2026-10-27/ (varian pp, 27 Okt).
Pemakaian: <python 3.10 + scikit-learn 1.5.2> -I analysis/analisis_dampak_topik.py
"""
import warnings
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import joblib, numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")
SRC = Path(ROOT / "outputs/korpus_bersih_15konfigurasi_2026-10-20")
HSRC = Path(ROOT / "outputs/paper_kurva_bobot_undersampling_2026-10-26")
PPSRC = Path(ROOT / "outputs/paper_nmf_per_periode_2026-10-27")  # NMF per periode (27 Okt)
OUT = Path(__file__).resolve().parent
SAVED = {"S1": "S1_baseline", "S2": "S2_document_weighting", "S3": "S3_temporal_tfidf", "S4": "S4_dual_level"}
EXTRA = ["b0.5", "b1.5", "b2", "us", "pp"]
LOW = ["2024-09", "2024-10", "2024-11", "2024-12"]
COS = 0.5

rows, topic_rows = [], []
peaks = {v: {"hilang": Counter(), "baru": Counter()} for v in list(SAVED)[1:] + EXTRA}
check = 0.0
for rs in (7, 21, 42, 84, 123):
    for k in (15, 20, 25):
        name = f"r{rs}_k{k}"; run = SRC / name
        H, P, top = {}, {}, {}
        for v, d in SAVED.items():
            H[v] = joblib.load(run / d / "nmf_model.joblib").components_
            P[v] = pd.read_csv(run / d / "soft_topic_prevalence_by_month.csv").set_index("year_month")
            top[v] = pd.read_csv(run / d / "top_features_top15.csv").top_features.tolist()
        Xa = sparse.load_npz(run / "X_base_all.npz").tocsr()
        led = pd.read_csv(run / "split_ledger.csv").sort_values("model_row"); am = led.year_month.astype(str).to_numpy()
        proto = joblib.load(run / "S1_baseline" / "nmf_model.joblib")
        feats = None

        def prevalence(Hk):
            proto.components_ = Hk; W = proto.transform(Xa); th = W / W.sum(1, keepdims=True)
            return pd.DataFrame(th).groupby(am).mean().rename(columns=lambda j: f"topic_{j + 1}")

        if name == "r42_k20":   # pemeriksaan: prevalensi S1 hitung ulang = berkas tersimpan
            check = float(np.abs(prevalence(H["S1"]).to_numpy() - P["S1"].to_numpy()).max())
        for v in EXTRA:
            H[v] = np.load((PPSRC if v == "pp" else HSRC) / f"{name}_{v}.npz")["H"]; P[v] = prevalence(H[v])
        lift = {v: P[v] / P[v].mean(axis=0) for v in P}
        peak = {v: lift[v].idxmax(axis=0).to_numpy() for v in P}
        unit = {v: H[v] / np.linalg.norm(H[v], axis=1, keepdims=True) for v in H}
        for v in P:
            out = {"run": name, "random_state": rs, "K": k, "varian": v,
                   "puncak_agustus": int((peak[v] == "2024-08").sum()), "puncak_juli_agustus": int(np.isin(peak[v], ["2024-07", "2024-08"]).sum()),
                   "puncak_sepdes": int(np.isin(peak[v], LOW).sum()),
                   "lift2_agustus": int((lift[v].loc["2024-08"] >= 2).sum()), "lift2_sepdes": int((lift[v].loc[LOW] >= 2).any(axis=0).sum())}
            if v != "S1":
                C = unit["S1"] @ unit[v].T; r, c = linear_sum_assignment(-C); sim = C[r, c]; bad = sim < COS
                out.update({"tanpa_padanan": int(bad.sum()), "kosinus_median": float(np.median(sim))})
                for i, j, s in zip(r[bad], c[bad], sim[bad]):
                    peaks[v]["hilang"][peak["S1"][i]] += 1; peaks[v]["baru"][peak[v][j]] += 1
                    if v in SAVED:
                        topic_rows.append({"run": name, "varian": v, "kosinus": s,
                                           "topik_S1": i + 1, "puncak_S1": peak["S1"][i], "lift_S1": float(lift["S1"].iloc[:, i].max()), "fitur_S1": top["S1"][i],
                                           f"topik_varian": j + 1, "puncak_varian": peak[v][j], "lift_varian": float(lift[v].iloc[:, j].max()), "fitur_varian": top[v][j]})
            rows.append(out)
        print("selesai:", name, flush=True)

df = pd.DataFrame(rows); df.to_csv(OUT / "dampak_topik_15konfigurasi.csv", index=False)
pd.DataFrame(topic_rows).to_csv(OUT / "dampak_topik_tanpa_padanan.csv", index=False)
pd.set_option("display.width", 250)
print(f"\nPemeriksaan: prevalensi S1 r42_k20 dihitung ulang vs berkas tersimpan, selisih maksimum {check:.2e}")
order = ["S1", "S2", "S3", "S4", "b0.5", "b1.5", "b2", "us", "pp"]
n_topics = df[df.varian == "S1"].K.sum()
print(f"\n== Jumlah atas 15 konfigurasi ({n_topics} topik per varian)")
print(df.groupby("varian")[["tanpa_padanan", "puncak_agustus", "puncak_juli_agustus", "puncak_sepdes", "lift2_agustus", "lift2_sepdes"]]
      .sum().reindex(order).to_string())
print("\n== Topik tanpa padanan per konfigurasi (min-maks) dan kosinus median")
print(df[df.varian != "S1"].groupby("varian").agg(min=("tanpa_padanan", "min"), maks=("tanpa_padanan", "max"),
                                                   kosinus_median=("kosinus_median", "median")).reindex(order[1:]).round(3).to_string())
print("\n== Bulan puncak topik tanpa padanan (hilang dari S1 / baru di varian)")
for v in order[1:]:
    print(f"{v}: hilang {dict(sorted(peaks[v]['hilang'].items()))}\n      baru   {dict(sorted(peaks[v]['baru'].items()))}")
