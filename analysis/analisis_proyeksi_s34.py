"""S3*/S4* pada ke-15 konfigurasi.

Basis S3/S4 dipelajari di ruang Temporal TF-IDF. Pada evaluasi utama, H dipakai langsung pada X_val ruang dasar.
Di sini H juga diproyeksikan balik ke ruang dasar (H / tidf, lalu norma tiap baris H dikembalikan), mengikuti
corpus1/eval_sim.py. Hanya membaca keluaran
outputs/korpus_bersih_15konfigurasi_2026-10-20/; tidak melatih ulang model.
Bootstrap sama dengan analisis_bobot_per_bulan.py: terstratifikasi per bulan, 1.000 kali, default_rng(0) per konfigurasi.
Pemakaian: <python 3.10 + scikit-learn 1.5.2> -I analysis/analisis_proyeksi_s34.py
"""
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import joblib, numpy as np, pandas as pd
from scipy import sparse
warnings.filterwarnings("ignore")
SRC = Path(ROOT / "outputs/korpus_bersih_15konfigurasi_2026-10-20")
OUT = Path(__file__).resolve().parent
SC = ["S1_baseline", "S2_document_weighting", "S3_temporal_tfidf", "S4_dual_level"]
KEYS = ["S1", "S2", "S3", "S4", "S3*", "S4*"]
B = 1000
rows, per_rows = [], []
for rs in (7, 21, 42, 84, 123):
    for k in (15, 20, 25):
        name = f"r{rs}_k{k}"; run = SRC / name
        Xv = sparse.load_npz(run / "X_base_valid.npz").tocsr()
        led = pd.read_csv(run / "split_ledger.csv"); v = led[led.split == "validation"].sort_values("split_row")
        vm = v.year_month.astype(str).to_numpy(); months = sorted(set(vm))
        tidf = pd.read_csv(run / "temporal_idf_train.csv").temporal_idf.to_numpy()
        xsq = np.asarray(Xv.multiply(Xv).sum(1)).ravel()

        def rowres(m, H):
            m.components_ = H; W = m.transform(Xv)
            return xsq + np.einsum("ij,jk,ik->i", W, H @ H.T, W) - 2 * np.sum(W * np.asarray(Xv @ H.T), 1)

        res = {}
        for s in SC:
            m = joblib.load(run / s / "nmf_model.joblib"); H = m.components_.copy()
            res[s[:2]] = rowres(m, H)
            if s[1] in "34":
                Hb = H / tidf; Hb *= (np.linalg.norm(H, axis=1) / np.linalg.norm(Hb, axis=1))[:, None]
                res[s[:2] + "*"] = rowres(m, Hb)
        idx = {p: np.flatnonzero(vm == p) for p in months}
        rre = lambda c, ix: np.sqrt(res[c][ix].sum() / xsq[ix].sum())
        for p in months:
            per_rows.append({"run": name, "random_state": rs, "K": k, "bulan": p,
                             **{f"rre_{c}": rre(c, idx[p]) for c in KEYS}})
        rng = np.random.default_rng(0)
        macro = {c: [] for c in KEYS}
        for b in range(B):
            ix = {p: rng.choice(idx[p], len(idx[p])) for p in months}
            for c in KEYS: macro[c].append(np.mean([rre(c, ix[p]) for p in months]))
        macro = {c: np.array(x) for c, x in macro.items()}
        allix = np.arange(len(vm))
        point = {c: np.mean([rre(c, idx[p]) for p in months]) for c in KEYS}
        for c in KEYS[1:]:
            d = macro[c] - macro["S1"]; d2 = macro[c] - macro["S2"]
            rows.append({"run": name, "random_state": rs, "K": k, "skenario": c,
                         "makro": point[c], "mikro": rre(c, allix),
                         "d_makro_vs_S1": point[c] - point["S1"],
                         "d_makro_vs_S1_lo": np.percentile(d, 2.5), "d_makro_vs_S1_hi": np.percentile(d, 97.5),
                         "d_makro_vs_S2": point[c] - point["S2"],
                         "d_makro_vs_S2_lo": np.percentile(d2, 2.5), "d_makro_vs_S2_hi": np.percentile(d2, 97.5),
                         "d_mikro_vs_S1": rre(c, allix) - rre("S1", allix)})
        print("selesai:", name, flush=True)

df = pd.DataFrame(rows); df.to_csv(OUT / "proyeksi_s34_15konfigurasi.csv", index=False)
pd.DataFrame(per_rows).to_csv(OUT / "proyeksi_s34_per_bulan_15konfigurasi.csv", index=False)
pd.set_option("display.width", 220)
print("\n== Pemeriksaan: S2/S3/S4 sama dengan per_bulan_15konfigurasi.csv")
old = pd.read_csv(OUT / "per_bulan_15konfigurasi.csv"); new = pd.DataFrame(per_rows)
mg = old.merge(new, on=["run", "bulan"], suffixes=("_lama", ""))
print("selisih maksimum:", max(float((mg[f"rre_{c}_lama"] - mg[f"rre_{c}"]).abs().max()) for c in ("S1", "S2", "S3", "S4")))
print("\n== Galat makro terhadap S1 dan S2 (15 konfigurasi)")
for c in KEYS[1:]:
    d = df[df.skenario == c]
    print(f"{c:4s}: d makro vs S1 rata-rata {d.d_makro_vs_S1.mean():+.4f} (rentang {d.d_makro_vs_S1.min():+.4f} s.d. {d.d_makro_vs_S1.max():+.4f}); "
          f"< S1 pada {int((d.d_makro_vs_S1 < 0).sum())}/15, IK < 0 pada {int((d.d_makro_vs_S1_hi < 0).sum())}/15, IK > 0 pada {int((d.d_makro_vs_S1_lo > 0).sum())}/15; "
          f"vs S2 rata-rata {d.d_makro_vs_S2.mean():+.4f}, < S2 pada {int((d.d_makro_vs_S2 < 0).sum())}/15 "
          f"(IK < 0 pada {int((d.d_makro_vs_S2_hi < 0).sum())}/15, IK > 0 pada {int((d.d_makro_vs_S2_lo > 0).sum())}/15); "
          f"d mikro vs S1 rata-rata {d.d_mikro_vs_S1.mean():+.4f}")
print("\n== Per K, rata-rata selisih makro terhadap S1")
print(df.pivot_table(index="K", columns="skenario", values="d_makro_vs_S1", aggfunc="mean").round(4).to_string())
print("\n== r42_k20"); print(df[df.run == "r42_k20"].round(4).to_string(index=False))
