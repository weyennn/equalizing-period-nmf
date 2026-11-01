"""Analisis tambahan untuk paper v2 (20 Oktober 2026): galat validasi per bulan dan interval bootstrap pada ke-15 konfigurasi.

Hanya membaca keluaran outputs/korpus_bersih_15konfigurasi_2026-10-20/ (model NMF, X_base_valid, split_ledger,
bobot dokumen); tidak melatih ulang model. Perhitungan galat sama dengan ringkas_15.py (H dipakai langsung pada S3/S4).
Bootstrap mengikuti audit6.py: terstratifikasi per bulan, 1.000 kali, default_rng(0) per konfigurasi; selisih makro dan mikro
dihitung berpasangan terhadap S1 pada sampel yang sama.
Pemakaian: <python 3.10 + scikit-learn 1.5.2> -I analysis/analisis_bobot_per_bulan.py
"""
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import joblib, numpy as np, pandas as pd
from scipy import sparse, stats
warnings.filterwarnings("ignore")
SRC = Path(ROOT / "outputs/korpus_bersih_15konfigurasi_2026-10-20")
OUT = Path(__file__).resolve().parent
SC = ["S1_baseline", "S2_document_weighting", "S3_temporal_tfidf", "S4_dual_level"]
SEPDES = ["2024-09", "2024-10", "2024-11", "2024-12"]
B = 1000
per_rows, boot_rows = [], []
for rs in (7, 21, 42, 84, 123):
    for k in (15, 20, 25):
        name = f"r{rs}_k{k}"; run = SRC / name
        Xv = sparse.load_npz(run / "X_base_valid.npz").tocsr()
        led = pd.read_csv(run / "split_ledger.csv"); v = led[led.split == "validation"].sort_values("split_row")
        vm = v.year_month.astype(str).to_numpy(); months = sorted(set(vm))
        w = pd.read_csv(run / "document_weights_train_by_month.csv").set_index("year_month")
        xsq = np.asarray(Xv.multiply(Xv).sum(1)).ravel()
        res = {}
        for s in SC:
            m = joblib.load(run / s / "nmf_model.joblib"); H = m.components_; W = m.transform(Xv)
            res[s[:2]] = xsq + np.einsum("ij,jk,ik->i", W, H @ H.T, W) - 2 * np.sum(W * np.asarray(Xv @ H.T), 1)
        idx = {p: np.flatnonzero(vm == p) for p in months}
        rre = lambda c, ix: np.sqrt(res[c][ix].sum() / xsq[ix].sum())
        for p in months:
            r = {"run": name, "random_state": rs, "K": k, "bulan": p, "n_train": int(w.loc[p, "n_documents"]),
                 "a_p": float(w.loc[p, "document_weight"]), "n_valid": len(idx[p])}
            for c in ("S1", "S2", "S3", "S4"): r[f"rre_{c}"] = rre(c, idx[p])
            per_rows.append(r)
        rng = np.random.default_rng(0)
        macro = {c: [] for c in ("S1", "S2", "S3", "S4")}
        for b in range(B):
            ix = {p: rng.choice(idx[p], len(idx[p])) for p in months}
            for c in macro: macro[c].append(np.mean([rre(c, ix[p]) for p in months]))
        micro = {c: [] for c in ("S1", "S2", "S3", "S4")}
        for b in range(B):
            ix = np.concatenate([rng.choice(idx[p], len(idx[p])) for p in months])
            for c in micro: micro[c].append(rre(c, ix))
        allix = np.arange(len(vm))
        for c in ("S2", "S3", "S4"):
            dm = np.array(macro[c]) - np.array(macro["S1"]); du = np.array(micro[c]) - np.array(micro["S1"])
            boot_rows.append({"run": name, "random_state": rs, "K": k, "skenario": c,
                              "d_makro": np.mean([rre(c, idx[p]) - rre("S1", idx[p]) for p in months]),
                              "d_makro_lo": np.percentile(dm, 2.5), "d_makro_hi": np.percentile(dm, 97.5),
                              "d_mikro": rre(c, allix) - rre("S1", allix),
                              "d_mikro_lo": np.percentile(du, 2.5), "d_mikro_hi": np.percentile(du, 97.5)})
        print("selesai:", name, flush=True)

per = pd.DataFrame(per_rows); boot = pd.DataFrame(boot_rows)
for c in ("S2", "S3", "S4"): per[f"d_{c}"] = per[f"rre_{c}"] - per["rre_S1"]
per["log_bobot_kuadrat"] = np.log(per.a_p ** 2)          # log(n_bar / n_p)
per.to_csv(OUT / "per_bulan_15konfigurasi.csv", index=False); boot.to_csv(OUT / "bootstrap_15konfigurasi.csv", index=False)

pd.set_option("display.width", 220)
print("\n== Bootstrap 95% (selisih terhadap S1), jumlah konfigurasi dengan interval tidak memuat nol")
for c in ("S2", "S3", "S4"):
    d = boot[boot.skenario == c]
    print(f"{c}: makro IK < 0 pada {int((d.d_makro_hi < 0).sum())}/15, IK > 0 pada {int((d.d_makro_lo > 0).sum())}/15; "
          f"mikro IK > 0 pada {int((d.d_mikro_lo > 0).sum())}/15")
print(boot[boot.run == "r42_k20"].round(4).to_string(index=False))
print("\n== Bobot bulan vs perubahan galat (180 titik = 15 konfigurasi x 12 bulan)")
for c in ("S2", "S4", "S3"):
    pr = stats.pearsonr(per.log_bobot_kuadrat, per[f"d_{c}"]); sp = stats.spearmanr(per.log_bobot_kuadrat, per[f"d_{c}"])
    within = per.groupby("run").apply(lambda g: stats.spearmanr(g.log_bobot_kuadrat, g[f"d_{c}"])[0])
    print(f"{c}: Pearson r = {pr[0]:.3f} (p = {pr[1]:.2g}); Spearman rho = {sp[0]:.3f} (p = {sp[1]:.2g}); "
          f"rho per konfigurasi negatif pada {int((within < 0).sum())}/15 (median {within.median():.3f}, rentang {within.min():.3f} s.d. {within.max():.3f})")
print("\n== Per bulan, S2 - S1 pada 15 konfigurasi")
g = per.groupby("bulan").agg(n_train=("n_train", "first"), a_p=("a_p", "first"), d_rata=("d_S2", "mean"), d_min=("d_S2", "min"),
                             d_max=("d_S2", "max"), turun=("d_S2", lambda x: int((x < 0).sum())), d_S4_rata=("d_S4", "mean"),
                             turun_S4=("d_S4", lambda x: int((x < 0).sum())))
print(g.round(4).to_string())
sd = per.groupby("run").agg(sd_S1=("rre_S1", "std"), sd_S2=("rre_S2", "std"))
print(f"\nSebaran galat antarbulan (SD 12 bulan): S1 rata-rata {sd.sd_S1.mean():.4f}, S2 {sd.sd_S2.mean():.4f}; "
      f"S2 lebih kecil pada {int((sd.sd_S2 < sd.sd_S1).sum())}/15")
