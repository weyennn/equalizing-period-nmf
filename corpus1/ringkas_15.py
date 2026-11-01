"""Ringkasan 15 konfigurasi (5 random state x K = 15, 20, 25) untuk korpus bersih.

Galat dihitung ulang dari model dan matriks validasi yang tersimpan (RRE per periode, makro, mikro). Juga dihitung: galat September-Desember,
rata-rata dan SD per K, jumlah konfigurasi dengan galat makro lebih rendah daripada S1, serta rentang c_v dan TD.
Konfigurasi yang belum selesai (tanpa run_manifest.json) dilewati. Pemakaian: python ringkas_15.py
"""
import json, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import joblib, numpy as np, pandas as pd
from scipy import sparse
warnings.filterwarnings("ignore")
O = ROOT / "outputs/korpus_bersih_15konfigurasi_2026-10-20"
SC = ["S1_baseline", "S2_document_weighting", "S3_temporal_tfidf", "S4_dual_level"]
SEPDES = ["2024-09", "2024-10", "2024-11", "2024-12"]
rows = []
for rs in (7, 21, 42, 84, 123):
    for k in (15, 20, 25):
        name = f"r{rs}_k{k}"; run = O / name
        if not (run / "run_manifest.json").exists():
            print("lewati (belum selesai):", name); continue
        Xv = sparse.load_npz(run / "X_base_valid.npz").tocsr()
        led = pd.read_csv(run / "split_ledger.csv"); v = led[led.split == "validation"].sort_values("split_row")
        vm = v.year_month.astype(str).to_numpy(); months = sorted(set(vm))
        assert all(p in months for p in SEPDES), months
        xsq = np.asarray(Xv.multiply(Xv).sum(1)).ravel()
        per = {}
        for s in SC:
            m = joblib.load(run / s / "nmf_model.joblib"); H = m.components_
            W = m.transform(Xv)
            res = xsq + np.einsum("ij,jk,ik->i", W, H @ H.T, W) - 2 * np.sum(W * np.asarray(Xv @ H.T), 1)
            pm = {p: float(np.sqrt(res[vm == p].sum() / xsq[vm == p].sum())) for p in months}
            sm = json.load(open(run / s / "scenario_summary.json"))
            per[s[:2]] = pm
            rows.append({"run": name, "random_state": rs, "K": k, "skenario": s[:2],
                         "rre_mikro": float(np.sqrt(res.sum() / xsq.sum())), "rre_makro": float(np.mean(list(pm.values()))),
                         "rre_sepdes": float(np.mean([pm[p] for p in SEPDES])),
                         "cv": sm["coherence_cv"], "td": sm["topic_diversity_top15"], "iter": sm["n_iter"],
                         "konvergen_tanpa_peringatan": bool(sm["converged_before_max_iter"] and not sm["convergence_warnings"])})
        for r in rows:
            if r["run"] == name:
                r["dmakro_vs_S1"] = r["rre_makro"] - float(np.mean(list(per["S1"].values())))
                r["dsepdes_vs_S1"] = r["rre_sepdes"] - float(np.mean([per["S1"][p] for p in SEPDES]))
                r["bulan_lebih_baik"] = int(sum(per[r["skenario"]][p] < per["S1"][p] for p in months))
        print("selesai:", name)
df = pd.DataFrame(rows); df.to_csv(O / "ringkasan_15_konfigurasi.csv", index=False)
n = df.run.nunique()
agg = df.groupby(["K", "skenario"]).agg(mikro=("rre_mikro", "mean"), mikro_sd=("rre_mikro", "std"),
                                        makro=("rre_makro", "mean"), makro_sd=("rre_makro", "std"),
                                        sepdes=("rre_sepdes", "mean"), dmakro=("dmakro_vs_S1", "mean"),
                                        makro_lebih_rendah=("dmakro_vs_S1", lambda x: int((x < 0).sum())),
                                        cv=("cv", "mean"), cv_sd=("cv", "std"), td=("td", "mean"), td_sd=("td", "std"),
                                        n=("run", "count"))
agg.to_csv(O / "ringkasan_15_per_K.csv")
pd.set_option("display.width", 220)
print(f"\nKonfigurasi selesai: {n} dari 15; model: {len(df)}; konvergen tanpa peringatan: {int(df.konvergen_tanpa_peringatan.sum())}")
print(agg.round(4).to_string())
for s in ("S2", "S3", "S4"):
    d = df[df.skenario == s]
    print(f"{s}: makro lebih rendah daripada S1 pada {int((d.dmakro_vs_S1 < 0).sum())} dari {len(d)} konfigurasi "
          f"(selisih {d.dmakro_vs_S1.min():+.4f} s.d. {d.dmakro_vs_S1.max():+.4f}); "
          f"Sep-Des lebih rendah pada {int((d.dsepdes_vs_S1 < 0).sum())} dari {len(d)}")
print(f"Rentang c_v seluruh skenario: {df.cv.min():.4f}-{df.cv.max():.4f}; Topic Diversity: {df.td.min():.4f}-{df.td.max():.4f}")
