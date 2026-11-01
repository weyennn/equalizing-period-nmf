"""Kurva eksponen bobot dan undersampling pada ke-15 konfigurasi.

Bobot periode dalam fungsi objektif dibuat (n_bar / n_p)^beta, artinya baris dikali (n_bar / n_p)^(beta / 2).
beta = 0 adalah S1 dan beta = 1 adalah S2 (model tersimpan dipakai apa adanya); yang dilatih di sini beta = 0,5; 1,5; 2.
Undersampling (US): dokumen latih tiap bulan diambil acak tanpa pengembalian sebanyak n bulan terkecil
(default_rng(random_state)), lalu NMF dilatih tanpa bobot. Vocabulary, IDF, split, dan X_val sama dengan run 15 konfigurasi.
NMF sama dengan build_locked_nmf (nndsvda, cd, Frobenius, max_iter 400, tanpa regularisasi, random_state konfigurasi).
Evaluasi dan bootstrap sama dengan analisis_bobot_per_bulan.py (H dipakai langsung, 1.000 resample per bulan, default_rng(0)).
Masukan: outputs/korpus_bersih_15konfigurasi_2026-10-20/ (hanya dibaca).
Keluaran: komponen H dan jumlah iterasi di outputs/paper_kurva_bobot_undersampling_2026-10-26/;
kurva_bobot_us_per_bulan.csv dan kurva_bobot_us_ringkasan.csv di folder skrip ini.
Pemakaian: <python 3.10 + scikit-learn 1.5.2> -I analysis/eksperimen_kurva_bobot_undersampling.py
"""
import time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import joblib, numpy as np, pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
SRC = Path(ROOT / "outputs/korpus_bersih_15konfigurasi_2026-10-20")
HOUT = Path(ROOT / "outputs/paper_kurva_bobot_undersampling_2026-10-26")
OUT = Path(__file__).resolve().parent
HOUT.mkdir(exist_ok=True)
BETAS = (0.5, 1.5, 2.0)
SEPDES = ["2024-09", "2024-10", "2024-11", "2024-12"]
B = 1000


def nmf(k, rs):
    return NMF(n_components=k, init="nndsvda", solver="cd", beta_loss="frobenius", max_iter=400,
               random_state=rs, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0)


per_rows, sum_rows = [], []
for rs in (7, 21, 42, 84, 123):
    for k in (15, 20, 25):
        name = f"r{rs}_k{k}"; run = SRC / name; t0 = time.time()
        X = sparse.load_npz(run / "X_base_train.npz").tocsr()
        led = pd.read_csv(run / "split_ledger.csv")
        tm = led[led.split == "train"].sort_values("split_row").year_month.astype(str).to_numpy()
        vm = led[led.split == "validation"].sort_values("split_row").year_month.astype(str).to_numpy()
        Xv = sparse.load_npz(run / "X_base_valid.npz").tocsr()
        months = sorted(set(tm)); counts = pd.Series(tm).value_counts().sort_index(); nbar = counts.mean()
        assert X.shape[0] == len(tm) and Xv.shape[0] == len(vm)
        a1 = np.sqrt(nbar / counts)[tm].to_numpy()
        assert abs(sparse.diags(a1) @ X - sparse.load_npz(run / "S2_document_weighting" / "X_train.npz")).max() < 1e-12

        def tvd(Xt, mt):
            e = np.asarray(Xt.multiply(Xt).sum(1)).ravel(); q = pd.Series(e).groupby(mt).sum(); q = q / q.sum()
            return 0.5 * float(np.abs(q - 1 / len(months)).sum())

        H, info = {}, {}
        for s, key, Xt in (("S1_baseline", "b0", X), ("S2_document_weighting", "b1", sparse.diags(a1) @ X)):
            m = joblib.load(run / s / "nmf_model.joblib"); H[key] = m.components_
            info[key] = {"n_iter": m.n_iter_, "n_train": len(tm), "warn": 0, "tvd": tvd(Xt, tm)}
        train_sets = {f"b{b:g}": (sparse.diags((nbar / counts)[tm].to_numpy() ** (b / 2)) @ X).tocsr() for b in BETAS}
        rng_us = np.random.default_rng(rs); nmin = int(counts.min())
        keep = np.sort(np.concatenate([rng_us.choice(np.flatnonzero(tm == p), nmin, replace=False) for p in months]))
        train_sets["us"] = X[keep]
        for key, Xt in train_sets.items():
            m = nmf(k, rs)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning); m.fit_transform(Xt)
            H[key] = m.components_
            info[key] = {"n_iter": m.n_iter_, "n_train": Xt.shape[0], "warn": len(caught),
                         "tvd": tvd(Xt, tm[keep] if key == "us" else tm)}
            np.savez_compressed(HOUT / f"{name}_{key}.npz", H=m.components_, n_iter=m.n_iter_)

        xsq = np.asarray(Xv.multiply(Xv).sum(1)).ravel(); res = {}
        proto = joblib.load(run / "S1_baseline" / "nmf_model.joblib")  # model terlatih dengan parameter yang sama; H diganti
        for key, Hk in H.items():
            proto.components_ = Hk; W = proto.transform(Xv)
            res[key] = xsq + np.einsum("ij,jk,ik->i", W, Hk @ Hk.T, W) - 2 * np.sum(W * np.asarray(Xv @ Hk.T), 1)
        idx = {p: np.flatnonzero(vm == p) for p in months}
        rre = lambda c, ix: np.sqrt(res[c][ix].sum() / xsq[ix].sum())
        for p in months:
            per_rows.append({"run": name, "random_state": rs, "K": k, "bulan": p, "n_train": int(counts[p]),
                             **{f"rre_{c}": rre(c, idx[p]) for c in res}})
        rng = np.random.default_rng(0)
        boots = {c: [] for c in res}
        for _ in range(B):
            ix = {p: rng.choice(idx[p], len(idx[p])) for p in months}
            for c in res: boots[c].append(np.mean([rre(c, ix[p]) for p in months]))
        boots = {c: np.array(v) for c, v in boots.items()}
        allix = np.arange(len(vm))
        for c in res:
            pm = np.array([rre(c, idx[p]) for p in months]); d1 = boots[c] - boots["b0"]; d2 = boots[c] - boots["b1"]
            top = np.argsort(-H[c], axis=1)[:, :15]
            sum_rows.append({"run": name, "random_state": rs, "K": k, "varian": c, **info[c],
                             "makro": pm.mean(), "makro_kuadrat": float(np.mean(pm ** 2)), "mikro": rre(c, allix),
                             "sepdes": float(np.mean([rre(c, idx[p]) for p in SEPDES])),
                             "agustus": rre(c, idx["2024-08"]), "bulan_terburuk": float(pm.max()),
                             "td": len(np.unique(top)) / top.size,
                             "d_makro_vs_b0_lo": np.percentile(d1, 2.5), "d_makro_vs_b0_hi": np.percentile(d1, 97.5),
                             "d_makro_vs_b1_lo": np.percentile(d2, 2.5), "d_makro_vs_b1_hi": np.percentile(d2, 97.5)})
        print(f"selesai: {name} ({time.time() - t0:.0f} dtk)", flush=True)

per = pd.DataFrame(per_rows); summ = pd.DataFrame(sum_rows)
base = summ[summ.varian == "b0"].set_index("run")
for col in ("makro", "makro_kuadrat", "mikro", "sepdes", "agustus", "bulan_terburuk"):
    summ[f"d_{col}"] = summ[col] - summ.run.map(base[col])
per.to_csv(OUT / "kurva_bobot_us_per_bulan.csv", index=False); summ.to_csv(OUT / "kurva_bobot_us_ringkasan.csv", index=False)

pd.set_option("display.width", 250)
order = ["b0", "b0.5", "b1", "b1.5", "b2", "us"]
print("\n== Pemeriksaan: b0 dan b1 sama dengan per_bulan_15konfigurasi.csv (S1, S2)")
old = pd.read_csv(OUT / "per_bulan_15konfigurasi.csv").merge(per, on=["run", "bulan"])
print("selisih maksimum:", float(max((old.rre_S1 - old.rre_b0).abs().max(), (old.rre_S2 - old.rre_b1).abs().max())))
print("\n== Rata-rata 15 konfigurasi, selisih terhadap S1 (b0)")
g = summ.groupby("varian")
tab = pd.DataFrame({"d_makro": g.d_makro.mean(), "d_makro_kuadrat": g.d_makro_kuadrat.mean(), "d_mikro": g.d_mikro.mean(),
                    "d_sepdes": g.d_sepdes.mean(), "d_agustus": g.d_agustus.mean(), "d_terburuk": g.d_bulan_terburuk.mean(),
                    "makro<S1": g.d_makro.apply(lambda x: int((x < 0).sum())),
                    "IK<0 vs S1": g.d_makro_vs_b0_hi.apply(lambda x: int((x < 0).sum())),
                    "IK>0 vs S2": g.d_makro_vs_b1_lo.apply(lambda x: int((x > 0).sum())),
                    "IK<0 vs S2": g.d_makro_vs_b1_hi.apply(lambda x: int((x < 0).sum())),
                    "td": g.td.mean(), "tvd": g.tvd.mean(), "iter_maks": g.n_iter.max(), "peringatan": g.warn.sum()}).reindex(order)
print(tab.round(4).to_string())
print("\n== Eksponen dengan galat makro terendah per konfigurasi (di antara b0..b2)")
cur = summ[summ.varian.isin(order[:5])]
print(cur.loc[cur.groupby("run").makro.idxmin()].varian.value_counts().to_string())
print(cur.loc[cur.groupby("run").makro_kuadrat.idxmin()].varian.value_counts().rename("menurut rata-rata kuadrat").to_string())
print("\n== Per K, selisih makro terhadap S1")
print(summ.pivot_table(index="K", columns="varian", values="d_makro", aggfunc="mean").reindex(columns=order).round(4).to_string())
