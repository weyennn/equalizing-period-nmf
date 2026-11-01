"""NMF per periode dua lapis (pembanding dari literatur) pada ke-15 konfigurasi.

Mengikuti pendekatan dua lapis Greene dan Cross (2017): lapis 1, NMF dengan K topik pada data latih setiap bulan secara
terpisah; lapis 2, NMF dengan K topik pada tumpukan topik bulanan (baris H tiap bulan dinormalkan L2), sehingga setiap
bulan menyumbang K topik yang setara. Basis lapis 2 dievaluasi seperti skenario lain (H dipakai langsung pada X_val ruang
dasar). Greene dan Cross memilih jumlah topik per jendela dengan koherensi; di sini K sama dengan K global agar sebanding.
NMF sama dengan build_locked_nmf (nndsvda, cd, Frobenius, max_iter 400, tanpa regularisasi, random_state konfigurasi).
Bootstrap sama dengan analisis_bobot_per_bulan.py (1.000 resample per bulan, default_rng(0)).
Masukan (hanya dibaca): outputs/korpus_bersih_15konfigurasi_2026-10-20/.
Keluaran: H dan jumlah iterasi di outputs/paper_nmf_per_periode_2026-10-27/; nmf_per_periode_ringkasan.csv dan
nmf_per_periode_per_bulan.csv di folder skrip ini.
Pemakaian: <python 3.10 + scikit-learn 1.5.2> -I analysis/eksperimen_nmf_per_periode.py
"""
import time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import joblib, numpy as np, pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
SRC = Path(ROOT / "outputs/korpus_bersih_15konfigurasi_2026-10-20")
HOUT = Path(ROOT / "outputs/paper_nmf_per_periode_2026-10-27")
OUT = Path(__file__).resolve().parent
HOUT.mkdir(exist_ok=True)
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
        months = sorted(set(tm)); assert X.shape[0] == len(tm) and Xv.shape[0] == len(vm)

        warn, iters, lapis1 = 0, [], []
        for p in months:
            m = nmf(k, rs)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning); m.fit(X[tm == p])
            warn += len(caught); iters.append(m.n_iter_)
            lapis1.append(m.components_ / np.linalg.norm(m.components_, axis=1, keepdims=True))
        Bm = np.vstack(lapis1)
        m2 = nmf(k, rs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning); m2.fit(Bm)
        warn += len(caught)
        Hpp = m2.components_
        np.savez_compressed(HOUT / f"{name}_pp.npz", H=Hpp, n_iter_lapis1=np.array(iters), n_iter_lapis2=m2.n_iter_)

        H = {"S1": joblib.load(run / "S1_baseline" / "nmf_model.joblib").components_,
             "S2": joblib.load(run / "S2_document_weighting" / "nmf_model.joblib").components_, "PP": Hpp}
        xsq = np.asarray(Xv.multiply(Xv).sum(1)).ravel(); res = {}
        proto = joblib.load(run / "S1_baseline" / "nmf_model.joblib")  # model terlatih dengan parameter yang sama; H diganti
        for key, Hk in H.items():
            proto.components_ = Hk; W = proto.transform(Xv)
            res[key] = xsq + np.einsum("ij,jk,ik->i", W, Hk @ Hk.T, W) - 2 * np.sum(W * np.asarray(Xv @ Hk.T), 1)
        idx = {p: np.flatnonzero(vm == p) for p in months}
        rre = lambda c, ix: np.sqrt(res[c][ix].sum() / xsq[ix].sum())
        for p in months:
            per_rows.append({"run": name, "random_state": rs, "K": k, "bulan": p, **{f"rre_{c}": rre(c, idx[p]) for c in res}})
        rng = np.random.default_rng(0); boots = {c: [] for c in res}
        for _ in range(B):
            ix = {p: rng.choice(idx[p], len(idx[p])) for p in months}
            for c in res: boots[c].append(np.mean([rre(c, ix[p]) for p in months]))
        boots = {c: np.array(v) for c, v in boots.items()}
        allix = np.arange(len(vm)); pm = {c: np.array([rre(c, idx[p]) for p in months]) for c in res}
        d1 = boots["PP"] - boots["S1"]; d2 = boots["PP"] - boots["S2"]
        sum_rows.append({"run": name, "random_state": rs, "K": k,
                         "iter_lapis1_maks": max(iters), "iter_lapis2": m2.n_iter_, "peringatan": warn,
                         **{f"makro_{c}": pm[c].mean() for c in res}, **{f"mikro_{c}": rre(c, allix) for c in res},
                         **{f"sepdes_{c}": float(np.mean([rre(c, idx[p]) for p in SEPDES])) for c in res},
                         **{f"agustus_{c}": rre(c, idx["2024-08"]) for c in res},
                         **{f"terburuk_{c}": float(pm[c].max()) for c in res},
                         "d_makro_PP_S1_lo": np.percentile(d1, 2.5), "d_makro_PP_S1_hi": np.percentile(d1, 97.5),
                         "d_makro_PP_S2_lo": np.percentile(d2, 2.5), "d_makro_PP_S2_hi": np.percentile(d2, 97.5)})
        print(f"selesai: {name} ({time.time() - t0:.0f} dtk)", flush=True)

per = pd.DataFrame(per_rows); s = pd.DataFrame(sum_rows)
per.to_csv(OUT / "nmf_per_periode_per_bulan.csv", index=False); s.to_csv(OUT / "nmf_per_periode_ringkasan.csv", index=False)
pd.set_option("display.width", 250)
print("\n== Pemeriksaan: S1 dan S2 sama dengan per_bulan_15konfigurasi.csv")
old = pd.read_csv(OUT / "per_bulan_15konfigurasi.csv").merge(per, on=["run", "bulan"], suffixes=("_lama", ""))
print("selisih maksimum:", float(max((old.rre_S1_lama - old.rre_S1).abs().max(), (old.rre_S2_lama - old.rre_S2).abs().max())))
print("\n== NMF per periode (PP) dibanding S1 dan S2, 15 konfigurasi")
for col in ("makro", "mikro", "sepdes", "agustus", "terburuk"):
    d1 = s[f"{col}_PP"] - s[f"{col}_S1"]; d2 = s[f"{col}_PP"] - s[f"{col}_S2"]
    print(f"{col:9s}: PP - S1 rata-rata {d1.mean():+.4f} (rentang {d1.min():+.4f} s.d. {d1.max():+.4f}), PP < S1 pada {int((d1 < 0).sum())}/15; "
          f"PP - S2 rata-rata {d2.mean():+.4f}, PP < S2 pada {int((d2 < 0).sum())}/15")
print(f"IK makro PP-S1 < 0 pada {int((s.d_makro_PP_S1_hi < 0).sum())}/15, > 0 pada {int((s.d_makro_PP_S1_lo > 0).sum())}/15; "
      f"PP-S2 < 0 pada {int((s.d_makro_PP_S2_hi < 0).sum())}/15, > 0 pada {int((s.d_makro_PP_S2_lo > 0).sum())}/15")
print(f"iterasi lapis 1 maks {s.iter_lapis1_maks.max()}, lapis 2 maks {s.iter_lapis2.max()}, peringatan konvergensi {s.peringatan.sum()}")
print("\n== Per K, selisih makro terhadap S1")
s["d_makro_PP"] = s.makro_PP - s.makro_S1; s["d_makro_S2"] = s.makro_S2 - s.makro_S1
print(s.groupby("K")[["d_makro_S2", "d_makro_PP"]].mean().round(4).to_string())
