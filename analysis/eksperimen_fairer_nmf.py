"""Fairer-NMF (Kassab et al. 2026) dengan bulan sebagai kelompok, korpus pertama.

Fairer-NMF versi MU (fairer_nmf.py, setara dengan kode resmi; lihat uji_fairer_nmf.py), aturan bobot kelompok
'decaying_lr', inisialisasi acak default_rng(random_state), 1.000 iterasi penuh. Penyimpangan dari pengaturan bawaan kode
resmi: pada data ini NMF-MU dengan rel_tol 1e-3 berhenti setelah dua iterasi (galat relatif 0,98; uji di
outputs/paper_fairer_nmf_2026-10-27/uji_pengaturan_bawaan/), sehingga galat acuan E_l terlalu besar. Karena itu
E_l dihitung dengan NMF yang konvergen pada setiap kelompok sendiri (pengaturan sama dengan model lain: nndsvda, cd,
400 iterasi), dan iterasi utama tidak dihentikan lebih awal. Pembanding solver: NMF biasa dengan MU yang sama
(acak, 1.000 iterasi) pada seluruh data latih (MU). Evaluasi sama dengan skenario lain (galat per bulan pada X_val,
bootstrap berpasangan 1.000 kali per bulan, default_rng(0)); ditambah galat berlebih maksimum pada data latih,
max_l (||X_l - W_l H|| - E_l) / ||X_l|| dengan W_l = argmin_{W>=0}, untuk S1, S2, MU, dan FN.
Masukan (hanya dibaca): outputs/korpus_bersih_15konfigurasi_2026-10-20/.
Keluaran: outputs/paper_fairer_nmf_2026-10-27/ (H dan hasil/ per konfigurasi); ringkasan fairer_nmf_* di folder ini.
Pemakaian (OMP_NUM_THREADS=1): <python 3.10 + scikit-learn 1.5.2> -I analysis/eksperimen_fairer_nmf.py satu <rs> <K>
                                ... eksperimen_fairer_nmf.py ringkas
"""
import sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import joblib, numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairer_nmf import fairer_nmf_mu, nmf_mu  # noqa: E402

SRC = Path(ROOT / "outputs/korpus_bersih_15konfigurasi_2026-10-20")
HOUT = Path(ROOT / "outputs/paper_fairer_nmf_2026-10-27")
HASIL = HOUT / "hasil"; HASIL.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).resolve().parent
SEPDES = ["2024-09", "2024-10", "2024-11", "2024-12"]
B, ITER = 1000, 1000


def galat_acuan(Xg, k, rs):
    m = NMF(n_components=k, init="nndsvda", solver="cd", beta_loss="frobenius", max_iter=400, random_state=rs,
            alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); W = m.fit_transform(Xg)
    H = m.components_; x2 = float(Xg.multiply(Xg).sum())
    return float(np.sqrt(max(x2 - 2 * np.sum(np.asarray(Xg @ H.T) * W) + np.sum((W.T @ W) * (H @ H.T)), 0.0)))


def satu(rs, k):
    name = f"r{rs}_k{k}"; run = SRC / name; t0 = time.time()
    X = sparse.load_npz(run / "X_base_train.npz").tocsr(); Xv = sparse.load_npz(run / "X_base_valid.npz").tocsr()
    Xa = sparse.load_npz(run / "X_base_all.npz").tocsr()
    led = pd.read_csv(run / "split_ledger.csv")
    tm = led[led.split == "train"].sort_values("split_row").year_month.astype(str).to_numpy()
    vm = led[led.split == "validation"].sort_values("split_row").year_month.astype(str).to_numpy()
    am = led.sort_values("model_row").year_month.astype(str).to_numpy()
    months = sorted(set(tm)); Xg = [X[tm == p] for p in months]
    E = np.array([galat_acuan(G, k, rs) for G in Xg])
    _, Hf, errs, _, n_iter = fairer_nmf_mu(Xg, k, np.random.default_rng(rs), max_iter=ITER, rel_tol=None, min_errs=E)
    _, Hmu, _ = nmf_mu(X, k, np.random.default_rng(rs), max_iter=ITER, rel_tol=None)
    np.savez_compressed(HOUT / f"{name}_fair.npz", H_FN=Hf, H_MU=Hmu, E=E, galat_berlebih_akhir=errs[:, -1], n_iter=n_iter)

    H = {"S1": joblib.load(run / "S1_baseline" / "nmf_model.joblib").components_,
         "S2": joblib.load(run / "S2_document_weighting" / "nmf_model.joblib").components_, "MU": Hmu, "FN": Hf}
    proto = joblib.load(run / "S1_baseline" / "nmf_model.joblib")   # model terlatih dengan parameter sama; H diganti

    def resid(Xm, Hk):
        proto.components_ = Hk; W = proto.transform(Xm); x = np.asarray(Xm.multiply(Xm).sum(1)).ravel()
        return x + np.einsum("ij,jk,ik->i", W, Hk @ Hk.T, W) - 2 * np.sum(W * np.asarray(Xm @ Hk.T), 1), x

    latih = {}
    for key, Hk in H.items():   # galat berlebih maksimum pada data latih (tujuan Fairer-NMF)
        r, x = resid(X, Hk)
        latih[key] = float(max((np.sqrt(r[tm == p].sum()) - e) / np.sqrt(x[tm == p].sum()) for p, e in zip(months, E)))
    res = {}; xsq = None
    for key, Hk in H.items():
        res[key], xsq = resid(Xv, Hk)
    idx = {p: np.flatnonzero(vm == p) for p in months}
    rre = lambda c, ix: np.sqrt(res[c][ix].sum() / xsq[ix].sum())
    per = [{"run": name, "random_state": rs, "K": k, "bulan": p, **{f"rre_{c}": rre(c, idx[p]) for c in res}} for p in months]
    rng = np.random.default_rng(0); boots = {c: [] for c in res}
    for _ in range(B):
        ix = {p: rng.choice(idx[p], len(idx[p])) for p in months}
        for c in res: boots[c].append(np.mean([rre(c, ix[p]) for p in months]))
    boots = {c: np.array(v) for c, v in boots.items()}
    allix = np.arange(len(vm)); pm = {c: np.array([rre(c, idx[p]) for p in months]) for c in res}
    lift, peak = {}, {}
    for key in ("S1", "FN"):
        proto.components_ = H[key]; Wa = proto.transform(Xa); s = Wa.sum(1, keepdims=True); s[s == 0] = np.nan
        prev = pd.DataFrame(Wa / s).groupby(am).mean(); lift[key] = prev / prev.mean(axis=0); peak[key] = lift[key].idxmax(axis=0).to_numpy()
    u1 = H["S1"] / np.linalg.norm(H["S1"], axis=1, keepdims=True); u2 = Hf / np.linalg.norm(Hf, axis=1, keepdims=True)
    C = u1 @ u2.T; r, c = linear_sum_assignment(-C); sim = C[r, c]
    top = np.argsort(-Hf, axis=1)[:, :15]
    row = {"run": name, "random_state": rs, "K": k, "n_iter": n_iter,
           **{f"latih_berlebih_maks_{c}": latih[c] for c in res},
           **{f"makro_{c}": pm[c].mean() for c in res}, **{f"mikro_{c}": rre(c, allix) for c in res},
           **{f"sepdes_{c}": float(np.mean([rre(c, idx[p]) for p in SEPDES])) for c in res},
           **{f"agustus_{c}": rre(c, idx["2024-08"]) for c in res}, **{f"terburuk_{c}": float(pm[c].max()) for c in res},
           **{f"d_makro_{c}_S1_{q}": np.percentile(boots[c] - boots["S1"], v) for c in ("MU", "FN") for q, v in (("lo", 2.5), ("hi", 97.5))},
           **{f"d_makro_{c}_S2_{q}": np.percentile(boots[c] - boots["S2"], v) for c in ("MU", "FN") for q, v in (("lo", 2.5), ("hi", 97.5))},
           "d_makro_FN_MU_lo": np.percentile(boots["FN"] - boots["MU"], 2.5), "d_makro_FN_MU_hi": np.percentile(boots["FN"] - boots["MU"], 97.5),
           "td_FN": len(np.unique(top)) / top.size, "tanpa_padanan": int((sim < 0.5).sum()), "kosinus_median": float(np.median(sim)),
           "puncak_agustus_FN": int((peak["FN"] == "2024-08").sum()), "puncak_sepdes_FN": int(np.isin(peak["FN"], SEPDES).sum()),
           "lift2_agustus_FN": int((lift["FN"].loc["2024-08"] >= 2).sum()), "lift2_sepdes_FN": int((lift["FN"].loc[SEPDES] >= 2).any(axis=0).sum())}
    pd.DataFrame(per).to_csv(HASIL / f"{name}_per_bulan.csv", index=False)
    pd.DataFrame([row]).to_csv(HASIL / f"{name}_ringkasan.csv", index=False)   # ditulis terakhir: tanda selesai
    print(f"selesai: {name} ({time.time() - t0:.0f} dtk)", flush=True)


def ringkas():
    names = [f"r{rs}_k{k}" for rs in (7, 21, 42, 84, 123) for k in (15, 20, 25)]
    done = [n for n in names if (HASIL / f"{n}_ringkasan.csv").exists()]
    s = pd.concat([pd.read_csv(HASIL / f"{n}_ringkasan.csv") for n in done], ignore_index=True)
    per = pd.concat([pd.read_csv(HASIL / f"{n}_per_bulan.csv") for n in done], ignore_index=True)
    s.to_csv(OUT / "fairer_nmf_ringkasan.csv", index=False); per.to_csv(OUT / "fairer_nmf_per_bulan.csv", index=False)
    old = pd.read_csv(OUT / "per_bulan_15konfigurasi.csv").merge(per, on=["run", "bulan"], suffixes=("_lama", ""))
    n = len(s)
    print(f"konfigurasi selesai: {n}/15; pemeriksaan S1/S2 vs per_bulan_15konfigurasi.csv: selisih maks "
          f"{float(max((old.rre_S1_lama - old.rre_S1).abs().max(), (old.rre_S2_lama - old.rre_S2).abs().max())):.1e}")
    for c in ("MU", "FN"):
        for col in ("makro", "mikro", "sepdes", "agustus", "terburuk"):
            d1 = s[f"{col}_{c}"] - s[f"{col}_S1"]; d2 = s[f"{col}_{c}"] - s[f"{col}_S2"]
            print(f"{c} {col:9s}: - S1 rata-rata {d1.mean():+.4f} (rentang {d1.min():+.4f} s.d. {d1.max():+.4f}), < S1 pada {int((d1 < 0).sum())}/{n}; "
                  f"- S2 rata-rata {d2.mean():+.4f}, < S2 pada {int((d2 < 0).sum())}/{n}")
        print(f"{c}: IK makro - S1 < 0 pada {int((s[f'd_makro_{c}_S1_hi'] < 0).sum())}/{n}, > 0 pada {int((s[f'd_makro_{c}_S1_lo'] > 0).sum())}/{n}; "
              f"- S2 > 0 pada {int((s[f'd_makro_{c}_S2_lo'] > 0).sum())}/{n}")
    d = s.makro_FN - s.makro_MU; dt = s.terburuk_FN - s.terburuk_MU
    print(f"FN - MU: makro rata-rata {d.mean():+.4f} (< MU pada {int((d < 0).sum())}/{n}; IK < 0 {int((s.d_makro_FN_MU_hi < 0).sum())}, > 0 {int((s.d_makro_FN_MU_lo > 0).sum())}); "
          f"terburuk rata-rata {dt.mean():+.4f} (< MU pada {int((dt < 0).sum())}/{n})")
    print("galat berlebih maks. data latih (rata-rata):", {c: round(float(s[f'latih_berlebih_maks_{c}'].mean()), 4) for c in ("S1", "S2", "MU", "FN")},
          "| FN terendah pada", int((s[[f'latih_berlebih_maks_{c}' for c in ('S1', 'S2', 'MU', 'FN')]].idxmin(axis=1) == 'latih_berlebih_maks_FN').sum()), f"/{n}")
    print(f"topik tanpa padanan {int(s.tanpa_padanan.sum())} ({s.tanpa_padanan.min()}–{s.tanpa_padanan.max()}), kosinus median {s.kosinus_median.median():.3f}; "
          f"puncak Agustus {int(s.puncak_agustus_FN.sum())}, Sep–Des {int(s.puncak_sepdes_FN.sum())}, lift2 Agustus {int(s.lift2_agustus_FN.sum())}, "
          f"lift2 Sep–Des {int(s.lift2_sepdes_FN.sum())}; TD {s.td_FN.mean():.4f}")


if len(sys.argv) > 1 and sys.argv[1] == "satu":
    satu(int(sys.argv[2]), int(sys.argv[3]))
elif len(sys.argv) > 1 and sys.argv[1] == "ringkas":
    ringkas()
else:
    sys.exit(__doc__)
