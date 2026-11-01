"""Fairer-NMF dengan galat acuan dari validasi silang (28 Oktober 2026), kedua korpus.

Menguji penjelasan di paper (Bagian 6.2): galat acuan E_l Fairer-NMF, yang dihitung pada data latih periode l sendiri,
terlalu optimistis untuk periode kecil, sehingga FN mengoreksi berlebihan ke arah periode kecil. Di sini E_l diganti dengan
galat validasi silang 5 lipatan di dalam data latih periode l (data validasi tidak disentuh): NMF K topik (nndsvda, cd,
400 iterasi, random_state konfigurasi) dilatih pada 4/5 dokumen latih periode, W lipatan sisanya diestimasi dengan H tetap
(transform), dan acuan_cv(l) = sqrt(jumlah residu kuadrat / jumlah norma kuadrat) atas kelima lipatan
(KFold acak, random_state konfigurasi); E_cv(l) = acuan_cv(l) * ||X_l||.
FNcv = fairer_nmf_mu dengan min_errs = E_cv; inisialisasi default_rng(random_state) dan 1.000 iterasi penuh sama dengan FN
(eksperimen_fairer_nmf.py, korpus2_huffpost.py fair).
Evaluasi sama dengan FN: galat per periode pada X_val untuk semua varian (S1, S2, beta, US, PP, MU, FN, FNcv), bootstrap
berpasangan 1.000 kali per periode (default_rng(0)) untuk FNcv terhadap S1, S2, dan FN; kriteria FN (galat berlebih
maksimum) pada data latih dengan acuan latih dan acuan_cv, dan pada data validasi dengan acuan latih dan acuan validasi
(acuan_val, acuan_cakupan_per_periode.csv); cakupan topik periode (basis NMF acuan periode, kosinus >= 0,5); pencocokan
topik dengan S1, periode puncak, lift; c_v dan keragaman topik (cara sama dengan cv_korpus1_varian.py dan
korpus2_huffpost.py). Pemeriksaan: E, galat S1/S2/FN per periode, dan c_v FN harus sama dengan hasil tersimpan.
Masukan (hanya dibaca): seperti analisis_acuan_validasi_cakupan.py, ditambah acuan_cakupan_per_periode.csv,
cv_korpus1_varian.csv, korpus2_fairer_nmf.csv, dan per-periode tersimpan di folder ini.
Keluaran: per konfigurasi di outputs/paper_fn_acuan_cv_2026-10-28/; ringkasan fn_acuan_cv_* di folder ini.
Pemakaian (OMP_NUM_THREADS=1, satu konfigurasi per proses):
  <python 3.10 + scikit-learn 1.5.2 + gensim 4.3.3> -I analysis/eksperimen_fn_acuan_cv.py satu <korpus 1|2> <rs> <K>
  ... eksperimen_fn_acuan_cv.py ringkas
"""
import json, sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
from types import SimpleNamespace
import joblib, numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from analisis_acuan_validasi_cakupan import SRC1, T6, data_korpus1, nmf  # noqa: E402
from fairer_nmf import fairer_nmf_mu  # noqa: E402

HOUT = T6 / "paper_fn_acuan_cv_2026-10-28"; HOUT.mkdir(parents=True, exist_ok=True)
RS, KS = (7, 21, 42, 84, 123), (15, 20, 25)
B, ITER, LIPAT = 1000, 1000, 5
KECIL = {1: ["2024-09", "2024-10", "2024-11", "2024-12"], 2: ["2019", "2020", "2021", "2022"]}
BESAR = {1: ["2024-08"], 2: ["2016", "2017"]}
warnings.filterwarnings("ignore", category=FutureWarning)


def data1(rs, k):
    X, Xv, tm, vm, H, E = data_korpus1(rs, k)
    run = SRC1 / f"r{rs}_k{k}"
    Xa = sparse.load_npz(run / "X_base_all.npz").tocsr()
    am = pd.read_csv(run / "split_ledger.csv").sort_values("model_row").year_month.astype(str).to_numpy()

    def cv(Hs):   # c_v korpus 1, seperti satu() di cv_korpus1_varian.py
        from cv_korpus1_varian import muat
        _, _, m, coh, _ = muat(rs, k)
        feats = joblib.load(run / "tfidf_vectorizer_train_only.joblib").get_feature_names_out()
        hasil = {}
        for key, Hk in Hs.items():
            topik = m.top_words(SimpleNamespace(components_=Hk), feats, m.TOP_N_WORDS)
            hasil[key] = (m.compute_cv_coherence(topik, coh, processes=1).coherence_cv, m.topic_diversity(topik))
        return hasil
    return X, Xv, tm, vm, H, E, Xa, am, cv


def data2(rs, k):
    # sama dengan data_korpus2() di analisis_acuan_validasi_cakupan.py, ditambah korpus koherensi (korpus2_huffpost.py)
    sys.path.insert(0, str(ROOT / "pipeline"))
    from preprocessing import clean_text
    from temporal_nmf_pipeline import compute_cv_coherence, encode_features_for_coherence, split_train_validation
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    data = ROOT / "data/raw/News_Category_Dataset_v3.json"
    rows = [json.loads(line) for line in open(data, encoding="utf-8")]
    df = pd.DataFrame([r for r in rows if r["category"] == "POLITICS"])
    df["text"] = (df.headline.fillna("") + " " + df.short_description.fillna("")).str.strip()
    df = df[df.text.str.len() > 0].drop_duplicates("link").drop_duplicates("text")
    sw = set(stopwords.words("english"))
    df["proc"] = [" ".join(w for w in clean_text(t).split() if w not in sw and len(w) > 2) for t in df.text]
    df = df[df.proc.str.split().str.len() >= 5].reset_index(drop=True); df["year"] = df.date.str[:4]
    assert len(df) == 35161
    tr, va = split_train_validation(df, 0.20, rs, stratify_column="year")
    vec = TfidfVectorizer(max_features=8000, min_df=10, max_df=0.60, ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sublinear_tf=True, smooth_idf=True, norm="l2",
                          dtype=np.float64)
    X = vec.fit_transform(tr.proc).tocsr(); Xv = vec.transform(va.proc).tocsr()
    ktr = X.getnnz(axis=1) > 0; kva = Xv.getnnz(axis=1) > 0
    tr, X, va, Xv = tr[ktr], X[ktr], va[kva], Xv[kva]
    tm, vm = tr.year.to_numpy(), va.year.to_numpy()
    h2 = T6 / "paper_korpus2_huffpost_2026-10-27"; name = f"r{rs}_k{k}"
    hs = np.load(h2 / f"{name}.npz"); fz = np.load(h2 / f"{name}_fair.npz")
    H = {**{v: hs[v] for v in ("S1", "S2", "b0.5", "b1.5", "b2", "US", "PP")}, "MU": fz["H_MU"], "FN": fz["H_FN"]}

    def cv(Hs):
        analyzer, vocab, feats = vec.build_analyzer(), vec.vocabulary_, vec.get_feature_names_out()
        coh = encode_features_for_coherence([f for f in analyzer(t) if f in vocab] for t in tr.proc)
        hasil = {}
        for key, Hk in Hs.items():
            top = np.argsort(-Hk, axis=1)[:, :15]
            hasil[key] = (compute_cv_coherence([[str(feats[j]) for j in t] for t in top], coh).coherence_cv,
                          len(np.unique(top)) / top.size)
        return hasil
    return X, Xv, tm, vm, H, fz["E"], sparse.vstack([X, Xv]).tocsr(), np.concatenate([tm, vm]), cv


def galat2(G, W, Hk):
    """||G - W Hk||_F^2 tanpa membentuk W Hk."""
    return max(float(G.multiply(G).sum()) - 2 * float(np.sum(np.asarray(G @ Hk.T) * W)) + float(np.sum((W.T @ W) * (Hk @ Hk.T))), 0.0)


def acuan_periode(G, k, rs):
    """Galat acuan latih (seperti FN), basis periode, dan acuan validasi silang LIPAT lipatan di dalam G."""
    m = nmf(k, rs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); W = m.fit_transform(G)
    Hp = m.components_; E = float(np.sqrt(galat2(G, W, Hp)))
    r2 = x2 = 0.0
    for a, b in KFold(LIPAT, shuffle=True, random_state=rs).split(np.arange(G.shape[0])):
        mf = nmf(k, rs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); mf.fit(G[a]); Wb = mf.transform(G[b])
        r2 += galat2(G[b], Wb, mf.components_); x2 += float(G[b].multiply(G[b]).sum())
    return E, Hp, float(np.sqrt(r2 / x2))


def satu(korpus, rs, k):
    name = f"k{korpus}_r{rs}_k{k}"; run = f"r{rs}_k{k}"; t0 = time.time()
    X, Xv, tm, vm, H, E_simpan, Xa, am, hitung_cv = (data1 if korpus == 1 else data2)(rs, k)
    periods = sorted(set(tm)); Xg = [X[tm == p] for p in periods]
    E, Hp, acv = [], [], []
    for G in Xg:
        e, h, a = acuan_periode(G, k, rs); E.append(e); Hp.append(h); acv.append(a)
    E = np.array(E); acv = np.array(acv); norma = np.array([np.sqrt(float(G.multiply(G).sum())) for G in Xg])
    selisih_E = float(np.abs(E - E_simpan).max())
    E_cv = acv * norma
    t1 = time.time()
    _, Hc, errs, _, n_iter = fairer_nmf_mu(Xg, k, np.random.default_rng(rs), max_iter=ITER, rel_tol=None, min_errs=E_cv)
    t_fn = time.time() - t1
    np.savez_compressed(HOUT / f"{name}.npz", H_FNcv=Hc, E=E, E_cv=E_cv, acuan_cv=acv, galat_berlebih_akhir=errs[:, -1], n_iter=n_iter)
    H = {**H, "FNcv": Hc}

    proto = nmf(k, rs).set_params(max_iter=1)   # model terlatih dengan parameter sama; H diganti (seperti skrip lain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); proto.fit(X)
    proto.set_params(max_iter=400)

    def resid(Xm, Hk):
        proto.components_ = Hk; Wm = proto.transform(Xm); x = np.asarray(Xm.multiply(Xm).sum(1)).ravel()
        return x + np.einsum("ij,jk,ik->i", Wm, Hk @ Hk.T, Wm) - 2 * np.sum(Wm * np.asarray(Xm @ Hk.T), 1), x

    latih = {}
    for key in ("S1", "S2", "MU", "FN", "FNcv"):   # galat berlebih maksimum pada data latih, dua acuan
        r_, x_ = resid(X, H[key]); g = np.array([np.sqrt(r_[tm == p].sum()) for p in periods])
        latih[key] = {"latih": float(np.max((g - E) / norma)), "cv": float(np.max((g - E_cv) / norma)),
                      "periode_maks_latih": periods[int(np.argmax((g - E) / norma))]}
    res = {}; xsq = None
    for key, Hk in H.items():
        res[key], xsq = resid(Xv, Hk)
    idx = {p: np.flatnonzero(vm == p) for p in periods}
    rre = lambda c, ix: float(np.sqrt(res[c][ix].sum() / xsq[ix].sum()))
    pm = {c: np.array([rre(c, idx[p]) for p in periods]) for c in res}

    ac = pd.read_csv(OUT / "acuan_cakupan_per_periode.csv", dtype={"periode": str})
    ac = ac[(ac.korpus == korpus) & (ac.run == run)].set_index("periode").reindex(periods)
    assert ac.acuan_val.notna().all() and float(np.abs(ac.acuan_latih.to_numpy() - E / norma).max()) < 1e-9
    aval = ac.acuan_val.to_numpy()
    val_krit = {c: {"latih": float(np.max(pm[c] - E / norma)), "val": float(np.max(pm[c] - aval)),
                    "periode_maks_val": periods[int(np.argmax(pm[c] - aval))]} for c in res}

    # pemeriksaan galat per periode S1, S2, FN terhadap hasil tersimpan
    if korpus == 1:
        lama = pd.read_csv(OUT / "fairer_nmf_per_bulan.csv", dtype={"bulan": str}).query("run == @run").set_index("bulan")
    else:
        lama = pd.read_csv(OUT / "korpus2_fairer_nmf_per_periode.csv", dtype={"tahun": str}).query("run == @run").set_index("tahun")
    lama = lama.reindex(periods)
    selisih_rre = float(max(np.abs(lama[f"rre_{c}"].to_numpy() - pm[c]).max() for c in ("S1", "S2", "FN")))

    rng = np.random.default_rng(0); bc = ("S1", "S2", "FN", "FNcv"); boots = {c: [] for c in bc}
    for _ in range(B):
        ix = {p: rng.choice(idx[p], len(idx[p])) for p in periods}
        for c in bc: boots[c].append(np.mean([rre(c, ix[p]) for p in periods]))
    boots = {c: np.array(v) for c, v in boots.items()}

    unit = lambda h: h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-300)
    cak = [(unit(h) @ unit(Hc).T).max(axis=1) for h in Hp]
    proto.components_ = H["S1"]; Wa = proto.transform(Xa); s = Wa.sum(1, keepdims=True); s[s == 0] = np.nan
    prev = pd.DataFrame(Wa / s).groupby(am).mean(); lift_s1 = prev / prev.mean(axis=0)
    proto.components_ = Hc; Wa = proto.transform(Xa); s = Wa.sum(1, keepdims=True); s[s == 0] = np.nan
    prev = pd.DataFrame(Wa / s).groupby(am).mean(); lift = prev / prev.mean(axis=0); peak = lift.idxmax(axis=0).to_numpy()
    C = unit(H["S1"]) @ unit(Hc).T; r, c = linear_sum_assignment(-C); sim = C[r, c]
    t2 = time.time(); cvh = hitung_cv({"FN": H["FN"], "FNcv": Hc}); t_cv = time.time() - t2
    if korpus == 1:
        cv_lama = pd.read_csv(OUT / "cv_korpus1_varian.csv").query("run == @run and varian == 'FN'").cv.iloc[0]
    else:
        cv_lama = pd.read_csv(OUT / "korpus2_fairer_nmf.csv").query("run == @run").cv_FN.iloc[0]

    kc, bs = KECIL[korpus], BESAR[korpus]
    per = [{"korpus": korpus, "run": run, "K": k, "periode": p, "n_latih": G.shape[0], "acuan_latih": E[i] / norma[i],
            "acuan_cv": acv[i], "acuan_val": aval[i], **{f"rre_{c}": pm[c][i] for c in res},
            "cakupan_05_FNcv": float((cak[i] >= 0.5).mean()), "cakupan_rata_FNcv": float(cak[i].mean())}
           for i, (p, G) in enumerate(zip(periods, Xg))]
    row = {"korpus": korpus, "run": run, "random_state": rs, "K": k, "n_iter": n_iter,
           "selisih_E_tersimpan": selisih_E, "selisih_rre_tersimpan": selisih_rre, "selisih_cv_FN_tersimpan": abs(cvh["FN"][0] - cv_lama),
           **{f"makro_{c}": float(pm[c].mean()) for c in res}, **{f"mikro_{c}": rre(c, np.arange(len(vm))) for c in res},
           **{f"kecil_{c}": float(np.mean([rre(c, idx[p]) for p in kc])) for c in res},
           **{f"besar_{c}": float(np.mean([rre(c, idx[p]) for p in bs])) for c in res},
           **{f"terburuk_{c}": float(pm[c].max()) for c in res},
           **{f"periode_terburuk_{c}": periods[int(np.argmax(pm[c]))] for c in ("S1", "S2", "FN", "FNcv")},
           **{f"d_makro_FNcv_{b}_{q}": float(np.percentile(boots["FNcv"] - boots[b], v)) for b in ("S1", "S2", "FN") for q, v in (("lo", 2.5), ("hi", 97.5))},
           **{f"krit_latih_acuanlatih_{c}": latih[c]["latih"] for c in latih}, **{f"krit_latih_acuancv_{c}": latih[c]["cv"] for c in latih},
           **{f"periode_krit_latih_{c}": latih[c]["periode_maks_latih"] for c in latih},
           **{f"krit_val_acuanlatih_{c}": val_krit[c]["latih"] for c in res}, **{f"krit_val_acuanval_{c}": val_krit[c]["val"] for c in res},
           **{f"periode_krit_val_{c}": val_krit[c]["periode_maks_val"] for c in ("S1", "S2", "FN", "FNcv")},
           "cv_FN": cvh["FN"][0], "cv_FNcv": cvh["FNcv"][0], "td_FNcv": cvh["FNcv"][1],
           "tanpa_padanan_FNcv": int((sim < 0.5).sum()), "kosinus_median_FNcv": float(np.median(sim)),
           "puncak_besar_FNcv": int(np.isin(peak, bs).sum()), "puncak_kecil_FNcv": int(np.isin(peak, kc).sum()),
           "lift2_kecil_FNcv": int((lift.loc[kc] >= 2).any(axis=0).sum()),
           "detik_fn": t_fn, "detik_cv": t_cv}
    pd.DataFrame(per).to_csv(HOUT / f"{name}_per_periode.csv", index=False)
    pd.DataFrame([row]).to_csv(HOUT / f"{name}_ringkasan.csv", index=False)   # ditulis terakhir: tanda selesai
    print(f"selesai: {name} ({time.time() - t0:.0f} dtk; FNcv {t_fn:.0f} dtk, c_v {t_cv:.0f} dtk); pemeriksaan: E {selisih_E:.1e}, "
          f"RRE {selisih_rre:.1e}, c_v FN {row['selisih_cv_FN_tersimpan']:.1e}", flush=True)
    print(pd.DataFrame(per)[["periode", "n_latih", "acuan_latih", "acuan_cv", "acuan_val", "rre_S1", "rre_S2", "rre_FN", "rre_FNcv"]].round(4).to_string(index=False))
    for c in ("S1", "S2", "FN", "FNcv"):
        print(f"{c:5s} makro {row[f'makro_{c}']:.4f} mikro {row[f'mikro_{c}']:.4f} kecil {row[f'kecil_{c}']:.4f} besar {row[f'besar_{c}']:.4f} "
              f"terburuk {row[f'terburuk_{c}']:.4f} ({row.get(f'periode_terburuk_{c}')}); krit. val (acuan val) {val_krit[c]['val']:.4f}")
    print(f"FNcv - S2 makro IK [{row['d_makro_FNcv_S2_lo']:+.4f}, {row['d_makro_FNcv_S2_hi']:+.4f}]; - FN [{row['d_makro_FNcv_FN_lo']:+.4f}, "
          f"{row['d_makro_FNcv_FN_hi']:+.4f}]; c_v FN {cvh['FN'][0]:.4f} FNcv {cvh['FNcv'][0]:.4f}; tanpa padanan {row['tanpa_padanan_FNcv']}")


def ringkas():
    f = [HOUT / f"k{c}_r{rs}_k{k}_ringkasan.csv" for c in (1, 2) for rs in RS for k in KS]
    s = pd.concat([pd.read_csv(x) for x in f if x.exists()], ignore_index=True)
    per = pd.concat([pd.read_csv(HOUT / f"k{r.korpus}_{r.run}_per_periode.csv", dtype={"periode": str}) for r in s.itertuples()], ignore_index=True)
    s.to_csv(OUT / "fn_acuan_cv_ringkasan.csv", index=False); per.to_csv(OUT / "fn_acuan_cv_per_periode.csv", index=False)
    for korpus in (1, 2):
        x = s[s.korpus == korpus]; n = len(x)
        if not n:
            continue
        print(f"\n== korpus {korpus}, {n} konfigurasi; pemeriksaan maks: E {x.selisih_E_tersimpan.max():.1e}, RRE {x.selisih_rre_tersimpan.max():.1e}, "
              f"c_v FN {x.selisih_cv_FN_tersimpan.max():.1e}; iterasi {x.n_iter.unique()}")
        y = per[per.korpus == korpus]
        print(y.groupby("periode")[["n_latih", "acuan_latih", "acuan_cv", "acuan_val"]].mean().round(4).to_string())
        for col in ("makro", "mikro", "kecil", "besar", "terburuk"):
            out = []
            for c in ("FN", "FNcv"):
                d1 = x[f"{col}_{c}"] - x[f"{col}_S1"]; d2 = x[f"{col}_{c}"] - x[f"{col}_S2"]
                out.append(f"{c} - S1 {d1.mean():+.4f} (< S1 {int((d1 < 0).sum())}/{n}), - S2 {d2.mean():+.4f} (< S2 {int((d2 < 0).sum())}/{n})")
            d3 = x[f"{col}_FNcv"] - x[f"{col}_FN"]
            print(f"{col:8s}: " + "; ".join(out) + f"; FNcv - FN {d3.mean():+.4f} (< FN {int((d3 < 0).sum())}/{n})")
        print(f"IK makro FNcv: - S1 < 0 {int((x.d_makro_FNcv_S1_hi < 0).sum())}, > 0 {int((x.d_makro_FNcv_S1_lo > 0).sum())}; "
              f"- S2 > 0 {int((x.d_makro_FNcv_S2_lo > 0).sum())}, < 0 {int((x.d_makro_FNcv_S2_hi < 0).sum())}; "
              f"- FN < 0 {int((x.d_makro_FNcv_FN_hi < 0).sum())}, > 0 {int((x.d_makro_FNcv_FN_lo > 0).sum())}")
        print("terburuk lebih tinggi dari S1: FN", int((x.terburuk_FN > x.terburuk_S1).sum()), "FNcv", int((x.terburuk_FNcv > x.terburuk_S1).sum()),
              "| periode terburuk FNcv:", x.periode_terburuk_FNcv.astype(str).value_counts().to_dict())
        b2 = "b2" if korpus == 1 else "S2"
        lebih = sum(all(r[f"{c}_{b2}"] < r[f"{c}_FNcv"] for c in ("makro", "mikro", "kecil", "besar", "terburuk")) for _, r in x.iterrows())
        print(f"{b2} lebih rendah dari FNcv pada kelima ukuran galat: {lebih}/{n}")
        for nama, kol in (("latih, acuan latih", "krit_latih_acuanlatih"), ("latih, acuan cv", "krit_latih_acuancv")):
            m = x[[f"{kol}_{c}" for c in ("S1", "S2", "MU", "FN", "FNcv")]]
            print(f"kriteria {nama}: rata-rata", {c.split('_')[-1]: round(float(m[c].mean()), 4) for c in m},
                  "| terendah:", m.idxmin(axis=1).str.split("_").str[-1].value_counts().to_dict())
        var = ["S1", "S2", "b0.5", "b1.5", "b2", "US", "PP", "MU", "FN", "FNcv"]
        for nama, kol in (("val, acuan latih", "krit_val_acuanlatih"), ("val, acuan val", "krit_val_acuanval")):
            m = x[[f"{kol}_{c}" for c in var]]
            print(f"kriteria {nama}: rata-rata", {c.split('_')[-1]: round(float(m[c].mean()), 4) for c in m},
                  "| terendah:", m.idxmin(axis=1).str.split("_").str[-1].value_counts().to_dict(),
                  "| FNcv < FN", int((m[f"{kol}_FNcv"] < m[f"{kol}_FN"]).sum()), "FNcv < S2", int((m[f"{kol}_FNcv"] < m[f"{kol}_S2"]).sum()))
        print("periode kriteria val maks (acuan val) FNcv:", x.periode_krit_val_FNcv.astype(str).value_counts().to_dict())
        dcv = x.cv_FNcv - x.cv_FN
        print(f"c_v FNcv - FN {dcv.mean():+.4f} (lebih tinggi {int((dcv > 0).sum())}/{n}); tanpa padanan FNcv {int(x.tanpa_padanan_FNcv.sum())} "
              f"(kosinus median {x.kosinus_median_FNcv.median():.3f}); puncak besar {int(x.puncak_besar_FNcv.sum())}, kecil {int(x.puncak_kecil_FNcv.sum())}, "
              f"lift2 kecil {int(x.lift2_kecil_FNcv.sum())}; TD {x.td_FNcv.mean():.4f}")
        kc = KECIL[korpus]; bs = BESAR[korpus]
        for nama, ps in (("kecil", kc), ("besar", bs), ("semua", sorted(y.periode.unique()))):
            z = y[y.periode.isin(ps)].groupby("run").cakupan_05_FNcv.mean()
            print(f"cakupan {nama}: FNcv {100 * z.mean():.1f}%")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "satu":
        satu(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "ringkas":
        ringkas()
    else:
        sys.exit(__doc__)
