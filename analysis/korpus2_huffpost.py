"""Korpus kedua paper (27 Oktober 2026): HuffPost News Category Dataset v3, kategori POLITICS 2014–2022, periode = tahun.

Protokol sama dengan korpus utama (outputs/korpus_bersih_15konfigurasi_2026-10-20/ dan skrip analisis paper):
- Teks = judul + ringkasan; prapemrosesan clean_text (pipeline/preprocessing.py) dengan stopword
  NLTK bahasa Inggris dan token > 2 huruf; dokumen dengan < 5 token dibuang; duplikat tautan dan teks dibuang.
- Pembagian 80/20 terstratifikasi tahun (split_train_validation) untuk random state 7, 21, 42, 84, 123; K = 15, 20, 25.
- TF-IDF sama (unigram+bigram, sublinear, smooth IDF, L2, min_df 10, max_df 0,60, maks. 8.000 fitur; dihitung dari data latih);
  baris kosong setelah vocabulary dibuang. Temporal TF-IDF dan bobot akar memakai fungsi pipeline asli.
- Varian: S1–S4, beta 0,5/1,5/2, undersampling (US), NMF per periode dua lapis (PP), dan S3*/S4* (proyeksi balik).
- NMF sama dengan build_locked_nmf; evaluasi dan bootstrap (1.000, terstratifikasi tahun, default_rng(0)) sama dengan korpus utama.
- Dampak topik: pencocokan Hungaria terhadap S1 (kosinus < 0,5 = tanpa padanan), tahun puncak = lift prevalensi tertinggi.
Masukan: data/raw/News_Category_Dataset_v3.json (hanya dibaca).
Keluaran: H dan hasil per konfigurasi di outputs/paper_korpus2_huffpost_2026-10-27/ (subfolder hasil/);
ringkasan korpus2_* di folder skrip ini.
Pemakaian (satu konfigurasi per proses, agar ringan dan bisa dilanjutkan; jalankan dengan OMP_NUM_THREADS=1):
  <python 3.10 + scikit-learn 1.5.2 + gensim 4.3.3> -I analysis/korpus2_huffpost.py satu <random_state> <K>
  ... korpus2_huffpost.py cv <random_state> <K>   (c_v varian lain dari H tersimpan, tanpa melatih ulang)
  ... korpus2_huffpost.py fair <random_state> <K> (Fairer-NMF, tahun sebagai kelompok; lihat fairer_nmf.py)
  ... korpus2_huffpost.py ringkas            (menggabungkan semua konfigurasi yang sudah selesai)
  ... korpus2_huffpost.py uji <folder>       (uji cepat r7_k15, bootstrap 50, keluaran ke folder lain)
"""
import json, sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

SRC = Path(ROOT / "pipeline")
sys.path.insert(0, str(SRC))
from preprocessing import clean_text  # noqa: E402
from temporal_nmf_pipeline import (apply_temporal_tfidf, compute_cv_coherence, compute_document_weights,  # noqa: E402
                                   compute_temporal_idf, encode_features_for_coherence, split_train_validation)
from nltk.corpus import stopwords  # noqa: E402

DATA = Path(ROOT / "data/raw/News_Category_Dataset_v3.json")
HOUT = Path(ROOT / "outputs/paper_korpus2_huffpost_2026-10-27")
OUT = Path(__file__).resolve().parent
B = 1000
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE == "uji":
    B, HOUT, OUT = 50, Path(sys.argv[2]), Path(sys.argv[2])
HASIL = HOUT / "hasil"
HASIL.mkdir(parents=True, exist_ok=True)
LOW, DOM = ["2019", "2020", "2021", "2022"], ["2016", "2017"]
BETAS = (0.5, 1.5, 2.0)
warnings.filterwarnings("ignore", category=FutureWarning)


def nmf(k, rs):
    return NMF(n_components=k, init="nndsvda", solver="cd", beta_loss="frobenius", max_iter=400,
               random_state=rs, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0)


def fit(X, k, rs):
    m = nmf(k, rs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning); m.fit(X)
    return m, sum(issubclass(c.category, ConvergenceWarning) for c in caught)


def tvd(X, labels):
    e = np.asarray(X.multiply(X).sum(1)).ravel(); q = pd.Series(e).groupby(labels).sum(); q = q / q.sum()
    return 0.5 * float(np.abs(q - 1 / len(q)).sum())


def siapkan_data():
    rows = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    df = pd.DataFrame([r for r in rows if r["category"] == "POLITICS"])
    info = {"politics": len(df)}
    df["text"] = (df.headline.fillna("") + " " + df.short_description.fillna("")).str.strip()
    df = df[df.text.str.len() > 0]; info["tidak_kosong"] = len(df)
    df = df.drop_duplicates("link"); info["tautan_unik"] = len(df)
    df = df.drop_duplicates("text"); info["teks_unik"] = len(df)
    sw = set(stopwords.words("english"))
    df["proc"] = [" ".join(w for w in clean_text(t).split() if w not in sw and len(w) > 2) for t in df.text]
    df = df[df.proc.str.split().str.len() >= 5].reset_index(drop=True); info["minimal_5_token"] = len(df)
    df["year"] = df.date.str[:4]
    info["per_tahun"] = {k: int(v) for k, v in df.year.value_counts().sort_index().items()}
    info["kata_per_dokumen_median"] = float(df.text.str.split().str.len().median())
    return df, info


def satu(df, rs, k):
    name = f"r{rs}_k{k}"; t0 = time.time()
    tr, va = split_train_validation(df, 0.20, rs, stratify_column="year")
    vec = TfidfVectorizer(max_features=8000, min_df=10, max_df=0.60, ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sublinear_tf=True, smooth_idf=True, norm="l2",
                          dtype=np.float64)
    Xtr = vec.fit_transform(tr.proc).tocsr(); Xva = vec.transform(va.proc).tocsr()
    ktr = Xtr.getnnz(axis=1) > 0; kva = Xva.getnnz(axis=1) > 0
    tr, Xtr, va, Xva = tr[ktr], Xtr[ktr], va[kva], Xva[kva]
    tm, vm = tr.year.to_numpy(), va.year.to_numpy(); periods = sorted(set(tm))
    pres = CountVectorizer(vocabulary=vec.vocabulary_, binary=True, ngram_range=(1, 2), token_pattern=vec.token_pattern,
                           lowercase=True).fit_transform(tr.proc).tocsr()
    tidf = compute_temporal_idf(pres, tm); Xtt = apply_temporal_tfidf(Xtr, tidf)
    w, audit = compute_document_weights(tm)
    counts = audit.set_index("year_month").n_documents; nbar = float(audit.n_bar.iloc[0])
    ap = audit.set_index("year_month").document_weight
    analyzer, vocab = vec.build_analyzer(), vec.vocabulary_
    coh_corpus = encode_features_for_coherence([f for f in analyzer(t) if f in vocab] for t in tr.proc)
    feats = vec.get_feature_names_out()
    Xall = sparse.vstack([Xtr, Xva]).tocsr(); am = np.concatenate([tm, vm])
    info = {"fitur": int(Xtr.shape[1]), "latih": int(Xtr.shape[0]), "validasi": int(Xva.shape[0]),
            "baris_kosong_latih": int((~ktr).sum()), "baris_kosong_validasi": int((~kva).sum()),
            "latih_per_tahun": {p: int(counts[p]) for p in periods},
            "validasi_per_tahun": {p: int((vm == p).sum()) for p in periods}, "a_p": {p: float(ap[p]) for p in periods}}
    A = sparse.diags(w)
    sets = {"S1": Xtr, "S2": (A @ Xtr).tocsr(), "S3": Xtt, "S4": (A @ Xtt).tocsr()}
    for b in BETAS:
        sets[f"b{b:g}"] = (sparse.diags((nbar / counts)[tm].to_numpy() ** (b / 2)) @ Xtr).tocsr()
    rng_us = np.random.default_rng(rs); nmin = int(counts.min())
    keep = np.sort(np.concatenate([rng_us.choice(np.flatnonzero(tm == p), nmin, replace=False) for p in periods]))
    sets["US"] = Xtr[keep]

    H, meta = {}, {}
    for key, Xs in sets.items():
        m, wn = fit(Xs, k, rs); H[key] = m.components_
        meta[key] = {"n_iter": m.n_iter_, "warn": wn, "tvd": tvd(Xs, tm[keep] if key == "US" else tm)}
        if key == "S1":
            proto = m
    lapis1, wpp, it1 = [], 0, []
    for p in periods:
        m, wn = fit(Xtr[tm == p], k, rs); wpp += wn; it1.append(m.n_iter_)
        lapis1.append(m.components_ / np.linalg.norm(m.components_, axis=1, keepdims=True))
    m, wn = fit(np.vstack(lapis1), k, rs); H["PP"] = m.components_
    meta["PP"] = {"n_iter": max(max(it1), m.n_iter_), "warn": wpp + wn, "tvd": np.nan}
    for key in ("S3", "S4"):
        Hb = H[key] / tidf; Hb *= (np.linalg.norm(H[key], axis=1) / np.linalg.norm(Hb, axis=1))[:, None]
        H[key + "*"] = Hb; meta[key + "*"] = meta[key]
    np.savez_compressed(HOUT / f"{name}.npz", **{key.replace("*", "_star"): v for key, v in H.items()})

    xsq = np.asarray(Xva.multiply(Xva).sum(1)).ravel(); res = {}
    for key, Hk in H.items():
        proto.components_ = Hk; Wv = proto.transform(Xva)
        res[key] = xsq + np.einsum("ij,jk,ik->i", Wv, Hk @ Hk.T, Wv) - 2 * np.sum(Wv * np.asarray(Xva @ Hk.T), 1)
    idx = {p: np.flatnonzero(vm == p) for p in periods}
    rre = lambda c, ix: np.sqrt(res[c][ix].sum() / xsq[ix].sum())
    per_rows = [{"run": name, "random_state": rs, "K": k, "tahun": p, "n_latih": int(counts[p]), "a_p": float(ap[p]),
                 "n_validasi": len(idx[p]), **{f"rre_{c}": rre(c, idx[p]) for c in res}} for p in periods]
    rng = np.random.default_rng(0); boots = {c: [] for c in res}
    for _ in range(B):
        ix = {p: rng.choice(idx[p], len(idx[p])) for p in periods}
        for c in res: boots[c].append(np.mean([rre(c, ix[p]) for p in periods]))
    boots = {c: np.array(v) for c, v in boots.items()}
    allix = np.arange(len(vm))
    lift, peak = {}, {}
    for key in ("S1", "S2", "S3", "S4", "b0.5", "b1.5", "b2", "US", "PP"):
        proto.components_ = H[key]; Wa = proto.transform(Xall); s = Wa.sum(1, keepdims=True); s[s == 0] = np.nan
        prev = pd.DataFrame(Wa / s).groupby(am).mean(); lift[key] = prev / prev.mean(axis=0); peak[key] = lift[key].idxmax(axis=0).to_numpy()
    unit = {key: H[key] / np.linalg.norm(H[key], axis=1, keepdims=True) for key in lift}
    sum_rows, imp_rows = [], []
    for c in res:
        pm = np.array([rre(c, idx[p]) for p in periods]); top = np.argsort(-H[c], axis=1)[:, :15]
        row = {"run": name, "random_state": rs, "K": k, "varian": c, **meta[c],
               "makro": pm.mean(), "makro_kuadrat": float(np.mean(pm ** 2)), "mikro": rre(c, allix),
               "rendah": float(np.mean([rre(c, idx[p]) for p in LOW])), "dominan": float(np.mean([rre(c, idx[p]) for p in DOM])),
               "terburuk": float(pm.max()), "td": len(np.unique(top)) / top.size,
               "d_makro_vs_S1_lo": np.percentile(boots[c] - boots["S1"], 2.5), "d_makro_vs_S1_hi": np.percentile(boots[c] - boots["S1"], 97.5),
               "d_makro_vs_S2_lo": np.percentile(boots[c] - boots["S2"], 2.5), "d_makro_vs_S2_hi": np.percentile(boots[c] - boots["S2"], 97.5)}
        if c in ("S1", "S2"):
            row["cv"] = compute_cv_coherence([[str(feats[j]) for j in t] for t in top], coh_corpus).coherence_cv
        if c in lift:
            row.update({"puncak_dominan": int(np.isin(peak[c], DOM).sum()), "puncak_rendah": int(np.isin(peak[c], LOW).sum()),
                        "lift2_dominan": int((lift[c].loc[DOM] >= 2).any(axis=0).sum()),
                        "lift2_rendah": int((lift[c].loc[LOW] >= 2).any(axis=0).sum())})
            if c != "S1":
                C = unit["S1"] @ unit[c].T; r, cc = linear_sum_assignment(-C); sim = C[r, cc]; bad = sim < 0.5
                row.update({"tanpa_padanan": int(bad.sum()), "kosinus_median": float(np.median(sim))})
                for i, j in zip(r[bad], cc[bad]):
                    imp_rows.append({"run": name, "varian": c, "puncak_S1": peak["S1"][i], "puncak_varian": peak[c][j],
                                     "fitur_S1": ", ".join(feats[np.argsort(-H["S1"][i])[:10]]),
                                     "fitur_varian": ", ".join(feats[np.argsort(-H[c][j])[:10]])})
        sum_rows.append(row)
    pd.DataFrame(per_rows).to_csv(HASIL / f"{name}_per_periode.csv", index=False)
    pd.DataFrame(imp_rows).to_csv(HASIL / f"{name}_topik_tanpa_padanan.csv", index=False)
    json.dump(info, open(HASIL / f"{name}_info.json", "w"), ensure_ascii=False, indent=1)
    pd.DataFrame(sum_rows).to_csv(HASIL / f"{name}_ringkasan.csv", index=False)  # ditulis terakhir: tanda konfigurasi selesai
    print(f"selesai: {name} ({time.time() - t0:.0f} dtk)", flush=True)


def cv_tambahan(df, rs, k):
    """c_v varian lain dari H tersimpan (tanpa melatih ulang); S1 dihitung ulang sebagai pemeriksaan."""
    name = f"r{rs}_k{k}"; t0 = time.time()
    tr, _ = split_train_validation(df, 0.20, rs, stratify_column="year")
    vec = TfidfVectorizer(max_features=8000, min_df=10, max_df=0.60, ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sublinear_tf=True, smooth_idf=True, norm="l2",
                          dtype=np.float64)
    Xtr = vec.fit_transform(tr.proc).tocsr(); tr = tr[Xtr.getnnz(axis=1) > 0]
    analyzer, vocab, feats = vec.build_analyzer(), vec.vocabulary_, vec.get_feature_names_out()
    coh_corpus = encode_features_for_coherence([f for f in analyzer(t) if f in vocab] for t in tr.proc)
    H = np.load(HOUT / f"{name}.npz")
    rows = []
    for key in ("S1", "b0.5", "b1.5", "b2", "US", "PP"):
        top = np.argsort(-H[key], axis=1)[:, :15]
        rows.append({"run": name, "varian": key,
                     "cv": compute_cv_coherence([[str(feats[j]) for j in t] for t in top], coh_corpus).coherence_cv})
    pd.DataFrame(rows).to_csv(HASIL / f"{name}_cv_tambahan.csv", index=False)
    print(f"selesai cv: {name} ({time.time() - t0:.0f} dtk)", flush=True)


def fair(df, rs, k):
    """Fairer-NMF (tahun sebagai kelompok) lewat fairer_nmf.py; pengaturan sama dengan eksperimen_fairer_nmf.py (korpus
    pertama): galat acuan E_l dari NMF konvergen (cd, nndsvda) per kelompok, 1.000 iterasi MU penuh, pembanding NMF-MU
    biasa (MU), dan galat berlebih maksimum pada data latih. Evaluasi seperti satu()."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fairer_nmf import fairer_nmf_mu, nmf_mu
    name = f"r{rs}_k{k}"; t0 = time.time()
    tr, va = split_train_validation(df, 0.20, rs, stratify_column="year")
    vec = TfidfVectorizer(max_features=8000, min_df=10, max_df=0.60, ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", sublinear_tf=True, smooth_idf=True, norm="l2",
                          dtype=np.float64)
    Xtr = vec.fit_transform(tr.proc).tocsr(); Xva = vec.transform(va.proc).tocsr()
    ktr = Xtr.getnnz(axis=1) > 0; kva = Xva.getnnz(axis=1) > 0
    tr, Xtr, va, Xva = tr[ktr], Xtr[ktr], va[kva], Xva[kva]
    tm, vm = tr.year.to_numpy(), va.year.to_numpy(); periods = sorted(set(tm))
    analyzer, vocab, feats = vec.build_analyzer(), vec.vocabulary_, vec.get_feature_names_out()
    coh_corpus = encode_features_for_coherence([f for f in analyzer(t) if f in vocab] for t in tr.proc)
    Xall = sparse.vstack([Xtr, Xva]).tocsr(); am = np.concatenate([tm, vm])
    Xg = [Xtr[tm == p] for p in periods]
    E = []
    for G in Xg:   # galat acuan: NMF konvergen pada kelompok sendiri
        m, _ = fit(G, k, rs); Wg = m.transform(G); Hg = m.components_; g2 = float(G.multiply(G).sum())
        E.append(float(np.sqrt(max(g2 - 2 * np.sum(np.asarray(G @ Hg.T) * Wg) + np.sum((Wg.T @ Wg) * (Hg @ Hg.T)), 0.0))))
    E = np.array(E)
    _, Hf, errs, _, n_iter = fairer_nmf_mu(Xg, k, np.random.default_rng(rs), max_iter=1000, rel_tol=None, min_errs=E)
    _, Hmu, _ = nmf_mu(Xtr, k, np.random.default_rng(rs), max_iter=1000, rel_tol=None)
    np.savez_compressed(HOUT / f"{name}_fair.npz", H_FN=Hf, H_MU=Hmu, E=E, galat_berlebih_akhir=errs[:, -1], n_iter=n_iter)
    Hs = np.load(HOUT / f"{name}.npz"); H = {"S1": Hs["S1"], "S2": Hs["S2"], "MU": Hmu, "FN": Hf}
    proto = nmf(k, rs).set_params(max_iter=1)                     # model terlatih dengan parameter sama; H diganti
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); proto.fit(Xtr)
    proto.set_params(max_iter=400)

    def resid(Xm, Hk):
        proto.components_ = Hk; Wm = proto.transform(Xm); x = np.asarray(Xm.multiply(Xm).sum(1)).ravel()
        return x + np.einsum("ij,jk,ik->i", Wm, Hk @ Hk.T, Wm) - 2 * np.sum(Wm * np.asarray(Xm @ Hk.T), 1), x

    latih = {}
    for key, Hk in H.items():   # galat berlebih maksimum pada data latih (tujuan Fairer-NMF)
        r_, x_ = resid(Xtr, Hk)
        latih[key] = float(max((np.sqrt(r_[tm == p].sum()) - e) / np.sqrt(x_[tm == p].sum()) for p, e in zip(periods, E)))
    res = {}; xsq = None
    for key, Hk in H.items():
        res[key], xsq = resid(Xva, Hk)
    idx = {p: np.flatnonzero(vm == p) for p in periods}
    rre = lambda c, ix: np.sqrt(res[c][ix].sum() / xsq[ix].sum())
    per = [{"run": name, "tahun": p, **{f"rre_{c}": rre(c, idx[p]) for c in res}} for p in periods]
    rng = np.random.default_rng(0); boots = {c: [] for c in res}
    for _ in range(B):
        ix = {p: rng.choice(idx[p], len(idx[p])) for p in periods}
        for c in res: boots[c].append(np.mean([rre(c, ix[p]) for p in periods]))
    boots = {c: np.array(v) for c, v in boots.items()}
    allix = np.arange(len(vm)); pm = {c: np.array([rre(c, idx[p]) for p in periods]) for c in res}
    lift, peak = {}, {}
    for key in ("S1", "FN"):
        proto.components_ = H[key]; Wa = proto.transform(Xall); s = Wa.sum(1, keepdims=True); s[s == 0] = np.nan
        prev = pd.DataFrame(Wa / s).groupby(am).mean(); lift[key] = prev / prev.mean(axis=0); peak[key] = lift[key].idxmax(axis=0).to_numpy()
    u1 = H["S1"] / np.linalg.norm(H["S1"], axis=1, keepdims=True); u2 = Hf / np.linalg.norm(Hf, axis=1, keepdims=True)
    C = u1 @ u2.T; r, c = linear_sum_assignment(-C); sim = C[r, c]
    top = np.argsort(-Hf, axis=1)[:, :15]
    row = {"run": name, "random_state": rs, "K": k, "n_iter": n_iter,
           **{f"latih_berlebih_maks_{c}": latih[c] for c in res},
           **{f"makro_{c}": pm[c].mean() for c in res}, **{f"mikro_{c}": rre(c, allix) for c in res},
           **{f"rendah_{c}": float(np.mean([rre(c, idx[p]) for p in LOW])) for c in res},
           **{f"dominan_{c}": float(np.mean([rre(c, idx[p]) for p in DOM])) for c in res},
           **{f"terburuk_{c}": float(pm[c].max()) for c in res},
           **{f"d_makro_{c}_S1_{q}": np.percentile(boots[c] - boots["S1"], v) for c in ("MU", "FN") for q, v in (("lo", 2.5), ("hi", 97.5))},
           **{f"d_makro_{c}_S2_{q}": np.percentile(boots[c] - boots["S2"], v) for c in ("MU", "FN") for q, v in (("lo", 2.5), ("hi", 97.5))},
           "d_makro_FN_MU_lo": np.percentile(boots["FN"] - boots["MU"], 2.5), "d_makro_FN_MU_hi": np.percentile(boots["FN"] - boots["MU"], 97.5),
           "cv_FN": compute_cv_coherence([[str(feats[j]) for j in t] for t in top], coh_corpus).coherence_cv,
           "cv_MU": compute_cv_coherence([[str(feats[j]) for j in t] for t in np.argsort(-Hmu, axis=1)[:, :15]], coh_corpus).coherence_cv,
           "td_FN": len(np.unique(top)) / top.size, "tanpa_padanan": int((sim < 0.5).sum()), "kosinus_median": float(np.median(sim)),
           "puncak_dominan_FN": int(np.isin(peak["FN"], DOM).sum()), "puncak_rendah_FN": int(np.isin(peak["FN"], LOW).sum()),
           "lift2_rendah_FN": int((lift["FN"].loc[LOW] >= 2).any(axis=0).sum())}
    pd.DataFrame(per).to_csv(HASIL / f"{name}_fair_per_periode.csv", index=False)
    pd.DataFrame([row]).to_csv(HASIL / f"{name}_fair.csv", index=False)   # ditulis terakhir: tanda selesai
    print(f"selesai fair: {name} ({time.time() - t0:.0f} dtk)", flush=True)


def ringkas():
    names = [f"r{rs}_k{k}" for rs in (7, 21, 42, 84, 123) for k in (15, 20, 25)]
    done = [n for n in names if (HASIL / f"{n}_ringkasan.csv").exists()]
    print(f"konfigurasi selesai: {len(done)}/15")
    s = pd.concat([pd.read_csv(HASIL / f"{n}_ringkasan.csv") for n in done], ignore_index=True)
    per = pd.concat([pd.read_csv(HASIL / f"{n}_per_periode.csv", dtype={"tahun": str}) for n in done], ignore_index=True)
    imp = [HASIL / f"{n}_topik_tanpa_padanan.csv" for n in done]
    imp = pd.concat([pd.read_csv(f) for f in imp if f.stat().st_size > 1], ignore_index=True)
    info = {n: json.load(open(HASIL / f"{n}_info.json")) for n in done}
    base = s[s.varian == "S1"].set_index("run")
    for col in ("makro", "makro_kuadrat", "mikro", "rendah", "dominan", "terburuk", "td"):
        s[f"d_{col}"] = s[col] - s.run.map(base[col])
    per.to_csv(OUT / "korpus2_per_periode.csv", index=False); s.to_csv(OUT / "korpus2_ringkasan.csv", index=False)
    imp.to_csv(OUT / "korpus2_topik_tanpa_padanan.csv", index=False)
    json.dump(info, open(OUT / "korpus2_info.json", "w"), ensure_ascii=False, indent=1)
    n = len(done)
    pd.set_option("display.width", 260)
    order = ["S2", "S3", "S3*", "S4", "S4*", "b0.5", "b1.5", "b2", "US", "PP"]
    g = s.groupby("varian")
    tab = pd.DataFrame({"d_makro": g.d_makro.mean(), "min": g.d_makro.min(), "maks": g.d_makro.max(),
                        "makro<S1": g.d_makro.apply(lambda x: int((x < 0).sum())), "IK<0": g.d_makro_vs_S1_hi.apply(lambda x: int((x < 0).sum())),
                        "IK>0": g.d_makro_vs_S1_lo.apply(lambda x: int((x > 0).sum())),
                        "IK>0 vs S2": g.d_makro_vs_S2_lo.apply(lambda x: int((x > 0).sum())), "IK<0 vs S2": g.d_makro_vs_S2_hi.apply(lambda x: int((x < 0).sum())),
                        "d_mikro": g.d_mikro.mean(), "mikro>S1": g.d_mikro.apply(lambda x: int((x > 0).sum())),
                        "d_rendah": g.d_rendah.mean(), "rendah<S1": g.d_rendah.apply(lambda x: int((x < 0).sum())), "d_dominan": g.d_dominan.mean(),
                        "d_terburuk": g.d_terburuk.mean(), "iter_maks": g.n_iter.max(), "peringatan": g.warn.sum()}).reindex(order)
    print(f"\n== Rata-rata {n} konfigurasi, selisih terhadap S1"); print(tab.round(4).to_string())
    mk = s.pivot(index="run", columns="varian", values="makro"); mq = s.pivot(index="run", columns="varian", values="makro_kuadrat")
    cur = ["S1", "b0.5", "S2", "b1.5", "b2"]
    print("\nEksponen dengan makro terendah:", mk[cur].idxmin(axis=1).value_counts().to_dict(),
          "| rata-rata kuadrat:", mq[cur].idxmin(axis=1).value_counts().to_dict())
    for v in ("b0.5", "US", "PP", "S4*"):
        print(f"{v} makro > S2: {int((mk[v] > mk['S2']).sum())}/{n} (rata-rata {float((mk[v] - mk['S2']).mean()):+.4f})")
    d = per.assign(d=per.rre_S2 - per.rre_S1)
    print(f"\nE3: periode berbobot > 1 turun {int(((d.a_p > 1) & (d.d < 0)).sum())}/{int((d.a_p > 1).sum())}; "
          f"berbobot < 1 naik {int(((d.a_p < 1) & (d.d > 0)).sum())}/{int((d.a_p < 1).sum())}; "
          f"korelasi Pearson log a_p^2 vs d = {np.corrcoef(np.log(d.a_p ** 2), d.d)[0, 1]:.2f}")
    print(d.groupby("tahun").agg(a_p=("a_p", "first"), d_rata=("d", "mean"), d_min=("d", "min"), d_maks=("d", "max"),
                                 turun=("d", lambda x: int((x < 0).sum()))).round(4).to_string())
    cv2 = s[s.varian == "S2"].set_index("run").cv
    print(f"\n== Kualitas topik: cv S2 - S1 rata-rata {float((cv2 - base.cv).mean()):+.4f} (S2 > S1 pada {int((cv2 > base.cv).sum())}/{n}); "
          f"rentang cv S1 {base.cv.min():.4f}-{base.cv.max():.4f}; TD S2 - S1 rata-rata {float(g.d_td.mean()['S2']):+.4f}")
    print(f"\n== Dampak topik (jumlah atas {n} konfigurasi)")
    imp_s = s[s.varian.isin(["S1", "S2", "S3", "S4", "b0.5", "b1.5", "b2", "US", "PP"])]
    print(imp_s.groupby("varian")[["tanpa_padanan", "puncak_dominan", "puncak_rendah", "lift2_dominan", "lift2_rendah"]].sum()
          .reindex(["S1", "S2", "S3", "S4", "b0.5", "b1.5", "b2", "US", "PP"]).to_string())
    print("kosinus median:", s[s.varian != "S1"].groupby("varian").kosinus_median.median().round(3).to_dict())
    print("fitur per konfigurasi:", {k: v["fitur"] for k, v in info.items()})
    cvf = [HASIL / f"{n}_cv_tambahan.csv" for n in done if (HASIL / f"{n}_cv_tambahan.csv").exists()]
    if cvf:
        c = pd.concat([pd.read_csv(f) for f in cvf], ignore_index=True); c.to_csv(OUT / "korpus2_cv_tambahan.csv", index=False)
        cp = c.pivot(index="run", columns="varian", values="cv")
        print(f"\n== c_v varian lain ({len(cp)} konfigurasi); pemeriksaan S1 dihitung ulang vs ringkasan: "
              f"selisih maks {float((cp.S1 - base.cv.reindex(cp.index)).abs().max()):.1e}")
        for v in ("b0.5", "b1.5", "b2", "US", "PP"):
            dv = cp[v] - cp.S1
            print(f"{v}: c_v - S1 rata-rata {dv.mean():+.4f} (lebih rendah pada {int((dv < 0).sum())}/{len(dv)})")
    ff = [HASIL / f"{n}_fair.csv" for n in done if (HASIL / f"{n}_fair.csv").exists()]
    if ff:
        f = pd.concat([pd.read_csv(x) for x in ff], ignore_index=True); f.to_csv(OUT / "korpus2_fairer_nmf.csv", index=False)
        pd.concat([pd.read_csv(HASIL / f"{r}_fair_per_periode.csv", dtype={"tahun": str}) for r in f.run], ignore_index=True) \
            .to_csv(OUT / "korpus2_fairer_nmf_per_periode.csv", index=False)
        m = len(f); chk = float((f.set_index("run").makro_S1 - base.makro.reindex(f.run)).abs().max())
        print(f"\n== Fairer-NMF (FN), {m} konfigurasi; pemeriksaan makro S1 vs ringkasan: selisih maks {chk:.1e}")
        for c in ("MU", "FN"):
            for col in ("makro", "mikro", "rendah", "dominan", "terburuk"):
                d1 = f[f"{col}_{c}"] - f[f"{col}_S1"]; d2 = f[f"{col}_{c}"] - f[f"{col}_S2"]
                print(f"{c} {col:9s}: - S1 rata-rata {d1.mean():+.4f} (rentang {d1.min():+.4f} s.d. {d1.max():+.4f}), < S1 pada {int((d1 < 0).sum())}/{m}; "
                      f"- S2 rata-rata {d2.mean():+.4f}, < S2 pada {int((d2 < 0).sum())}/{m}")
            print(f"{c}: IK makro - S1 < 0 pada {int((f[f'd_makro_{c}_S1_hi'] < 0).sum())}/{m}, > 0 pada {int((f[f'd_makro_{c}_S1_lo'] > 0).sum())}/{m}; "
                  f"- S2 > 0 pada {int((f[f'd_makro_{c}_S2_lo'] > 0).sum())}/{m}")
            dcv = f.set_index("run")[f"cv_{c}"] - base.cv.reindex(f.run)
            print(f"{c}: c_v - S1 rata-rata {dcv.mean():+.4f} (lebih rendah pada {int((dcv < 0).sum())}/{m})")
        d = f.makro_FN - f.makro_MU; dt = f.terburuk_FN - f.terburuk_MU
        print(f"FN - MU: makro rata-rata {d.mean():+.4f} (< MU pada {int((d < 0).sum())}/{m}; IK < 0 {int((f.d_makro_FN_MU_hi < 0).sum())}, "
              f"> 0 {int((f.d_makro_FN_MU_lo > 0).sum())}); terburuk rata-rata {dt.mean():+.4f} (< MU pada {int((dt < 0).sum())}/{m})")
        lc = [f"latih_berlebih_maks_{c}" for c in ("S1", "S2", "MU", "FN")]
        print("galat berlebih maks. data latih (rata-rata):", {c: round(float(f[c].mean()), 4) for c in lc},
              "| FN terendah pada", int((f[lc].idxmin(axis=1) == "latih_berlebih_maks_FN").sum()), f"/{m}; TD FN {f.td_FN.mean():.4f}")
        print(f"topik tanpa padanan {int(f.tanpa_padanan.sum())} ({f.tanpa_padanan.min()}–{f.tanpa_padanan.max()}), kosinus median "
              f"{f.kosinus_median.median():.3f}; puncak 2016–2017 {int(f.puncak_dominan_FN.sum())}, puncak 2019–2022 {int(f.puncak_rendah_FN.sum())}, "
              f"lift2 2019–2022 {int(f.lift2_rendah_FN.sum())}")


if MODE == "ringkas":
    ringkas()
elif MODE == "cv":
    df, _ = siapkan_data()
    cv_tambahan(df, int(sys.argv[2]), int(sys.argv[3]))
elif MODE == "fair":
    df, _ = siapkan_data()
    fair(df, int(sys.argv[2]), int(sys.argv[3]))
elif MODE in ("satu", "uji"):
    df, info = siapkan_data()
    json.dump(info, open(HASIL / "data_info.json", "w"), ensure_ascii=False, indent=1)
    rs, k = (7, 15) if MODE == "uji" else (int(sys.argv[2]), int(sys.argv[3]))
    satu(df, rs, k)
    if MODE == "uji":
        ringkas()
else:
    sys.exit(__doc__)
