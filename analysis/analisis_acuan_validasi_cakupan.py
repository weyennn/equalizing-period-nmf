"""Galat acuan Fairer-NMF pada data validasi dan cakupan topik per periode (27 Oktober 2026), kedua korpus.

Untuk setiap periode l, NMF K topik dengan pengaturan galat acuan Fairer-NMF (nndsvda, cd, 400 iterasi, random_state
konfigurasi) dilatih ulang pada data latih periode itu saja; E_l dicek sama dengan E tersimpan di *_fair.npz.
1. Acuan validasi: RRE dokumen validasi periode l terhadap basis periode itu sendiri, H_l (W lewat transform, H tetap).
   Dengan acuan ini, kriteria Fairer-NMF pada data validasi = max_l (RRE_val(l) - acuan_val(l)), dibandingkan dengan
   acuan data latih E_l / ||X_l|| (galat_acuan_fn_per_periode.csv).
2. Cakupan topik: untuk setiap topik periode (baris H_l), kosinus tertinggi terhadap topik basis global setiap varian;
   dilaporkan rata-ratanya dan bagian topik periode dengan kosinus >= 0,5 (ambang yang sama dengan pencocokan topik).
Masukan (hanya dibaca): korpus 1 outputs/korpus_bersih_15konfigurasi_2026-10-20/ (matriks, ledger, model S1/S2),
paper_kurva_bobot_undersampling_2026-10-26/, paper_nmf_per_periode_2026-10-27/, paper_fairer_nmf_2026-10-27/;
korpus 2 dibangun ulang seperti korpus2_huffpost.py (pembagian dan TF-IDF sama), H dari paper_korpus2_huffpost_2026-10-27/.
Keluaran: per konfigurasi di outputs/paper_acuan_validasi_cakupan_2026-10-27/; ringkasan acuan_cakupan_* di folder ini.
Pemakaian (OMP_NUM_THREADS=1, satu konfigurasi per proses):
  <python 3.10 + scikit-learn 1.5.2> -I analysis/analisis_acuan_validasi_cakupan.py satu <korpus 1|2> <rs> <K>
  ... analisis_acuan_validasi_cakupan.py ringkas
"""
import json, sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import joblib, numpy as np, pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF

OUT = Path(__file__).resolve().parent
T6 = Path(ROOT / "outputs")
SRC1 = T6 / "korpus_bersih_15konfigurasi_2026-10-20"
HOUT = T6 / "paper_acuan_validasi_cakupan_2026-10-27"; HOUT.mkdir(parents=True, exist_ok=True)
VARIAN = ["S1", "S2", "b0.5", "b1.5", "b2", "US", "PP", "MU", "FN"]
RS, KS = (7, 21, 42, 84, 123), (15, 20, 25)
warnings.filterwarnings("ignore", category=FutureWarning)


def nmf(k, rs, it=400):
    return NMF(n_components=k, init="nndsvda", solver="cd", beta_loss="frobenius", max_iter=it, random_state=rs,
               alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0)


def data_korpus1(rs, k):
    run = SRC1 / f"r{rs}_k{k}"
    X = sparse.load_npz(run / "X_base_train.npz").tocsr(); Xv = sparse.load_npz(run / "X_base_valid.npz").tocsr()
    led = pd.read_csv(run / "split_ledger.csv")
    tm = led[led.split == "train"].sort_values("split_row").year_month.astype(str).to_numpy()
    vm = led[led.split == "validation"].sort_values("split_row").year_month.astype(str).to_numpy()
    kb = T6 / "paper_kurva_bobot_undersampling_2026-10-26"; name = f"r{rs}_k{k}"
    fz = np.load(T6 / "paper_fairer_nmf_2026-10-27" / f"{name}_fair.npz")
    H = {"S1": joblib.load(run / "S1_baseline" / "nmf_model.joblib").components_,
         "S2": joblib.load(run / "S2_document_weighting" / "nmf_model.joblib").components_,
         **{f"b{b}": np.load(kb / f"{name}_b{b}.npz")["H"] for b in ("0.5", "1.5", "2")},
         "US": np.load(kb / f"{name}_us.npz")["H"],
         "PP": np.load(T6 / "paper_nmf_per_periode_2026-10-27" / f"{name}_pp.npz")["H"],
         "MU": fz["H_MU"], "FN": fz["H_FN"]}
    return X, Xv, tm, vm, H, fz["E"]


def data_korpus2(rs, k):
    # sama dengan siapkan_data() dan awal satu()/fair() di korpus2_huffpost.py (modul itu tidak bisa diimpor: ada blok utama)
    sys.path.insert(0, str(ROOT / "pipeline"))
    from preprocessing import clean_text
    from temporal_nmf_pipeline import split_train_validation
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
    h2 = T6 / "paper_korpus2_huffpost_2026-10-27"; name = f"r{rs}_k{k}"
    hs = np.load(h2 / f"{name}.npz"); fz = np.load(h2 / f"{name}_fair.npz")
    H = {**{v: hs[v] for v in VARIAN[:7]}, "MU": fz["H_MU"], "FN": fz["H_FN"]}
    return X, Xv, tr.year.to_numpy(), va.year.to_numpy(), H, fz["E"]


def satu(korpus, rs, k):
    name = f"k{korpus}_r{rs}_k{k}"; t0 = time.time()
    X, Xv, tm, vm, H, E_simpan = (data_korpus1 if korpus == 1 else data_korpus2)(rs, k)
    assert all(h.shape == (k, X.shape[1]) for h in H.values())
    proto = nmf(k, rs, it=1)                         # model terlatih dengan parameter sama; H diganti (seperti korpus2_huffpost.py)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); proto.fit(X)
    proto.set_params(max_iter=400)
    unit ={v: h / np.linalg.norm(h, axis=1, keepdims=True) for v, h in H.items()}
    rows = []
    for p, e_simpan in zip(sorted(set(tm)), E_simpan):
        G, Gv = X[tm == p], Xv[vm == p]
        m = nmf(k, rs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); W = m.fit_transform(G)
        Hp = m.components_; g2 = float(G.multiply(G).sum())
        e = float(np.sqrt(max(g2 - 2 * np.sum(np.asarray(G @ Hp.T) * W) + np.sum((W.T @ W) * (Hp @ Hp.T)), 0.0)))
        proto.components_ = Hp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); Wv = proto.transform(Gv)
        v2 = float(Gv.multiply(Gv).sum())
        rv = v2 - 2 * np.sum(np.asarray(Gv @ Hp.T) * Wv) + np.sum((Wv.T @ Wv) * (Hp @ Hp.T))
        up = Hp / np.maximum(np.linalg.norm(Hp, axis=1, keepdims=True), 1e-300)
        row = {"korpus": korpus, "run": f"r{rs}_k{k}", "K": k, "periode": p, "n_latih": G.shape[0], "n_val": Gv.shape[0],
               "E": e, "selisih_E_tersimpan": abs(e - e_simpan), "acuan_latih": e / np.sqrt(g2),
               "acuan_val": float(np.sqrt(max(rv, 0.0) / v2)), "topik_periode_kosong": int((np.linalg.norm(Hp, axis=1) == 0).sum())}
        for v, u in unit.items():
            best = (up @ u.T).max(axis=1)
            row[f"cakupan_rata_{v}"] = float(best.mean()); row[f"cakupan_05_{v}"] = float((best >= 0.5).mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(HOUT / f"{name}.csv", index=False)   # ditulis terakhir: tanda selesai
    print(f"selesai: {name} ({time.time() - t0:.0f} dtk); selisih E maks {max(r['selisih_E_tersimpan'] for r in rows):.1e}", flush=True)


def ringkas():
    f = [HOUT / f"k{c}_r{rs}_k{k}.csv" for c in (1, 2) for rs in RS for k in KS]
    d = pd.concat([pd.read_csv(x, dtype={"periode": str}) for x in f if x.exists()], ignore_index=True)
    d.to_csv(OUT / "acuan_cakupan_per_periode.csv", index=False)
    # galat validasi per periode setiap varian
    a = pd.read_csv(OUT / "fairer_nmf_per_bulan.csv", dtype={"bulan": str})
    b = pd.read_csv(OUT / "kurva_bobot_us_per_bulan.csv", dtype={"bulan": str}).rename(columns={"rre_us": "rre_US"})
    c = pd.read_csv(OUT / "nmf_per_periode_per_bulan.csv", dtype={"bulan": str})
    r1 = a.merge(b[["run", "bulan", "rre_b0.5", "rre_b1.5", "rre_b2", "rre_US"]], on=["run", "bulan"]) \
          .merge(c[["run", "bulan", "rre_PP"]], on=["run", "bulan"]).rename(columns={"bulan": "periode"}).assign(korpus=1)
    k2 = pd.read_csv(OUT / "korpus2_per_periode.csv", dtype={"tahun": str})
    f2 = pd.read_csv(OUT / "korpus2_fairer_nmf_per_periode.csv", dtype={"tahun": str})
    r2 = k2.merge(f2[["run", "tahun", "rre_MU", "rre_FN"]], on=["run", "tahun"]).rename(columns={"tahun": "periode"}).assign(korpus=2)
    rre = pd.concat([r1[["korpus", "run", "periode"] + [f"rre_{v}" for v in VARIAN]],
                     r2[["korpus", "run", "periode"] + [f"rre_{v}" for v in VARIAN]]], ignore_index=True)
    d = d.merge(rre, on=["korpus", "run", "periode"], how="left"); assert d[[f"rre_{v}" for v in VARIAN]].notna().all().all()
    kecil = {1: ["2024-09", "2024-10", "2024-11", "2024-12"], 2: ["2019", "2020", "2021", "2022"]}
    besar = {1: ["2024-08"], 2: ["2016", "2017"]}
    for korpus in (1, 2):
        x = d[d.korpus == korpus]; m = x.run.nunique()
        print(f"\n== korpus {korpus}, {m} konfigurasi; selisih E maks {x.selisih_E_tersimpan.max():.1e}; topik periode kosong {int(x.topik_periode_kosong.sum())}")
        print(x.groupby("periode").agg(n_latih=("n_latih", "first"), acuan_latih=("acuan_latih", "mean"), acuan_val=("acuan_val", "mean"),
                                       rre_S1=("rre_S1", "mean")).round(4).to_string())
        for nama, acuan in (("acuan latih", "acuan_latih"), ("acuan validasi", "acuan_val")):
            ex = {v: x[f"rre_{v}"] - x[acuan] for v in VARIAN}
            mx = pd.DataFrame({v: ex[v].groupby(x.run).max() for v in VARIAN})
            arg = {v: x.periode.loc[ex[v].groupby(x.run).idxmax()].value_counts().to_dict() for v in ("S1", "S2", "FN")}
            print(f"kriteria FN pada validasi ({nama}), rata-rata:", {v: round(float(mx[v].mean()), 4) for v in VARIAN})
            print("   terendah:", mx.idxmin(axis=1).value_counts().to_dict(), "| FN < S1", int((mx.FN < mx.S1).sum()), "FN < S2", int((mx.FN < mx.S2).sum()),
                  "FN < b1.5", int((mx.FN < mx["b1.5"]).sum()), "FN < b2", int((mx.FN < mx.b2).sum()), "| periode maks:", arg)
        for ukur in ("cakupan_rata", "cakupan_05"):
            for nama, ps in (("kecil", kecil[korpus]), ("besar", besar[korpus]), ("semua", sorted(x.periode.unique()))):
                y = x[x.periode.isin(ps)].groupby("run")[[f"{ukur}_{v}" for v in VARIAN]].mean()
                dd = {v: f"{float((y[f'{ukur}_{v}'] - y[f'{ukur}_S1']).mean()):+.4f} ({int((y[f'{ukur}_{v}'] > y[f'{ukur}_S1']).sum())})" for v in VARIAN[1:]}
                print(f"{ukur} {nama:5s}: S1 {float(y[f'{ukur}_S1'].mean()):.4f}; - S1 (lebih tinggi pada):", dd)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "satu":
        satu(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "ringkas":
        ringkas()
    else:
        sys.exit(__doc__)
