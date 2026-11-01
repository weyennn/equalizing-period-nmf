"""c_v dan keragaman topik varian lain korpus pertama (27 Oktober 2026): beta 0,5/1,5/2, undersampling (US), NMF per periode
(PP), NMF-MU (MU), dan Fairer-NMF (FN), dari H tersimpan tanpa melatih ulang.

Protokol sama dengan S1–S4 (08_run_weighting_experiments.py): teks latih = preprocessed_text baris latih (urutan split_row),
korpus koherensi = build_coherence_corpus (token analyzer vectorizer latih yang ada di vocabulary, bigram satu token),
15 fitur teratas per topik lewat top_words, c_v Gensim satu model per varian (processes=1). Pemeriksaan: matriks latih
dibangun ulang dari teks harus sama dengan X_base_train.npz, dan c_v serta keragaman S1 dihitung ulang harus sama dengan
S1_S4_metrics_summary.csv.
Masukan (hanya dibaca): outputs/korpus_bersih_15konfigurasi_2026-10-20/ (dataset, ledger, vectorizer, model S1),
paper_kurva_bobot_undersampling_2026-10-26/, paper_nmf_per_periode_2026-10-27/, paper_fairer_nmf_2026-10-27/.
Keluaran: outputs/paper_cv_korpus1_2026-10-27/r<rs>_k<K>.csv; ringkasan cv_korpus1_varian.csv di folder ini.
Pemakaian (OMP_NUM_THREADS=1, satu konfigurasi per proses):
  <python 3.10 + scikit-learn 1.5.2 + gensim 4.3.3> -I analysis/cv_korpus1_varian.py satu <rs> <K>
  ... cv_korpus1_varian.py cepat <rs> <K>   (hasil sama, korpus dipindai sekali untuk semua varian; lihat hitung_jendela)
  ... cv_korpus1_varian.py uji <rs> <K>     (cara cepat dibandingkan dengan CSV cara lama, tanpa menimpa)
  ... cv_korpus1_varian.py ringkas
"""
import importlib.util, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
from types import SimpleNamespace
import joblib, numpy as np, pandas as pd
from scipy import sparse

SRC = Path(ROOT / "pipeline"); sys.path.insert(0, str(SRC))
T6 = Path(ROOT / "outputs")
SRC1 = T6 / "korpus_bersih_15konfigurasi_2026-10-20"
HOUT = T6 / "paper_cv_korpus1_2026-10-27"; HOUT.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).resolve().parent
VARIAN = ["b0.5", "b1.5", "b2", "US", "PP", "MU", "FN"]
RS, KS = (7, 21, 42, 84, 123), (15, 20, 25)


def runner():
    spec = importlib.util.spec_from_file_location("runner", SRC / "08_run_weighting_experiments.py")
    m = importlib.util.module_from_spec(spec); sys.modules["runner"] = m; spec.loader.exec_module(m)
    return m


def muat(rs, k):
    name = f"r{rs}_k{k}"; run = SRC1 / name; m = runner()
    ds = Path(json.load(open(run / "run_manifest.json"))["dataset"])
    df = m.build_preprocessed_frame(m.load_dataset(ds))
    led = pd.read_csv(run / "split_ledger.csv")
    tr = led[led.split == "train"].sort_values("split_row")
    train_texts = df.loc[tr.model_row.to_numpy(), "preprocessed_text"].tolist()
    vec = joblib.load(run / "tfidf_vectorizer_train_only.joblib")
    X = sparse.load_npz(run / "X_base_train.npz").tocsr(); Xr = vec.transform(train_texts).tocsr()
    assert X.shape == Xr.shape and abs(X - Xr).max() < 1e-12, "teks latih tidak cocok dengan X_base_train"
    coh = m.build_coherence_corpus(vec, train_texts); feats = vec.get_feature_names_out()
    kb = T6 / "paper_kurva_bobot_undersampling_2026-10-26"; fz = np.load(T6 / "paper_fairer_nmf_2026-10-27" / f"{name}_fair.npz")
    H = {"S1": joblib.load(run / "S1_baseline" / "nmf_model.joblib").components_,
         **{f"b{b}": np.load(kb / f"{name}_b{b}.npz")["H"] for b in ("0.5", "1.5", "2")},
         "US": np.load(kb / f"{name}_us.npz")["H"],
         "PP": np.load(T6 / "paper_nmf_per_periode_2026-10-27" / f"{name}_pp.npz")["H"],
         "MU": fz["H_MU"], "FN": fz["H_FN"]}
    topik = {key: m.top_words(SimpleNamespace(components_=Hk), feats, m.TOP_N_WORDS) for key, Hk in H.items()}
    return name, run, m, coh, topik


def simpan(name, run, rs, k, rows, t0, uji=False):
    ref = pd.read_csv(run / "S1_S4_metrics_summary.csv").set_index("scenario").loc["S1_baseline"]
    rows[0]["selisih_cv_tersimpan"] = abs(rows[0]["cv"] - ref.coherence_cv)
    rows[0]["selisih_td_tersimpan"] = abs(rows[0]["td"] - ref.topic_diversity_top15)
    if uji:   # bandingkan dengan hasil cara lama tanpa menimpa
        lama = pd.read_csv(HOUT / f"{name}.csv").set_index("varian")
        baru = pd.DataFrame(rows).set_index("varian")
        print(f"uji {name}: selisih c_v maks {(baru.cv - lama.cv).abs().max():.1e}, TD maks {(baru.td - lama.td).abs().max():.1e}")
        return
    pd.DataFrame(rows).to_csv(HOUT / f"{name}.csv", index=False)   # ditulis terakhir: tanda selesai
    print(f"selesai: {name} ({time.time() - t0:.0f} dtk); S1 c_v selisih {rows[0]['selisih_cv_tersimpan']:.1e}", flush=True)


def satu(rs, k):
    t0 = time.time(); name, run, m, coh, topik = muat(rs, k); rows = []
    for key, topics in topik.items():
        c = m.compute_cv_coherence(topics, coh, processes=1)
        rows.append({"run": name, "random_state": rs, "K": k, "varian": key, "cv": c.coherence_cv,
                     "cakupan_kamus": c.coverage, "td": m.topic_diversity(topics)})
        print(f"  {key}: c_v {c.coherence_cv:.4f} ({time.time() - t0:.0f} dtk)", flush=True)
    simpan(name, run, rs, k, rows, t0)


W = 110   # jendela geser c_v Gensim


def hitung_jendela(coh, kata):
    """Kemunculan bersama per jendela untuk kata `kata` (token), dengan satu kali pindai korpus.

    Meniru WordOccurrenceAccumulator Gensim 4.3.3: setiap dokumen memberi max(L - 110 + 1, 1) jendela (dokumen yang lebih
    pendek dari jendela atau kosong tetap satu jendela), dan pada setiap geseran kata di tepi kiri dianggap keluar walaupun
    masih muncul di dalam jendela, sebelum kata di tepi kanan masuk (_slide_window). Keanggotaan kata di jendela hanya
    bergantung pada posisi kata itu sendiri, dan jumlah jendela tidak bergantung pada himpunan kata, sehingga hitungan untuk
    gabungan kata teratas semua varian sama dengan hitungan per varian."""
    idx = {t: i for i, t in enumerate(kata)}; C = np.zeros((len(kata), len(kata))); nj = 0
    for doc in coh:
        n = max(len(doc) - W + 1, 1); nj += n
        pw = [(p, idx[t]) for p, t in enumerate(doc) if t in idx]
        if not pw:
            continue
        p, w = np.array(pw).T; ids, w = np.unique(w, return_inverse=True)
        if n == 1:
            B = np.ones((len(ids), 1))
        else:
            E = np.full((len(ids), n), -1, np.int8); E[:, 0] = 0; E[w[p < W], 0] = 1
            s = p + 1 < n; E[w[s], p[s] + 1] = 0          # keluar dari tepi kiri
            s = p >= W; E[w[s], p[s] - W + 1] = 1          # masuk dari tepi kanan
            j = np.where(E >= 0, np.arange(n), 0); np.maximum.accumulate(j, axis=1, out=j)
            B = np.take_along_axis(E, j, axis=1).astype(np.float64)
        C[np.ix_(ids, ids)] += B @ B.T
    return np.rint(C).astype(np.int64), nj


def cepat(rs, k, uji=False):
    """Sama dengan satu(), tetapi korpus dipindai sekali untuk semua varian; c_v tetap dihitung oleh CoherenceModel Gensim
    dari hitungan itu."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
    from gensim.topic_coherence.text_analysis import WordOccurrenceAccumulator
    import temporal_nmf_pipeline as tp
    t0 = time.time(); name, run, m, coh, topik = muat(rs, k)
    dic = Dictionary(coh); enc = {key: tp.encode_topic_features(t) for key, t in topik.items()}
    kata = sorted({f for e in enc.values() for t in e for f in t}); pos = {dic.token2id[t]: i for i, t in enumerate(kata)}
    C, nj = hitung_jendela(coh, kata); print(f"  pindai {len(kata)} kata, {nj} jendela ({time.time() - t0:.0f} dtk)", flush=True)
    rows = []
    for key, e in enc.items():
        cov = tp.coherence_feature_coverage(e, dic.token2id)[2]
        assert np.isclose(cov, 1.0)
        cm = CoherenceModel(topics=e, texts=coh, dictionary=dic, coherence="c_v", processes=1)
        acc = WordOccurrenceAccumulator(unique_ids_from_segments(cm.measure.seg(cm.topics)), dic)
        inv = np.empty(len(acc.id2contiguous), int)
        for wid, c in acc.id2contiguous.items():
            inv[c] = pos[wid]
        Cv = C[np.ix_(inv, inv)].astype(np.uint32)
        acc._occurrences = Cv.diagonal().copy(); acc._co_occurrences = sparse.csr_matrix(Cv); acc.num_docs = nj
        cm._accumulator = acc
        rows.append({"run": name, "random_state": rs, "K": k, "varian": key, "cv": float(cm.get_coherence()),
                     "cakupan_kamus": cov, "td": m.topic_diversity(topik[key])})
    simpan(name, run, rs, k, rows, t0, uji)


def ringkas():
    f = [HOUT / f"r{rs}_k{k}.csv" for rs in RS for k in KS]
    d = pd.concat([pd.read_csv(x) for x in f if x.exists()], ignore_index=True)
    d.to_csv(OUT / "cv_korpus1_varian.csv", index=False)
    cv = d.pivot(index="run", columns="varian", values="cv"); td = d.pivot(index="run", columns="varian", values="td")
    s = pd.concat([pd.read_csv(SRC1 / r / "S1_S4_metrics_summary.csv").assign(run=r) for r in cv.index]) \
          .pivot(index="run", columns="scenario", values="coherence_cv")
    m = len(cv)
    print(f"== {m} konfigurasi; selisih c_v S1 maks {d.selisih_cv_tersimpan.max():.1e}, TD S1 maks {d.selisih_td_tersimpan.max():.1e}")
    print(f"S1 c_v rata-rata {cv.S1.mean():.4f} (rentang {cv.S1.min():.4f}–{cv.S1.max():.4f}); "
          f"S2 - S1 {float((s.S2_document_weighting - s.S1_baseline).mean()):+.4f}")
    for v in VARIAN:
        dv = cv[v] - cv.S1; d2 = cv[v] - s.S2_document_weighting.reindex(cv.index); dt = td[v] - td.S1
        print(f"{v:4s}: c_v - S1 {dv.mean():+.4f} (rentang {dv.min():+.4f} s.d. {dv.max():+.4f}; lebih rendah {int((dv < 0).sum())}/{m}); "
              f"- S2 {d2.mean():+.4f} (lebih rendah {int((d2 < 0).sum())}/{m}); TD - S1 {dt.mean():+.4f}")
    allcv = pd.concat([cv, s[["S2_document_weighting", "S3_temporal_tfidf", "S4_dual_level"]]], axis=1)
    print(f"rentang c_v semua model: {allcv.min().min():.4f}–{allcv.max().max():.4f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "satu":
        satu(int(sys.argv[2]), int(sys.argv[3]))
    elif mode in ("cepat", "uji"):
        cepat(int(sys.argv[2]), int(sys.argv[3]), uji=mode == "uji")
    elif mode == "ringkas":
        ringkas()
    else:
        sys.exit(__doc__)
