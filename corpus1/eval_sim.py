"""Evaluasi simulasi korpus bersih (logika sama dengan audit8.py/audit7.py) dan perbandingan dengan run utama.

Pemakaian: python eval_sim.py <run_dir_bersih> <out.json>
"""
import copy, json, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import joblib, numpy as np, pandas as pd
from scipy import sparse
warnings.filterwarnings("ignore")

R = Path(sys.argv[1]); OUT = Path(sys.argv[2])
ORIG = Path(ROOT / "outputs/global_nmf_temporal_locked_rerun_hardened_2026-08-20")
DS0 = ROOT / "data/processed/politik_full_year_2024_experiment_qc.csv"
SC = ["S1_baseline", "S2_document_weighting", "S3_temporal_tfidf", "S4_dual_level"]
ds = pd.read_csv(DS0, usecols=["Link", "Content_clean"])
flag = dict(zip(ds.Link, ds.Content_clean.fillna("").str.lower().str.contains("melanjutkan membaca|pilihan editor|berita iklan", regex=True)))


def evaluate(run: Path, boot=1000):
    Xv = sparse.load_npz(run / "X_base_valid.npz").tocsr()
    led = pd.read_csv(run / "split_ledger.csv"); v = led[led.split == "validation"].sort_values("split_row")
    vm = v.year_month.astype(str).to_numpy(); vs = v.source.to_numpy()
    tm = v.Link.map(flag).fillna(False).to_numpy().astype(bool)  # dokumen yang bertemplat pada korpus asli
    tidf = pd.read_csv(run / "temporal_idf_train.csv").temporal_idf.to_numpy()
    xsq = np.asarray(Xv.multiply(Xv).sum(1)).ravel()

    def rowres(m, H):
        mm = copy.deepcopy(m); mm.components_ = H; W = mm.transform(Xv)
        return xsq + np.einsum("ij,jk,ik->i", W, H @ H.T, W) - 2 * np.sum(W * np.asarray(Xv @ H.T), 1)

    res = {}; summ = {}
    for s in SC:
        m = joblib.load(run / s / "nmf_model.joblib"); H = m.components_; res[s[:2]] = rowres(m, H)
        if s[1] in "34":
            Hb = H / tidf; Hb *= (np.linalg.norm(H, axis=1) / np.linalg.norm(Hb, axis=1))[:, None]
            res[s[:2] + "*"] = rowres(m, Hb)
        summ[s[:2]] = json.load(open(run / s / "scenario_summary.json"))
    keys = list(res); months = sorted(set(vm)); idx = {p: np.flatnonzero(vm == p) for p in months}
    rre = lambda c, ix: float(np.sqrt(res[c][ix].sum() / xsq[ix].sum()))
    out = {"per_month": {c: [rre(c, idx[p]) for p in months] for c in keys}, "months": months}
    out["micro"] = {c: rre(c, np.arange(len(vm))) for c in keys}
    out["macro"] = {c: float(np.mean(out["per_month"][c])) for c in keys}
    out["n_months_better_than_S1"] = {c: int(sum(a < b for a, b in zip(out["per_month"][c], out["per_month"]["S1"]))) for c in keys}
    rng = np.random.default_rng(0); B = {c: [] for c in keys if c != "S1"}; Bm = {c: [] for c in B}
    for _ in range(boot):
        ix = {p: rng.choice(idx[p], len(idx[p])) for p in months}
        mac = {c: np.mean([rre(c, ix[p]) for p in months]) for c in keys}
        allix = np.concatenate(list(ix.values())); mic = {c: rre(c, allix) for c in keys}
        for c in B: B[c].append(mac[c] - mac["S1"]); Bm[c].append(mic[c] - mic["S1"])
    out["macro_diff_ci"] = {} if not boot else {c: [float(np.mean(B[c])), *map(float, np.percentile(B[c], [2.5, 97.5]))] for c in B}
    out["micro_diff_ci"] = {} if not boot else {c: [float(np.mean(Bm[c])), *map(float, np.percentile(Bm[c], [2.5, 97.5]))] for c in Bm}
    nt = {p: np.flatnonzero((vm == p) & ~tm) for p in months}
    out["macro_excl_orig_template_docs"] = {c: float(np.mean([rre(c, nt[p]) for p in months])) for c in keys}
    out["per_source"] = {c: {s: rre(c, np.flatnonzero(vs == s)) for s in ["detik", "kompas", "tempo"]} for c in keys}
    out["cv"] = {k: v.get("coherence_cv") for k, v in summ.items()}
    out["td"] = {k: v.get("topic_diversity_top15") for k, v in summ.items()}
    out["iters"] = {k: v.get("n_iter") for k, v in summ.items()}
    out["tvd"] = {k: v.get("tvd_train_energy") for k, v in summ.items()}
    out["summary_raw_keys"] = sorted(summ["S1"].keys())
    # konsentrasi sumber dan kata kunci
    conc = {}; feats = {}
    for s in SC:
        a = pd.read_csv(run / s / "projected_document_topic_assignments.csv")
        ct = pd.crosstab(a.dominant_topic, a.source, normalize="index"); n = a.dominant_topic.value_counts()
        conc[s[:2]] = [(int(k), r.idxmax(), round(float(r.max()), 3), int(n[k])) for k, r in ct.iterrows() if r.max() >= 0.70]
        f = pd.read_csv(run / s / "top_features_top15.csv"); feats[s[:2]] = dict(zip(f.topic_id.astype(int), f.top_features))
    out["source_concentrated_topics"] = conc; out["top_features"] = feats
    return out


clean = evaluate(R)
json.dump(clean, open(OUT, "w"), indent=1, ensure_ascii=False)
orig = evaluate(ORIG, boot=0) if not (OUT.parent / "orig_eval.json").exists() else json.load(open(OUT.parent / "orig_eval.json"))
json.dump(orig, open(OUT.parent / "orig_eval.json", "w"), indent=1, ensure_ascii=False)
json.dump(clean, open(OUT, "w"), indent=1, ensure_ascii=False)
rows = []
for c in ["S1", "S2", "S3", "S4", "S3*", "S4*"]:
    rows.append({"skenario": c,
                 "mikro_asli": orig["micro"][c], "mikro_bersih": clean["micro"][c],
                 "makro_asli": orig["macro"][c], "makro_bersih": clean["macro"][c],
                 "dmakro_vs_S1_bersih": clean["macro"][c] - clean["macro"]["S1"],
                 "dmakro_vs_S1_asli": orig["macro"][c] - orig["macro"]["S1"],
                 "IK95_bersih": clean["macro_diff_ci"].get(c), "bulan_lebih_baik_bersih": clean["n_months_better_than_S1"][c],
                 "cv_asli": orig["cv"].get(c), "cv_bersih": clean["cv"].get(c),
                 "td_asli": orig["td"].get(c), "td_bersih": clean["td"].get(c),
                 "topik_sumber70_asli": len(orig["source_concentrated_topics"].get(c, [])) if c in orig["source_concentrated_topics"] else None,
                 "topik_sumber70_bersih": len(clean["source_concentrated_topics"].get(c, [])) if c in clean["source_concentrated_topics"] else None})
df = pd.DataFrame(rows); df.to_csv(OUT.with_suffix(".csv"), index=False)
pd.set_option("display.width", 250); print(df.round(4).to_string())
print("summary keys:", clean["summary_raw_keys"])
