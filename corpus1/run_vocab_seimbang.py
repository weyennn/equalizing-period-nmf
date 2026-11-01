"""Rerun S1–S4 dengan vocabulary seimbang antarbulan (simulasi; protokol lain sama).

Vocabulary utama: 8.000 istilah dengan frekuensi tertinggi pada seluruh data latih (didominasi Agustus).
Di sini: kandidat tetap memenuhi min_df=10 dan max_df=0,6 pada data latih, lalu dipilih bergiliran (round-robin)
dari peringkat frekuensi istilah tiap bulan sampai 8.000 istilah unik, sehingga setiap bulan menyumbang
istilah teratasnya secara setara. IDF tetap dipelajari dari data latih saja.
Pemakaian: python run_vocab_seimbang.py <clean.csv> <out_dir> <vocab_out.json>
"""
import json, sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

SRC = Path(ROOT / "pipeline"); sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("runner", SRC / "08_run_weighting_experiments.py")
m = importlib.util.module_from_spec(spec); sys.modules["runner"] = m; spec.loader.exec_module(m)
ds, out, vocab_out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
m.EXPECTED_DATASET_SHA256 = m.sha256_file(ds)

# split identik dengan run (fungsi yang sama, seed sama)
df = m.build_preprocessed_frame(m.load_dataset(ds))
train, _ = m.split_train_validation(df, validation_fraction=m.VALIDATION_FRACTION, random_state=42)
texts = train["preprocessed_text"].tolist(); months = train["year_month"].astype(str).to_numpy()
cv = CountVectorizer(min_df=m.MIN_DF, max_df=m.MAX_DF, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
C = cv.fit_transform(texts).tocsr(); terms = np.array(cv.get_feature_names_out())
ranks = {}
for p in sorted(set(months)):
    tf = np.asarray(C[months == p].sum(0)).ravel()
    ranks[p] = [j for j in np.argsort(-tf, kind="stable") if tf[j] > 0]
vocab, seen, pos = [], set(), {p: 0 for p in ranks}
while len(vocab) < m.MAX_FEATURES:
    for p in sorted(ranks):
        while pos[p] < len(ranks[p]) and ranks[p][pos[p]] in seen:
            pos[p] += 1
        if pos[p] < len(ranks[p]) and len(vocab) < m.MAX_FEATURES:
            j = ranks[p][pos[p]]; seen.add(j); vocab.append(terms[j])
vocab = sorted(vocab)
json.dump(vocab, open(vocab_out, "w"), ensure_ascii=False)
print("vocab", len(vocab), "kandidat", len(terms))


def build_vectorizer(config):
    return TfidfVectorizer(vocabulary=vocab, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                           sublinear_tf=True, smooth_idf=True, norm="l2", dtype=np.float64)


m.build_vectorizer = build_vectorizer
cfg = m.RunConfig(smoke=False, smoke_rows=0, max_features=m.MAX_FEATURES, min_df=m.MIN_DF, max_df=m.MAX_DF, n_topics=20, random_state=42)
m.run(cfg, dataset_path=ds, output_dir=out)
