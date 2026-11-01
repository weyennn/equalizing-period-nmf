# Period weighting for global NMF topic models

Code for the manuscript *Equalizing Period Contributions in Global NMF Topic Modeling: Square-Root Document
Weighting and Per-Period Evaluation on Imbalanced News Corpora* (Yayang Matira and Yohanes Suyanto, Universitas
Gadjah Mada).

Square-root document weighting scales each TF-IDF row of period *p* by `a_p = sqrt(mean(n) / n_p)`, which turns the
Frobenius objective of global NMF into the sum of per-period mean squared residuals. The repository contains the
experiment pipeline, the construction of both corpora, every analysis reported in the paper, and the result files
from which the tables and figures were taken.

The scripts were written during an MSc thesis, so file names, comments, and console output are in Indonesian. The
table below and the docstring at the top of each script describe what it does.

## Layout

| Folder | Contents |
|---|---|
| `pipeline/` | Experiment runner for scenarios S1–S4 (`08_run_weighting_experiments.py`), TF-IDF, NMF, Temporal TF-IDF, evaluation and coherence helpers (`temporal_nmf_pipeline.py`), and Indonesian preprocessing (`preprocessing.py`, `custom_stopwords_id.txt`) |
| `corpus1/` | Construction of the first corpus (merge, audit, political-keyword filter, boilerplate removal), the runs of the 15 configurations, and the reference-configuration analyses (raw versus cleaned corpus, balanced vocabulary) |
| `analysis/` | All further analyses of the paper, the second corpus, Fairer NMF, and their result files (`*.csv`) |
| `data/` | `corpus1_documents.csv` and `corpus2_documents.csv`: links and dates of the documents of both corpora, in dataset order. Downloaded data go to `data/raw/`, derived CSV files to `data/processed/` (both not tracked) |
| `outputs/` | Created by the runs (models, matrices, per-configuration results; not tracked) |

## Environment

Python 3.10 with the versions used for all reported results:

```
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

All runs were single-threaded (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`). The shell runners are `zsh` scripts; they skip configurations whose results already exist,
so they can be stopped and restarted. Set `PYTHON` (or pass the interpreter as the first argument where the usage line
says so) to choose the Python executable.

## Data

**First corpus.** 2024 articles from the national news channels of Detik, Kompas, and Tempo in the Indonesia News
Dataset (Kaggle, CC BY-NC 4.0, <https://doi.org/10.34740/kaggle/ds/5673686>): January to 5 September from the first
release and October to December from the second. `data/corpus1_documents.csv` lists the 48,673 documents (link, date,
source). The construction steps are, in order:

1. `corpus1/merge_politik_full_year.py`: merges the political articles of the two releases (inputs
   `data/processed/politik_merge_dates_fixed.csv` and `data/raw/dataset_fixed.csv`, prepared from the two releases with
   earlier scripts that are not included; the link list identifies the resulting documents).
2. `corpus1/01_audit_clean_dataset.py`: removes empty content and duplicate links.
3. `corpus1/03_qc_filter_experiment_dataset.py`: the keyword rule (29 non-political and 68 political keywords, listed in
   the script), short documents, and duplicates; output `politik_full_year_2024_experiment_qc.csv`
   (SHA-256 `8d384717ce9f8576816d43312c8dd6815ad05df3110fea32fc0cad43a65b0187`).
4. `corpus1/select_spans.py` and `corpus1/clean.py`: removal of scraped template text and inserted related-article
   titles (rules in the docstring of `clean.py`, the 236 spans in `spans_C_final.json`); output
   `politik_full_year_2024_experiment_qc_bersih.csv`
   (SHA-256 `953df38eae223c17e1ea82502db5d9b8a9c3ce9b09f0fe98969778a9e8d6c44f`), the corpus used for all results.

**Second corpus.** POLITICS category of the HuffPost News Category Dataset (Kaggle, CC BY 4.0,
<https://www.kaggle.com/datasets/rmisra/news-category-dataset>). Place `News_Category_Dataset_v3.json` in `data/raw/`;
`analysis/korpus2_huffpost.py` applies the filters itself. `data/corpus2_documents.csv` lists the 35,161 items.

## Reproducing the results

First corpus, scenarios S1–S4 in the 15 configurations (random states 7, 21, 42, 84, 123; K = 15, 20, 25):

```
for rs in 7 21 42 84 123; do zsh corpus1/jalankan_gelombang.sh $rs; done
python -I corpus1/ringkas_15.py
```

The further analyses read these runs from `outputs/korpus_bersih_15konfigurasi_2026-10-20/` and write their result
files next to themselves in `analysis/`. Their usage lines are in the docstrings; scripts with a `satu <rs> <K>` mode
run one configuration per call and have a `ringkas` mode that summarizes all 15.

| Paper | Script | Result files |
|---|---|---|
| Tables 1–2 (documents per period) | `data/corpus*_documents.csv`, `analysis/korpus2_huffpost.py` | `analysis/korpus2_info.json` |
| Table 3, Fig. 1 (macro–micro trade-off) | `corpus1/ringkas_15.py`, `analysis/analisis_bobot_per_bulan.py` | `analysis/bootstrap_15konfigurasi.csv` |
| Table 4, Fig. 2a (per-month changes, E3) | `analysis/analisis_bobot_per_bulan.py` | `analysis/per_bulan_15konfigurasi.csv` |
| Table 5, Fig. 2b (exponents, undersampling) | `analysis/eksperimen_kurva_bobot_undersampling.py` | `analysis/kurva_bobot_us_*.csv` |
| Table 5 (per-period NMF) | `analysis/eksperimen_nmf_per_periode.py` | `analysis/nmf_per_periode_*.csv` |
| Table 5, Section 5.4 (Fairer NMF, MU) | `analysis/fairer_nmf.py`, `analysis/eksperimen_fairer_nmf.py`, `analysis/jalankan_fairer_nmf.sh` | `analysis/fairer_nmf_*.csv` |
| Section 5.5 (S3\*, S4\*) | `analysis/analisis_proyeksi_s34.py` | `analysis/proyeksi_s34_*.csv` |
| Table 6 (topics without counterpart, peaks) | `analysis/analisis_dampak_topik.py` | `analysis/dampak_topik_*.csv` |
| Table 7, Fig. 3, Sections 5.4 and 6.2 (coverage, FN criterion on validation data) | `analysis/analisis_acuan_validasi_cakupan.py` | `analysis/acuan_cakupan_per_periode.csv` |
| Section 6.2 (reference errors of FN) | `analysis/analisis_galat_acuan_fn.py` | `analysis/galat_acuan_fn_per_periode.csv` |
| Tables 5, 7, 10, Sections 5.4, 5.10, 6.2 (Fairer NMF with cross-validated reference errors, FN-CV) | `analysis/eksperimen_fn_acuan_cv.py`, `analysis/jalankan_fn_acuan_cv.sh` | `analysis/fn_acuan_cv_*.csv` |
| Table 8 (boilerplate) | uncleaned reference run (below), then `corpus1/eval_sim.py` on run `r42_k20` of the 15 configurations | |
| Table 9 (balanced vocabulary) | `corpus1/run_vocab_seimbang.py`, `corpus1/eval_sim.py` | |
| Section 5.9 (coherence of all variants) | `analysis/cv_korpus1_varian.py`, `analysis/jalankan_cv_korpus1.sh` | `analysis/cv_korpus1_varian.csv` |
| Table 10 (second corpus) | `analysis/korpus2_huffpost.py`, `analysis/jalankan_korpus2.sh`, `analysis/jalankan_korpus2_cv.sh` | `analysis/korpus2_*.csv` |

The uncleaned reference run of Table 8 is
`python -I pipeline/08_run_weighting_experiments.py --dataset data/processed/politik_full_year_2024_experiment_qc.csv
--output-dir outputs/global_nmf_temporal_locked_rerun_hardened_2026-08-20` (random state 42, K = 20).

`analysis/uji_fairer_nmf.py` checks the sparse reimplementation of Fairer NMF against the official code on random data
(it needs a copy of `fairnmf.py` from <https://github.com/ErinGeorge/Fairer-NMF>). `analysis/cv_korpus1_varian.py`
has a `cepat` mode that scans the coherence corpus once for all variants and an `uji` mode that checks it against the
per-variant computation; both give identical $c_v$.

## License

The code of this repository is released under the MIT License (`LICENSE`). `analysis/fairer_nmf.py` is derived from
the Fairer NMF code of Kassab et al., also under the MIT License; see `THIRD_PARTY_NOTICES.md`. The datasets keep their
own licenses (CC BY-NC 4.0 and CC BY 4.0), and the article texts are not redistributed here: `data/` contains only the
links, dates, and sources of the documents used, and `corpus1/spans_C_final.json` the template and related-article
title spans removed during cleaning. These files are derived from the datasets and are shared under their licenses.
