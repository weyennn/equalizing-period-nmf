"""Uji kesetaraan fairer_nmf.py dengan kode resmi Fairer-NMF (FairNMF_MU dan NMF di fairnmf.py, ErinGeorge/Fairer-NMF).

Kedua versi dijalankan pada data acak kecil dengan generator acak yang sama; kode resmi memakai matriks padat, versi kita
matriks sparse. cvxpy (hanya untuk versi AM) diganti modul kosong agar fairnmf.py dapat diimpor.
Pemakaian: <python> -I analysis/uji_fairer_nmf.py <jalur fairnmf.py resmi>
"""
import importlib.util, sys, types
from pathlib import Path
import numpy as np
from scipy import sparse

sys.modules["cvxpy"] = types.ModuleType("cvxpy")
spec = importlib.util.spec_from_file_location("fairnmf_resmi", sys.argv[1]); resmi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resmi)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairer_nmf import fairer_nmf_mu, nmf_mu  # noqa: E402

gen = np.random.default_rng(123)
Xs = []
for n in (40, 15, 25):   # tiga kelompok, 60 fitur, baris dinormalkan seperti TF-IDF
    X = sparse.random(n, 60, density=0.15, random_state=gen, format="csr")
    X = sparse.diags(1 / np.maximum(np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()), 1e-12)) @ X
    Xs.append(X.toarray())

resmi.rng = np.random.default_rng(7)
W_r, H_r, e_r = resmi.NMF(Xs[0], 4, max_iter=300, rel_tol=1e-3)
W_k, H_k, e_k = nmf_mu(sparse.csr_matrix(Xs[0]), 4, np.random.default_rng(7), max_iter=300, rel_tol=1e-3)
print(f"NMF: iterasi {len(e_r)} vs {len(e_k)}; selisih maks H {np.abs(H_r - H_k).max():.2e}, galat {np.abs(e_r - e_k).max():.2e}")

resmi.rng = np.random.default_rng(11)
Ws_r, H_r, errs_r, min_r = resmi.FairNMF_MU(Xs, 4, max_iter=300, rel_tol=1e-3, c_rule="decaying_lr")
Ws_k, H_k, errs_k, min_k, it_k = fairer_nmf_mu([sparse.csr_matrix(X) for X in Xs], 4, np.random.default_rng(11), max_iter=300, rel_tol=1e-3)
print(f"FairNMF: iterasi {errs_r.shape[1]} vs {it_k}; selisih maks min_errs {np.abs(min_r - min_k).max():.2e}, "
      f"H {np.abs(H_r - H_k).max():.2e}, W {max(np.abs(a - b).max() for a, b in zip(Ws_r, Ws_k)):.2e}, "
      f"galat {np.abs(errs_r - errs_k).max():.2e}")
