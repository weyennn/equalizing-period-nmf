"""Fairer-NMF (Kassab et al. 2026), versi multiplicative updates, untuk matriks sparse (27 Oktober 2026).

Diturunkan dari FairNMF_MU dan NMF di https://github.com/ErinGeorge/Fairer-NMF (fairnmf.py; lisensi MIT,
Copyright (c) 2025 Erin George; lihat THIRD_PARTY_NOTICES.md). Aturan pembaruan,
inisialisasi acak, urutan pengambilan bilangan acak, aturan bobot kelompok 'decaying_lr', galat acuan (rata-rata 5 run
NMF-MU per kelompok) dan kriteria berhenti sama dengan kode resmi. Bedanya hanya cara menghitung: galat Frobenius
dihitung dari ||X||^2 - 2<X H^T, W> + <W^T W, H H^T> dan perkalian blok tertimbang dijumlahkan per kelompok, sehingga
X tidak perlu dibuat padat dan W @ H tidak pernah dibentuk. Kesetaraan dengan kode resmi diuji di uji_fairer_nmf.py.
"""
import numpy as np
from scipy import sparse

MACHINE_EPSILON = np.finfo(np.float32).eps


def _err(X, x2, W, H, HHt=None):
    """||X - W H||_F tanpa membentuk W H."""
    HHt = H @ H.T if HHt is None else HHt
    cross = float(np.sum(np.asarray(X @ H.T) * W))
    return float(np.sqrt(max(x2 - 2.0 * cross + float(np.sum((W.T @ W) * HHt)), 0.0)))


def nmf_mu(X, num_topics, rng, max_iter=1000, rel_tol=1e-3):
    """Setara dengan NMF() kode resmi (MU, inisialisasi acak seragam, berhenti bila perubahan galat relatif < rel_tol)."""
    X = sparse.csr_matrix(X); x2 = float(X.multiply(X).sum())
    W = rng.random((X.shape[0], num_topics)); H = rng.random((num_topics, X.shape[1]))
    errs = []
    for i in range(max_iter):
        denom = W.T @ W @ H; denom[denom < MACHINE_EPSILON] = MACHINE_EPSILON
        H *= np.asarray(X.T @ W).T / denom
        HHt = H @ H.T
        denom = W @ HHt; denom[denom < MACHINE_EPSILON] = MACHINE_EPSILON
        W *= np.asarray(X @ H.T) / denom
        errs.append(_err(X, x2, W, H, HHt))
        if rel_tol is not None and i > 0 and abs(errs[-2] - errs[-1]) / abs(errs[-1]) < rel_tol:
            break
    return W, H, np.array(errs)


def fairer_nmf_mu(Xs, num_topics, rng, max_iter=1000, rel_tol=1e-3, n_ref=5, min_errs=None):
    """Setara dengan FairNMF_MU(Xs, num_topics, max_iter, rel_tol, c_rule='decaying_lr') kode resmi.

    Xs: daftar matriks (sparse/padat) per kelompok dengan kolom sama. Mengembalikan Ws, H, errs (kelompok x iterasi;
    galat relatif berlebih (||X_l - W_l H|| - E_l) / ||X_l||), min_errs (E_l), dan jumlah iterasi.
    min_errs: bila diberikan, galat acuan E_l dipakai apa adanya dan tidak dihitung dengan NMF-MU (perluasan, 27 Okt:
    pada data kita NMF-MU dengan rel_tol 1e-3 berhenti setelah dua iterasi sehingga E_l terlalu besar).
    rel_tol=None: selalu max_iter iterasi.
    """
    Xs = [sparse.csr_matrix(X) for X in Xs]
    x2 = np.array([float(X.multiply(X).sum()) for X in Xs]); norms = np.sqrt(x2)
    if min_errs is None:
        min_errs = np.zeros(len(Xs))
        for i, X in enumerate(Xs):   # galat acuan: rata-rata 5 run NMF pada kelompok sendiri
            opt = []
            for _ in range(n_ref):
                W, H, _ = nmf_mu(X, num_topics, rng, max_iter=max_iter, rel_tol=rel_tol)
                opt.append(_err(X, x2[i], W, H))
            min_errs[i] = np.nanmean(opt)
    min_errs = np.asarray(min_errs, dtype=float)
    Ws = [rng.random((X.shape[0], num_topics)) for X in Xs]
    H = rng.random((num_topics, Xs[0].shape[1]))
    coefs = np.ones(len(Xs)) / len(Xs)
    errs = np.zeros((len(Xs), max_iter))
    HHt = H @ H.T
    cur = (np.array([_err(X, a, W, H, HHt) for X, a, W in zip(Xs, x2, Ws)]) - min_errs) / norms
    for it in range(max_iter):
        coefs[np.argmax(cur)] += 1                      # c_rule = 'decaying_lr'
        s = (coefs / norms) ** 2                         # W_step^T X_step = sum_l (c_l/n_l)^2 W_l^T X_l
        num = sum(sl * np.asarray(X.T @ W).T for sl, X, W in zip(s, Xs, Ws))
        WtW = sum(sl * (W.T @ W) for sl, W in zip(s, Ws))
        denom = WtW @ H; denom[denom < MACHINE_EPSILON] = MACHINE_EPSILON
        H *= num / denom
        HHt = H @ H.T
        for j, X in enumerate(Xs):
            denom = Ws[j] @ HHt; denom[denom < MACHINE_EPSILON] = MACHINE_EPSILON
            Ws[j] *= np.asarray(X @ H.T) / denom
        cur = (np.array([_err(X, a, W, H, HHt) for X, a, W in zip(Xs, x2, Ws)]) - min_errs) / norms
        errs[:, it] = cur
        if it > 0 and rel_tol is not None and np.max(np.abs(errs[:, it - 1] - errs[:, it]) / np.abs(errs[:, it])) < rel_tol:
            break
    return Ws, H, errs[:, :it + 1], min_errs, it + 1
