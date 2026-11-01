"""Galat acuan relatif Fairer-NMF per periode (27 Oktober 2026), untuk Pembahasan "Average versus worst case".

E_l (galat NMF konvergen K topik pada data latih periode l sendiri) dibagi ||X_l||_F = sqrt(n_l), karena setiap baris
TF-IDF ber-norma 1 dan tidak ada baris kosong. Dirata-rata atas 15 konfigurasi per korpus.
Masukan: *_fair.npz (E) di outputs/paper_fairer_nmf_2026-10-27/ dan paper_korpus2_huffpost_2026-10-27/,
n latih per periode dari kurva_bobot_us_per_bulan.csv dan korpus2_per_periode.csv.
Keluaran: galat_acuan_fn_per_periode.csv di folder skrip ini.
Pemakaian: <python> -I analysis/analisis_galat_acuan_fn.py
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent
H1 = Path(ROOT / "outputs/paper_fairer_nmf_2026-10-27")
H2 = Path(ROOT / "outputs/paper_korpus2_huffpost_2026-10-27")
p1 = pd.read_csv(OUT / "kurva_bobot_us_per_bulan.csv", dtype={"bulan": str})
p2 = pd.read_csv(OUT / "korpus2_per_periode.csv", dtype={"tahun": str})
rows = []
for rs in (7, 21, 42, 84, 123):
    for k in (15, 20, 25):
        name = f"r{rs}_k{k}"
        for korpus, H, p, col, ncol in ((1, H1, p1, "bulan", "n_train"), (2, H2, p2, "tahun", "n_latih")):
            E = np.load(H / f"{name}_fair.npz")["E"]
            q = p[p.run == name].sort_values(col)
            assert len(q) == len(E)
            for per, n, e in zip(q[col], q[ncol], E):
                rows.append({"korpus": korpus, "run": name, "K": k, "periode": per, "n_latih": n, "E": e, "E_rel": e / np.sqrt(n)})
d = pd.DataFrame(rows)
d.to_csv(OUT / "galat_acuan_fn_per_periode.csv", index=False)
for korpus in (1, 2):
    x = d[d.korpus == korpus]
    print(f"== korpus {korpus}: korelasi log n vs E_rel {np.corrcoef(np.log(x.n_latih), x.E_rel)[0, 1]:.2f}")
    print(x.groupby("periode").agg(n_latih=("n_latih", "first"), E_rel=("E_rel", "mean")).round(5).to_string())
