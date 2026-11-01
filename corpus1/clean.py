"""Bersihkan teks templat halaman dan potongan tautan terkait dari Content_clean.

Simulasi terpisah: dataset asli tidak diubah; keluaran ditulis ke CSV baru.
Aturan:
  A. Teks templat halaman (penanda eksplisit):
     - Tempo : "Scroll ke bawah untuk melanjutkan membaca",
               "Baca berita dengan sedikit iklan, klik di sini"
     - Detik : "[Gambas:...]", "(Simak|Lihat|Tonton) (juga) Video ...[Gambas:...]",
               "Baca selengkapnya di (sini|halaman selanjutnya|halaman berikutnya)( dan di sini)"
  B. Potongan tautan terkait dengan penanda (Tempo):
     - "Pilihan Editor: <judul>" / "Selanjutnya: <judul>" di akhir artikel (≤40 kata sampai akhir) dihapus
       sampai akhir; selain itu hanya penanda + ≤20 kata berikutnya.
     - "Terkini: <judul>" dihapus melalui aturan C; penanda sisa dihapus.
  C. Potongan tautan terkait tanpa penanda (spans_C_final.json, dibentuk oleh select_spans.py): rentang ≥8 token
     yang muncul identik pada ≥15 dokumen Tempo, ≥90% kemunculan dari Tempo, ≥60% kata berhuruf awal kapital,
     tidak memuat "TEMPO.CO", dan ≥70% kemunculannya berada di batas kalimat (diawali akhir kalimat/":" dan
     diikuti huruf kapital). Frasa jabatan/nama lembaga yang menyatu dalam kalimat tidak terpilih.
"""
from __future__ import annotations
import json, re, sys
import pandas as pd

SRC = sys.argv[1]; OUT = sys.argv[2]; SPANS = sys.argv[3]; REPORT = sys.argv[4]

df = pd.read_csv(SRC)
c = df["Content_clean"].fillna("").astype(str)
src = df["source"].astype(str)

A = [
    r"Scroll ke bawah untuk melanjutkan membaca",
    r"Baca berita dengan sedikit iklan,? klik di sini",
    r"(?:Simak|Lihat|Tonton)(?: juga)? [Vv]ideo\b[^\[]{0,200}\[Gambas:[^\]]*\]",
    r"\[Gambas:[^\]]*\]",
    r"Baca selengkapnya di (?:sini(?: dan di sini)?|halaman selanjutnya|halaman berikutnya)",
]
A_RE = [re.compile(p) for p in A]

sel = [tuple(x) for x in json.load(open(SPANS))]
sel.sort(key=lambda x: -len(x[0]))  # panjang dulu agar rentang gabungan terhapus utuh


def tail_marker(t: str, marker: str) -> str:
    out = t
    while True:
        j = out.find(marker)
        if j < 0:
            return out
        rest = out[j + len(marker):]
        if len(rest.split()) <= 40:
            out = out[:j]
        else:
            out = out[:j] + " " + " ".join(rest.split()[20:])


stats = {"A": 0, "B": 0, "C": 0}
docs = {"A": set(), "B": set(), "C": set()}
new = []
for i, t in enumerate(c):
    o = t
    for r in A_RE:
        t2 = r.sub(" ", t)
        if t2 != t:
            docs["A"].add(i)
        t = t2
    for m in ("Pilihan Editorial:", "Pilihan Editor:", "Pilihan Editor ", "Selanjutnya:"):
        t2 = tail_marker(t, m)
        if t2 != t:
            docs["B"].add(i)
        t = t2
    if src[i] == "tempo":
        for s, _ in sel:
            if s in t:
                t = t.replace(s, " "); docs["C"].add(i)
        t2 = re.sub(r"\bTerkini:\s*", " ", t)
        if t2 != t:
            docs["B"].add(i)
        t = t2
    t = re.sub(r"\s{2,}", " ", t).strip()
    new.append(t)

df["Content_clean_original"] = c
df["Content_clean"] = new
changed = df["Content_clean"] != df["Content_clean_original"]
df["n_words_removed"] = c.str.split().str.len() - df["Content_clean"].str.split().str.len()
df.drop(columns=["Content_clean_original"]).to_csv(OUT, index=False)

df["m"] = pd.to_datetime(df["parsed_date"]).dt.to_period("M").astype(str)
resid = {k: int(df["Content_clean"].str.contains(k, regex=False).sum()) for k in
         ["melanjutkan membaca", "Pilihan Editor", "berita iklan", "[Gambas", "Selanjutnya:", "Terkini:",
          "Tol Sigli-Banda Aceh", "Ahmad Luthfi Rival", "Insan Olahraga Raih Prestasi"]}
rep = {
    "rows": len(df), "docs_changed": int(changed.sum()),
    "docs_rule": {k: len(v) for k, v in docs.items()},
    "n_spans_rule_C": len(sel),
    "words_removed_total": int(df["n_words_removed"].sum()),
    "words_removed_share": float(df["n_words_removed"].sum() / c.str.split().str.len().sum()),
    "changed_by_source": df[changed].groupby("source").size().to_dict(),
    "changed_by_source_month": df[changed].groupby(["source", "m"]).size().unstack(0).fillna(0).astype(int).to_dict(),
    "residual_docs_with_phrase": resid,
    "min_words_after": int(df["Content_clean"].str.split().str.len().min()),
}
json.dump({"report": rep, "spans_rule_C": sel}, open(REPORT, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "changed_by_source_month"}, ensure_ascii=False, indent=1))
