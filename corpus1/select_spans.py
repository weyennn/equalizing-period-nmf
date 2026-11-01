"""Bentuk daftar rentang potongan tautan terkait tanpa penanda (aturan C pada clean.py).

1. Tambang rentang: gabungkan 8-gram token (dipisah spasi) yang muncul pada >=15 dokumen menjadi rentang
   maksimal per dokumen, lalu hitung frekuensi dokumen setiap rentang (>=15).
2. Pilih rentang judul berita sisipan: >=8 token, diawali huruf kapital/angka/kutip, tidak berakhir koma
   atau kata hubung, tidak memuat "TEMPO.CO", >=60% kata berhuruf awal kapital, >=90% kemunculan dari Tempo,
   dan >=70% kemunculan di batas kalimat pada kedua sisi. Varian dengan token pertama berakhiran titik
   atau token terakhir yang ikut tergabung dicoba dipangkas.
Pemakaian: python select_spans.py <dataset.csv> <spans_C_final.json>
"""
import collections, json, re, sys
import pandas as pd

df = pd.read_csv(sys.argv[1], usecols=["Content_clean", "source"])
c_all = df["Content_clean"].fillna("").tolist()
src = df["source"].tolist()
N = 8
toks = [t.split() for t in c_all]
cnt = collections.Counter()
for t in toks:
    cnt.update(set(tuple(t[j:j + N]) for j in range(len(t) - N + 1)))
gs = {g for g, n in cnt.items() if n >= 15}
spans = collections.Counter()
for t in toks:
    j, L, found = 0, len(t), set()
    while j <= L - N:
        if tuple(t[j:j + N]) in gs:
            k = j
            while k <= L - N and tuple(t[k:k + N]) in gs:
                k += 1
            found.add(" ".join(t[j:k + N - 1])); j = k + N - 1
        else:
            j += 1
    spans.update(found)

MON = r"(?:(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember) )?2024\. "
tempo = [t for t, s in zip(c_all, src) if s == "tempo"]
ser = pd.Series(c_all); src_s = pd.Series(src)


def bstat(s):
    pre = post = k = 0
    for t in tempo:
        j = t.find(s)
        if j < 0:
            continue
        k += 1
        b = t[:j].rstrip(); a = t[j + len(s):].lstrip()
        if j == 0 or re.search(r'[.!?"”’):]$|\d{4}$', b):
            pre += 1
        if a == "" or re.match(r'["“‘(A-Z0-9]', a):
            post += 1
    return k, (pre / k if k else 0), (post / k if k else 0)


keep = {}
for s, n in spans.items():
    if n < 15:
        continue
    s2 = re.sub(r"^Terkini: ", "", re.sub("^" + MON, "", s).strip())
    if len(s2.split()) < 8 or "TEMPO.CO" in s2:
        continue
    w = re.findall(r"[A-Za-z][\w'’-]*", s2)
    if sum(x[0].isupper() for x in w) / max(1, len(w)) < 0.6:
        continue
    hit = ser.str.contains(s2, regex=False)
    if hit.sum() < 15 or (src_s[hit] == "tempo").mean() < 0.9:
        continue
    ws = s2.split(); variants = [s2]
    if ws[0].endswith("."):
        variants.append(" ".join(ws[1:]))
    variants += [" ".join(v.split()[:-1]) for v in list(variants)]
    for v in variants:
        if len(v.split()) < 8 or not re.match(r'^[A-Z0-9"“\']', v) or re.search(r"[,]$| (pada|di|itu|dan|yang)$", v):
            continue
        k, p, q = bstat(v)
        if k >= 15 and p >= 0.7 and q >= 0.7:
            keep[v] = k; break
out = sorted(keep.items(), key=lambda x: -x[1])
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=0)
print(len(out), "rentang terpilih")
