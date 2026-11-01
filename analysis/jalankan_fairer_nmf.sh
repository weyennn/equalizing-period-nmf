#!/bin/zsh
# Fairer-NMF pada kedua korpus: satu konfigurasi per proses, berurutan, 1 utas CPU, prioritas rendah.
# Konfigurasi yang sudah punya berkas hasil dilewati, jadi aman dihentikan dan dijalankan ulang.
# Pemakaian: zsh analysis/jalankan_fairer_nmf.sh <python 3.10 dengan scikit-learn 1.5.2 dan gensim 4.3.3>
PY=${1:-${PYTHON:-python3}}
D=${0:A:h}
H1=${0:A:h}/../outputs/paper_fairer_nmf_2026-10-27/hasil
H2=${0:A:h}/../outputs/paper_korpus2_huffpost_2026-10-27/hasil
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for rs in 7 21 42 84 123; do
  for k in 15 20 25; do
    if [[ -f $H1/r${rs}_k${k}_ringkasan.csv ]]; then echo "lewati korpus 1: r${rs}_k${k}"; continue; fi
    nice -n 19 $PY -I $D/eksperimen_fairer_nmf.py satu $rs $k || { echo "gagal korpus 1: r${rs}_k${k}"; exit 1; }
  done
done
nice -n 19 $PY -I $D/eksperimen_fairer_nmf.py ringkas
for rs in 7 21 42 84 123; do
  for k in 15 20 25; do
    if [[ -f $H2/r${rs}_k${k}_fair.csv ]]; then echo "lewati korpus 2: r${rs}_k${k}"; continue; fi
    nice -n 19 $PY -I $D/korpus2_huffpost.py fair $rs $k || { echo "gagal korpus 2: r${rs}_k${k}"; exit 1; }
  done
done
nice -n 19 $PY -I $D/korpus2_huffpost.py ringkas
