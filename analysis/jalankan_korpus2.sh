#!/bin/zsh
# Menjalankan korpus2_huffpost.py satu konfigurasi per proses, berurutan, 1 utas CPU, prioritas rendah.
# Konfigurasi yang sudah punya hasil/<nama>_ringkasan.csv dilewati, jadi aman dihentikan dan dijalankan ulang.
# Pemakaian: zsh analysis/jalankan_korpus2.sh <python 3.10 dengan scikit-learn 1.5.2 dan gensim 4.3.3>
PY=${1:-${PYTHON:-python3}}
SKRIP=${0:A:h}/korpus2_huffpost.py
HASIL=${0:A:h}/../outputs/paper_korpus2_huffpost_2026-10-27/hasil
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for rs in 7 21 42 84 123; do
  for k in 15 20 25; do
    if [[ -f $HASIL/r${rs}_k${k}_ringkasan.csv ]]; then echo "lewati: r${rs}_k${k}"; continue; fi
    nice -n 19 $PY -I $SKRIP satu $rs $k || { echo "gagal: r${rs}_k${k}"; exit 1; }
  done
done
nice -n 19 $PY -I $SKRIP ringkas
