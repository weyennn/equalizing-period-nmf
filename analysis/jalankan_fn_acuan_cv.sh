#!/bin/zsh
# Fairer-NMF dengan galat acuan validasi silang (eksperimen_fn_acuan_cv.py) pada kedua korpus: satu konfigurasi per proses,
# berurutan, 1 utas CPU, prioritas rendah. Konfigurasi yang sudah punya berkas ringkasan dilewati, jadi aman dihentikan dan
# dijalankan ulang.
# Pemakaian: zsh analysis/jalankan_fn_acuan_cv.sh <python 3.10 dengan scikit-learn 1.5.2 dan gensim 4.3.3>
PY=${1:-${PYTHON:-python3}}
D=${0:A:h}
H=${0:A:h}/../outputs/paper_fn_acuan_cv_2026-10-28
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for korpus in 1 2; do
  for rs in 7 21 42 84 123; do
    for k in 15 20 25; do
      if [[ -f $H/k${korpus}_r${rs}_k${k}_ringkasan.csv ]]; then echo "lewati: korpus $korpus r${rs}_k${k}"; continue; fi
      L=$H/k${korpus}_r${rs}_k${k}_keluaran.txt
      nice -n 19 $PY -I $D/eksperimen_fn_acuan_cv.py satu $korpus $rs $k > $L 2>&1 || { echo "gagal: korpus $korpus r${rs}_k${k} (lihat $L)"; exit 1; }
      head -1 $L
    done
  done
done
nice -n 19 $PY -I $D/eksperimen_fn_acuan_cv.py ringkas
