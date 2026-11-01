#!/bin/zsh
# c_v varian lain korpus pertama (mode cepat: korpus dipindai sekali per konfigurasi), satu konfigurasi per proses, satu thread, nice 19; konfigurasi yang selesai dilewati.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=${PYTHON:-python3}
D=${0:A:h}; H=${0:A:h}/../outputs/paper_cv_korpus1_2026-10-27
for rs in 7 21 42 84 123; do for k in 15 20 25; do
  [[ -f $H/r${rs}_k${k}.csv ]] || nice -n 19 $PY -I $D/cv_korpus1_varian.py cepat $rs $k | grep selesai
done; done
nice -n 19 $PY -I $D/cv_korpus1_varian.py ringkas
