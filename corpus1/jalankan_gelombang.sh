#!/bin/zsh
# Jalankan satu gelombang (satu random state, K = 15, 20, 25) secara berurutan.
# Pemakaian: ./jalankan_gelombang.sh <random_state>
# Folder hasil: outputs/korpus_bersih_15konfigurasi_2026-10-20/r<rs>_k<K>; run yang selesai punya run_manifest.json. Run tidak dimulai bila ruang kosong < 2 GB
# dan dihentikan bila ruang kosong turun di bawah 1 GB selama berjalan.
S=${0:A:h}
O=$S/../outputs/korpus_bersih_15konfigurasi_2026-10-20
mkdir -p $O
PY=${PYTHON:-python3}
CSV=$S/../data/processed/politik_full_year_2024_experiment_qc_bersih.csv
RS=$1
cd $O
for K in 15 20 25; do
  D=r${RS}_k${K}
  if [[ -f $D/run_manifest.json ]]; then echo "$(date '+%F %T') lewati $D (sudah selesai)" >> status.log; continue; fi
  FREE=$(df -k $O | awk 'NR==2{print $4}')
  if (( FREE < 2*1024*1024 )); then echo "$(date '+%F %T') BERHENTI: ruang kosong < 2 GB sebelum $D" >> status.log; exit 1; fi
  rm -rf $D
  echo "$(date '+%F %T') mulai $D" >> status.log
  $PY $S/run_bersih.py $CSV $D $RS $K > $D.log 2>&1 &
  PID=$!
  # pengaman: hentikan run bila ruang kosong < 1 GB selama berjalan (diperiksa tiap 30 detik)
  while kill -0 $PID 2>/dev/null; do
    FREE=$(df -k $O | awk 'NR==2{print $4}')
    if (( FREE < 1024*1024 )); then
      pkill -P $PID; kill $PID 2>/dev/null
      echo "$(date '+%F %T') DIHENTIKAN $D: ruang kosong < 1 GB" >> status.log; exit 1
    fi
    sleep 30
  done
  wait $PID
  if [[ -f $D/run_manifest.json ]]; then echo "$(date '+%F %T') selesai $D" >> status.log
  else echo "$(date '+%F %T') GAGAL $D (lihat $D.log)" >> status.log; exit 1; fi
done
echo "$(date '+%F %T') gelombang r=$RS selesai" >> status.log
