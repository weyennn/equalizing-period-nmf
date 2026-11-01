"""Run S1–S4 pada korpus bersih dengan protokol terkunci. Pemakaian: run_bersih.py <csv> <out_dir> <random_state> <K>"""
import sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # akar repositori
SRC = Path(ROOT / "pipeline"); sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('runner', SRC / '08_run_weighting_experiments.py')
m = importlib.util.module_from_spec(spec); sys.modules['runner'] = m; spec.loader.exec_module(m)
ds, out, rs, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
m.EXPECTED_DATASET_SHA256 = m.sha256_file(ds)  # korpus bersih (SHA 953df38e...)
cfg = m.RunConfig(smoke=False, smoke_rows=0, max_features=m.MAX_FEATURES, min_df=m.MIN_DF, max_df=m.MAX_DF, n_topics=k, random_state=rs)
m.run(cfg, dataset_path=ds, output_dir=out)
