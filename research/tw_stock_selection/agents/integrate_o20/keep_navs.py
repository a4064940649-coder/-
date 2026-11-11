#!/usr/bin/env python3
"""integrate_o20 中間檔瘦身：data/ 只留 100 萬一般／壓力的 13 j 均值淨值（base 與 M2，float32），其餘在 scratchpad（題目結束時刪除）。python3 -I -B keep_navs.py"""
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/integrate_o20')
N = pd.read_parquet(SCR / 'navs_v7x_o20.parquet')
k = [c for c in N.columns if '|100萬|std' in c]
(HERE / 'data/v7x_navs_jmean_subset.parquet').unlink()
N[k].astype('float32').to_csv(HERE / 'data/v7x_navs_jmean_100w.csv.gz', float_format='%.6g')
print(len(k), (HERE / 'data/v7x_navs_jmean_100w.csv.gz').stat().st_size)
