#!/usr/bin/env python3
"""除錯：比對單一缺口格的逐月來源（我的提及 vs 研究員的 mentions_P3 輸出資料）。"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]] + sys.argv[1:]
import importlib.util
spec = importlib.util.spec_from_file_location('rp', HERE / 'r_proxy.py'); rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
R = pd.read_parquet(HERE.parent / 'data' / 'mentions_P3.parquet')
R['p'] = pd.PeriodIndex(R.ym, freq='M')
for arg in sys.argv[1:]:
    sig, code = arg.split(',')
    d = pd.Timestamp(sig); m = pd.Period(d, 'M'); cut = d + pd.Timedelta(hours=13, minutes=30)
    print('=====', sig, code, rp.proxies(d, code))
    for p in [m - 1, m - 2, m - 3, m - 4]:
        print('  month', p, 'off', rp.off_yoy(d, code, p), 'news', rp.news_val(d, code, p), 'base', rp.LV.get((code, p - 12)))
        g = rp.MG.get((code, p))
        if g is not None:
            g = g[g.ts <= cut]
            print('   mine:', g[['src', 'ts', 'level_k', 'yoy', 'snip']].head(4).to_string().replace('\n', '\n   '))
        r = R[(R.code == code) & (R.p == p) & (R.ts <= cut)].sort_values('ts')
        if len(r): print('   theirs:', r[['src', 'ts', 'rev_k', 'yoy', 'snip']].head(4).to_string().replace('\n', '\n   '))
