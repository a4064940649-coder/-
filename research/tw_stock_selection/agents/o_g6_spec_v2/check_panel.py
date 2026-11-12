#!/usr/bin/env python3
"""SPEC v2 小型驗證（只讀，不算報酬）：兩個價格面板的日期、欄位、GOOD 日，與 TAIEX 日曆的差異。python3 -I -B check_panel.py"""
import json
from pathlib import Path
import pandas as pd
AG = Path('/home/user/-/research/tw_stock_selection/agents')
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
out = {}; idx = {}
for nm, p in (('paper', SCR / 'panel/close_raw.parquet'), ('v7', AG / 'o_v7_listing_exec_engine/data/close_raw.parquet')):
    C = pd.read_parquet(p); nobs = C.notna().sum(axis=1)
    out[nm] = dict(n_days=len(C), first=str(C.index[0].date()), last=str(C.index[-1].date()), n_cols=C.shape[1],
                   n_cols_00=int(sum(c.startswith('00') for c in C.columns)), non4=[c for c in C.columns if len(c) != 4][:10],
                   not_good_days=[str(d.date()) for d in C.index[nobs < 300]][:20],
                   n_days_2024_01_02_to_02_14=int(((C.index >= '2024-01-02') & (C.index <= '2024-02-14')).sum()),
                   cols_sorted=list(C.columns) == sorted(C.columns))
    idx[nm] = C.index
a, b = idx['paper'], idx['v7']
out['idx_equal'] = bool(len(a) == len(b) and (a == b).all())
out['only_v7'] = [str(d.date()) for d in b.difference(a)][:40]; out['n_only_v7'] = len(b.difference(a)); out['n_only_paper'] = len(a.difference(b))
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']
out['taiex_first_last_n'] = [str(tx.min().date()), str(tx.max().date()), len(tx)]
out['v7_not_in_taiex'] = [str(d.date()) for d in b.difference(pd.DatetimeIndex(tx))][:20]
out['taiex_not_in_v7_2006on'] = [str(d.date()) for d in pd.DatetimeIndex(tx[tx >= b[0]]).difference(b)][:20]
pj = json.load(open(AG / 'rev_announce/p_delay.json')); out['p_delay_keys'] = {k: (v if not isinstance(v, (list, dict)) else len(v)) for k, v in pj.items()}
Path(__file__).with_name('check_panel.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False))
