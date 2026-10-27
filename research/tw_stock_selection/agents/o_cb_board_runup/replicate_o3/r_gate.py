#!/usr/bin/env python3
"""複製員：閘門 (a) 事件數。"""
import json
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
V = pd.read_parquet(P / 'value.parquet'); C = pd.read_parquet(P / 'close_raw.parquet')
good = C.notna().sum(axis=1) >= 300
Vg = V[good]
V60 = Vg.rolling(60, min_periods=50).mean()
SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
FIN = set(SL.index[SL['industry'] == '金融保險業']) | {c for c in C.columns if c[:2] == '28' or c in ('5876', '5880')}
ev = pd.read_csv(HERE / 'my_events.csv', dtype={'code': str}, parse_dates=['board'])
f = pd.read_csv(HERE.parent.parent / 'o_issuance_avoid/fsc_cases_all.csv', dtype=str)
f['item2'] = f['item'].fillna('').str.replace(r'\s+', '', regex=True)
f['recv'] = pd.to_datetime(f['recv'], errors='coerce')
cb = f[f.item2.str.contains('轉換公司債') & ~f.item2.str.contains('海外|私募|補辦|外國') & (f.ctype == '上市')].copy()
e = ev[(ev.kind == 'CB') & (ev.board >= '2010-01-01') & (ev.board <= '2018-12-31')].copy()
def liq(r):
    if r.code not in V60.columns: return np.nan
    s = V60[r.code]; s = s[s.index < r.board]
    return s.iloc[-1] if len(s) else np.nan
def listed(r):
    if r.code not in C.columns: return False
    s = C[r.code]; s = s[(s.index < r.board) & (s.index >= r.board - pd.Timedelta(days=30))]
    return s.notna().any()
e['v60'] = [liq(r) for r in e.itertuples()]
e['listed'] = [listed(r) for r in e.itertuples()]
e['fin'] = e.code.isin(FIN)
def outcome(r):
    if r.board.year == 2016 or (r.board.year == 2015 and r.board.month >= 11): 
        g = cb[(cb.code == r.code) & (cb.recv >= r.board) & ((cb.recv - r.board).dt.days <= 180)]
        if not len(g): return 'fsc_gap'
    g = cb[(cb.code == r.code) & (cb.recv >= r.board) & ((cb.recv - r.board).dt.days <= 180)]
    if not len(g): return 'not_filed'
    return 'issued' if (g.result == '生效').any() else 'withdrawn'
e['outcome'] = [outcome(r) for r in e.itertuples()]
ok = e.listed & (e.v60 >= 3e7)
res = {'n_all_CB_2010_2018': len(e), 'gate_a_exfin': int((ok & ~e.fin).sum()), 'gate_a_incfin': int(ok.sum()),
       'by_year': e[ok & ~e.fin].groupby(e.board.dt.year).size().to_dict(),
       'by_outcome': e[ok & ~e.fin].outcome.value_counts().to_dict()}
# 上界：FSC 2010～2018 上市國內 CB（流動性、非金融）＋ FSC 缺年 2016 用董事會事件
fx = cb[(cb.recv >= '2010-01-01') & (cb.recv <= '2018-12-31')].drop_duplicates(['code', 'recv']).copy()
def liq2(code, d):
    if code not in V60.columns: return np.nan
    s = V60[code]; s = s[s.index < d]; return s.iloc[-1] if len(s) else np.nan
fx['v60'] = [liq2(c, d - pd.Timedelta(days=19)) for c, d in zip(fx.code, fx.recv)]
fx_ok = fx[(fx.v60 >= 3e7) & ~fx.code.isin(FIN)]
nf = e[ok & ~e.fin & e.outcome.isin(['not_filed', 'fsc_gap'])]
res['fsc_2010_2018_ok'] = len(fx_ok); res['upper_bound'] = len(fx_ok) + len(nf)
print(json.dumps(res, default=str))
