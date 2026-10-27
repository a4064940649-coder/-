#!/usr/bin/env python3
"""診斷（看完主結果後才寫，不影響判定）：不設 20 檔上限的「日曆時間事件組合」——每日把所有進行中的事件交易等權（每日再平衡、
收盤到收盤近似，進場日用開盤→收盤、出場日用前收→開盤），毛報酬；用來區分「訊號本身沒有優勢」與「20 檔上限抽樣」。"""
import json, sys, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa
L = M.L
IDX, COLS = L.IDX, L.COLS
GOOD = (L.adj.notna().sum(axis=1) >= 300).to_numpy(); gi = np.where(GOOD)[0]; G = IDX[gi]; T = len(G)
ACg = L.AC[gi]; AOg = L.AO[gi]
CI = {c: j for j, c in enumerate(COLS)}
navs = pd.read_parquet(HERE / 'nav.parquet'); EW = navs['EW_pool'].dropna(); E50 = navs['0050'].dropna()
_s = pd.Series(G, index=G)
MD = pd.DatetimeIndex([g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([G.year, G.month]) if (g.dt.day >= 11).any()])
MD = MD[MD >= pd.Timestamp('2004-07-01')]
def mwin(nav):
    nd = nav.reindex(MD); return (nd.shift(-1) / nd - 1).iloc[:-1]
W = {'select': ('2005-01-01', '2019-04-30'), 'valid': ('2019-05-01', '2026-10-31'), 'y2024_2025': ('2024-01-01', '2025-12-31')}
out = {}
for vn in ('V1X1', 'V2X1', 'V1X2', 'V2X2'):
    tr = pd.read_csv(HERE / f'events_{vn}.csv.gz', dtype={'code': str})
    S = np.zeros(T); Cn = np.zeros(T)
    for r in tr.itertuples():
        j = CI[r.code]; e, x = int(r.e), int(r.x)
        if x <= e: continue
        rr = np.empty(x - e + 1)
        rr[0] = ACg[e, j] / AOg[e, j] - 1
        if x - e > 1: rr[1:x - e] = ACg[e + 1:x, j] / ACg[e:x - 1, j] - 1
        rr[-1] = (AOg[x, j] if r.why != 'open_at_end' else ACg[x, j]) / ACg[x - 1, j] - 1
        rr = np.nan_to_num(rr)
        S[e:x + 1] += rr; Cn[e:x + 1] += 1
    dr = np.where(Cn > 0, S / np.maximum(Cn, 1), 0.0)
    nav = pd.Series(np.cumprod(1 + dr), index=G)
    o = {'mean_active': round(float(Cn[G >= pd.Timestamp('2005-01-01')].mean()), 1), 'share_days_no_position': round(float((Cn[G >= pd.Timestamp('2005-01-01')] == 0).mean()), 3)}
    for k, (a, b) in W.items():
        m = (nav.index >= a) & (nav.index <= b); n_ = nav[m]; yrs = (n_.index[-1] - n_.index[0]).days / 365.25
        pr = mwin(nav); pr = pr[(pr.index >= a) & (pr.index <= b)].dropna()
        ex = pr - mwin(EW).reindex(pr.index); ex5 = pr - mwin(E50).reindex(pr.index)
        o[k] = {'CAGR_gross': round(float((n_.iloc[-1] / n_.iloc[0]) ** (1 / yrs) - 1), 4),
                'maxDD': round(float((n_ / n_.cummax() - 1).min()), 4), 't_vs_EW_m': L.tstat(ex), 't_vs_0050_m': L.tstat(ex5),
                'excess_vs_EW_pct_m': round(float(ex.mean() * 100), 2)}
    out[vn] = o
    print(vn, o, flush=True)
(HERE / 'results_diag.json').write_text(json.dumps({'calendar_time_event_portfolio_gross_uncapped': out,
    'note': '看完主結果後才做的診斷，不參與判定；毛報酬、每日等權再平衡、不設檔數上限'}, ensure_ascii=False, indent=1), encoding='utf-8')
