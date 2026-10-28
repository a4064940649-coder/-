#!/usr/bin/env python3
"""步驟 1：缺口格清冊（不看報酬）。2013-01～2021-12 每個換股訊號日，前 300 池中營收以外都合格（ok_other）但 V4 的 y3 缺值的股月。
分類：merger（併購下市）／deterior（經營惡化等其他下市）／nondelist（非下市：逐格查原因）。
標記：V60 排名、52 週比百分位、B10（非 AB）實際入選、缺的是哪幾個營收月、缺水準或缺基期。輸出 gap_cells_2013_2021.csv。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rglib as G
K, L, V = G.K, G.L, G.V
b = G.base()
S = G.mk()
B10 = pd.read_csv(G.V4DIR / 'plan_V4_B10.csv', dtype=str)
P4 = pd.read_csv(G.V4DIR / 'plan_V4.csv', dtype=str)
pk10 = {r.signal: set(r.picks.split(',')) for r in B10.itertuples()}
pk4 = {r.signal: set(r.picks.split(',')) for r in P4.itertuples()}
# 營收水準（長期 sii＋taiwanan＋rev_5y）用來判斷缺水準或缺基期
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str}); d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d['pri'] = (d['market'] != 'sii').astype(int); d = d.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
LV = d.pivot(index='p', columns='code', values='revenue_k_twd')
r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str}); r5['p'] = pd.PeriodIndex(r5['ym'], freq='M')
L5 = r5.drop_duplicates(['p', 'code'], keep='last').pivot(index='p', columns='code', values='revenue')
def lvl(c, p):
    for X in (LV, L5):
        if c in X.columns and p in X.index and np.isfinite(X.at[p, c]) and X.at[p, c] > 0: return True
    return False
rows = []
for dd in S.sig_days:
    if not (G.W0 <= dd <= G.W1): continue
    cs = S.cross_section(dd)
    ok = cs['ok_other']; y3 = S.y3(dd); y3b = S.y3(dd, 1)
    miss = ok & ~np.isfinite(y3)
    t = cs['t']; v60 = S.V60[t]
    order = np.argsort(-np.where(np.isfinite(v60), v60, -np.inf), kind='stable')
    rank = np.empty(K.N, int); rank[order] = np.arange(1, K.N + 1)
    s52p = pd.Series(np.where(ok, cs['s52'], np.nan)).rank(pct=True).to_numpy()
    Y = S.Y5 if dd >= V.SWITCH else S.YL
    m = pd.Period(dd, 'M')
    sd = str(dd.date())
    for j in np.where(miss)[0]:
        c = K.COLS[j]
        mm = [m - 1, m - 2, m - 3]
        ym = [bool(c in Y.columns and p in Y.index and np.isfinite(Y.at[p, c])) for p in mm]
        lv = [lvl(c, p) for p in mm]; lb = [lvl(c, p - 12) for p in mm]
        rows.append(dict(sig=sd, code=c, name=L.DL_NAME.get(c, V.SL['name'].get(c) if c in V.SL.index else ''),
                         type=G.gap_type(c, dd), liq_rank=int(rank[j]), s52=float(cs['s52'][j]), s52_pct=float(s52p[j]),
                         y3_back1_finite=bool(np.isfinite(y3b[j])),
                         n_yoy_have=int(sum(ym)), m1_yoy=ym[0], m2_yoy=ym[1], m3_yoy=ym[2],
                         n_level_miss=int(sum(not x for x in lv)), n_base_miss=int(sum(not x for x in lb)),
                         in_B10=c in pk10.get(sd, set()), in_V4=c in pk4.get(sd, set()), src='Y5' if dd >= V.SWITCH else 'YL'))
X = pd.DataFrame(rows)
X.to_csv(HERE / 'gap_cells_2013_2021.csv', index=False)
print(len(X)); print(X.groupby([X.sig.str[:4], 'type']).size().unstack(fill_value=0))
print('B10 picks', X.in_B10.sum(), X[X.in_B10].groupby('type').size().to_dict())
print(X[X.type == 'nondelist'].groupby(['code', 'name']).size())
print(X.groupby('type')[['n_level_miss', 'n_base_miss', 'n_yoy_have']].mean())
