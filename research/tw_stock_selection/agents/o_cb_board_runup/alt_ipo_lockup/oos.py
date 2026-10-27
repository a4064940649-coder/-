# -*- coding: utf-8 -*-
"""樣本外（只跑一次）：python3 -I -B oos.py v2｜v3。斷言 PREREG 已存在且 sha256 相符。
v2＝main_v2 正式引擎；v3＝o_infra_v3_rebuild 的 V3（P1 下市股價格、R1 營收、X1 併購出清、Dhi 處置）。"""
import os, sys, json, hashlib
MODE = sys.argv[1]; assert MODE in ('v2', 'v3')
if MODE == 'v3': os.environ['ALT_V3'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert RES.get('PREREG') and hashlib.sha256(RES['PREREG'].encode('utf-8')).hexdigest() == RES['PREREG_sha256'], '先執行 prereg.py'
import numpy as np, pandas as pd
import alib as A
V, K, O = A.V, A.K, A.O
IDX, N, COLS = A.IDX, A.N, A.COLS
assert A.sha(HERE / 'listings.csv') in RES['PREREG'], 'listings.csv 與凍結版不同'

SEGS = {'SEL': ('2006-04-01', '2019-04-30'), 'VAL': ('2019-05-01', '2026-10-12'), 'VAL1': ('2019-05-01', '2023-12-31'),
        'HOLD': ('2024-01-01', '2026-10-12'), 'ALL': ('2006-04-01', '2026-10-12')}

def avoid(kindmask, lo=2, hi=4):
    def f(d, e, e2):
        a = A.age_years(d); return (a >= lo) & (a < hi) & kindmask
    return f

if MODE == 'v3':
    npatch = O.apply_patch()
    E = O.load_events(); H = O.load_disposal_hist()
    R3 = json.loads((A.AG / 'o_infra_v3_rebuild' / 'results.json').read_text(encoding='utf-8'))
    EXTRA = {pd.Timestamp(k): v for k, v in R3['PREREG_params']['D_hi_extra'].items()}
    O.set_disposal('all_plus', H, EXTRA)
    YL, Y5, _ = V.load_revenue(); YR = O.load_rdel()
    S = O.V3('gapfix', YR, Y5, exfin=True)
    XR = O.exit_rules(E, 'main')
    sim = lambda plan, b, s, t0, t1: O.simulate(plan, b, s, t0, t1, xrules=XR)
else:
    npatch = 0
    S = A.strategy(exfin=True)
    sim = lambda plan, b, s, t0, t1: K.simulate(plan, b, s, t0, t1)
t0, t1, sd = S.bounds()
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}

def monthly(nav):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index]); nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]
def win(s, a, b):
    s = s.dropna(); a, b = pd.Timestamp(a), pd.Timestamp(b)
    return s[(s.index >= a) & (s.index <= b - pd.Timedelta(days=20))]

base = S.plan(10, 'official')
plans = {'base': base}
plans['A5'], sk = A.plan_avoid(S, 10, avoid(A.IPOMASK))
plans['PLACEBO_transfer'], _ = A.plan_avoid(S, 10, avoid(A.TRMASK))
ew, _ = sim([(p[0], p[2]) for p in base], 0, 0, t0, t1)
navs = {'EW': ew}; out = {'mode': MODE, 'patched_cells': npatch, 'runs': {}}
E50m = None
for c, (bb, ss) in COST.items():
    for k, pl in plans.items():
        nav, lg = sim([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
        navs[f'{k}|{c}'] = nav
    m0 = monthly(navs[f'base|{c}'])
    bd = K.E50.reindex(m0.index.union([sd[-1]])); b50 = (K.E50.reindex(pd.DatetimeIndex([d for d in sd if d in navs['EW'].index])))
    b50 = (b50.shift(-1) / b50 - 1).iloc[:-1]
    for k in plans:
        nav = navs[f'{k}|{c}']; m = monthly(nav)
        rec = {}
        for sg, (a, b) in SEGS.items():
            st = V.seg_stats(nav, ew, a, b, sd, yearly=(sg == 'ALL'))
            x = win(m - b50, a, b)
            st['ex0050_mean_pp'] = float(x.mean() * 100); st['ex0050_sd_pp'] = float(x.std(ddof=1) * 100)
            if k != 'base':
                st['paired'] = A.msum(win(m - m0, a, b))
            rec[sg] = st
        out['runs'][f'{k}|{c}'] = rec

# ---------- 受影響股與剔除最大貢獻 3 檔 ----------
def contrib_table(pl_v, pl_b, a, b):
    _, per_v = V.stock_contrib(pl_v, t1); _, per_b = V.stock_contrib(pl_b, t1)
    dv = {d: (pk, r) for d, pk, r in per_v}; dbb = {d: (pk, r) for d, pk, r in per_b}
    rows = []
    for d in dbb:
        if not (pd.Timestamp(a) <= d <= pd.Timestamp(b) - pd.Timedelta(days=20)) or d not in dv: continue
        pv, rv = dv[d]; pb, rb = dbb[d]
        sv, sb = set(pv), set(pb)
        for j, r in zip(pv, rv):
            if j not in sb: rows.append(dict(d=d, code=COLS[j], side='added', r=r, c=r / len(pv)))
        for j, r in zip(pb, rb):
            if j not in sv: rows.append(dict(d=d, code=COLS[j], side='skipped', r=r, c=-r / len(pb)))
    return pd.DataFrame(rows)
rob = {}
for sg in ('SEL', 'VAL', 'ALL'):
    T = contrib_table(plans['A5'], base, *SEGS[sg])
    if not len(T): continue
    byc = T.groupby('code').c.sum().sort_values(ascending=False)
    top3 = list(byc.index[:3])
    g = T.groupby('d').c.sum()
    g_ex = T[~T.code.isin(top3)].groupby('d').c.sum()
    dall = win(monthly(navs['A5|normal']) - monthly(navs['base|normal']), *SEGS[sg])
    # 把剔除 3 檔的貢獻從配對差中扣掉（其餘月份配對差不變）
    adj = dall.copy()
    for d, v in (g - g_ex.reindex(g.index).fillna(0)).items():
        if d in adj.index: adj[d] -= v
    sk_ = T[T.side == 'skipped']; ad_ = T[T.side == 'added']
    rob[sg] = dict(n_skipped=int(len(sk_)), n_added=int(len(ad_)), skipped_mean_ret_pp=float(sk_.r.mean() * 100), added_mean_ret_pp=float(ad_.r.mean() * 100),
                   gross_diff_sum_pp=float(T.c.sum() * 100), top3=[(c, round(float(byc[c]) * 100, 2)) for c in top3],
                   bottom3=[(c, round(float(v) * 100, 2)) for c, v in byc.tail(3).items()],
                   paired_ex_top3=A.msum(adj), skipped_codes=sorted(set(sk_.code)),
                   skipped_industry=pd.Series([V.SL['industry'].get(c, '?') for c in sk_.code]).value_counts().head(8).to_dict(),
                   by_year=T.groupby(T.d.dt.year).c.sum().mul(100).round(2).to_dict())
    T.to_csv(HERE / f'oos_{MODE}_affected_{sg}.csv', index=False)
out['robust'] = rob
out['paired_by_year_normal'] = (monthly(navs['A5|normal']) - monthly(navs['base|normal'])).dropna().groupby(lambda x: x.year).sum().mul(100).round(3).to_dict()
out['young_share_picks'] = {sg: float(np.mean([np.mean(((A.age_years(p[3])[p[1]] >= 2) & (A.age_years(p[3])[p[1]] < 4) & A.IPOMASK[p[1]])) for p in base
                                               if pd.Timestamp(SEGS[sg][0]) <= p[3] <= pd.Timestamp(SEGS[sg][1]) and len(p[1])])) for sg in SEGS}
A.dump(out, f'oos_{MODE}.json')
pd.DataFrame(navs).to_parquet(HERE / f'oos_navs_{MODE}.parquet')
def brief(k):
    r = out['runs'][k]; return {sg: (round(v['CAGR'], 2), round(v['maxDD'], 1), round(v['t_0050'], 2), v.get('paired', {}).get('mean_pp'), v.get('paired', {}).get('t')) for sg, v in r.items()}
for k in out['runs']: print(k, brief(k))
print(json.dumps(rob, ensure_ascii=False, default=A.conv)[:3000])
