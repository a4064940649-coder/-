#!/usr/bin/env python3
"""(1) 除權息日 ret=0 的敏感度；(2) rev_announce A／B／AB 版獨立重建（期限日曆自己算）。輸出 results_extra.json。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ia_core as K

T0 = time.time(); res = {}
b, s = K.rates(0.001)
YL = K.yoy_from_long(); Y5 = K.yoy_from_rev5y()
SEG = {'P1': ('2006-04-01', '2020-12-31'), 'P2': ('2021-01-01', '2026-10-31'), 'P2b': ('2021-03-01', '2026-10-31'), 'V24': ('2024-01-01', '2026-10-31')}

def evaluate(S, n, G=None, plan=None):
    t0 = K.IDX.get_loc(S.sig_days[0]); t1 = K.IDX.get_loc(S.end_day); sd = S.sig_days + [S.end_day]
    pl = plan if plan is not None else S.plan(n)
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1, ret_override=G)
    nav, _ = K.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1, ret_override=G)
    return nav, ew, sd

# ---------- (1) 除權息日 ----------
F = K.Fn
gap_mean = 0.00576   # checks.py：2021-10 起 flag 2 除權息日「開盤／參考價 −1」平均（真實值）
G1 = K.GROW.copy()
m1 = (F == 1) & K.TRADED & np.isfinite(K.On)
G1[m1] = (K.Cn[m1] / K.On[m1]) * (1 + gap_mean)
G2 = K.GROW.copy(); G2[(F == 1) & K.TRADED] = 1.00163   # 用真實除權息日平均報酬
exd = {}
for tag, Y, st in (('P1_long', YL, '2005-01-01'), ('P2_rev5y', Y5, '2021-01-01')):
    S = K.Strategy(Y, st)
    for n in (30, 10):
        for gname, G in (('base_ret0', None), ('fix_open_close_gap', G1), ('fix_mean', G2)):
            nav, ew, sd = evaluate(S, n, G)
            segs = ('P1',) if tag == 'P1_long' else ('P2b',)
            o = {k: K.short(K.stats(nav, *SEG[k], sd, K.E50, ew)) for k in segs}
            exd[f'{tag}|n{n}|{gname}'] = o
            print('exday', tag, n, gname, o, flush=True)
res['exday_sensitivity'] = exd

# ---------- (2) rev_announce A／B／AB ----------
sl = pd.read_csv(K.X.parent / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
FIN = set(sl.index[sl['industry'] == '金融保險業']) | {c for c in K.COLS if c[:2] == '28' or c in ('5876', '5880')}
RA = K.HERE.parent / 'rev_announce'
pdl = json.loads((RA / 'p_delay.json').read_text(encoding='utf-8'))        # 資料：2026-09 各產業延後機率（rev_announce 由公告時點資料估計）
INS15 = set(pdl['INS15'])
fs = pd.read_csv(RA / 'firstseen_2026-09.csv', dtype=str, index_col=0)     # 資料：2026-09 第一次被抓到的時間分類
ann_ok_2609 = set(fs.index[fs['cat'] == 'A_before'])
ind = [sl['industry'].get(c) if isinstance(sl['industry'].get(c), str) else '（stock_list 無）' for c in K.COLS]
pcol = np.array([1.0 if c in FIN else pdl['p_by_industry'].get(i, pdl['p_overall']) for c, i in zip(K.COLS, ind)])
is15 = np.array([c in INS15 for c in K.COLS])
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']
TCAL = pd.DatetimeIndex(sorted(set(tx) | set(K.IDX)))
def case_of(d):
    rm = pd.Period(d, 'M') - 1
    if str(rm) in ('2019-01', '2023-03'): return 'D>S'
    if str(rm) == '2020-01': return 'D=S'
    tenth = pd.Timestamp(d.year, d.month, 10)
    D = TCAL[TCAL >= tenth][0]
    return 'D<S' if D < d else ('D=S' if D == d else 'D>S')
S_tmp = K.Strategy(YL, '2005-01-01')
cases = {d: case_of(d) for d in S_tmp.sig_days + [S_tmp.end_day]}
res['deadline_case_counts'] = pd.Series(list(cases.values())).value_counts().to_dict()
cal_ra = pd.read_csv(RA / 'deadline_calendar.csv', dtype=str).set_index('signal_day')['case']
mism = [str(d.date()) for d, c in cases.items() if str(d.date()) in cal_ra.index and cal_ra[str(d.date())] != c]
res['deadline_case_mismatch_vs_rev_announce'] = mism
print('cases', res['deadline_case_counts'], 'mismatch', mism, flush=True)

def make_unavail(mode, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    cache = {}
    def U(d):
        if d in cache: return cache[d]
        rm = str(pd.Period(d, 'M') - 1); c = cases.get(d, 'D<S')
        u = np.zeros(K.N, bool)
        if rm == '2026-09':
            u = ~np.array([x in ann_ok_2609 for x in K.COLS])
        elif c == 'D>S':
            u[:] = True
        elif c == 'D=S':
            if mode == 'worst': u[:] = True
            elif mode == 'mc': u = rng.random(K.N) < pcol
        if '2026-01' <= rm <= '2026-08': u |= is15
        cache[d] = u
        return u
    return U

ab = {}
NMC = 40
for n in (10, 30):
    variants = {'BASE': dict(), 'A_exFIN': dict(fin_exclude=FIN), 'B_worst': dict(unavail=make_unavail('worst')),
                'AB_worst': dict(fin_exclude=FIN, unavail=make_unavail('worst'))}
    for v, kw in variants.items():
        S = K.Strategy(YL, '2005-01-01', **kw)
        nav, ew, sd = evaluate(S, n)
        ab[f'{v}_n{n}'] = {k: K.short(K.stats(nav, *SEG[k], sd, K.E50, ew)) for k in ('P1', 'P2', 'V24')}
        print('AB', v, n, {k: (x['CAGR'], x['t_0050']) for k, x in ab[f'{v}_n{n}'].items()}, round(time.time() - T0), flush=True)
    for v, fx in (('B_mc', None), ('AB_mc', FIN)):
        rets = []
        for seed in range(NMC):
            S = K.Strategy(YL, '2005-01-01', fin_exclude=fx, unavail=make_unavail('mc', seed))
            nav, ew, sd = evaluate(S, n)
            rets.append(nav.pct_change())
        avg = pd.concat(rets, axis=1).mean(axis=1).fillna(0)
        navm = (1 + avg).cumprod()
        cg = []
        for r in rets:
            nj = (1 + r.fillna(0)).cumprod(); cg.append([K.stats(nj, *SEG[k], sd)['CAGR'] for k in ('P1', 'P2')])
        cg = np.array(cg) * 100
        ab[f'{v}_n{n}'] = {k: K.short(K.stats(navm, *SEG[k], sd, K.E50, ew)) for k in ('P1', 'P2', 'V24')}
        ab[f'{v}_n{n}']['mc_CAGR_p05_p95'] = {'P1': [round(float(np.percentile(cg[:, 0], 5)), 1), round(float(np.percentile(cg[:, 0], 95)), 1)],
                                              'P2': [round(float(np.percentile(cg[:, 1], 5)), 1), round(float(np.percentile(cg[:, 1], 95)), 1)]}
        print('AB', v, n, {k: (x['CAGR'], x['t_0050']) for k, x in ab[f'{v}_n{n}'].items() if k in SEG}, ab[f'{v}_n{n}']['mc_CAGR_p05_p95'], round(time.time() - T0), flush=True)
res['rev_announce_rebuild'] = ab
(K.HERE / 'results_extra.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print('done', round(time.time() - T0))
