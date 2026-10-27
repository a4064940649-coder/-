# -*- coding: utf-8 -*-
"""反方審查（資料與統計，o3）：只讀研究員檔案，不寫回任何研究員檔案。
1) A5 配對差的自相關與 Newey-West t（v2／v3 淨值）
2) A5 挑選期配對差對單一年份（2017）的依賴
3) A1（跳過 1～3 年新上市股）在挑選期的集中度：剔除貢獻最大 3 檔；以及 A1 在驗證期（研究員未看）的方向
4) 多重檢定：~36 個檢定下 A1 t=-3.04 的 Bonferroni／Holm 門檻
5) 閘門上界：FSC 自身缺月（2010-08～12、2013-12 等）未計入
用法：python3 -I -B review_stat_o3.py"""
import os, sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
ALT = HERE / 'alt_ipo_lockup'
sys.path.insert(0, str(ALT))
import numpy as np, pandas as pd

def nw_t(x, L=None):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if L is None: L = int(math.floor(4 * (n / 100) ** (2 / 9)))
    m = x.mean(); e = x - m
    g0 = (e @ e) / n; s = g0
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); s += 2 * w * (e[l:] @ e[:-l]) / n
    return float(m / math.sqrt(s / n)), L

def ac1(x):
    x = pd.Series(x).dropna(); return float(x.autocorr(1))

out = {}
SEGS = {'SEL': ('2006-04-01', '2019-04-30'), 'VAL': ('2019-05-01', '2026-10-12'), 'ALL': ('2006-04-01', '2026-10-12')}
import alib as A
V, K = A.V, A.K
S = A.strategy(exfin=True)
t0, t1, sd = S.bounds()
def monthly(nav):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index]); nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]
def win(s, a, b):
    s = s.dropna(); a, b = pd.Timestamp(a), pd.Timestamp(b)
    return s[(s.index >= a) & (s.index <= b - pd.Timedelta(days=20))]

# ---- 1) A5 paired NW t ----
for m in ('v2', 'v3'):
    nv = pd.read_parquet(ALT / f'oos_navs_{m}.parquet')
    d = monthly(nv['A5|normal']) - monthly(nv['base|normal'])
    for sg, (a, b) in SEGS.items():
        x = win(d, a, b)
        out[f'A5_{m}_{sg}'] = dict(mean_pp=float(x.mean() * 100), t_iid=A.tstat(x), t_nw=nw_t(x)[0], ac1=ac1(x),
                                   n=int(len(x)), n_nonzero=int((x.abs() > 1e-9).sum()))
    xs = win(d, *SEGS['SEL'])
    yr = xs.groupby(xs.index.year).sum() * 100
    out[f'A5_{m}_SEL_by_year'] = yr.round(3).to_dict()
    ex17 = xs[xs.index.year != 2017]
    out[f'A5_{m}_SEL_ex2017'] = dict(mean_pp=float(ex17.mean() * 100), t=A.tstat(ex17))

# ---- 3) A1 full period ----
import iterate as IT
def contrib_table(pl_v, pl_b, a, b):
    _, per_v = V.stock_contrib(pl_v, t1); _, per_b = V.stock_contrib(pl_b, t1)
    dv = {d: (pk, r) for d, pk, r in per_v}; dbb = {d: (pk, r) for d, pk, r in per_b}
    rows = []
    for d in dbb:
        if not (pd.Timestamp(a) <= d <= pd.Timestamp(b) - pd.Timedelta(days=20)) or d not in dv: continue
        pv, rv = dv[d]; pb, rb = dbb[d]; sv, sb = set(pv), set(pb)
        for j, r in zip(pv, rv):
            if j not in sb: rows.append(dict(d=d, code=A.COLS[j], side='added', r=r, c=r / len(pv)))
        for j, r in zip(pb, rb):
            if j not in sv: rows.append(dict(d=d, code=A.COLS[j], side='skipped', r=r, c=-r / len(pb)))
    return pd.DataFrame(rows)

bb, ss = K.rates(0.001)
base = S.plan(10, 'official')
nav0, _ = K.simulate([(p[0], p[1]) for p in base], bb, ss, t0, t1)
m0 = monthly(nav0)
for ver in ('A1', 'A3', 'A5'):
    pl, sk = A.plan_avoid(S, 10, IT.VERS[ver])
    nav, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
    d = monthly(nav) - m0
    for sg, (a, b) in SEGS.items():
        x = win(d, a, b)
        T = contrib_table(pl, base, a, b)
        rec = dict(mean_pp=float(x.mean() * 100), t_iid=A.tstat(x), t_nw=nw_t(x)[0], ac1=ac1(x), n=int(len(x)),
                   n_nonzero=int((x.abs() > 1e-9).sum()))
        if len(T):
            byc = T.groupby('code').c.sum().sort_values()
            # A1 為負：剔除「讓配對差最負」的 3 檔（被跳過的贏家）
            worst3 = list(byc.index[:3]); best3 = list(byc.index[-3:])
            g = T.groupby('d').c.sum()
            for lab, drop in (('ex_worst3', worst3), ('ex_best3', best3)):
                g_ex = T[~T.code.isin(drop)].groupby('d').c.sum()
                adj = x.copy()
                for dd, v in (g - g_ex.reindex(g.index).fillna(0)).items():
                    if dd in adj.index: adj[dd] -= v
                rec[lab] = dict(codes=drop, contrib_pp=[round(float(byc[c]) * 100, 2) for c in drop], mean_pp=float(adj.mean() * 100), t=A.tstat(adj))
            skp = T[T.side == 'skipped']; add = T[T.side == 'added']
            rec['skipped_mean_pp'] = float(skp.r.mean() * 100); rec['added_mean_pp'] = float(add.r.mean() * 100)
            rec['n_skipped_rows'] = int(len(skp))
            rec['gross_diff_sum_pp'] = float(T.c.sum() * 100); rec['paired_sum_pp'] = float(x.sum() * 100)
            rec['by_year_pp'] = T.groupby(T.d.dt.year).c.sum().mul(100).round(2).to_dict()
        out[f'{ver}_v2_{sg}'] = rec
    print(ver, {sg: (round(out[f'{ver}_v2_{sg}']['mean_pp'], 3), round(out[f'{ver}_v2_{sg}']['t_iid'], 2), round(out[f'{ver}_v2_{sg}']['t_nw'], 2)) for sg in SEGS}, flush=True)

# ---- 4) multiple testing ----
from statistics import NormalDist
nd = NormalDist()
for k in (6, 36):
    out[f'bonferroni_two_sided_t_crit_k{k}'] = nd.inv_cdf(1 - 0.05 / (2 * k))

# ---- 5) gate upper bound: FSC own coverage gaps ----
FCOV = json.load(open(HERE.parent / 'o_issuance_avoid' / 'events_build_summary.json'))['fsc_year_coverage']
miss_days = 0
for y in range(2010, 2019):
    if str(y) not in FCOV: miss_days += 365; continue
    a, b = pd.Timestamp(FCOV[str(y)][0]), pd.Timestamp(FCOV[str(y)][1])
    miss_days += (a - pd.Timestamp(f'{y}-01-01')).days + (pd.Timestamp(f'{y}-12-31') - b).days
out['fsc_missing_days_2010_2018'] = miss_days
S_ = json.load(open(HERE / 'build_summary.json'))
ub = S_['gate_a']['upper_bound']
fsc_n = ub['fsc_liquid_exfin']
cov_days = 9 * 365 - miss_days
out['fsc_liquid_rate_per_covered_year'] = fsc_n / (cov_days / 365)
out['ub_scaled_if_fsc_complete'] = fsc_n * 9 * 365 / cov_days
print(json.dumps(out, ensure_ascii=False, indent=1, default=A.conv))

# ---- 6) 面板首日群聚（2010-01-04 有 28 檔同日「上市」，全標為轉上市）：疑為資料起點而非真實上市日 ----
if __name__ == '__main__' and os.environ.get('REVIEW_PART6', '1') == '1':
    LSr = pd.read_csv(ALT / 'listings.csv', dtype={'code': str}, parse_dates=['first'])
    nw = LSr[LSr.new]
    cl = nw['first'].value_counts(); cl = cl[cl >= 10]
    out6 = {'clusters': {str(k.date()): dict(n=int(v), kinds=nw[nw['first'] == k].kind.value_counts().to_dict()) for k, v in cl.items()}}
    bad = set(nw[nw['first'].isin(cl.index)].code)
    BAD = np.isin(A.COLS, list(bad))
    def mkA1clean(lo, hi):
        def f(d, e, e2):
            a = A.age_years(d); return (a >= lo) & (a < hi) & A.NEWMASK & ~BAD
        return f
    pl, _ = A.plan_avoid(S, 10, mkA1clean(1, 3))
    nav, _ = K.simulate([(p[0], p[1]) for p in pl], bb, ss, t0, t1)
    d = monthly(nav) - m0
    for sg, (a, b) in SEGS.items():
        x = win(d, a, b)
        out6[f'A1_clean_{sg}'] = dict(mean_pp=float(x.mean() * 100), t=A.tstat(x), t_nw=nw_t(x)[0])
    # 原版入選股中，被誤標年資的群聚股在 1～3 年窗內出現幾次
    cnt = 0
    for p in base:
        ag = A.age_years(p[3])[p[1]]
        cnt += int(((ag >= 1) & (ag < 3) & BAD[p[1]]).sum())
    out6['base_picks_age1to3_in_cluster'] = cnt
    print(json.dumps(out6, ensure_ascii=False, indent=1, default=A.conv))
