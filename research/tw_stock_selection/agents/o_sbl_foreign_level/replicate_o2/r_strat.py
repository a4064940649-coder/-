import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, math
import numpy as np, pandas as pd
import r_lib as L
K, IDX, COLS, N = L.K, L.IDX, L.COLS, L.N

D = L.build()
SD = D['sig_days']; END = D['end']
t0 = IDX.get_loc(SD[0]); t1 = IDX.get_loc(END)
REF = pd.read_parquet(L.AG / 'main_v2/navs_gapfix.parquet')
navs = {}
main_full = L.sim(D['main_plan'], 'normal', D['t0'], D['t1'])
navs['main'] = main_full.iloc[t0 - D['t0']:] / main_full.iloc[t0 - D['t0']]
ms = L.sim(D['main_plan'], 'stress', D['t0'], D['t1']); navs['main_stress'] = ms.iloc[t0 - D['t0']:] / ms.iloc[t0 - D['t0']]
ew = REF['EXFIN|EW'].reindex(IDX[t0:t1 + 1]); navs['EW'] = ew / ew.iloc[0]
e50 = K.E50.iloc[t0:t1 + 1]; navs['0050'] = e50 / e50.iloc[0]
B1 = L.b1_plan(D)
for kind in ('normal', 'stress', 'gross'):
    navs[f'B1_{kind}'] = L.sim(B1, kind, t0, t1)
C2 = L.c_plan(D, 0.2)
navs['C2'] = L.sim(C2, 'normal', t0, t1)
# 鄰近版本
for w in (0.1, 0.3):
    navs[f'C_w{w}'] = L.sim(L.c_plan(D, w), 'normal', t0, t1)
for per in (5, 15):
    navs[f'B1_n{per}'] = L.sim(L.b1_plan(D, per), 'normal', t0, t1)
for k in range(3):
    navs[f'B1_g{k}'] = L.sim(L.b1_plan(D, 10, only_group=k), 'normal', t0, t1)
    navs[f'B1_g{k}_gross'] = L.sim(L.b1_plan(D, 10, only_group=k), 'gross', t0, t1)
    navs[f'EW_g{k}'] = L.sim(L.ew_group_plan(D, k), 'gross', t0, t1)
NAV = pd.DataFrame(navs)
# C1：每個訊號日收盤重設 80/20
pts = list(SD) + [END]
def win(s):
    x = s.reindex(pd.DatetimeIndex(pts)); return (x.shift(-1) / x - 1).iloc[:-1]
c1 = [1.0]; idx = [SD[0]]
seg = []
for i in range(len(SD)):
    a, b = pts[i], pts[i + 1]
    m = NAV.loc[a:b, 'main'] / NAV.loc[a, 'main']; bb = NAV.loc[a:b, 'B1_normal'] / NAV.loc[a, 'B1_normal']
    seg.append(c1[-1] * (0.8 * m + 0.2 * bb).iloc[1:])
    c1.append(seg[-1].iloc[-1])
NAV['C1'] = pd.concat([pd.Series([1.0], index=[SD[0]])] + seg)
NAV.to_parquet(L.HERE / 'my_navs.parquet')
W = pd.DataFrame({c: win(NAV[c]) for c in NAV.columns})

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 else float('nan')

def daily(s, a, b):
    x = s.loc[a:b]; x = x / x.iloc[0]
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return dict(CAGR=round((x.iloc[-1] ** (1 / yrs) - 1) * 100, 2), MDD=round(float((x / x.cummax() - 1).min()) * 100, 2))

# 期間定義（訊號窗起點）
P = {
    'SEL26': (pd.Timestamp('2021-10-01'), pd.Timestamp('2023-11-30')),
    'SEL27': (pd.Timestamp('2021-10-01'), pd.Timestamp('2023-12-31')),
    'A20': (pd.Timestamp('2024-01-01'), pd.Timestamp('2025-09-30')),
    'A19': (pd.Timestamp('2024-03-01'), pd.Timestamp('2025-09-30')),
    'A19b': (pd.Timestamp('2024-01-01'), pd.Timestamp('2025-08-31')),
    'B12': (pd.Timestamp('2025-10-01'), pd.Timestamp('2026-09-30')),
    'VAL32': (pd.Timestamp('2024-01-01'), pd.Timestamp('2026-09-30')),
    'VAL31': (pd.Timestamp('2024-03-01'), pd.Timestamp('2026-09-30')),
    'ALL': (pd.Timestamp('2021-10-01'), pd.Timestamp('2026-09-30')),
}
DAILY = {'SEL': (SD[0], pd.Timestamp('2024-02-15')), 'SEL_dec': (SD[0], pd.Timestamp('2023-12-29')),
         'A': (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-10-13')), 'B': (pd.Timestamp('2025-10-13'), END),
         'VAL': (pd.Timestamp('2024-02-15'), END), 'ALL': (SD[0], END)}
def sel(p):
    a, b = P[p]; return W[(W.index >= a) & (W.index <= b)]
R = {'n_sig': len(SD)}
def ex(col, bench, p):
    w = sel(p); x = w[col] - w[bench]
    return dict(n=int(x.notna().sum()), mean=round(x.mean() * 100, 3), t=round(tstat(x), 2))
for p in P:
    R[f'B1_vs_EW|{p}'] = ex('B1_normal', 'EW', p)
    R[f'B1s_vs_EW|{p}'] = ex('B1_stress', 'EW', p)
    R[f'B1g_vs_EW|{p}'] = ex('B1_gross', 'EW', p)
    R[f'B1_vs_0050|{p}'] = ex('B1_normal', '0050', p)
    R[f'B1_vs_main|{p}'] = ex('B1_normal', 'main', p)
    R[f'C2_vs_main|{p}'] = ex('C2', 'main', p)
    R[f'C2_vs_0050|{p}'] = ex('C2', '0050', p)
    R[f'main_vs_0050|{p}'] = ex('main', '0050', p)
    w = sel(p)
    R[f'corr_B1ex_mainex|{p}'] = round(float((w.B1_normal - w.EW).corr(w.main - w.EW)), 3)
    R[f'groups|{p}'] = {k: ex(f'B1_g{k}', f'EW_g{k}', p) for k in range(3)}
    R[f'groups_gross|{p}'] = {k: ex(f'B1_g{k}_gross', f'EW_g{k}', p) for k in range(3)}
    for c in ('C_w0.1', 'C_w0.3'): R[f'{c}_vs_main|{p}'] = ex(c, 'main', p)
    for c in ('B1_n5', 'B1_n15'): R[f'{c}_vs_EW|{p}'] = ex(c, 'EW', p)
    # 追蹤誤差（窗報酬差，年化）
    for c in ('C2', 'main'):
        x = w[c] - w['0050']; R[f'TE_{c}|{p}'] = round(float(x.std(ddof=1) * math.sqrt(12) * 100), 2)
    # 日頻追蹤誤差
for dk, (a, b) in DAILY.items():
    R[f'daily|{dk}'] = {c: daily(NAV[c], a, b) for c in ('main', 'C2', 'C1', 'B1_normal', 'B1_stress', 'EW', '0050')}
    dr = NAV.loc[a:b].pct_change().dropna()
    R[f'TEd|{dk}'] = {c: round(float((dr[c] - dr['0050']).std() * math.sqrt(252) * 100), 2) for c in ('main', 'C2')}
# 換手：C2 每月換掉幾檔
ch = [len(set(C2[i][1]) - set(C2[i - 1][1])) for i in range(1, len(C2))]
R['C2_avg_replaced'] = round(float(np.mean(ch)), 2)
chm = [len(set(D['main_plan'][i][1]) - set(D['main_plan'][i - 1][1])) for i in range(1, len(D['main_plan'])) if D['main_plan'][i][3] >= SD[1]]
R['main_avg_replaced'] = round(float(np.mean(chm)), 2)
# C2 vs main 名單差異
mp = {p[3]: set(p[1]) for p in D['main_plan']}
R['C2_diff_from_main_avg'] = round(float(np.mean([len(set(p[1]) - mp[p[3]]) for p in C2])), 2)

# 去掉前 5 大貢献股（毛，開盤→開盤，等權，替換為同期其他持股均值）
def per_window(plan):
    ex_ = [p[0] for p in plan]; out = []
    for i, (t, pk, _, d) in enumerate(plan):
        t2 = ex_[i + 1] if i + 1 < len(ex_) else IDX.get_loc(END)
        r = K.Cn[t, pk] / K.On[t, pk] * np.prod(K.GROW[t + 1:t2 + 1, pk], axis=0)
        if i + 1 < len(ex_):
            tr = K.TRADED[t2, pk] & np.isfinite(K.On[t2, pk])
            r = np.where(tr, r * K.On[t2, pk] / np.where(tr, K.Cn[t2, pk], 1.0), r)
        out.append((d, pk, r - 1))
    return out
pwB = per_window(B1)
ewp = per_window([(c['e'], np.where(c['el'])[0], None, d) for d, c in D['cs'].items()])
ewr = {d: r.mean() for d, pk, r in ewp}
for p in ('VAL32', 'VAL31', 'SEL26', 'ALL'):
    a, b = P[p]
    ws = [(d, pk, r) for d, pk, r in pwB if a <= d <= b]
    con = {}
    for d, pk, r in ws:
        for j, x in zip(pk, r): con[j] = con.get(j, 0) + (x - ewr[d]) / len(pk)
    top5 = sorted(con, key=con.get, reverse=True)[:5]
    g1 = []; g0 = []
    for d, pk, r in ws:
        g0.append(r.mean() - ewr[d])
        m = np.isin(pk, top5); rr = r.copy()
        if m.any(): rr[m] = rr[~m].mean()
        g1.append(rr.mean() - ewr[d])
    R[f'B1_gross_oo|{p}'] = dict(mean=round(np.mean(g0) * 100, 3), t=round(tstat(g0), 2))
    R[f'B1_ex_top5|{p}'] = dict(top5=[COLS[j] for j in top5], mean=round(np.mean(g1) * 100, 3), t=round(tstat(g1), 2))
    # 版本2：直接把那 5 檔從名單剔除（等權其餘）
    g2 = []
    for d, pk, r in ws:
        m = ~np.isin(pk, top5); g2.append(r[m].mean() - ewr[d])
    R[f'B1_drop_top5|{p}'] = dict(mean=round(np.mean(g2) * 100, 3), t=round(tstat(g2), 2))
(L.HERE / 'my_strat.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
for k, v in R.items(): print(k, v)
