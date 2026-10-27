#!/usr/bin/env python3
"""反方審查（資料與統計）：只讀研究員檔案，不寫入任何研究員檔案。輸出到 stdout。
python3 -I -B review_stat_o2.py
"""
import sys, json, math
from pathlib import Path
HERE = Path('/home/user/-/research/tw_stock_selection/agents/o_sbl_foreign_level')
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import sbl_lib as L
from scipy import stats as st

K, V, IDX, COLS, N = L.K, L.V, L.IDX, L.COLS, L.N
S = L.strategy(True)
ALLSD = S.sig_days + [S.end_day]
END = S.end_day
SLD = V.SL['industry']
elec = {'半導體業', '電子零組件業', '電腦及週邊設備業', '光電業', '通信網路業', '其他電子業', '電子通路業', '資訊服務業'}
Praw = pd.read_parquet(HERE / 'fo_panel.parquet').reindex(index=IDX, columns=COLS).to_numpy(dtype=float)

rows = []
for d in [x for x in S.sig_days if pd.Timestamp('2021-10-01') <= x <= L.VAL[1]]:
    cs = S.cross_section(d); el = cs['el']
    if el.sum() < 50: continue
    t, e = cs['t'], cs['e']
    i = ALLSD.index(d); nd = ALLSD[i + 1]
    e2 = IDX.get_loc(nd) + 1 if nd != END else IDX.get_loc(nd)
    f = L.fo_features(t); g = L.terciles(f['mcap'], el); idx = np.where(el)[0]
    r = L.open_to_open(e, e2, idx, to_open=nd != END)
    pr = lambda x: pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()
    s52p, y3p = pr(cs['s52']), pr(cs['y3'])
    for k, j in enumerate(idx):
        rows.append(dict(d=d, code=COLS[j], g=g[j], lvl=f['lvl'][j], raw=Praw[t - 1, j], r=r[k], mc=f['mcap'][j],
                         s52p=s52p[j], y3p=y3p[j], elec=SLD.get(COLS[j], '') in elec, ind=SLD.get(COLS[j], '?')))
D = pd.DataFrame(rows)
D['per'] = np.where(D.d <= L.SEL[1], 'SEL', np.where(D.d <= L.VAL_A[1], 'VAL_A', 'HOLD'))

def summ(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return {'n': len(x)}
    return {'n': len(x), 'mean%': round(x.mean() * 100, 2), 't': round(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))), 2)}

def spread(Dx, col='lvl', within='g', q=5, legs=False):
    out = {}
    for d, x in Dx.groupby('d'):
        x = x[np.isfinite(x[col]) & np.isfinite(x.r)]
        sp, lo, hi = [], [], []
        for gg, y in x.groupby(within):
            if (isinstance(gg, (int, np.integer)) and gg < 0) or len(y) < 10: continue
            b = pd.qcut(y[col].rank(method='first'), q, labels=False)
            m = y.r.groupby(b).mean()
            sp.append(m.iloc[-1] - m.iloc[0]); hi.append(m.iloc[-1] - y.r.mean()); lo.append(m.iloc[0] - y.r.mean())
        if sp: out[d] = (np.mean(sp), np.mean(hi), np.mean(lo))
    o = pd.DataFrame(out, index=['spr', 'Q5_minus_grp', 'Q1_minus_grp']).T
    return o

def by_per(o, cols=('spr', 'Q5_minus_grp', 'Q1_minus_grp')):
    res = {}
    for p, (a, b) in {'SEL': L.SEL, 'VAL': L.VAL, 'VAL_A': L.VAL_A, 'HOLD': L.HOLD}.items():
        z = o[(o.index >= a) & (o.index <= b)]
        res[p] = {c: summ(z[c]) for c in cols}
    return res

R = {}
o = spread(D)
R['1_replicate_and_legs'] = by_per(o)
R['1b_yearly_spread'] = {int(y): summ(z.spr) for y, z in o.groupby(o.index.year)}
# 去掉最好的 3 個月
for p, (a, b) in {'SEL': L.SEL, 'VAL': L.VAL}.items():
    z = o[(o.index >= a) & (o.index <= b)].spr.sort_values()
    R[f'1c_{p}_spread_drop_best3_months'] = summ(z.iloc[:-3])
    R[f'1c_{p}_spread_median%'] = round(float(z.median()) * 100, 2)
    R[f'1c_{p}_frac_positive'] = round(float((z > 0).mean()), 2)

# 2. 回推截零／接近零的股票：Q1 是否被回推誤差灌滿
D['near0'] = D.raw < 0.005
R['2_near0_share_in_pool'] = D.groupby('per').near0.mean().round(3).to_dict()
R['2_raw_negative_share'] = D.assign(neg=D.raw < 0).groupby('per').neg.mean().round(3).to_dict()
R['2_lvl_nan_share_in_pool(survivorship)'] = D.assign(na=~np.isfinite(D.lvl)).groupby('per').na.mean().round(3).to_dict()
R['2b_spread_excl_near0'] = by_per(spread(D[~D.near0]), ('spr',))

# 3. 產業：只看非電子、只看電子；以及市值×電子 雙重分組
R['3a_non_elec_spread'] = by_per(spread(D[~D.elec]), ('spr', 'Q5_minus_grp', 'Q1_minus_grp'))
R['3b_elec_spread'] = by_per(spread(D[D.elec]), ('spr',))
D['g_e'] = D.g.astype(str) + D.elec.astype(str)
D.loc[D.g < 0, 'g_e'] = None
R['3c_within_size_x_elec'] = by_per(spread(D.dropna(subset=['g_e']), within='g_e'), ('spr', 'Q5_minus_grp'))

# 4. 控制主策略因子（52 週比、營收）與 log 市值後的殘差
res = []
for d, x in D.groupby('d'):
    x = x[np.isfinite(x.lvl) & np.isfinite(x.mc) & np.isfinite(x.s52p) & np.isfinite(x.y3p)].copy()
    Xm = np.c_[np.ones(len(x)), np.log(x.mc), x.s52p, x.y3p, x.elec.astype(float)]
    beta = np.linalg.lstsq(Xm, x.lvl.to_numpy(), rcond=None)[0]
    x['res'] = x.lvl - Xm @ beta
    res.append(x)
DR = pd.concat(res)
R['4_residual_lvl_within_size'] = by_per(spread(DR, col='res'), ('spr', 'Q5_minus_grp', 'Q1_minus_grp'))

# 4b. 逐一控制：只控 52 週比；只控 log 市值＋電子
for lab, cols in [('s52_only', ['s52p']), ('logmc_elec', ['lmc', 'elecf']), ('s52_y3', ['s52p', 'y3p'])]:
    res = []
    for d, x in D.groupby('d'):
        x = x[np.isfinite(x.lvl) & np.isfinite(x.mc) & np.isfinite(x.s52p) & np.isfinite(x.y3p)].copy()
        x['lmc'] = np.log(x.mc); x['elecf'] = x.elec.astype(float)
        Xm = np.c_[np.ones(len(x)), x[cols].to_numpy(float)]
        beta = np.linalg.lstsq(Xm, x.lvl.to_numpy(), rcond=None)[0]
        x['res'] = x.lvl - Xm @ beta; res.append(x)
    R[f'4b_residual_{lab}'] = by_per(spread(pd.concat(res), col='res'), ('spr', 'Q5_minus_grp'))
# 4c. 52 週比本身在市值組內的 Q5−Q1（對照：外資價差是否只是動能的影子）
R['4c_s52p_within_size'] = by_per(spread(D, col='s52p'), ('spr',))

# 5. 市值分組 × 期別：效果所在
for gg, nm in [(0, 'small'), (1, 'mid'), (2, 'large')]:
    R[f'5_group_{nm}'] = by_per(spread(D[D.g == gg]), ('spr', 'Q5_minus_grp'))

# 6. 多重檢定：B1 挑選期 t 2.39，Spearman 平均 + 14 個試驗（6 版 + 8 鄰近）＋ explore 至少 12 個切面
for m in (6, 14, 26):
    p1 = 2 * (1 - st.t.cdf(2.39, df=25))
    R[f'6_B1_sel_t2.39_bonferroni_m{m}_p'] = round(min(1, p1 * m), 3)
    R[f'6_t_needed_bonf_m{m}'] = round(float(st.t.ppf(1 - 0.025 / m, df=25)), 2)
# B1 驗證期 t 0.98 → 單邊 p
R['6b_B1_val_t0.98_one_sided_p'] = round(float(1 - st.t.cdf(0.98, df=30)), 3)
# C2 挑選期勝出差距：C2 t 0.54 vs C1 0.49；兩期配對差標準誤
R['6c_C2_val_mean0.13_t0.15_implied_se%'] = round(0.13 / 0.15, 2)

# 7. 前 5 大貢獻股在長短價差裡的角色（驗證期，Q5 組）
top5 = {'2360', '6442', '2308', '2454', '3711'}
R['7_spread_VAL_excl_B1_top5'] = by_per(spread(D[~D.code.isin(top5)]), ('spr', 'Q5_minus_grp'))
# Q1 組裡最常出現的產業（驗證期）
q1 = []
for d, x in D[D.d >= L.VAL[0]].groupby('d'):
    for gg, y in x[np.isfinite(x.lvl)].groupby('g'):
        if gg < 0 or len(y) < 10: continue
        b = pd.qcut(y.lvl.rank(method='first'), 5, labels=False)
        q1.append(y[b.values == 0])
Q1 = pd.concat(q1)
R['7b_VAL_Q1_industry_top'] = Q1.ind.value_counts(normalize=True).head(6).round(3).to_dict()
R['7c_VAL_Q1_mean_raw_lvl%'] = round(float(Q1.raw.mean() * 100), 2)

print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
