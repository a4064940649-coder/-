# -*- coding: utf-8 -*-
"""替代題探索：只用挑選期（訊號日 2007-01～2019-03，持有期在 2019-04-30 以前結束）。輸出 explore.json。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import alib as A
V, K = A.V, A.K
IDX, N = A.IDX, A.N

S = A.strategy(exfin=True)
sd = S.sig_days
rows = []          # 每個訊號日 × 合格股
for i, d in enumerate(sd[:-1]):
    if d < pd.Timestamp('2006-04-01'): continue
    e2 = IDX.get_loc(sd[i + 1]) + 1
    if IDX[e2] > A.SEL_END: break
    cs = S.cross_section(d); el = cs['el']
    if el.sum() < 50: continue
    e = cs['e']; t = cs['t']
    js = np.where(el)[0]
    r = A.per_ret(e, e2, js)
    age = A.age_years(d)[js]
    pr52 = pd.Series(cs['s52'][js]).rank(pct=True).to_numpy()
    pry = pd.Series(cs['y3'][js]).rank(pct=True).to_numpy()
    sc = cs['sc'][js]
    rank = pd.Series(-sc).rank(method='first').to_numpy()
    for k, j in enumerate(js):
        rows.append(dict(d=d, j=j, r=r[k], age=age[k], kind=A.KIND[j], hot=bool(A.HOT[j]), pr52=pr52[k], pry=pry[k], sc=sc[k], rank=rank[k],
                         lv=np.log(S.V60[t, j]), elec=bool(A.ELEC[j]), anniv=A.half_year_anniv_in(j, e, e2),
                         anniv48=A.half_year_anniv_in(j, e, e2, months=(24, 30, 36, 42, 48))))
X = pd.DataFrame(rows)
X = X[np.isfinite(X.r) & ~X.kind.isin(['reorg', 'dr'])]
X['dec'] = X.groupby('d').sc.transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False))
X['y13'] = (X.age >= 1) & (X.age < 3)
X['y35'] = (X.age >= 3) & (X.age < 5)
X['ipo13'] = X.y13 & (X.kind == 'ipo')
X['tr13'] = X.y13 & (X.kind == 'transfer')
X['old5'] = X.age >= 5
X['old3'] = X.age >= 3
X.to_parquet(A.HERE / 'explore_panel.parquet')

def fm(df, xs, ctrl=('pr52', 'pry', 'lv', 'elec')):
    """Fama-MacBeth：每月橫斷面 OLS（r ~ 1 + xs + 控制），回傳各係數月序列的平均與 t（pp）。"""
    out = {k: [] for k in xs}
    for d, g in df.groupby('d'):
        Z = np.column_stack([np.ones(len(g))] + [g[c].astype(float).to_numpy() for c in list(xs) + list(ctrl)])
        if any(g[c].sum() == 0 for c in xs): continue
        b, *_ = np.linalg.lstsq(Z, g['r'].to_numpy(), rcond=None)
        for k, c in enumerate(xs): out[c].append(b[1 + k])
    return {c: A.msum(v) for c, v in out.items()}

def matched(df, grp, ref):
    """同訊號日、同分數十分位內：grp 平均 − ref 平均；每月取各十分位差的加權平均（權重＝grp 檔數）。"""
    s = []
    for d, g in df.groupby('d'):
        num = 0.0; w = 0
        for q, h in g.groupby('dec'):
            a = h[h[grp]]; b = h[h[ref]]
            if len(a) and len(b): num += len(a) * (a.r.mean() - b.r.mean()); w += len(a)
        if w: s.append(num / w)
    return A.msum(s)

def resid_diff(df, grp, ref, ctrl=('pr52', 'pry', 'lv', 'elec')):
    s = []
    for d, g in df.groupby('d'):
        Z = np.column_stack([np.ones(len(g))] + [g[c].astype(float).to_numpy() for c in ctrl])
        b, *_ = np.linalg.lstsq(Z, g['r'].to_numpy(), rcond=None)
        e = g['r'].to_numpy() - Z @ b
        a = e[g[grp].to_numpy()]; c = e[g[ref].to_numpy()]
        if len(a) and len(c): s.append(a.mean() - c.mean())
    return A.msum(s)

P = X[X.d >= A.P1_START]
ex = {}
ex['sample'] = dict(sig_days=int(P.d.nunique()), first=str(P.d.min().date()), last=str(P.d.max().date()), n_obs=int(len(P)),
                    avg_names=float(P.groupby('d').size().mean()),
                    avg_y13=float(P.groupby('d').y13.sum().mean()), avg_ipo13=float(P.groupby('d').ipo13.sum().mean()), avg_tr13=float(P.groupby('d').tr13.sum().mean()),
                    avg_y35=float(P.groupby('d').y35.sum().mean()), hot_median_ipo_count=A.HOT_MED, hot_years=A.HOTYEAR)
ex['raw_mean_by_age_pp'] = {k: float(P[P[k]].r.mean() * 100) for k in ['y13', 'ipo13', 'tr13', 'y35', 'old5', 'old3']}
ex['P1_fm'] = dict(base=fm(P, ['y13', 'y35']), ipo_tr=fm(P, ['ipo13', 'tr13', 'y35']))
ex['P1_matched_decile'] = dict(y13_vs_old3=matched(P, 'y13', 'old3'), y13_vs_old5=matched(P, 'y13', 'old5'),
                               ipo13_vs_old3=matched(P, 'ipo13', 'old3'), tr13_vs_old3=matched(P, 'tr13', 'old3'), y35_vs_old5=matched(P, 'y35', 'old5'))
ex['P1_resid'] = dict(y13_vs_old3=resid_diff(P, 'y13', 'old3'), ipo13_vs_old3=resid_diff(P, 'ipo13', 'old3'), tr13_vs_old3=resid_diff(P, 'tr13', 'old3'),
                      y35_vs_old5=resid_diff(P, 'y35', 'old5'))
# P2：持有期內含上市後半年週年（12/18/24/30/36 月）
P['ipo13_ann'] = P.ipo13 & P.anniv; P['ipo13_noann'] = P.ipo13 & ~P.anniv
ex['P2'] = dict(resid_ann=resid_diff(P, 'ipo13_ann', 'old3'), resid_noann=resid_diff(P, 'ipo13_noann', 'old3'),
                n_ann=int(P.ipo13_ann.sum()), n_noann=int(P.ipo13_noann.sum()))
P['ipo25'] = (P.age >= 2) & (P.age < 5) & (P.kind == 'ipo')
P['ipo25_ann'] = P.ipo25 & P.anniv48; P['ipo25_noann'] = P.ipo25 & ~P.anniv48
ex['P2_oldrule_2to5y'] = dict(note='舊制（屆滿 2 年後每半年領回 1/5）候選日 24/30/36/42/48 月；探索中途追加（看過 age 3～4 年負值之後），只當描述',
                              ann=resid_diff(P, 'ipo25_ann', 'old5'), noann=resid_diff(P, 'ipo25_noann', 'old5'), n_ann=int(P.ipo25_ann.sum()), n_noann=int(P.ipo25_noann.sum()))
# P3：熱市 vs 冷市（上市年 IPO 家數 ≥ 挑選期中位數）
P['ipo13_hot'] = P.ipo13 & P.hot; P['ipo13_cold'] = P.ipo13 & ~P.hot
ex['P3'] = dict(hot=resid_diff(P, 'ipo13_hot', 'old3'), cold=resid_diff(P, 'ipo13_cold', 'old3'), n_hot=int(P.ipo13_hot.sum()), n_cold=int(P.ipo13_cold.sum()),
                placebo_transfer=ex['P1_resid']['tr13_vs_old3'])
# 分年資細格
cells = {}
for lo, hi in [(1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 5)]:
    P['cell'] = (P.age >= lo) & (P.age < hi) & (P.kind == 'ipo')
    cells[f'ipo_{lo}-{hi}'] = resid_diff(P, 'cell', 'old5')
ex['age_cells_resid_vs_old5'] = cells
# 分數十分位：年輕股在哪個分數區（主策略入選＝第 10 分位前段）
ex['y13_by_decile'] = {int(q): dict(n=int(g.y13.sum()), share=float(g.y13.mean()), diff_pp=float((g[g.y13].r.mean() - g[g.old3].r.mean()) * 100) if g.y13.any() else None)
                       for q, g in P.groupby('dec')}
# 年代分段
ex['P1_resid_by_era'] = {era: resid_diff(P[(P.d >= a) & (P.d <= b)], 'ipo13', 'old3') for era, (a, b) in
                         {'2007-2012': ('2007-01-01', '2012-12-31'), '2013-2019Q1': ('2013-01-01', '2019-04-30')}.items()}

# ---------- 集保候選解除日（半年週年）±20 日路徑：上市 1～3 年初次上市股、60 日均值 ≥ 1,000 萬 ----------
S52, V60 = S.S52, S.V60
R = np.where(K.TRADED, np.clip(np.nan_to_num(K.Rn, nan=0.0), -0.5, 0.5), np.nan)
# 同池等權（前 300）
tr_ = [IDX.get_loc(x) for x in K.rebalance_days('2005-01-01')]
PEW = np.full(len(IDX), np.nan)
for k, t in enumerate(tr_):
    t2 = tr_[k + 1] if k + 1 < len(tr_) else len(IDX) - 1
    v = V60[t]; px = K.Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10)
    o = np.argsort(-np.where(base, v, -np.inf), kind='stable')[:min(300, int(base.sum()))]
    m = np.zeros(N, bool); m[o] = True; m &= base
    with np.errstate(all='ignore'): PEW[t + 1:t2 + 1] = np.nanmean(R[t + 1:t2 + 1][:, m], axis=1)
ev = []
for j in np.where(A.NEWMASK)[0]:
    f = pd.Timestamp(A.FIRST[j])
    for mth in (6, 12, 18, 24, 30, 36):
        x = f + pd.DateOffset(months=mth)
        t0 = int(IDX.searchsorted(x))
        if t0 - 21 < 0 or t0 + 21 >= len(IDX) or IDX[t0 + 20] > A.SEL_END or IDX[t0] < pd.Timestamp('2005-01-01'): continue
        if not (np.isfinite(V60[t0 - 1, j]) and V60[t0 - 1, j] >= 1e7): continue
        rr = R[t0 - 20:t0 + 21, j]
        if np.isfinite(rr).mean() < 0.7: continue
        ab = np.nan_to_num(rr) - np.nan_to_num(PEW[t0 - 20:t0 + 21])
        ev.append(dict(code=A.COLS[j], kind=A.KIND[j], m=mth, date=IDX[t0], pre=ab[:20].sum(), post5=ab[20:26].sum(), post20=ab[20:41].sum(), path=np.cumsum(ab)))
EV = pd.DataFrame(ev)
def cl(v, keys):
    s = pd.Series(np.asarray(v, float), index=pd.Index(keys)).dropna(); g = s.groupby(level=0).mean()
    return dict(n=int(len(s)), mean_pp=float(s.mean() * 100), median_pp=float(s.median() * 100), t_cl=A.tstat(g.to_numpy()), n_cl=int(len(g)))
lk = {}
for kind in ('ipo', 'transfer'):
    for mth in (6, 12, 18, 24, 30, 36):
        g = EV[(EV.kind == kind) & (EV.m == mth)]
        if len(g) < 5: continue
        km = g.date.dt.to_period('M')
        lk[f'{kind}_{mth}m'] = dict(pre20=cl(g.pre, km), post0_5=cl(g.post5, km), post0_20=cl(g.post20, km),
                                    path_pp=[round(float(x) * 100, 2) for x in np.mean(np.stack(g.path.to_numpy()), axis=0)[::5]])
ex['lockup_anniv_CAR'] = lk
ex['lockup_rules_note'] = ('集保規則查證（本環境連不上 TWSE／法規網站，只能用新聞鏡像與搜尋摘要）：'
                           '早期上市規定「上市屆滿二年後領回 1/5、之後每半年 1/5」（ithome 2002 前後報導）；'
                           '2016-05 證交所把科技事業改為「屆滿一年領回 1/2、二年全數」；2018-07 櫃買／證交所改為「每屆滿半年領回 1/4」（CTBC 新聞鏡像 2018-07-12）；'
                           '2018 年櫃買一般公司為「集保 1 年、滿半年領回 1/2」。一般上市公司各年度確切版本未能逐年查證，所以用上市後 6/12/18/24/30/36 個月週年當候選解除日。')

# ---------- 主策略 n10 入選股 ----------
pl = S.plan(10, 'official')
pk = []
for i, (e, pick, _, d) in enumerate(pl[:-1]):
    e2 = pl[i + 1][0]
    if IDX[e2] > A.SEL_END or d < pd.Timestamp('2006-04-01'): continue
    r = A.per_ret(e, e2, pick); age = A.age_years(d)[pick]
    for k, j in enumerate(pick):
        pk.append(dict(d=d, code=A.COLS[j], j=j, r=r[k], age=age[k], kind=A.KIND[j], anniv=A.half_year_anniv_in(j, e, e2)))
PK = pd.DataFrame(pk)
PK['y13'] = (PK.age >= 1) & (PK.age < 3) & PK.kind.isin(['ipo', 'transfer'])
PK['ipo13'] = PK.y13 & (PK.kind == 'ipo')
def paired(flag):
    s = []
    for d, g in PK.groupby('d'):
        a = g[g[flag]]; b = g[~g[flag]]
        if len(a) and len(b): s.append(a.r.mean() - b.r.mean())
    return A.msum(s)
con = PK.groupby('code').r.sum() / 10
top20 = con.sort_values(ascending=False).head(20)
young_codes = set(PK[PK.y13].code)
ex['P4_picks'] = dict(n_pick_months=int(len(PK)), share_y13=float(PK.y13.mean()), share_ipo13=float(PK.ipo13.mean()), n_y13=int(PK.y13.sum()),
                      paired_y13=paired('y13'), paired_ipo13=paired('ipo13'),
                      y13_mean_pp=float(PK[PK.y13].r.mean() * 100), other_mean_pp=float(PK[~PK.y13].r.mean() * 100),
                      top20=[(c, round(float(v) * 100, 2), c in young_codes) for c, v in top20.items()],
                      n_top20_young=int(sum(c in young_codes for c in top20.index)),
                      young_pick_contrib_pp=float(PK[PK.y13].r.sum() / 10 * 100), total_contrib_pp=float(PK.r.sum() / 10 * 100))
PK.to_csv(A.HERE / 'explore_picks_n10.csv', index=False)
A.dump(ex, 'explore.json')
import json
print(json.dumps(ex, ensure_ascii=False, indent=1, default=A.conv)[:12000])
