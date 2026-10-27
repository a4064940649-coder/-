#!/usr/bin/env python3
"""探索（只用挑選期 SEL：訊號日 < 2019-01-11；任何報酬窗的終點不晚於 2019-01 換股成交日）。寫 explore.json。"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcore as L
sys.path.pop(0)
import numpy as np, pandas as pd
K, V, O = L.K, L.V, L.O
T0_ = time.time()
def log(*a): print(f'[{time.time()-T0_:5.0f}s]', *a, flush=True)
SPLIT = L.SPLIT
SIGS = [d for d in L.SIG if d < SPLIT]
EXS = L.EXEC[:len(SIGS) + 1]          # 最後一個＝2019-01 成交日（只當終點）
TMAX = EXS[-1]
assert L.SIG[len(SIGS)] == SPLIT

# 開盤還原指數（停牌用前一日收盤還原值）
MY = K.MYADJ.to_numpy()
IO = MY * K.On / K.Cn
MYf = pd.DataFrame(MY).ffill().to_numpy()
IOf = np.where(np.isfinite(IO), IO, np.vstack([np.full((1, L.N), np.nan), MYf[:-1]]))
def fwd(i, h, idx):
    if i + h >= len(EXS): return None
    a, b = EXS[i], EXS[i + h]
    assert b <= TMAX
    return IOf[b, idx] / IOf[a, idx] - 1

def spread_series(P=50, h=1, key='sc', score='55', sub=None, q=0.2):
    rows = []
    for i, d in enumerate(SIGS):
        cs = L.large_cs(d, P, score)
        pool = cs['pool']
        if sub is not None: pool = pool[sub[pool]]
        if len(pool) < 10: continue
        r = fwd(i, h, pool)
        if r is None: continue
        x = cs[key][pool] if key != 'sc' else cs['sc'][pool]
        ok = np.isfinite(r) & np.isfinite(x)
        if ok.sum() < 10: continue
        rk = pd.Series(x[ok]).rank(pct=True).to_numpy(); rr = r[ok]
        top, bot = rr[rk > 1 - q], rr[rk <= q]
        ic = pd.Series(x[ok]).rank().corr(pd.Series(rr).rank())
        rows.append((d, top.mean() - bot.mean(), top.mean() - rr.mean(), ic))
    df = pd.DataFrame(rows, columns=['d', 'q5q1', 'top_ew', 'ic']).set_index('d')
    return df

def summ(df, h=1):
    o = {}
    for c in ('q5q1', 'top_ew', 'ic'):
        x = df[c].dropna()
        o[c] = {'mean_pp' if c != 'ic' else 'mean': round(float(x.mean()) * (100 if c != 'ic' else 1), 3), 't_naive': round(K.tstat(x), 2), 'n': int(len(x))}
        if h > 1 and c != 'ic':
            o[c]['per_month_pp'] = round(float(x.mean()) * 100 / h, 3)
            # 重疊窗：用非重疊子樣本（每 h 個月取一個）的 t
            o[c]['t_nonoverlap'] = round(K.tstat(x.iloc[::h]), 2)
    return o

EX = {'scope': 'SEL 訊號日 2006-04～2018-12；終點不晚於 ' + str(L.IDX[TMAX].date()), 'n_signals': len(SIGS)}
# (a) 五分位價差
a = {}
for key, lab in (('sc', '合成 0.5/0.5'), ('s52', '52 週比'), ('y3', '營收 3 月年增率')):
    a[lab] = {'全池 P50': summ(spread_series(50, 1, key)), '電子': summ(spread_series(50, 1, key, sub=L.ELEC)), '非電子': summ(spread_series(50, 1, key, sub=~L.ELEC))}
    for P in (30, 80): a[lab][f'P{P}'] = summ(spread_series(P, 1, key))
# 分段
for lab, lo, hi in (('2006-2012', '2006-01-01', '2012-12-31'), ('2013-2018', '2013-01-01', '2018-12-31')):
    df = spread_series(50, 1, 'sc'); a['合成 0.5/0.5'][f'P50 {lab}'] = summ(df[(df.index >= lo) & (df.index <= hi)])
EX['a_quintile_spread_largepool_h1'] = a
log('a done')

# (a2) 資訊傳遞速度：大型池 vs 主策略池中非大型
def mid_series(h, q=0.2):
    rows = []
    for i, d in enumerate(SIGS):
        cs = SR_cs(d); lp = set(L.large_cs(d, 50)['pool'].tolist())
        pool = np.array([j for j in np.where(cs['el'])[0] if j not in lp], int)
        r = fwd(i, h, pool)
        if r is None: continue
        s52, y3 = cs['s52'][pool], cs['y3'][pool]
        sc = 0.5 * pd.Series(s52).rank(pct=True).to_numpy() + 0.5 * pd.Series(y3).rank(pct=True).to_numpy()
        ok = np.isfinite(r) & np.isfinite(sc)
        rk = pd.Series(sc[ok]).rank(pct=True).to_numpy(); rr = r[ok]
        rows.append((d, rr[rk > 1 - q].mean() - rr[rk <= q].mean(), rr[rk > 1 - q].mean() - rr.mean(), np.nan))
    return pd.DataFrame(rows, columns=['d', 'q5q1', 'top_ew', 'ic']).set_index('d')
SR_cs = L.SR.cross_section
a2 = {}
for h in (1, 2, 3, 6):
    a2[f'h{h}'] = {'大型池 P50': summ(spread_series(50, h, 'sc'), h), '主策略池非大型（第 51 名以後）': summ(mid_series(h), h)}
EX['a2_info_speed'] = a2
log('a2 done')

# (b) L5 的周轉、集中度、2330
pl = L.plan_leg(50, 5, upto=SPLIT)
cnt = {}
chg = []
prev = None
for p in pl:
    s = set(L.COLS[j] for j in p[1])
    for c in s: cnt[c] = cnt.get(c, 0) + 1
    if prev is not None: chg.append(len(s - prev) / 5)
    prev = s
tot = sum(cnt.values())
EX['b_L5'] = {'n_months': len(pl), 'avg_names_replaced_frac': round(float(np.mean(chg)), 3),
              'months_with_2330': cnt.get('2330', 0), 'share_months_2330': round(cnt.get('2330', 0) / len(pl), 3),
              'HHI_name_share': round(sum((v / tot) ** 2 for v in cnt.values()), 4), 'n_unique_names': len(cnt),
              'top15': sorted(((c, V.SL['name'].get(c, ''), v) for c, v in cnt.items()), key=lambda x: -x[2])[:15],
              'elec_share_of_slots': round(sum(v for c, v in cnt.items() if L.ELEC[L.CI[c]]) / tot, 3),
              'pool_elec_share': round(float(np.mean([L.ELEC[p[4]].mean() for p in pl])), 3)}
log('b done', EX['b_L5']['share_months_2330'])

# (c)(d) NAV（只到 2019-01 訊號日收盤）
t1 = L.T_SPLIT
mp = L.main_plan(SPLIT)
navM, _ = L.sim(L.wplan_main(mp), L.NORMAL, t1)
navL, _ = L.sim(L.wplan_leg(pl), L.NORMAL, t1)
navC, _ = L.sim(L.wplan_mix(mp, pl, 0.3), L.NORMAL, t1)
navT, _ = L.sim(O.tsmc_plan(mp, 0.3), L.NORMAL, t1)
ewP = L.pool_ew(pl, t1)
sd = [d for d in L.SD if d <= SPLIT]
w = lambda n: L.wr(n, sd)
E = w(K.E50); TS = w(O.TSMC_IDX)
xL, xM, xC, xT = w(navL) - E, w(navM) - E, w(navC) - E, w(navT) - E
xLP = w(navL) - w(ewP)
z = TS - E
c = {'corr_Lex0050_vs_MAINex0050': L.corr(xL, xM), 'corr_LexPool_vs_MAINex0050': L.corr(xLP, xM),
     'beta_L_ex0050_on_2330m0050': L.beta(xL, z), 'beta_MAIN_ex0050_on_2330m0050': L.beta(xM, z),
     'beta_LexPool_on_2330m0050': L.beta(xLP, z), 'beta_combo_ex0050_on_2330m0050': L.beta(xC, z), 'beta_TSMC30_ex0050_on_2330m0050': L.beta(xT, z),
     'L_vs_pool_EW': L.mstat(xLP), 'L_vs_0050': L.mstat(xL), 'pool_EW_vs_0050': L.mstat(w(ewP) - E), 'MAIN_vs_0050': L.mstat(xM)}
for lab, lo, hi in (('2006-2012', '2006-01-01', '2012-12-31'), ('2013-2018', '2013-01-01', '2018-12-31')):
    sl = lambda s: s[(s.index >= lo) & (s.index <= hi)]
    c[f'corr_{lab}'] = L.corr(sl(xL), sl(xM)); c[f'L_vs_pool_{lab}'] = L.mstat(sl(xLP))
EX['c_corr_beta'] = c
EX['d_decomp_V01_w30_vs_MAIN'] = L.decomp(xC, xM)
EX['d_decomp_TSMC30_vs_MAIN'] = L.decomp(xT, xM)
log('c/d done', c['corr_Lex0050_vs_MAINex0050'], c['L_vs_pool_EW'])

# (e) 子樣本：0050 贏池等權的年份
yr = lambda s: s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)
yE, yP = yr(E), yr(w(ewP))
lead = (yE > yP)
e_ = {'years_0050_beats_poolEW': [int(y) for y in lead.index[lead]], 'years_other': [int(y) for y in lead.index[~lead]]}
yy = xLP.index.year
e_['L_vs_pool_when_capweight_leads'] = L.mstat(xLP[np.isin(yy, e_['years_0050_beats_poolEW'])])
e_['L_vs_pool_other_years'] = L.mstat(xLP[np.isin(yy, e_['years_other'])])
e_['L_vs_0050_when_capweight_leads'] = L.mstat(xL[np.isin(yy, e_['years_0050_beats_poolEW'])])
e_['MAIN_vs_0050_when_capweight_leads'] = L.mstat(xM[np.isin(yy, e_['years_0050_beats_poolEW'])])
e_['combo_minus_MAIN_when_capweight_leads'] = L.mstat((xC - xM)[np.isin(yy, e_['years_0050_beats_poolEW'])])
e_['combo_minus_MAIN_other_years'] = L.mstat((xC - xM)[np.isin(yy, e_['years_other'])])
# 電子 vs 非電子入選股：每檔期報酬 − 池等權（月平均）
per = L.leg_period_returns(pl, t1)
el_, ne_ = [], []
for d, pk, wt, r, pool, rp in per:
    m = L.ELEC[pk]; ew = rp.mean()
    if m.any(): el_.append((d, r[m].mean() - ew))
    if (~m).any(): ne_.append((d, r[~m].mean() - ew))
e_['elec_picks_minus_poolEW'] = L.mstat(pd.Series(dict(el_)))
e_['nonelec_picks_minus_poolEW'] = L.mstat(pd.Series(dict(ne_)))
EX['e_subsamples'] = e_
# 輔助：臺灣50 成分（2015-04～2018-12）
plT = L.plan_leg(50, 5, members_fn=L.tw50_members, upto=SPLIT)
plT = [p for p in plT if p[3] >= L.TW50_START]
first = plT[0][3]
sdT = [d for d in sd if d >= first]
navLT, _ = L.sim(L.wplan_leg(L.plan_leg(50, 5, members_fn=lambda d: L.tw50_members(d) if d >= L.TW50_START else None, upto=SPLIT)), L.NORMAL, t1)
# 只取 2015-04 以後的窗
plT_full = L.plan_leg(50, 5, members_fn=lambda d: L.tw50_members(d) if d >= L.TW50_START else None, upto=SPLIT)
ewT = L.pool_ew(plT_full, t1)
xT50 = (w(navLT) - w(ewT)); xT50 = xT50[xT50.index >= first]
xLP15 = xLP[xLP.index >= first]
EX['aux_tw50_2015_2018'] = {'first_signal': str(first.date()), 'pool_size_mean': round(float(np.mean([len(p[4]) for p in plT])), 1),
                            'L5_vs_TW50pool_EW': L.mstat(xT50), 'main_version_L5_vs_pool_same_months': L.mstat(xLP15),
                            'overlap_with_P50_pool': round(float(np.mean([len(set(a[4]) & set(b[4])) / len(a[4]) for a, b in zip(plT, [q for q in pl if q[3] >= first])])), 3)}
log('e done')
(HERE / 'explore.json').write_text(L.dumps(EX), encoding='utf-8')
log('written')
