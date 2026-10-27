#!/usr/bin/env python3
"""事後診斷（凍結之後、只用挑選期 2006-07～2013-12，不影響判定）：python3 -I -B diag.py → results.json 的 DIAG

1. 24 版零成本 t（成本吃掉多少）
2. 進場時點拆解：訊號日收盤進場（不可成交的理想值）vs 隔日開盤進場（引擎），都不扣成本
3. 選中版本 AVG|B1|n30：逐年、剔除最大貢獻 1 檔、持股產業／流動性分布
4. 合成（各半、n10）在挑選期內的樣子（僅描述，判定規則不打開）
5. P3 的挑選期版本：安聯 2006-04～2013-12 對第二腿因子的載荷（計畫的 P3 是 2014～2020，屬驗證期，不做）
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as O
V, K = O.V, O.K
T0 = time.time()
res = json.loads((O.HERE / 'results.json').read_text(encoding='utf-8'))
assert res.get('PREREG') and res['OOS']['status'] == 'NOT_OPENED'
BEST = res['PREREG']['selected_version']

R = O.revenue()
S = V.V2('gapfix', R['YL'], R['Y5'], exfin=True)
L = O.Leg(S)
d0 = O.leg_start(L)
days = [d for d in S.sig_days if d0 <= d <= O.SEL_END]
t1 = K.IDX.get_loc(K.IDX[K.IDX <= O.SEL_END][-1]); sd = days + [K.IDX[t1]]
NORMAL, STRESS = K.rates(0.001), K.rates('stress')
plm = [p for p in S.plan(10, 'official') if p[3] in set(days)]
t0 = plm[0][0]
ew, _ = K.simulate([(p[0], p[2]) for p in plm], 0, 0, t0, t1)
wr_ew = O.window_returns(ew, sd)
E50 = K.E50.reindex(ew.index)
out = {}

# 1. 零成本
z = {}; XG = []
for v in O.ALL_VERSIONS:
    rev, band, n = v.split('|')
    pl = L.plan(rev, band, int(n[1:]), days)
    nav, _ = K.simulate([(p[0], p[1]) for p in pl], 0, 0, t0, t1)
    ex = (O.window_returns(nav, sd) - wr_ew).to_numpy()
    z[v] = {'t_EW_gross': O.tstat(ex), 'mean_ex_gross_pct_m': float(ex.mean() * 100)}; XG.append(ex)
z['maxt_bootstrap_gross'] = O.maxt_bootstrap(np.column_stack(XG), B=5000, mean_block=6, seed=0)
navm0, _ = K.simulate([(p[0], p[1]) for p in plm], 0, 0, t0, t1)
exm0 = (O.window_returns(navm0, sd) - wr_ew).to_numpy()
z['MAIN_n10'] = {'t_EW_gross': O.tstat(exm0), 'mean_ex_gross_pct_m': float(exm0.mean() * 100)}
out['zero_cost'] = z

# 2. 進場時點：訊號日收盤→下一訊號日收盤（等權、不扣成本）vs 引擎（隔日開盤）
A = K.MYADJ.ffill().to_numpy()
ti = [K.IDX.get_loc(d) for d in days]; te = ti[1:] + [t1]
def close_entry(pl):
    xs = []
    for p, a, b in zip(pl, ti, te):
        el = p[2]
        xs.append(np.nanmean(A[b, p[1]] / A[a, p[1]] - 1) - np.nanmean(A[b, el] / A[a, el] - 1))
    return np.array(xs)
tim = {}
for v in (BEST, 'AVG|B1|n10', 'AVG|B2|n10', 'ACC|B3|n10'):
    rev, band, n = v.split('|'); pl = L.plan(rev, band, int(n[1:]), days)
    ce = close_entry(pl)
    tim[v] = {'close_entry_mean_ex_pct': float(ce.mean() * 100), 'close_entry_t': O.tstat(ce),
              'open_entry_gross_mean_ex_pct': z[v]['mean_ex_gross_pct_m'], 'open_entry_gross_t': z[v]['t_EW_gross']}
ce = close_entry(plm)
tim['MAIN_n10'] = {'close_entry_mean_ex_pct': float(ce.mean() * 100), 'close_entry_t': O.tstat(ce),
                   'open_entry_gross_mean_ex_pct': z['MAIN_n10']['mean_ex_gross_pct_m'], 'open_entry_gross_t': z['MAIN_n10']['t_EW_gross']}
out['entry_timing'] = tim

# 3. 選中版本細節
rev, band, n = BEST.split('|'); n = int(n[1:])
pl = L.plan(rev, band, n, days)
nav, lg = K.simulate([(p[0], p[1]) for p in pl], *NORMAL, t0, t1)
navs, _ = K.simulate([(p[0], p[1]) for p in pl], *STRESS, t0, t1)
navm, _ = K.simulate([(p[0], p[1]) for p in plm], *NORMAL, t0, t1)
st = O.seg(nav, ew, O.SEL[0], O.SEL_END, sd, yearly=True)
sm = O.seg(navm, ew, O.SEL[0], O.SEL_END, sd, yearly=True)
se = K.stats(ew, O.SEL[0], O.SEL_END, sd); s5 = K.stats(E50, O.SEL[0], O.SEL_END, sd)
out['best'] = {'version': BEST, 'normal': st, 'stress': O.seg(navs, ew, O.SEL[0], O.SEL_END, sd),
               'yearly': {y: {'leg': st['yearly'][y], 'main_n10': sm['yearly'].get(y), 'EW': se['yearly'].get(y), '0050': s5['yearly'].get(y)} for y in st['yearly']},
               'top_contrib': V.top_contrib(pl, t1, O.SEL[0], O.SEL_END)}
ex = (O.window_returns(nav, sd) - wr_ew)
out['best']['t_EW_ex2013'] = O.tstat(ex[ex.index.year != 2013].to_numpy())
wr_b = O.window_returns(nav, sd); wr_mm = O.window_returns(navm, sd); wr_5 = O.window_returns(E50, sd)
out['best']['paired_vs_main_n10'] = {'mean_pct_m': float((wr_b - wr_mm).mean() * 100), 't': O.tstat((wr_b - wr_mm).to_numpy())}
out['best']['mean_ex_EW_pct_m'] = float(ex.mean() * 100); out['best']['mean_ex_0050_pct_m'] = float((wr_b - wr_5).mean() * 100)
out['best']['n_windows'] = int(len(ex)); out['best']['n_stock_months'] = int(sum(len(p[1]) for p in pl)); out['best']['turnover'] = lg['turnover']
out['best']['main_n10_same_period'] = {k: sm[k] for k in ('CAGR', 'maxDD', 't_0050', 't_EW', 'E50_CAGR', 'EW_CAGR')}
out['best']['t_EW_2013_only'] = O.tstat(ex[ex.index.year == 2013].to_numpy())
ind = {}; liq = []; el_share = []
for p in pl:
    for j in p[1]: ind[O.INDUSTRY[j]] = ind.get(O.INDUSTRY[j], 0) + 1
    el_share.append(O.ELEC[p[1]].mean())
    cs = S.cross_section(p[3]); v60 = S.V60[cs['t']]
    rk = pd.Series(np.where(cs['el'], v60, np.nan)).rank(ascending=False, method='first').to_numpy()
    liq += rk[p[1]].tolist()
tot = sum(ind.values())
out['best']['industry_share'] = {k: round(v / tot, 3) for k, v in sorted(ind.items(), key=lambda x: -x[1])[:10]}
out['best']['elec_share'] = float(np.mean(el_share)); out['best']['liq_rank_median'] = float(np.median(liq))
out['best']['share_liq_101_300'] = float(np.mean(np.array(liq) > 100))
# 剔除最大貢獻股後的訊號窗 t（該股該期報酬換成同期其他持股平均；不扣成本的近似）
con, per = V.stock_contrib(pl, t1, a=O.SEL[0], b=O.SEL_END)
top = max(con, key=con.get)
gx = []
for (d, pk, r), a, b in zip(per, ti, te):
    rr = r.copy(); m = np.array([K.COLS[j] == top for j in pk])
    if m.any() and (~m).any(): rr[m] = rr[~m].mean()
    gx.append(rr.mean())
ewg = O.window_returns(ew, sd).to_numpy()
# 注意：per 的期間是成交日開盤→下一成交日開盤，與等權的訊號窗差一天，只作近似；第一期等權沒有對應窗（淨值從第一個成交日起算），去掉
out['best']['ex_top_gross_t_approx'] = O.tstat(np.array(gx[1:]) - ewg)
out['best']['with_top_gross_t_approx'] = O.tstat(np.array([r.mean() for _, _, r in per][1:]) - ewg)

# 4. 挑選期內的合成（主 n10 ＋ 同訊號同區間 n10 各半）
pl10 = L.plan(rev, band, 10, days)
cp = O.combo_plan(plm, pl10, 0.5)
comb = {}
for c, (b, s) in {'normal': NORMAL, 'stress': STRESS}.items():
    nc, _ = O.simulate_w(cp, b, s, t0, t1)
    nm, _ = K.simulate([(p[0], p[1]) for p in plm], b, s, t0, t1)
    nl, _ = K.simulate([(p[0], p[1]) for p in pl10], b, s, t0, t1)
    comb[c] = {'combo': O.seg(nc, ew, O.SEL[0], O.SEL_END, sd), 'main_n10': O.seg(nm, ew, O.SEL[0], O.SEL_END, sd),
               'leg_n10': O.seg(nl, ew, O.SEL[0], O.SEL_END, sd)}
exl = (O.window_returns(K.simulate([(p[0], p[1]) for p in pl10], *NORMAL, t0, t1)[0], sd) - wr_ew).to_numpy()
exm = (O.window_returns(navm, sd) - wr_ew).to_numpy()
comb['corr_leg_n10_main_exEW'] = O.corr(exl, exm)
out['combo_in_selection'] = comb

# 5. P3（挑選期版）：安聯
M = pd.read_csv('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/winners_attr/allianz_monthly.csv', index_col=0, parse_dates=True)
dts = list(M.index)
def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b; n_, k = X.shape; Sx = np.zeros((k, k)); Lg = 3
    for l in range(Lg + 1):
        w = 1 - l / (Lg + 1); G = (X[l:] * e[l:, None]).T @ (X[:n_ - l] * e[:n_ - l, None])
        Sx += G if l == 0 else w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X); Vv = XtXi @ Sx @ XtXi
    return b, b / np.sqrt(np.diag(Vv)), 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
def leg_m(navx, ewx):
    o = []
    for i in range(len(dts) - 1):
        a, b = dts[i], dts[i + 1]
        if a < navx.index[0] or b > navx.index[-1]: o.append(np.nan); continue
        o.append((navx.asof(b) / navx.asof(a)) - (ewx.asof(b) / ewx.asof(a)))
    return o + [np.nan]
p3 = {}
nav10, _ = K.simulate([(p[0], p[1]) for p in pl10], *NORMAL, t0, t1)
M['LEGx'] = leg_m(nav, ew); M['LEG10x'] = leg_m(nav10, ew)
d = M.loc[:'2013-12-31'].dropna(subset=['LEGx', 'FUND'])
d = d.iloc[:-1] if d.index[-1] == pd.Timestamp('2013-12-11') else d      # 最後一窗延伸到 2014-01，排除
y = (d.FUND - d['0050']).values
Fm = pd.DataFrame({'EWm': d.EW - d['0050'], 'ELEC': d.ELEC, 'C1x': d.C1 - d.EW, 'C2x': d.C2 - d.EW})
for lab, extra in {'M3': None, 'M3+LEG(n30)': 'LEGx', 'M3+LEG(n10)': 'LEG10x'}.items():
    F = Fm.copy()
    if extra: F[extra] = d[extra].values
    bb, tt, r2 = ols(y, F.values)
    p3[lab] = {'alpha_m_pct': float(bb[0] * 100), 'alpha_t': float(tt[0]), 'r2': float(r2),
               'betas': {f: [float(bb[i + 1]), float(tt[i + 1])] for i, f in enumerate(F.columns)}}
p3['n'] = len(d); p3['window'] = [str(d.index[0].date()), str(d.index[-1].date())]
p3['note'] = '計畫的 P3 是 2014～2020（驗證期），閘門未開故不做；這裡只是挑選期描述。安聯在 2006～2013 本來就沒有殘差 α。'
out['P3_selection_period_only'] = p3
out['runtime_s'] = round(time.time() - T0, 1)
res['DIAG_posthoc_selection_only'] = O.r4(out)
(O.HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(O.r4(out), ensure_ascii=False, indent=1))
