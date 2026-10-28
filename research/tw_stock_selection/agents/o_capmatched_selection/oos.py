#!/usr/bin/env python3
"""步驟 5：驗證（2023-01～2026-10）與對照；全部在 PREREG 之後一次跑完。
python3 -I -B oos.py           ：正式（斷言 results.json 的 PREREG 存在、所有雜湊相符）
python3 -I -B oos.py --smoke   ：凍結前的程式冒煙測試，只用 ≤ 2023-01 換股日的資料（期間改成挑選期內的假切分），輸出 results_smoke.json
輸出 results_oos.json（之後由 summarize.py 併入 results.json）與 navs_oos.parquet。"""
import sys, json, math, time, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cmlib as L, cmstrat as T, cmexec as X

SMOKE = '--smoke' in sys.argv
HERE = L.HERE
if not SMOKE:
    R0 = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    PR = R0['PREREG']; assert PR and PR.get('sha256_files'), 'PREREG 不存在'
    CHK = R0['PREREG_AMEND']['sha256_files'] if 'PREREG_AMEND' in R0 else PR['sha256_files']     # 凍結後引擎修正（v2）要重新登記
    for f, h in CHK.items():
        assert L.sha(HERE / f) == h, f'{f} 與 PREREG 雜湊不符'
    FROZEN = PR['frozen_version']
else:
    FROZEN = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))['round1']['chosen']

if SMOKE:
    END = [d for d in L.official_days('2022-12-01', '2023-01-31') if d.year == 2023][0]
    START = pd.Timestamp('2019-01-01')
    PER = {'SEL': ('2019-01-01', '2020-12-31'), 'VAL': ('2021-01-01', '2022-12-31'), 'POST2019': ('2019-01-01', '2022-12-31'),
           'PRE': ('2019-01-01', '2019-12-31'), 'ALL': ('2019-01-01', '2022-12-31')}
    ODD_DATE = pd.Timestamp('2020-10-26')
else:
    END = pd.Timestamp('2026-10-12')
    START = pd.Timestamp('2006-04-01')
    PER = {'SEL': ('2019-01-01', '2022-12-31'), 'VAL': ('2023-01-01', '2026-10-12'), 'POST2019': ('2019-01-01', '2026-10-12'),
           'PRE': ('2006-04-01', '2018-12-31'), 'ALL': ('2006-04-01', '2026-10-12'), 'HOLD24': ('2024-01-01', '2026-10-12')}
    ODD_DATE = pd.Timestamp('2020-10-26')
assert END <= pd.Timestamp('2026-10-12')
DAYS = [d for d in L.official_days(START, END) if d < END] + [END]
DAYS19 = [d for d in DAYS if d >= pd.Timestamp('2019-01-01')]
G = L.grow_exfix()
nv4 = pd.read_parquet(L.RAW / 'navs' / 'navs_v4.parquet'); nv6 = pd.read_parquet(L.RAW / 'navs' / 'navs_v6.parquet'); ni6 = pd.read_parquet(L.RAW / 'navs' / 'integrate_o6_navs.parquet')
COMP = {'0050': L.E50, 'AB_V4': nv4['AB_V4|normal'], 'AB_V4_B10': nv4['AB_V4_B10|normal'], 'AB_V6': nv6['AB_V6|normal'], 'POOL_EW': nv4['AB_V4|EW'],
        'M_E50_30': ni6['M_E50_30|AB|normal'], 'M_2330_30': ni6['M_2330_30|AB|normal']}
MAIN = COMP['AB_V4']
OUT = {'frozen': FROZEN, 'smoke': SMOKE, 'END': str(END.date()), 'periods': PER}
t_start = time.time()
def log(*a): print(f'[{time.time() - t_start:6.0f}s]', *a, flush=True)

def stats_all(nav, days=DAYS, bench=L.E50):
    o = {}
    for k, (a, b) in PER.items():
        if pd.Timestamp(a) < nav.index[0] - pd.Timedelta(days=40): continue
        o[k] = L.rel_stats(nav, bench, days, a, b)
    return o

def paired(nav, other, days, a, b):
    ra = L.win_ret(nav, days); rb = L.win_ret(other.reindex(nav.index).ffill(), days).reindex(ra.index)
    x = (ra - rb); x = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()
    return {'mean_pp': float(x.mean() * 100), 't': L.tstat(x), 'n': int(len(x))}

def mix(nav_s, a, days):
    """0050 a ＋策略 (1−a)，每個換股日收盤再平衡；轉移金額付 0050 單邊成本（手續費 6 折＋0.05% 滑價，賣出另加 0.1% 稅的一半近似）。"""
    e = L.E50.reindex(nav_s.index).ffill(); s = nav_s
    D = set(pd.DatetimeIndex(days)); out = np.empty(len(s)); ve, vs = a, 1 - a; c = L.FEE + 0.0005 + 0.0005
    for i in range(len(s)):
        if i > 0:
            ve *= e.iloc[i] / e.iloc[i - 1]; vs *= s.iloc[i] / s.iloc[i - 1]
        tot = ve + vs
        if s.index[i] in D and i > 0:
            tr = abs(ve - a * tot); tot -= tr * c; ve, vs = a * tot, (1 - a) * tot
        out[i] = ve + vs
    return pd.Series(out, index=s.index)

# ---------------- 1. 全部版本（一般／壓力成本），全期一次跑 ----------------
NAV = {}; INFO = {}; PLAN = {}
rep_nav, _, rep_plan, _ = T.run(('REP', {}), DAYS, END, 'normal', G); NAV['REP'] = rep_nav
for v in T.VERSIONS:
    for cost in ('normal', 'stress'):
        nav, lg, plan, info = T.run(v, DAYS, END, cost, G)
        NAV[f'{v}|{cost}'] = nav
        if cost == 'normal': INFO[v] = pd.DataFrame(info); PLAN[v] = plan; OUT.setdefault('turnover', {})[v] = lg['turnover']
log('versions done')
VR = {}
for v in T.VERSIONS:
    VR[v] = {}
    for cost in ('normal', 'stress'):
        nav = NAV[f'{v}|{cost}']
        VR[v][cost] = {'vs_0050': stats_all(nav)}
        if cost == 'normal':
            VR[v][cost]['vs_REP'] = {k: paired(nav, rep_nav, DAYS, a, b) for k, (a, b) in PER.items()}
            VR[v][cost]['vs_MAIN'] = {k: paired(nav, MAIN, DAYS, a, b) for k, (a, b) in PER.items()}
OUT['versions'] = VR
OUT['REP_vs_0050'] = stats_all(rep_nav)
OUT['comparators'] = {k: stats_all(v.dropna()) for k, v in COMP.items() if k != '0050'}
FN = NAV[f'{FROZEN}|normal']; FS = NAV[f'{FROZEN}|stress']

# ---------------- 2. 核心＋衛星 ----------------
COMBO = {'C50': mix(FN, 0.5, DAYS), 'C70': mix(FN, 0.7, DAYS)}
COMBO_S = {'C50': mix(FS, 0.5, DAYS), 'C70': mix(FS, 0.7, DAYS)}
OUT['combos'] = {k: {'normal': stats_all(v), 'stress': stats_all(COMBO_S[k]), 'vs_MAIN': {p: paired(v, MAIN, DAYS, a, b) for p, (a, b) in PER.items()}} for k, v in COMBO.items()}

# ---------------- 3. P1 層內選股力（挑選、驗證、結構前）----------------
b_, s_ = L.rates('normal'); RT = b_ + s_
def tier_resid(days):
    rows = []; prevw = {}
    for i in range(len(days) - 1):
        d = days[i]; cs = T.cross_section(d); e0 = cs['e']
        e1 = L.IDX.get_loc(days[i + 1]) + 1 if days[i + 1] < END else L.IDX.get_loc(END)
        f = L.window_factors_all(e0, min(e1, L.IDX.get_loc(END)), G)
        for k, (a, bb) in {'T1_2_10': (2, 10), 'T2_11_50': (11, 50), 'T3_51_300': (51, 300)}.items():
            Tm = (cs['rank'] >= a) & (cs['rank'] <= bb); el = Tm & cs['elig'] & np.isfinite(f)
            idx = T.score_order(cs, el)
            if len(idx) < 3: continue
            top = idx[:max(1, int(math.ceil(0.3 * len(idx))))]; mc = cs['mc']
            wt = mc[top] / mc[top].sum(); rtop = float((wt * (f[top] - 1)).sum()); rel = float(((mc[el] / mc[el].sum()) * (f[el] - 1)).sum())
            nw = dict(zip(top.tolist(), wt.tolist())); pw = prevw.get(k, {})
            tov = 0.5 * sum(abs(nw.get(j, 0) - pw.get(j, 0)) for j in set(nw) | set(pw)); prevw[k] = nw
            rows.append(dict(d=d, tier=k, net=rtop - rel - (tov * RT if pw else b_), gross=rtop - rel,
                             hi=float(np.nanmean(np.where(L.Cn[cs['t'], top] >= 500, f[top] - 1, np.nan))) if (L.Cn[cs['t'], top] >= 500).any() else np.nan))
    return pd.DataFrame(rows)
TR = tier_resid(DAYS)
P1 = {}
for k, (a, b) in PER.items():
    z = TR[(TR.d >= a) & (TR.d <= b)]
    P1[k] = {t: {'net_pp': float(g.net.mean() * 100), 't_net': L.tstat(g.net), 'gross_pp': float(g.gross.mean() * 100), 'n': int(len(g))} for t, g in z.groupby('tier')}
OUT['P1_tier_residual'] = P1
log('P1 done')

# ---------------- 4. P4 結構前後：本題−主策略 的月差，與對（0050−同池等權）的 β ----------------
def regress(y, x):
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 12: return None
    A = np.column_stack([np.ones(len(d)), d.iloc[:, 1]]); bb, *_ = np.linalg.lstsq(A, d.iloc[:, 0], rcond=None)
    res = d.iloc[:, 0] - A @ bb; s2 = res @ res / (len(d) - 2); se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    return {'alpha_pp': float(bb[0] * 100), 'beta': float(bb[1]), 't_beta': float(bb[1] / se[1]), 'n': int(len(d))}
dF = L.win_ret(FN, DAYS) - L.win_ret(MAIN.reindex(FN.index).ffill(), DAYS)
spread = L.win_ret(L.E50.reindex(FN.index).ffill(), DAYS) - L.win_ret(COMP['POOL_EW'].reindex(FN.index).ffill(), DAYS)
P4 = {}
for k in ('PRE', 'POST2019', 'SEL', 'VAL', 'ALL'):
    if k not in PER: continue
    a, b = PER[k]; m = (dF.index >= pd.Timestamp(a)) & (dF.index <= pd.Timestamp(b))
    P4[k] = {'diff_mean_pp': float(dF[m].mean() * 100), 't': L.tstat(dF[m].dropna()), 'reg_on_0050_minus_EW': regress(dF[m], spread[m])}
OUT['P4'] = P4
OUT['P4_note'] = '2006～2018 的市值為回推（2021-10 以前沒有配股資訊、2015-09 以前成分為規則重建，與實際 2015-09 名單一致率 0.88）；只看方向。'

# ---------------- 5. P5 台積電 ----------------
I = INFO[FROZEN]
w2 = L.win_ret(pd.Series(np.cumprod(G[:, L.J2330]), index=L.IDX), DAYS); w5 = L.win_ret(L.E50, DAYS)
act = pd.Series(I.act2330.to_numpy()[:len(w2)], index=w2.index[:len(I)])
contrib = act * (w2 - w5.reindex(w2.index)).reindex(act.index)
P5 = {}
for k, (a, b) in PER.items():
    m = (contrib.index >= pd.Timestamp(a)) & (contrib.index <= pd.Timestamp(b))
    ex = VR[FROZEN]['normal']['vs_0050'].get(k, {}).get('mean_pp', np.nan)
    P5[k] = {'tsmc_contrib_pp_m': float(contrib[m].mean() * 100), 'excess_pp_m': ex, 'share': float(contrib[m].mean() * 100 / ex) if ex and np.isfinite(ex) and ex != 0 else None,
             'max_abs_act2330': float(I.act2330[(pd.to_datetime(I.d) >= pd.Timestamp(a)) & (pd.to_datetime(I.d) <= pd.Timestamp(b))].abs().max())}
fam, kw = T.VERSIONS[FROZEN]
navX, _, _, _ = T.run((fam, dict(kw, ex2330=True)), DAYS, END, 'normal', G)
repX, _, _, _ = T.run(('REP', {'ex2330': True}), DAYS, END, 'normal', G)
P5['ex_tsmc_skeleton_vs_ex_tsmc_replica'] = {k: paired(navX, repX, DAYS, a, b) for k, (a, b) in PER.items()}
xt = pd.read_csv(L.RAW / 'screener' / 'ex_tsmc.csv', parse_dates=['date']).set_index('date')
wr = I.set_index(pd.to_datetime(I.d)).w2330
cm = pd.concat([wr, xt.w2330 / 100], axis=1, keys=['strategy', 'actual0050']).dropna()
P5['strategy_w2330_vs_actual_0050_2025_09_on'] = {'n': int(len(cm)), 'mean_gap_pp': float((cm.strategy - cm.actual0050).mean() * 100) if len(cm) else None}
OUT['P5'] = P5
NAV['EXTSMC'] = navX; NAV['REP_EXTSMC'] = repX
log('P4 P5 done')

# ---------------- 6. P6 探索 ----------------
P6 = {}
for lab, a, b in [('pre_odd', PER['ALL'][0], ODD_DATE - pd.Timedelta(days=1)), ('post_odd', ODD_DATE, PER['ALL'][1])]:
    z = TR[(TR.d >= a) & (TR.d <= b)]
    P6[lab] = {t: {'top_hi500_minus_tier_cap_pp': float((g.hi.dropna()).mean() * 100) if g.hi.notna().any() else None, 'n_hi': int(g.hi.notna().sum()), 'net_pp_all': float(g.net.mean() * 100)} for t, g in z.groupby('tier')}
OUT['P6_exploratory'] = P6

# ---------------- 7. j 掃描（每個 j 實際重跑換股與成本；2019 年起）----------------
JS = {}
for j in range(13):
    dj = [d for d in L.jscan_days(j, '2018-12-01', END) if pd.Timestamp('2018-12-01') <= d < END and d in L.IDX]
    dj = [d for d in dj if d >= pd.Timestamp('2019-01-01')] + [END]
    nav, lg, _, _ = T.run(FROZEN, dj, END, 'normal', G)
    o = {'n_days': len(dj) - 1}
    for nm, nv_ in [('F', nav), ('C50', mix(nav, 0.5, dj)), ('C70', mix(nav, 0.7, dj))]:
        for k in ('SEL', 'VAL', 'POST2019'):
            r = L.rel_stats(nv_, L.E50, dj, *PER[k]); o[f'{nm}|{k}'] = {'mean_pp': r['mean_pp'], 't': r['t'], 'IR': r['IR'], 'CAGR': r['CAGR']}
    JS[j] = o
js_sum = {}
for nm in ('F', 'C50', 'C70'):
    for k in ('SEL', 'VAL', 'POST2019'):
        ts = np.array([JS[j][f'{nm}|{k}']['t'] for j in JS]); ms = np.array([JS[j][f'{nm}|{k}']['mean_pp'] for j in JS])
        js_sum[f'{nm}|{k}'] = {'t_median': float(np.median(ts)), 't_p10': float(np.percentile(ts, 10)), 'mean_pp_median': float(np.median(ms)), 'mean_pp_p10': float(np.percentile(ms, 10)),
                               't_min': float(ts.min()), 't_max': float(ts.max())}
OUT['jscan'] = {'by_j': JS, 'summary': js_sum}
log('jscan done')

# ---------------- 8. 端點、壓力、滑價兩平點 ----------------
def endpoints(nav, days, a, b):
    x = L.excess_series(nav, L.E50, days, a, b)
    o = {'full': {'mean_pp': float(x.mean() * 100), 't': L.tstat(x)}}
    for k in (1, 3, 5):
        z = x.drop(x.nlargest(k).index); o[f'drop_best{k}'] = {'mean_pp': float(z.mean() * 100), 't': L.tstat(z)}
    z = x[x.index >= x.index[0] + pd.DateOffset(years=1)]; o['drop_first_year'] = {'mean_pp': float(z.mean() * 100), 't': L.tstat(z)}
    z = x.iloc[:-1]; o['drop_last'] = {'mean_pp': float(z.mean() * 100), 't': L.tstat(z)}
    s3 = x[~((x.index >= pd.Timestamp('2024-02-15')) & (x.index <= pd.Timestamp('2025-09-23')))]
    o['ex_src3_windows'] = {'mean_pp': float(s3.mean() * 100), 't': L.tstat(s3), 'n': int(len(s3))}
    return o
EP = {}
for nm, nv_ in [('F', FN), ('C50', COMBO['C50']), ('C70', COMBO['C70'])]:
    EP[nm] = {k: endpoints(nv_, DAYS, *PER[k]) for k in ('VAL', 'POST2019', 'SEL')}
OUT['endpoints'] = EP
# 滑價兩平點（2019 年起重跑；一般手續費＋單邊滑價 x）
plan19, _ = T.build_plan(FROZEN, DAYS19)
t0 = L.IDX.get_loc(DAYS19[0]); t1 = L.IDX.get_loc(END)
def ex_at(x, k):
    nav, _ = L.simulate_w(plan19, *L.rates(('slip', x)), t0, t1, G=G)
    return L.rel_stats(nav, L.E50, DAYS19, *PER[k])['mean_pp']
BE = {}
for k in ('VAL', 'POST2019', 'SEL'):
    lo, hi = 0.0, 0.05
    if ex_at(hi, k) > 0: BE[k] = '>5%'
    elif ex_at(lo, k) <= 0: BE[k] = '<0（零滑價也不贏）'
    else:
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if ex_at(mid, k) > 0: lo = mid
            else: hi = mid
        BE[k] = lo
OUT['breakeven_slip_oneway'] = BE
log('endpoints done')

# ---------------- 9. 可執行版（2019 年起；4 級距 × FIX／GROW × o5／tick）----------------
EXE = {}
for tier_name, tier in X.TIERS.items():
    for ver in ('GROW', 'FIX'):
        for model in ('o5', 'tick'):
            for nm, ew in [('F', 0.0), ('C50', 0.5), ('C70', 0.7)]:
                nav, lg = X.simulate_exec(plan19, t0, t1, tier, ver, model, etf_weight=ew, G=G)
                EXE[f'{nm}|{tier_name}|{ver}|{model}'] = {k: {kk: L.rel_stats(nav, L.E50, DAYS19, *PER[k])[kk] for kk in ('CAGR', 'mean_pp', 't', 'IR')} for k in ('SEL', 'VAL', 'POST2019')}
OUT['executable'] = EXE
log('exec done')

# ---------------- 10. 穩健性：逐年、產業、最大貢獻股、面板口徑 ----------------
def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 2) for y, g in r.groupby(r.index.year)}
OUT['yearly'] = {'F': yearly(FN), 'C50': yearly(COMBO['C50']), 'C70': yearly(COMBO['C70']), 'REP': yearly(rep_nav), '0050': yearly(L.E50.reindex(FN.index).ffill()),
                 'AB_V4': yearly(MAIN.reindex(FN.index).dropna())}
# 產業與個股的主動貢獻（窗近似：換股日主動權重 ×（個股窗報酬 − 複製窗報酬））
pl = PLAN[FROZEN]; rows = []
for i in range(len(DAYS) - 1):
    d = DAYS[i]; cs = T.cross_section(d); e0 = cs['e']
    if e0 not in pl: continue
    e1 = L.IDX.get_loc(DAYS[i + 1]) + 1 if DAYS[i + 1] < END else L.IDX.get_loc(END)
    f = L.window_factors_all(e0, min(e1, L.IDX.get_loc(END)), G)
    js, ws = pl[e0]; w = np.zeros(L.N); w[js] = ws; a_ = w - cs['R']
    rr = float(np.nansum(cs['R'] * (np.nan_to_num(f, nan=1.0) - 1)))
    for j in np.where(np.abs(a_) > 1e-9)[0]:
        rows.append((d, L.COLS[j], L.SL['industry'].get(L.COLS[j], '?'), a_[j] * ((f[j] if np.isfinite(f[j]) else 1.0) - 1 - rr)))
C = pd.DataFrame(rows, columns=['d', 'code', 'ind', 'c'])
RB = {}
for k in ('SEL', 'VAL', 'POST2019', 'PRE'):
    if k not in PER: continue
    a, b = PER[k]; z = C[(C.d >= a) & (C.d <= b)]; nw = z.d.nunique()
    if nw == 0: continue
    ind = (z.groupby('ind').c.sum() / nw * 100).sort_values()
    st = (z.groupby('code').c.sum() / nw * 100).sort_values()
    tot = float(z.c.sum() / nw * 100)
    RB[k] = {'active_contrib_total_pp_m': tot, 'by_industry_top5': ind.tail(5).round(3).to_dict(), 'by_industry_bottom5': ind.head(5).round(3).to_dict(),
             'top_stock': [st.index[-1], float(st.iloc[-1])], 'excess_minus_top_stock_pp_m': VR[FROZEN]['normal']['vs_0050'][k]['mean_pp'] - float(st.iloc[-1]),
             'top5_stocks': st.tail(5).round(3).to_dict(), 'elec_share_of_active': float(z[z.ind.isin(L.ELEC_IND)].c.sum() / z.c.sum()) if z.c.sum() != 0 else None}
OUT['robust_contrib'] = RB
navP, _, _, _ = T.run(FROZEN, DAYS19, END, 'normal', L.GROW)
OUT['panel_convention'] = {'vs_0050': {k: L.rel_stats(navP, L.E50, DAYS19, *PER[k]) for k in ('SEL', 'VAL', 'POST2019')},
                           'vs_MAIN_same_convention': {k: paired(navP, MAIN, DAYS19, *PER[k]) for k in ('SEL', 'VAL', 'POST2019')}}
# 主策略同期 IR（判定 b）
OUT['main_IR'] = {k: L.rel_stats(MAIN.dropna(), L.E50, DAYS, *PER[k])['IR'] for k in ('SEL', 'VAL', 'POST2019')}

# ---------------- 11. 檢定數、MDE ----------------
xv = L.excess_series(FN, L.E50, DAYS, *PER['VAL'])
sd = float(xv.std(ddof=1)); n = len(xv)
from math import erfc, sqrt
tv = VR[FROZEN]['normal']['vs_0050']['VAL']['t']
p1 = 0.5 * erfc(tv / sqrt(2)) if np.isfinite(tv) else np.nan
OUT['multiple_testing'] = {'n_versions': len(T.VERSIONS), 'n_j': 13, 'n_eff': len(T.VERSIONS) * 13, 'VAL_t': tv, 'p_one_sided': p1,
                           'bonferroni_p': min(1.0, p1 * len(T.VERSIONS) * 13) if np.isfinite(p1) else None}
OUT['MDE80_pp_m'] = {'VAL': 2.8 * sd / math.sqrt(n) * 100, 'n': n, 'sd_pp_m': sd * 100,
                     'design_note': '計畫值：TE 8%／年、46 個月 → MDE80 約 0.95pp／月'}

# ---------------- 12. 判定（事前門檻）----------------
def judge(navkey, stat):
    vs = stat
    a = vs['SEL']['IR'] >= 0.3 and vs['SEL']['t'] >= 1.5
    b = vs['VAL']['mean_pp'] > 0 and vs['VAL']['t'] >= 1.0 and vs['POST2019']['IR'] > OUT['main_IR']['POST2019']
    c = P1['SEL'].get('T3_51_300', {}).get('net_pp', -1) > 0 and P1['VAL'].get('T3_51_300', {}).get('net_pp', -1) > 0
    sh = P5['VAL']['share']; d = (sh is None) or (abs(sh) < 0.2)
    nm = 'F' if navkey == FROZEN else navkey
    e = (OUT['jscan']['summary'][f'{nm}|VAL']['mean_pp_median'] > 0 and EP[nm]['VAL']['drop_best3']['mean_pp'] > 0)
    stress = (VR[FROZEN]['stress']['vs_0050']['VAL']['mean_pp'] if navkey == FROZEN else OUT['combos'][navkey]['stress']['VAL']['mean_pp'])
    e = e and stress > 0
    unusable = (vs['VAL']['mean_pp'] <= 0) or (P1['SEL'].get('T3_51_300', {}).get('net_pp', 1) <= 0 and P1['VAL'].get('T3_51_300', {}).get('net_pp', 1) <= 0) or (sh is not None and sh >= 0.5)
    verdict = '待觀察' if (a and b and c and d and e) else '不可用'
    return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'stress_VAL_mean_pp': stress, 'unusable_trigger': unusable, 'verdict': verdict}
OUT['judgement'] = {FROZEN: judge(FROZEN, VR[FROZEN]['normal']['vs_0050']),
                    'C50': judge('C50', OUT['combos']['C50']['normal']), 'C70': judge('C70', OUT['combos']['C70']['normal'])}
fn = 'results_smoke.json' if SMOKE else 'results_oos.json'
(HERE / fn).write_text(L.dumps(OUT), encoding='utf-8')
navdf = pd.DataFrame({**{k: v for k, v in NAV.items()}, 'C50': COMBO['C50'], 'C70': COMBO['C70']})
navdf.to_parquet(HERE / ('navs_smoke.parquet' if SMOKE else 'navs_oos.parquet'))
log('written', fn)
