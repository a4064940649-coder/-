# -*- coding: utf-8 -*-
"""zheng2-signal 兩桶規則：獨立重建回測、槓桿對照、穩健性。
執行：python3 -I -B run.py   → results.json
"""
import sys
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_ext_zheng2_signal')
import itertools
import json
import math

import numpy as np
import pandas as pd

import z2lib as z

OUT = '/home/user/-/research/tw_stock_selection/agents/o_ext_zheng2_signal/'
FEE = 0.001425 * 0.28                    # 使用者券商 2.8 折 = 0.0399%
COSTS = {
    'user': (FEE + 0.0005, FEE + 0.001 + 0.0005),            # 手續費 2.8 折＋滑價 0.05%；賣出加 ETF 證交稅 0.1%
    'author': (0.000855, 0.001855),                          # 作者：6 折、無滑價
    'stress': (0.001425 + 0.003, 0.001425 + 0.001 + 0.003),  # 牌價手續費＋滑價 0.3%
    'zero': (0.0, 0.0),
}
R = {}
df, info = z.load()
R['data'] = dict(info, rows=len(df), first=str(df.index[0].date()), last=str(df.index[-1].date()),
                 bad_dates_dropped=z.BAD_DATES, L_fee_assumed=z.L_FEE)
N = len(df)
idx = df.index
REAL_L = idx > z.L_START

# ───────── 1. 合成 00631L 檢驗 ─────────
def ann(r, d):
    y = (d[-1] - d[0]).days / 365.25
    return math.exp(np.log1p(r).sum() / y) - 1

syn = {}
d = df[REAL_L]
for nm in ('rL_synA', 'rL_synB'):
    yy = pd.DataFrame({'real': np.log1p(d.rL_real), 'syn': np.log1p(d[nm])}).groupby(d.index.year).sum()
    seg = {}
    for lo, hi in (('2014-11', '2019-12'), ('2020-01', '2023-12'), ('2024-01', '2026-12')):
        dd = d[lo:hi]
        seg[lo + '~' + hi] = {'real': round(ann(dd.rL_real, dd.index) * 100, 1), 'syn': round(ann(dd[nm], dd.index) * 100, 1)}
    syn[nm] = {'real_cagr': round(ann(d.rL_real, d.index) * 100, 2), 'syn_cagr': round(ann(d[nm], d.index) * 100, 2),
               'corr_daily': round(float(d.rL_real.corr(d[nm])), 4),
               'te_ann_pct': round(float((d.rL_real - d[nm]).std() * math.sqrt(252) * 100), 2),
               'yearly_real_minus_syn_pp': (np.expm1(yy.real) - np.expm1(yy.syn)).mul(100).round(1).to_dict(), 'segments': seg}
X = np.column_stack([np.ones(len(d)), d.r50, d.rtw])
b = np.linalg.lstsq(X, d.rL_real.values, rcond=None)[0]
syn['reg_real_on_0050_taiex'] = {'beta_0050': round(float(b[1]), 3), 'beta_taiex': round(float(b[2]), 3)}
R['synthetic_check'] = syn

# 報酬序列：2014-10-31 之後用真實 00631L，之前用合成（主：A＝2×0050；敏感度：B＝2×加權）
rL_A = df.rL_real.where(REAL_L, df.rL_synA).values
rL_B = df.rL_real.where(REAL_L, df.rL_synB).values
rL_allsynA, rL_allsynB = df.rL_synA.values, df.rL_synB.values
r50 = df.r50.values
rf = df.rf.values
close = df.tw_c.values
bench = pd.Series(np.cumprod(1 + r50), index=idx)       # 0050 含息買進持有（不扣成本）
rf_s = pd.Series(rf, index=idx)

dfA = df.copy()
dfA['rL'] = rL_A
osp = z.add_open_split(dfA, 'r50', 'rL')
# 開盤缺口代理的檢驗（有真實開盤價的 2020-01~2025-05）
oc = {}
for nm in ('50', 'L'):
    m = osp['real_open_' + nm]
    oc[nm] = {'n': int(m.sum()), 'corr_proxy_vs_real_overnight': round(float(np.corrcoef(osp['ov' + nm][m], osp['ov_proxy_' + nm][m])[0, 1]), 3),
              'std_real': round(float(osp['ov' + nm][m].std()) * 100, 3), 'std_proxy': round(float(osp['ov_proxy_' + nm][m].std()) * 100, 3)}
R['open_proxy_check'] = oc

S2000 = int(idx.searchsorted(pd.Timestamp('2000-01-04')))
S2003 = int(idx.searchsorted(pd.Timestamp('2003-07-02')))
S2015 = int(idx.searchsorted(pd.Timestamp('2015-09-01')))
PERIODS = {'full_2003': ('2003-07-02', None), 'from_2019': ('2019-01-01', None), 'from_2024': ('2024-01-01', None),
           'p2006_2012': ('2006-01-01', '2012-12-31'), 'p2013_2018': ('2013-01-01', '2018-12-31'),
           'p2003_2005': ('2003-07-02', '2005-12-31'), 'realL_2014_11': ('2014-11-03', None),
           'author_10y': ('2016-11-10', '2026-11-10')}


def run(wL, w50, start=S2003, cost='user', rL=rL_A, **k):
    buy, sell = COSTS[cost]
    r = z.simulate(rL, r50, rf, wL, w50, start, buy=buy, sell=sell, osp=osp, **k)
    r['eq'] = pd.Series(r['eq'], index=idx[start:])
    r['expo'] = pd.Series(r['expo'], index=idx[start:])
    return r


def const(wl, w5):
    return np.full(N, wl), np.full(N, w5)


def period_stats(r, periods=PERIODS, b=bench):
    out = {}
    for k, (lo, hi) in periods.items():
        st = z.stats(r['eq'], b, rf_s, lo, hi)
        if st:
            e = r['expo'][lo:hi] if hi else r['expo'][lo:]
            st['avg_expo'] = round(float(e.mean()), 3)
            out[k] = st
    return out


def fixed_mix_for(E):
    """平均曝險 E 的固定比例：E>=1 用 0050+00631L；E<1 用 0050+現金。"""
    if E >= 1:
        wl = min(E - 1, 1.0)
        return const(wl, 1 - wl)
    return const(0.0, E)


def lev_control(r, lo, hi, cost='user', rL=rL_A, band=0.10):
    """同平均曝險的固定比例對照（自該期間起點重新起跑，10pp 再平衡，同成本）。"""
    e = r['expo'][lo:hi] if hi else r['expo'][lo:]
    E = float(e.mean())
    s = int(idx.searchsorted(pd.Timestamp(lo)))
    fm = run(*fixed_mix_for(E), start=max(s - 1, 0), cost=cost, rL=rL, band=band)
    st_fm = z.stats(fm['eq'], bench, rf_s, lo, hi)
    st_vs = z.stats(r['eq'], fm['eq'], rf_s, lo, hi)
    return {'avg_expo': round(E, 3), 'fixed_mix': {k: st_fm[k] for k in ('cagr', 'vol', 'sharpe', 'mdd', 'excess_cagr', 't_month')},
            'strategy': {k: st_vs[k] for k in ('cagr', 'vol', 'sharpe', 'mdd')},
            'strategy_minus_fixed_cagr_pp': st_vs['excess_cagr'], 't_month_vs_fixed': st_vs['t_month'],
            'n_month': st_vs['n_month'], 'fixed_n_reb': fm['n_reb']}, fm


# ───────── 2. 訊號（逐行對照作者 replay）與作者 10 年回測複製 ─────────
wL, w50, lv, on = z.replay(close)
chg = pd.Series((np.diff(wL) != 0) | (np.diff(w50) != 0), index=idx[1:])
A = json.load(open(z.RAW + 'author_bt.json'))
rep = {}
for k, (a, b_) in {'s0050': const(0, 1), 's5050': const(.5, 0), 'sL': const(1, 0), 'blend': (wL, w50),
                   'ladder': z.replay(close, bucket=1.0)[:2]}.items():
    r = z.simulate(rL_A, r50, rf, a, b_, S2015, buy=0.000855, sell=0.001855, cash_rate=0.01)
    e = pd.Series(r['eq'], index=idx[S2015:])['2016-11-10':'2026-11-10']
    st = z.stats(e)
    au = [x for x in A['bt']['stats'] if x['key'] == k][0]
    rep[k] = {'mine_cagr': st['cagr'], 'author_cagr': au['cagr'], 'mine_mdd': st['mdd'], 'author_mdd': au['mdd']}
    if k == 'blend':
        rep['blend_yearly_mine'] = z.yearly(e)
        rep['blend_yearly_author'] = {int(y['year']): y['blend'] for y in A['bt']['yearly']}
rep['signal_changes_10y_mine'] = int(chg['2016-11-11':'2026-11-10'].sum())
rep['signal_changes_10y_author'] = A['bt']['changes']
rep['latest_target_mine'] = {'date': str(idx[-2].date()), 'wL': float(wL[-2]), 'w50': float(w50[-2])}
rep['latest_target_author'] = {'date': A['asof'], 'wL': A['today']['wL'], 'w50': A['today']['w50']}
# 作者 replay 只用約 12.5 年資料起算（初始狀態 on=True、lvl=0）：與完整歷史起算的差異
i0 = int(idx.searchsorted(pd.Timestamp('2014-05-10')))
wl2, w52, _, _ = z.replay(close[i0:])
j0 = int(idx.searchsorted(pd.Timestamp('2016-11-10')))
rep['short_history_init_diff_days'] = int(((wl2[j0 - i0:] != wL[j0:]) | (w52[j0 - i0:] != w50[j0:])).sum())
R['author_replication'] = rep
R['signal_changes_dates_since_2016_11'] = [str(x.date()) for x in chg[chg]['2016-11-11':].index]
R['signal_changes_per_year_2003'] = round(float(chg['2003-07-02':].sum()) / ((idx[-1] - pd.Timestamp('2003-07-02')).days / 365.25), 2)
st_s = pd.DataFrame({'lvl': lv, 'on': on}, index=idx)['2003-07-02':]
R['state_time_share_2003'] = {'trend_on': round(float(st_s.on.mean()), 3), 'lvl0': round(float((st_s.lvl == 0).mean()), 3),
                              'lvl1': round(float((st_s.lvl == 1).mean()), 3), 'lvl2': round(float((st_s.lvl == 2).mean()), 3)}

# ───────── 3. 主重建：可執行口徑 ─────────
main = {}
runs = {}
specs = {
    'blend_close_user': dict(cost='user'),                 # 主：T+1 收盤、使用者成本、現金=短利
    'blend_open_user': dict(cost='user', mode='open'),     # T+1 開盤（2020-01~2025-05 真實開盤價，其餘加權缺口推估）
    'blend_close_author_cost_cash1': dict(cost='author', cash_rate=0.01),
    'blend_close_stress': dict(cost='stress'),
    'blend_close_zero': dict(cost='zero'),
    'blend_close_user_cash0': dict(cost='user', cash_rate=0.0),
    'blend_close_user_synB': dict(cost='user', rL=rL_B),
    'blend_close_user_ALLsynA': dict(cost='user', rL=rL_allsynA),   # 2014 後也用合成（顯示「2×0050 示意」的灌水）
    'blend_close_user_ALLsynB': dict(cost='user', rL=rL_allsynB),
}
for k, kw in specs.items():
    r = run(wL, w50, **kw)
    runs[k] = r
    main[k] = {'periods': period_stats(r), 'n_reb': r['n_reb'], 'turnover_x': round(r['turnover'], 2),
               'cost_drag_total_pct': round(r['cost_sum'] * 100, 2), 'yearly': z.yearly(r['eq'])}
for j in range(1, 6):
    r = run(wL, w50, j=j)
    runs['blend_close_user_j%d' % j] = r
    main['blend_close_user_j%d' % j] = {'periods': period_stats(r), 'n_reb': r['n_reb']}
# 元件與對照
wl_c, w5_c = z.replay(close, bucket=1.0)[:2]      # 只用逆勢桶
wl_t, w5_t = z.replay(close, bucket=0.0)[:2]      # 只用順勢桶
for k, (a, b_) in {'contra_only': (wl_c, w5_c), 'trend_only': (wl_t, w5_t), 'hold_L': const(1, 0),
                   's5050_L_cash': const(.5, 0)}.items():
    r = run(a, b_)
    runs[k] = r
    main[k] = {'periods': period_stats(r), 'n_reb': r['n_reb'], 'yearly': z.yearly(r['eq'])}
# 不用槓桿的變體：00631L 一律換成 0050（逆勢桶恆為 0050；順勢桶 0050／現金）
w_nl = 0.5 + 0.5 * on.astype(float)
r = run(np.zeros(N), w_nl)
runs['no_leverage_variant'] = r
main['no_leverage_variant'] = {'periods': period_stats(r), 'n_reb': r['n_reb'], 'yearly': z.yearly(r['eq'])}
main['bench_0050_yearly'] = z.yearly(bench[idx[S2003 - 1]:])
R['main'] = main
M = runs['blend_close_user']

# j 掃描摘要（G1）
jsum = {}
for p in ('full_2003', 'from_2019'):
    ex = [main['blend_close_user']['periods'][p]['excess_cagr']] + [main['blend_close_user_j%d' % j]['periods'][p]['excess_cagr'] for j in range(1, 6)]
    ts = [main['blend_close_user']['periods'][p]['t_month']] + [main['blend_close_user_j%d' % j]['periods'][p]['t_month'] for j in range(1, 6)]
    jsum[p] = {'excess_by_j': ex, 't_by_j': ts, 'median_excess': round(float(np.median(ex)), 2), 'median_t': round(float(np.median(ts)), 2)}
R['j_scan'] = jsum

# ───────── 4. 槓桿 vs 擇時 ─────────
lev = {}
for p, (lo, hi) in PERIODS.items():
    lev[p], _ = lev_control(M, lo, hi)
# 壓力成本與開盤口徑下的對照
lev['full_2003_open'], _ = lev_control(runs['blend_open_user'], *PERIODS['full_2003'])
lev['full_2003_synB'], _ = lev_control(runs['blend_close_user_synB'], *PERIODS['full_2003'], rL=rL_B)
# 固定曝險前緣
front = {}
for p in ('full_2003', 'from_2019', 'realL_2014_11'):
    lo, hi = PERIODS[p]
    s = int(idx.searchsorted(pd.Timestamp(lo)))
    rows = []
    for E in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0):
        fm = run(*fixed_mix_for(E), start=max(s - 1, 0))
        st = z.stats(fm['eq'], bench, rf_s, lo, hi)
        rows.append({'E': E, 'cagr': st['cagr'], 'mdd': st['mdd'], 'vol': st['vol'], 'sharpe': st['sharpe']})
    front[p] = rows
lev['fixed_exposure_frontier'] = front
# 兩因子迴歸：策略月超額 ~ a + b1(0050−rf) + b2(00631L−rf)
def alpha_reg(r, lo, hi, rL=rL_A):
    e = r['eq']
    lser = pd.Series(np.cumprod(1 + rL), index=idx)
    rfm = pd.Series(np.cumprod(1 + rf), index=idx)
    sl = slice(pd.Timestamp(lo), pd.Timestamp(hi) if hi else None)
    i_ = e.index.searchsorted(pd.Timestamp(lo))
    base = e.index[max(i_ - 1, 0)]
    def mret(s):
        s = s[base:pd.Timestamp(hi)] if hi else s[base:]
        return z.monthly(s)
    y = mret(e) - mret(rfm)
    x1 = mret(bench) - mret(rfm)
    x2 = mret(lser) - mret(rfm)
    out = {}
    for nm, Xc in (('on_0050_only', [x1]), ('on_0050_and_L', [x1, x2])):
        X = np.column_stack([np.ones(len(y))] + [c.values for c in Xc])
        beta, res, rk, sv = np.linalg.lstsq(X, y.values, rcond=None)
        resid = y.values - X @ beta
        s2 = resid @ resid / (len(y) - X.shape[1])
        cov = s2 * np.linalg.inv(X.T @ X)
        out[nm] = {'alpha_month_pct': round(float(beta[0]) * 100, 3), 'alpha_ann_pct': round(float(beta[0]) * 1200, 2),
                   't_alpha': round(float(beta[0] / math.sqrt(cov[0, 0])), 2), 'betas': [round(float(x), 3) for x in beta[1:]],
                   'n': len(y)}
    return out
lev['alpha_regression'] = {p: alpha_reg(M, *PERIODS[p]) for p in ('full_2003', 'from_2019', 'realL_2014_11', 'p2006_2012', 'p2013_2018')}
# 元件各自對同曝險固定比例
for nm in ('contra_only', 'trend_only'):
    lev[nm] = {p: lev_control(runs[nm], *PERIODS[p])[0] for p in ('full_2003', 'from_2019', 'p2006_2012', 'p2013_2018')}
R['leverage_vs_timing'] = lev

# ───────── 5. 2000 年起（含 2000、2008 兩次大空頭；0050 於 2003-07 前為加權＋股息率代理） ─────────
P2000 = {'from_2000': ('2000-01-04', None), 'p2000_2012': ('2000-01-04', '2012-12-31'), 'p2000_2005': ('2000-01-04', '2005-12-31'),
         'p2009_on': ('2009-01-01', None)}
s00 = {}
for nm, (a, b_, rl) in {'blend_synA': (wL, w50, rL_A), 'blend_synB': (wL, w50, rL_B), 'contra_only': (wl_c, w5_c, rL_A),
                        'trend_only': (wl_t, w5_t, rL_A), 'hold_L': (*const(1, 0), rL_A), 's5050_L_cash': (*const(.5, 0), rL_A)}.items():
    r = run(a, b_, start=S2000, rL=rl)
    runs['y2000_' + nm] = r
    s00[nm] = {'periods': period_stats(r, P2000), 'yearly': z.yearly(r['eq']) if nm.startswith('blend') else None}
s00['lev_control_from_2000'], _ = lev_control(runs['y2000_blend_synA'], '2000-01-04', None)
s00['lev_control_2000_2012'], _ = lev_control(runs['y2000_blend_synA'], '2000-01-04', '2012-12-31')
au = json.load(open(z.RAW + 'author_stress.json'))
s00['author_stress_stats'] = au['stats']
# 空頭段累計
bears = {'2000-02-17~2001-10-03': ('2000-02-17', '2001-10-03'), '2007-10-29~2008-11-20': ('2007-10-29', '2008-11-20'),
         '2011-02-08~2011-12-19': ('2011-02-08', '2011-12-19'), '2020-01-20~2020-03-19': ('2020-01-20', '2020-03-19'),
         '2022-01-04~2022-10-25': ('2022-01-04', '2022-10-25'), '2025-01-22~2025-04-09': ('2025-01-22', '2025-04-09')}
bs = {}
e00 = runs['y2000_blend_synA']['eq']
for k, (lo, hi) in bears.items():
    bs[k] = {'blend': round((e00[hi] / e00[lo] - 1) * 100, 1), 'b0050': round((bench[hi] / bench[lo] - 1) * 100, 1),
             'L': round((runs['y2000_hold_L']['eq'][hi] / runs['y2000_hold_L']['eq'][lo] - 1) * 100, 1),
             'avg_expo': round(float(runs['y2000_blend_synA']['expo'][lo:hi].mean()), 2)}
s00['bear_segments'] = bs
R['from_2000'] = s00

# ───────── 6. 穩健性 ─────────
rob = {}
# 6a 拿掉最好 3 年
def drop_best3(r, b=bench, lo='2003-07-02'):
    e = r['eq']
    i_ = e.index.searchsorted(pd.Timestamp(lo))
    base = e.index[max(i_ - 1, 0)]
    xm = z.monthly(e[base:]) - z.monthly(b[base:])
    yr = xm.groupby(level=0).sum()
    best = list(yr.sort_values().index[-3:])
    keep = xm[~xm.index.get_level_values(0).isin(best)]
    return {'best3_years': [int(x) for x in best], 't_all': round(z.tstat(xm.values), 2), 't_ex_best3': round(z.tstat(keep.values), 2),
            'mean_month_excess_ex_best3_pct': round(float(keep.mean()) * 100, 3), 'n': len(keep)}
rob['drop_best3_vs_0050'] = drop_best3(M)
_, fm_full = lev_control(M, *PERIODS['full_2003'])
rob['drop_best3_vs_fixed_mix'] = drop_best3(M, b=fm_full['eq'])

# 6b 參數鄰域（MA × BUF × 階梯 × 高點視窗）
LADDERS = {'20/30(作者)': ((0.20, 0.60), (0.30, 1.00)), '15/25': ((0.15, 0.60), (0.25, 1.00)), '25/35': ((0.25, 0.60), (0.35, 1.00)),
           '15/30': ((0.15, 0.60), (0.30, 1.00)), '20/35': ((0.20, 0.60), (0.35, 1.00)),
           '10/20/30(原始三階)': ((0.10, 0.30), (0.20, 0.60), (0.30, 1.00)), '30單階': ((0.30, 1.00),)}
grid = []
xm_trials_0050, xm_trials_fm = [], []
for ma_n, buf, (ln, lad), hw in itertools.product((150, 175, 200, 225, 250), (0.0, 0.01, 0.02, 0.03, 0.04, 0.05), LADDERS.items(), (126, 252, 504)):
    a, b_, _, _ = z.replay(close, ladder=lad, high_win=hw, ma_n=ma_n, buf=buf)
    r = run(a, b_)
    row = {'ma': ma_n, 'buf': buf, 'ladder': ln, 'hw': hw, 'n_reb': r['n_reb']}
    for p in ('full_2003', 'from_2019', 'p2006_2012', 'p2013_2018'):
        st = z.stats(r['eq'], bench, rf_s, *PERIODS[p])
        row[p] = {'cagr': st['cagr'], 'ex': st['excess_cagr'], 't': st['t_month'], 'mdd': st['mdd']}
    lc, fm = lev_control(r, *PERIODS['full_2003'])
    row['E'] = lc['avg_expo']
    row['vs_fixed_ex'] = lc['strategy_minus_fixed_cagr_pp']
    row['vs_fixed_t'] = lc['t_month_vs_fixed']
    lc19, _ = lev_control(r, *PERIODS['from_2019'])
    row['vs_fixed_ex_2019'] = lc19['strategy_minus_fixed_cagr_pp']
    row['vs_fixed_t_2019'] = lc19['t_month_vs_fixed']
    base = idx[S2003 - 1]
    x0 = (z.monthly(r['eq'][base:]) - z.monthly(bench[base:])).values
    x1 = (z.monthly(r['eq'][base:]) - z.monthly(fm['eq'][base:])).values
    xm_trials_0050.append(x0.mean() / x0.std(ddof=1))
    xm_trials_fm.append(x1.mean() / x1.std(ddof=1))
    grid.append(row)
g = pd.DataFrame([{**{k: v for k, v in r.items() if not isinstance(v, dict)}, **{p + '_' + kk: vv for p in ('full_2003', 'from_2019', 'p2006_2012', 'p2013_2018') for kk, vv in r[p].items()}} for r in grid])
g.to_csv(OUT + 'grid_neighborhood.csv', index=False)
def dist(s):
    return {'min': round(float(s.min()), 2), 'p10': round(float(s.quantile(.1)), 2), 'median': round(float(s.median()), 2),
            'p90': round(float(s.quantile(.9)), 2), 'max': round(float(s.max()), 2), 'share_pos': round(float((s > 0).mean()), 3)}
me = g[(g.ma == 200) & (g.buf == 0.03) & (g.ladder == '20/30(作者)') & (g.hw == 252)].iloc[0]
rob['grid'] = {'n': len(g),
               'author_point': {k: (float(me[k]) if not isinstance(me[k], str) else me[k]) for k in ('full_2003_ex', 'full_2003_t', 'from_2019_ex', 'from_2019_t', 'vs_fixed_ex', 'vs_fixed_t', 'vs_fixed_ex_2019', 'vs_fixed_t_2019', 'E', 'full_2003_mdd')},
               'author_rank_pct_full_ex': round(float((g.full_2003_ex < me.full_2003_ex).mean()), 3),
               'author_rank_pct_vs_fixed_ex': round(float((g.vs_fixed_ex < me.vs_fixed_ex).mean()), 3),
               'author_rank_pct_author10y_proxy_2019_ex': round(float((g.from_2019_ex < me.from_2019_ex).mean()), 3),
               'excess_vs_0050_full': dist(g.full_2003_ex), 't_vs_0050_full': dist(g.full_2003_t),
               'excess_vs_0050_2019': dist(g.from_2019_ex), 't_vs_0050_2019': dist(g.from_2019_t),
               'excess_vs_0050_2006_2012': dist(g.p2006_2012_ex), 'excess_vs_0050_2013_2018': dist(g.p2013_2018_ex),
               'excess_vs_fixed_full': dist(g.vs_fixed_ex), 't_vs_fixed_full': dist(g.vs_fixed_t),
               'excess_vs_fixed_2019': dist(g.vs_fixed_ex_2019), 't_vs_fixed_2019': dist(g.vs_fixed_t_2019),
               'share_t_vs_fixed_ge2': round(float((g.vs_fixed_t >= 2).mean()), 3),
               'share_t_vs_0050_ge3': round(float((g.full_2003_t >= 3).mean()), 3),
               'by_ma_median_vs_fixed_ex': g.groupby('ma').vs_fixed_ex.median().round(2).to_dict(),
               'by_buf_median_vs_fixed_ex': g.groupby('buf').vs_fixed_ex.median().round(2).to_dict(),
               'by_ladder_median_vs_fixed_ex': g.groupby('ladder').vs_fixed_ex.median().round(2).to_dict(),
               'by_ladder_median_ex_0050': g.groupby('ladder').full_2003_ex.median().round(2).to_dict(),
               'by_hw_median_vs_fixed_ex': g.groupby('hw').vs_fixed_ex.median().round(2).to_dict(),
               'by_ma_median_n_reb': g.groupby('ma').n_reb.median().to_dict()}
# 一維鄰域（其餘維持作者值）
one = {}
for nm, sub in (('ma', g[(g.buf == 0.03) & (g.ladder == '20/30(作者)') & (g.hw == 252)]), ('buf', g[(g.ma == 200) & (g.ladder == '20/30(作者)') & (g.hw == 252)]),
                ('ladder', g[(g.ma == 200) & (g.buf == 0.03) & (g.hw == 252)]), ('hw', g[(g.ma == 200) & (g.buf == 0.03) & (g.ladder == '20/30(作者)')])):
    one[nm] = sub[[nm, 'full_2003_cagr', 'full_2003_ex', 'full_2003_t', 'full_2003_mdd', 'from_2019_ex', 'from_2019_t', 'E', 'vs_fixed_ex', 'vs_fixed_t', 'vs_fixed_ex_2019', 'n_reb']].to_dict('records')
rob['one_dim'] = one
# DRIFT 與 BUCKET
dr = []
for band in (0.03, 0.05, 0.10, 0.15, 0.20, 1.0):
    r = run(wL, w50, band=band)
    st = period_stats(r, {k: PERIODS[k] for k in ('full_2003', 'from_2019')})
    lc, _ = lev_control(r, *PERIODS['full_2003'], band=band)
    dr.append({'band': band, 'n_reb': r['n_reb'], 'full_ex': st['full_2003']['excess_cagr'], 'full_t': st['full_2003']['t_month'],
               'ex_2019': st['from_2019']['excess_cagr'], 't_2019': st['from_2019']['t_month'], 'E': lc['avg_expo'], 'vs_fixed_ex': lc['strategy_minus_fixed_cagr_pp'], 'vs_fixed_t': lc['t_month_vs_fixed']})
rob['drift'] = dr
bk = []
for bucket in (0.0, 0.25, 0.5, 0.75, 1.0):
    a, b_, _, _ = z.replay(close, bucket=bucket)
    r = run(a, b_)
    st = period_stats(r, {k: PERIODS[k] for k in ('full_2003', 'from_2019')})
    lc, _ = lev_control(r, *PERIODS['full_2003'])
    bk.append({'contra_share': bucket, 'cagr': st['full_2003']['cagr'], 'mdd': st['full_2003']['mdd'], 'full_ex': st['full_2003']['excess_cagr'], 'full_t': st['full_2003']['t_month'],
               'ex_2019': st['from_2019']['excess_cagr'], 'E': lc['avg_expo'], 'vs_fixed_ex': lc['strategy_minus_fixed_cagr_pp'], 'vs_fixed_t': lc['t_month_vs_fixed']})
rob['bucket'] = bk
# Deflated Sharpe
base = idx[S2003 - 1]
x0 = (z.monthly(M['eq'][base:]) - z.monthly(bench[base:])).values
x1 = (z.monthly(M['eq'][base:]) - z.monthly(fm_full['eq'][base:])).values
rob['deflated_sharpe_vs_0050'] = z.deflated_sharpe(x0, xm_trials_0050)
rob['deflated_sharpe_vs_fixed_mix'] = z.deflated_sharpe(x1, xm_trials_fm)
R['robustness'] = rob

# ───────── 7. 逐年表 ─────────
yt = pd.DataFrame({'blend': pd.Series(main['blend_close_user']['yearly']), 'b0050': pd.Series(main['bench_0050_yearly']),
                   'fixed_mix': pd.Series(z.yearly(fm_full['eq'])), 'hold_L': pd.Series(main['hold_L']['yearly']),
                   'contra': pd.Series(main['contra_only']['yearly']), 'trend': pd.Series(main['trend_only']['yearly'])})
yt['blend_minus_0050'] = (yt.blend - yt.b0050).round(1)
yt['blend_minus_fixed'] = (yt.blend - yt.fixed_mix).round(1)
ye = M['expo'].groupby(M['expo'].index.year).mean().round(2)
yt['avg_expo'] = ye
R['yearly_table'] = {int(k): v for k, v in yt.to_dict('index').items()}
R['win_years_vs_0050'] = '%d/%d' % (int((yt.blend_minus_0050 > 0).sum()), len(yt))
R['win_years_vs_fixed'] = '%d/%d' % (int((yt.blend_minus_fixed > 0).sum()), len(yt))

# 流動性（G7）：hschi 2025-05 成交金額
try:
    cap = {}
    for c in ('0050', '00631L'):
        h = pd.read_csv('/home/user/hschi1106/stock_ga/data/stock_dayK/%s.csv' % c, usecols=['date', 'Trading_money']).tail(60)
        cap[c] = {'median_daily_value_twd_bn_last60': round(float(h.Trading_money.median()) / 1e9, 2), 'asof': str(h.date.iloc[-1])}
    R['liquidity'] = cap
except Exception as ex:  # noqa: BLE001
    R['liquidity'] = {'error': str(ex)}

json.dump(R, open(OUT + 'results.json', 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('done')
