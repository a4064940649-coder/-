#!/usr/bin/env python3
"""重測 r1_1：52 週高點策略「2 月改持 0050」濾網與月份表的重跑。

以 s5_52w_by_month.py 完全相同的策略建構為基礎（t−1 訊號、t 收盤成交、前 300 流動性、前 30 檔、成本 0.5%、
鎖漲停只限制新買進），另外：
  (a) filter_Feb_to_0050：持有月份 = 2 月時改持 0050，並多扣 0.5%（與 s5 其他濾網同口徑）；
      同時重跑 base、Jan_to_EW、Jan_to_0050、Feb_to_EW、FebDec_to_EW 作對照。兩期報告年化、MDD、對 0050／等權 t。
  (b) 各持有月份「策略淨報酬 − 0050」、「等權 − 0050」、「策略 − 等權」月份表（與 s5 一樣剔除 2024-02 缺口期）。
  (c) 驗證期 3 月逐筆：策略、等權、0050 報酬、持股 src 分布（持有期間股票-日的 src 比例）、flag 3/4/5 次數、
      持股報酬極值，檢查 t_vs0 ≈ 8 是否由資料造成。
輸出：recheck_r1_1.json（同資料夾）。
"""
import json
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from scipy import stats

adj, C, H, V, SRC, FLAG = load('adj', 'close_raw', 'high', 'value', 'src', 'flag')
me = month_ends(adj.index); R = period_returns(adj, me); B = bench_returns(e50_series(), me)
S52 = (adj / adj.rolling(252, min_periods=240).max()).shift(1).reindex(me)
el, v60 = eligibility(me, C, V, lag=1)
el &= S52.notna()
lock = locked_limit_up(C, H).reindex(me).fillna(False)
U = top_liq(el, v60, 300)
picks = topn_picks(S52.where(U), lock, 30, 30, R.index)
pp = pick_portfolio(picks, R)
strat = pp['net']
EW = pool_ew(R, U)
nxt = dict(zip(me[:-1], me[1:]))
hm = lambda s: pd.Series([nxt[t].month for t in s.index], index=s.index)
periods = {'IS': (IS_START, IS_END), 'OOS': (OOS_START, OOS_END)}
gap_drop = [t for t in strat.index if nxt[t].year == 2024 and nxt[t].month == 2]
print('dropped (gap) periods:', [(str(t.date()), str(nxt[t].date())) for t in gap_drop])

res = {'gap_dropped': [(str(t.date()), str(nxt[t].date())) for t in gap_drop]}

# ---------- (a) 濾網 ----------
def filt(months, alt):
    r = strat.copy(); m = hm(r); sw = m.isin(months)
    r[sw] = alt.reindex(r.index)[sw] - 0.005
    return r
filters = {'base': None, 'Feb_to_0050': ([2], B), 'Feb_to_EW': ([2], EW), 'FebDec_to_EW': ([2, 12], EW),
           'Jan_to_0050': ([1], B), 'Jan_to_EW': ([1], EW)}
for fn, spec in filters.items():
    r = strat if spec is None else filt(*spec)
    for pn, (a, b) in periods.items():
        o = eval_monthly(r, me, B, EW, a, b)
        res[f'filter_{fn}|{pn}'] = o
        print(f'{pn:3s} {fn:13s}', fmt(o))

# ---------- (b) 月份表 ----------
def table(x):
    x = x.dropna().drop(gap_drop, errors='ignore'); m = hm(x); out = {}
    for k in range(1, 13):
        a_, b_ = x[m == k], x[m != k]
        out[k] = {'n': int(len(a_)), 'mean_pct': round(float(a_.mean() * 100), 2), 't_vs0': tstat(a_),
                  't_vs_other': round(float(stats.ttest_ind(a_, b_, equal_var=False).statistic), 2) if len(a_) > 2 else None}
    return out
series = {'strat_minus_0050': strat - B.reindex(strat.index), 'EW_minus_0050': EW - B.reindex(EW.index),
          'strat_minus_EW': strat - EW.reindex(strat.index), 'EW_raw': EW, 'strat_raw': strat,
          '0050_raw': B}
for pn, (a, b) in periods.items():
    for sn, s in series.items():
        res[f'{sn}_by_month|{pn}'] = table(cut(s, a, b))
    for sn in ['strat_minus_0050', 'strat_minus_EW', 'EW_minus_0050']:
        t = res[f'{sn}_by_month|{pn}']
        print(f'{pn} {sn:17s}', ' '.join(f"{k}:{v['mean_pct']:+.2f}(t0 {v['t_vs0']:+.2f},to {v['t_vs_other']:+.2f},n{v['n']})" for k, v in t.items()))
    f2 = {sn: res[f'{sn}_by_month|{pn}'][2] for sn in series}
    print(f'{pn} Feb detail:', f2)

# ---------- (c) 驗證期 3 月逐筆 ----------
rows = []
oos = cut(strat, OOS_START, OOS_END)
for t in oos.index:
    if nxt[t].month != 3: continue
    t1 = nxt[t]; p = picks[t]
    win = (SRC.index > t) & (SRC.index <= t1)
    sw = SRC.loc[win, p]
    vc = sw.stack().value_counts(normalize=True).round(3).to_dict()
    src_t = SRC.loc[t, p].value_counts().to_dict() if t in SRC.index else {}
    fw = FLAG.loc[win, p].stack()
    fl = {int(k): int(v) for k, v in fw[fw.isin([3, 4, 5])].value_counts().to_dict().items()}
    rs = R.loc[t, p].dropna()
    rows.append({'t': str(t.date()), 'end': str(t1.date()), 'strat_net_pct': round(strat[t] * 100, 2),
                 'strat_gross_pct': round(pp['gross'][t] * 100, 2), 'EW_pct': round(EW[t] * 100, 2),
                 '0050_pct': round(B[t] * 100, 2), 'ex_EW_pct': round((strat[t] - EW[t]) * 100, 2),
                 'n_hold': len(rs), 'src_share_stockdays': {int(k): v for k, v in vc.items()},
                 'src_at_entry_t': {int(k): int(v) for k, v in src_t.items()},
                 'flag345_in_window': fl, 'hold_ret_min_pct': round(rs.min() * 100, 1),
                 'hold_ret_max_pct': round(rs.max() * 100, 1), 'hold_ret_median_pct': round(rs.median() * 100, 1),
                 'pool_n': int(U.loc[t].sum()), 'pool_median_pct': round(float(R.loc[t][U.loc[t]].median() * 100), 2),
                 'top_vs_pool_median_pct': round(float((rs.median() - R.loc[t][U.loc[t]].median()) * 100), 2)})
res['OOS_March_rows'] = rows
mx = pd.Series([r['ex_EW_pct'] for r in rows])
res['OOS_March_ex_EW_stats'] = {'mean': round(mx.mean(), 2), 'std': round(mx.std(ddof=1), 2), 't': tstat(mx),
                                'min': mx.min(), 'max': mx.max()}
for r in rows: print(r)
print('OOS March ex EW:', res['OOS_March_ex_EW_stats'])
# 不含來源 3 期間的 3 月
ns = [r['ex_EW_pct'] for r in rows if r['t'] < '2024-02-01' or r['t'] > '2025-09-23']
res['OOS_March_ex_EW_excl_src3'] = {'n': len(ns), 'mean': round(float(np.mean(ns)), 2), 't': tstat(pd.Series(ns))}
print('OOS March excl src3 periods:', res['OOS_March_ex_EW_excl_src3'])
# 挑選期 3 月對照
im = cut(strat - EW.reindex(strat.index), IS_START, IS_END).dropna(); im = im[hm(im) == 3] * 100
res['IS_March_ex_EW'] = {'n': len(im), 'mean': round(im.mean(), 2), 'std': round(im.std(), 2), 't': tstat(im)}
print('IS March ex EW:', res['IS_March_ex_EW'])

(OUTDIR / 'recheck_r1_1.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
