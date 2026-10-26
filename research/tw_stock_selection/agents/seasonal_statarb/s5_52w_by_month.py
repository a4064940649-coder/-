#!/usr/bin/env python3
"""檢定 5：52 週高點策略的月份差異（是否需要季節濾網）。

事前固定：
  52 週高點策略（同主研究 bt_52w_oos.py 的 B 版，但訊號與可投資條件用 t−1 資訊、t 收盤成交）：
    訊號 = 還原價 / 過去 252 日最高還原價；流動性前 300 名中取訊號最高 30 檔等權，月底換股，成本 0.5%。
  依「持有月份」分組：
    (i)  前 30 檔淨報酬 − 同池（前 300）等權 的月平均、對 0 的 t、與其他月份差的 Welch t
    (ii) 前 300 名內五分位 Q5−Q1 的月平均與 t
  門檻：挑選期（2005-01～2019-04）某月份 (i) 與其他月份差 t ≤ −2（或 1 月 (i) 平均為負且 t ≤ −2）
        → 定義濾網「該月改持有同池等權（不付額外換股成本以外的成本）」，到驗證期檢查；否則結論為不需季節濾網。
  另報告「1 月改持 0050」與「1 月改持同池等權」兩個濾網在兩期的結果（不論是否顯著，作為檢查）。
"""
import json
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from scipy import stats

adj, C, H, V = load('adj', 'close_raw', 'high', 'value')
me = month_ends(adj.index); R = period_returns(adj, me); B = bench_returns(e50_series(), me)
S52 = (adj / adj.rolling(252, min_periods=240).max()).shift(1).reindex(me)
el, v60 = eligibility(me, C, V, lag=1)
el &= S52.notna()
lock = locked_limit_up(C, H).reindex(me).fillna(False)
U = top_liq(el, v60, 300)
rr = run_sort(S52, R, U, nq=5, top_n=30, rt_cost=0.005, min_names=30)    # 五分位用
pp = pick_portfolio(topn_picks(S52.where(U), lock, 30, 30, R.index), R)    # 鎖漲停只限制新買進
rr.update({'best_net': pp['net'], 'best_gross': pp['gross'], 'turnover': pp['turnover']})
rr['universe'] = pool_ew(R, U)
nxt = dict(zip(me[:-1], me[1:]))
hm = lambda s: pd.Series([nxt[t].month for t in s.index], index=s.index)   # 持有月份（期末月份）
ex = (rr['best_net'] - rr['universe']).dropna()
sp = (rr['Q5'] - rr['Q1']).dropna()
drop = [t for t in ex.index if nxt[t] == pd.Timestamp('2024-02-29') or (nxt[t].year == 2024 and nxt[t].month == 2)]
ex, sp = ex.drop(drop, errors='ignore'), sp.drop(drop, errors='ignore')

def table(x):
    m = hm(x); out = {}
    for k in range(1, 13):
        a, b = x[m == k], x[m != k]
        out[k] = {'n': int(len(a)), 'mean_pct': round(float(a.mean() * 100), 2), 't_vs0': tstat(a),
                  't_vs_other': round(float(stats.ttest_ind(a, b, equal_var=False).statistic), 2) if len(a) > 2 else None}
    return out

res = {}
periods = {'IS': (IS_START, IS_END), 'OOS': (OOS_START, OOS_END)}
for pn, (a, b) in periods.items():
    res[f'excess_vs_EW_by_month|{pn}'] = table(cut(ex, a, b))
    res[f'Q5_Q1_by_month|{pn}'] = table(cut(sp, a, b))
    res[f'base_strategy|{pn}'] = eval_monthly(rr['best_net'], me, B, rr['universe'], a, b)

# 濾網
sigm = [k for k, v in res['excess_vs_EW_by_month|IS'].items() if v['t_vs_other'] is not None and v['t_vs_other'] <= -2]
res['IS_negative_months_t<=-2'] = sigm
def filt(months, alt):
    r = rr['best_net'].copy(); m = hm(r)
    sw = m.isin(months)
    # 換到替代資產 = 一次賣出 + 下月再買回：多付約一次全額換股成本 0.5%（保守）
    r[sw] = alt.reindex(r.index)[sw] - 0.005
    return r
filters = {'Jan_to_EW': ([1], rr['universe']), 'Jan_to_0050': ([1], B)}
if sigm: filters['ISsig_to_EW'] = (sigm, rr['universe'])
for fn, (ms, alt) in filters.items():
    r = filt(ms, alt)
    for pn, (a, b) in periods.items():
        res[f'filter_{fn}|{pn}'] = eval_monthly(r, me, B, rr['universe'], a, b)

for pn in periods:
    print(pn, 'base', fmt(res[f'base_strategy|{pn}']))
    print('  ex vs EW by month:', ' '.join(f"{k}:{v['mean_pct']:+.2f}({v['t_vs_other']:+.1f})" for k, v in res[f'excess_vs_EW_by_month|{pn}'].items()))
    print('  Q5-Q1 by month   :', ' '.join(f"{k}:{v['mean_pct']:+.2f}({v['t_vs_other']:+.1f})" for k, v in res[f'Q5_Q1_by_month|{pn}'].items()))
    for fn in filters: print('  filter', fn, fmt(res[f'filter_{fn}|{pn}']))
print('IS neg months:', sigm)
(OUTDIR / 'results_s5.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
