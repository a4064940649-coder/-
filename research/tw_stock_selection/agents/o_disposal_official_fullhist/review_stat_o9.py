#!/usr/bin/env python3
"""反方審查（資料與統計，o9）：只讀研究員產出，不改任何檔。python3 -I -B review_stat_o9.py"""
import sys, math, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
d = pd.read_csv(H / 'trades_all.csv.gz', parse_dates=['date', 'exit_date'], dtype={'code': str})
O = d[d.seg == 'OOS'].copy()


def t(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 else None


def S(df, col='net'):
    if len(df) == 0: return {'n': 0}
    m = df.groupby(df.date.dt.to_period('M'))[col].mean()
    return {'n': len(df), 'mean': round(df[col].mean() * 100, 2), 'med': round(df[col].median() * 100, 2),
            't_month': t(m), 'months': len(m)}


def nw_t(x, L=6):
    x = np.asarray(x, float); n = len(x); e = x - x.mean()
    g0 = (e @ e) / n; s = g0
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return round(float(x.mean() / math.sqrt(s / n)), 2)


out = {}
out['OOS_all'] = S(O)
# 1. 「沒被看過」：與近似名單（disposal_bear_longhistory 已用來算報酬）不重疊的子集
un = O[O.overlap_proxy == False]; se = O[O.overlap_proxy == True]
out['unseen'] = S(un); out['seen_overlap'] = S(se)
pre14 = O.date < '2014-01-01'; surv = O.date >= '2019-05-16'
out['unseen_pre2014'] = S(un[un.date < '2014-01-01']); out['unseen_2014+'] = S(un[un.date >= '2014-01-01'])
out['unseen_pre_survivor_window'] = S(un[un.date < '2019-05-16'])
out['OOS_pre_survivor_window'] = S(O[~surv]); out['OOS_survivor_window'] = S(O[surv])
out['OOS_2014_to_2019-05-15'] = S(O[(~pre14) & (~surv)])
# 2. 逐年刪除、刪最好 k 年
yrs = sorted(O.date.dt.year.unique())
ym = {y: O[O.date.dt.year == y].net.sum() for y in yrs}
best = sorted(ym, key=lambda y: -ym[y])
out['drop_best_k_years'] = {k: {'dropped': best[:k], **S(O[~O.date.dt.year.isin(best[:k])])} for k in (1, 2, 3, 4)}
out['leave_one_year_out_min_t'] = min((S(O[O.date.dt.year != y])['t_month'], int(y)) for y in yrs)
# 3. 股票集中度
cs = O.groupby('code').net.sum().sort_values(ascending=False)
out['n_stocks'] = int(len(cs)); out['top10_stock_share_of_sum'] = round(float(cs.head(10).sum() / cs.sum()), 3)
for k in (5, 10, 20):
    out[f'drop_top{k}_stocks'] = S(O[~O.code.isin(cs.index[:k])])
ev = O.net.sort_values(ascending=False)
for k in (10, 20):
    out[f'drop_top{k}_events'] = S(O.drop(ev.index[:k]))
# 對稱截尾：去最好與最差各 5%
lo, hi = O.net.quantile([0.05, 0.95])
out['trim5_both_tails'] = S(O[(O.net > lo) & (O.net < hi)])
w = O.copy(); w['net'] = w.net.clip(lo, hi); out['winsor5'] = S(w)
# 4. 重疊樣本：日曆時間組合（每日等權持有中事件的對 EW 超額無法重建 → 以進場月序列做 NW 與補零月）
m = O.groupby(O.date.dt.to_period('M')).net.mean()
full = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq='M'))
out['month_series'] = {'n_months_with_events': int(m.notna().sum()), 'calendar_months': len(full),
                       't_iid': t(m), 't_NW6_event_months': nw_t(m.values)}
# 進場月內「事件數加權」相當於每筆 t；同日事件群聚
dd = O.groupby('date').net.mean(); out['t_by_entry_day'] = t(dd)
# 同期重疊：持有期互相重疊的事件佔比
O2 = O.sort_values('date'); ov = 0
for i, r in enumerate(O2.itertuples()):
    ov += ((O2.date < r.exit_date) & (O2.exit_date > r.date)).sum() - 1 > 0
out['share_events_overlapping_another_holding'] = round(ov / len(O2), 3)
# 5. 多重檢定：P1 單尾 p 與 Holm／族群調整
from scipy.stats import norm
p1 = 1 - norm.cdf(out['OOS_all']['t_month'])
out['P1_p_one_sided'] = round(float(p1), 4)
# 此題組內處置規則累計看過的版本：本題 6 迭代＋46 衛星；中心內 disposal_* 共 7 組 → Bonferroni 粗估
out['P1_bonferroni_x7_groups'] = round(min(1, p1 * 7), 3)
# 6. 成本後：V7X 可執行 OOS 與 unseen
out['exec_V7X_OOS'] = S(O, 'net_x'); out['exec_V7X_unseen'] = S(un, 'net_x')
out['exec_V7X_pre_survivor'] = S(O[~surv], 'net_x'); out['exec_stress_OOS'] = S(O, 'net_x_stress')
out['exec_V7X_pre2014'] = S(O[pre14], 'net_x'); out['exec_V7X_2014+'] = S(O[~pre14], 'net_x')
# 7. 下市與持有中下市
out['delisted_OOS'] = int(O.delisted.sum()); out['delayed_OOS'] = int(O.delayed.sum())
# 8. 逐年（淨，筆數，均值）
out['OOS_yearly'] = {int(y): S(g) for y, g in O.groupby(O.date.dt.year)}
# 9. 全期（含挑選、描述）去掉 2019-05 之後的存活者窗 + 已看過的 2021-10 之後
F = d[d.seg != 'PREPAY']
out['FULL_2006_2019-05_only'] = S(F[F.date < '2019-05-16'])
out['FULL_since_2019-05'] = S(F[F.date >= '2019-05-16'])
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
