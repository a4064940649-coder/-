#!/usr/bin/env python3
"""檢定 4：大盤季節性（0050 含息）。

事前固定：
  月報酬 = 0050 含息還原價月底對月底。挑選期 2005-01～2019-04（172 個月）、驗證期 2019-05～2026-10。
  (a) 12 個日曆月：平均、對 0 的 t、與「其他月份」平均差的 Welch t。
  (b) 除息旺季：7＋8 月（月報酬）vs 其他月份。
  (c) 年底作帳：12 月 vs 其他月份。
  (d) 農曆年後：年假（每年 1/10～2/28 間最長休市空檔）後前 20 個交易日（主）／前 10 個交易日 的報酬，
      對「同期間所有 20（10）日報酬的平均」做 t；附：年假前 10 個交易日（封關前）。
  另以同池等權（流動性前 300 名，月底換股）做 (a)～(c) 作為中小型股的對照。
  門檻：挑選期「與其他月份差」t ≥ 2（或 ≤ −2）才到驗證期檢查並做擇時組合
       （正效應：只在該時段持有 0050，其餘現金；負效應：該時段持現金，其餘持有 0050；現金報酬 0）。
  註：12 個月同時檢定，偶然出現一個 |t|≥2 的機率約 46%，見 FINDINGS。
"""
import json, math
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from scipy import stats

e = e50_series()
e = e[e.index >= '2004-01-01']
mm = e.groupby([e.index.year, e.index.month]).last()
mm.index = pd.PeriodIndex([pd.Period(year=y, month=m, freq='M') for y, m in mm.index])
mr = (mm / mm.shift(1) - 1).dropna()
# 2024-01 0050 有資料（ETF 檔獨立），不受面板缺口影響
IS = mr[(mr.index >= pd.Period('2005-01', 'M')) & (mr.index <= pd.Period('2019-04', 'M'))]
OOS = mr[(mr.index >= pd.Period('2019-05', 'M')) & (mr.index <= pd.Period('2026-10', 'M'))]

def welch(x, y):
    x, y = pd.Series(x).dropna(), pd.Series(y).dropna()
    if len(x) < 3: return float('nan')
    return round(float(stats.ttest_ind(x, y, equal_var=False).statistic), 2)

def month_table(s):
    out = {}
    for m in range(1, 13):
        x = s[s.index.month == m]; y = s[s.index.month != m]
        out[m] = {'n': int(len(x)), 'mean_pct': round(float(x.mean() * 100), 2), 'hit': round(float((x > 0).mean()), 2),
                  't_vs0': tstat(x), 't_vs_other': welch(x, y)}
    return out

def window_test(s, months):
    x = s[s.index.month.isin(months)]; y = s[~s.index.month.isin(months)]
    return {'n': int(len(x)), 'mean_pct': round(float(x.mean() * 100), 2), 'other_mean_pct': round(float(y.mean() * 100), 2), 't_vs_other': welch(x, y), 't_vs0': tstat(x)}

res = {'0050': {}, 'poolEW': {}}
for nm, s in [('IS', IS), ('OOS', OOS)]:
    res['0050'][nm] = {'months': month_table(s), 'JulAug_exdiv': window_test(s, [7, 8]), 'Dec_window_dressing': window_test(s, [12])}

# 同池等權（前 300 名）月報酬
adj, C, V = load('adj', 'close_raw', 'value')
me = month_ends(adj.index); R = period_returns(adj, me)
el, v60 = eligibility(me, C, V, lag=1); U = top_liq(el, v60, 300)
pw = pool_ew(R, U); nxt = dict(zip(me[:-1], me[1:]))
pw.index = pd.PeriodIndex([pd.Period(nxt[t], 'M') for t in pw.index]); pw = pw.drop(pd.Period('2024-02', 'M'), errors='ignore')
for nm, (a, b) in [('IS', ('2005-01', '2019-04')), ('OOS', ('2019-05', '2026-10'))]:
    s = pw[(pw.index >= pd.Period(a, 'M')) & (pw.index <= pd.Period(b, 'M'))]
    res['poolEW'][nm] = {'months': month_table(s), 'JulAug_exdiv': window_test(s, [7, 8]), 'Dec_window_dressing': window_test(s, [12])}

# 農曆年
d = e.index
lny = {}
for y in range(2004, 2027):
    w = d[(d >= f'{y}-01-10') & (d <= f'{y}-02-28')]
    if len(w) < 2: continue
    gaps = pd.Series(w[1:] - w[:-1], index=w[1:])
    lny[y] = gaps.idxmax()          # 年假後第一個交易日
ev = {}
pos = {x: i for i, x in enumerate(d)}
for H in (20, 10):
    fwd = e.shift(-H) / e - 1        # 從 t 收盤起 H 日；年後：用年前最後收盤 → 年後第 H 日收盤（含開紅盤跳空）
    rows = []
    for y, first in lny.items():
        i0 = pos[first] - 1          # 年前最後交易日
        if i0 + H >= len(d): continue
        rows.append({'year': y, 'post': float(e.iloc[i0 + H] / e.iloc[i0] - 1), 'pre10': float(e.iloc[i0] / e.iloc[i0 - 10] - 1),
                     'start': str(d[i0].date())})
    df = pd.DataFrame(rows).set_index('year')
    for nm, (a, b) in [('IS', (2005, 2019)), ('OOS', (2020, 2026))]:
        x = df.loc[a:b]
        base = fwd[(fwd.index >= f'{a}-01-01') & (fwd.index <= f'{b}-12-31')].dropna()
        mu, sd = base.mean(), base.std()
        o = {'n': int(len(x)), f'post{H}_mean_pct': round(float(x.post.mean() * 100), 2), f'all_{H}d_mean_pct': round(float(mu * 100), 2),
             't_vs_all': round(float((x.post.mean() - mu) / (x.post.std(ddof=1) / math.sqrt(len(x)))), 2),
             'hit': round(float((x.post > 0).mean()), 2), 'by_year_pct': {int(k): round(v * 100, 1) for k, v in x.post.items()}}
        if H == 10:
            base10 = (e / e.shift(10) - 1); base10 = base10[(base10.index >= f'{a}-01-01') & (base10.index <= f'{b}-12-31')].dropna()
            o['pre10_mean_pct'] = round(float(x.pre10.mean() * 100), 2)
            o['pre10_t_vs_all'] = round(float((x.pre10.mean() - base10.mean()) / (x.pre10.std(ddof=1) / math.sqrt(len(x)))), 2)
        ev[f'LNY_post{H}|{nm}'] = o
res['LNY'] = ev
res['LNY_dates'] = {y: str(v.date()) for y, v in lny.items()}

# 門檻檢查與擇時
sig = []
for nm, tab in res['0050']['IS']['months'].items():
    if abs(tab['t_vs_other']) >= 2: sig.append(('month', nm, tab['t_vs_other']))
for k in ('JulAug_exdiv', 'Dec_window_dressing'):
    if abs(res['0050']['IS'][k]['t_vs_other']) >= 2: sig.append((k, None, res['0050']['IS'][k]['t_vs_other']))
for k in ('LNY_post20|IS', 'LNY_post10|IS'):
    if abs(ev[k]['t_vs_all']) >= 2: sig.append((k, None, ev[k]['t_vs_all']))
res['IS_significant'] = [list(map(str, s)) for s in sig]
timing = {}
for kind, m, t in sig:
    if kind != 'month': continue
    for nm, s in [('IS', IS), ('OOS', OOS)]:
        strat = s.where(s.index.month == m, 0.0) if t > 0 else s.where(s.index.month != m, 0.0)
        yrs = len(s) / 12
        def cg(x): return round(float((1 + x).prod() ** (1 / yrs) - 1), 4)
        def dd(x): eq = (1 + x).cumprod(); return round(float((eq / eq.cummax() - 1).min()), 4)
        timing[f'month{m}_{"only" if t > 0 else "avoid"}|{nm}'] = {'CAGR': cg(strat), 'MDD': dd(strat), '0050_CAGR': cg(s), '0050_MDD': dd(s),
                                                                  't_diff_in_OOS_month': res['0050'][nm]['months'][m]['t_vs_other']}
res['timing'] = timing

def show(tag, tab):
    print(tag, ' '.join(f"{m}:{v['mean_pct']:+.2f}({v['t_vs_other']:+.1f})" for m, v in tab['months'].items()))
    print('   JulAug', tab['JulAug_exdiv'], '\n   Dec', tab['Dec_window_dressing'])
for k in ('0050', 'poolEW'):
    for nm in ('IS', 'OOS'): show(f'{k} {nm}', res[k][nm])
for k, v in ev.items(): print(k, {a: b for a, b in v.items() if a != 'by_year_pct'})
print('IS significant:', res['IS_significant']); print('timing:', timing)
(OUTDIR / 'results_s4.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
