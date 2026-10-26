#!/usr/bin/env python3
"""重測 r1_0：seasonal_statarb「7～8 月除息旺季、農曆年後都不顯著」。

事前固定（與 s4 相同定義，不新增參數）：
  A. 重算 0050 7+8 月 vs 其他月份 Welch t（挑選 2005-01～2019-04、驗證 2019-05～2026-10）。
  B. 重算農曆年後 20／10 日（年前最後收盤 → 年後第 H 日收盤）對同期全部 H 日報酬平均的 t。
  C. 補做 s4 事前規則要求、但沒跑的農曆年擇時組合：
     只在 [年前最後收盤, 年後第 H 日收盤] 持有 0050、其餘現金(0) vs 全期持有 0050。
     期間：挑選期日報酬 2005-01-01～2019-04-30（事件 2005～2019）、驗證期 2019-05-01～2026-10-22（事件 2020～2026）。
     報告：CAGR、MDD、年化波動、曝險比例、同曝險隨機持有的期望 CAGR、扣成本（每次來回 0.5%／1%）、
           窗內 vs 窗外日報酬 Welch t。
  D. 同池等權（流動性前 300，月底換股，s4 同一套程式）2 月：逐年表、兩期 t_vs0、對其他月份 Welch t，
     並對照 0050 2 月與「等權 − 0050」價差。
"""
import json, math
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from scipy import stats

def welch(x, y):
    x, y = pd.Series(x).dropna(), pd.Series(y).dropna()
    if len(x) < 3: return float('nan')
    return round(float(stats.ttest_ind(x, y, equal_var=False).statistic), 2)

out = {}
e = e50_series(); e = e[e.index >= '2004-01-01']
mm = e.groupby([e.index.year, e.index.month]).last()
mm.index = pd.PeriodIndex([pd.Period(year=y, month=m, freq='M') for y, m in mm.index])
mr = (mm / mm.shift(1) - 1).dropna()
PER = {'IS': ('2005-01', '2019-04'), 'OOS': ('2019-05', '2026-10')}
def pcut(s, a, b): return s[(s.index >= pd.Period(a, 'M')) & (s.index <= pd.Period(b, 'M'))]

# ---- A. 7+8 月
out['A_JulAug_0050'] = {}
for nm, (a, b) in PER.items():
    s = pcut(mr, a, b); x = s[s.index.month.isin([7, 8])]; y = s[~s.index.month.isin([7, 8])]
    out['A_JulAug_0050'][nm] = {'n': int(len(x)), 'mean_pct': round(x.mean() * 100, 2), 'other_pct': round(y.mean() * 100, 2), 't_vs_other': welch(x, y)}

# ---- B. 農曆年事件
d = e.index; pos = {x: i for i, x in enumerate(d)}
lny = {}
for y in range(2005, 2027):
    w = d[(d >= f'{y}-01-10') & (d <= f'{y}-02-28')]
    gaps = pd.Series(w[1:] - w[:-1], index=w[1:]); lny[y] = gaps.idxmax()
out['LNY_first_day'] = {y: str(v.date()) for y, v in lny.items()}
EVY = {'IS': (2005, 2019), 'OOS': (2020, 2026)}
out['B_LNY_event'] = {}
for H in (20, 10):
    fwd = e.shift(-H) / e - 1
    rows = {y: float(e.iloc[pos[f] - 1 + H] / e.iloc[pos[f] - 1] - 1) for y, f in lny.items() if pos[f] - 1 + H < len(d)}
    ser = pd.Series(rows)
    for nm, (a, b) in EVY.items():
        x = ser.loc[a:b]; base = fwd[(fwd.index >= f'{a}-01-01') & (fwd.index <= f'{b}-12-31')].dropna()
        out['B_LNY_event'][f'post{H}|{nm}'] = {'n': int(len(x)), 'mean_pct': round(x.mean() * 100, 2), 'all_mean_pct': round(base.mean() * 100, 2),
                                               't_vs_all': round((x.mean() - base.mean()) / (x.std(ddof=1) / math.sqrt(len(x))), 2),
                                               'hit': round(float((x > 0).mean()), 2), 'median_pct': round(x.median() * 100, 2),
                                               'by_year_pct': {int(k): round(v * 100, 1) for k, v in x.items()}}
        if nm == 'OOS':
            x2 = x.drop(2024)
            out['B_LNY_event'][f'post{H}|{nm}']['ex2024_mean_pct'] = round(x2.mean() * 100, 2)

# ---- C. 農曆年擇時組合（日頻）
r = e.pct_change()
DPER = {'IS': ('2005-01-01', '2019-04-30'), 'OOS': ('2019-05-01', '2026-10-22')}
def stats_daily(x, cost_events=None, rt=0.0):
    x = x.copy()
    if cost_events is not None and rt:
        for t in cost_events:
            if t in x.index: x.loc[t] -= rt
    eq = (1 + x).cumprod(); yrs = len(x) / 252
    return {'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), 'MDD': round(float((eq / eq.cummax() - 1).min()), 4),
            'vol': round(float(x.std() * math.sqrt(252)), 4), 'total': round(float(eq.iloc[-1] - 1), 4)}
out['C_LNY_timing'] = {}
for H in (20, 10):
    inpos = pd.Series(False, index=d); exit_days = []
    for y, f in lny.items():
        i0 = pos[f] - 1
        if i0 + H >= len(d): continue
        inpos.iloc[i0 + 1: i0 + H + 1] = True; exit_days.append(d[i0 + H])
    for nm, (a, b) in DPER.items():
        rr = r[(r.index >= a) & (r.index <= b)].dropna(); ip = inpos.reindex(rr.index)
        strat = rr.where(ip, 0.0); expo = float(ip.mean())
        bh = stats_daily(rr)
        o = {'exposure': round(expo, 4), 'n_events': int(sum((t >= pd.Timestamp(a)) & (t <= pd.Timestamp(b)) for t in exit_days)),
             'strat': stats_daily(strat), 'strat_cost0.5%': stats_daily(strat, exit_days, 0.005), 'strat_cost1%': stats_daily(strat, exit_days, 0.01),
             '0050_buyhold': bh,
             'exposure_matched_CAGR': round(float((1 + rr.mean() * expo) ** 252 - 1), 4),
             'in_window_daily_mean_bp': round(float(rr[ip].mean() * 1e4), 2), 'out_window_daily_mean_bp': round(float(rr[~ip].mean() * 1e4), 2),
             'welch_in_vs_out': welch(rr[ip], rr[~ip])}
        sy = strat.groupby(strat.index.year).apply(lambda z: (1 + z).prod() - 1); by = rr.groupby(rr.index.year).apply(lambda z: (1 + z).prod() - 1)
        o['yearly_pct'] = {int(k): {'strat': round(sy[k] * 100, 1), '0050': round(by[k] * 100, 1)} for k in sy.index}
        out['C_LNY_timing'][f'LNY_post{H}_only|{nm}'] = o

# ---- D. 同池等權 2 月
adj, C, V = load('adj', 'close_raw', 'value')
me = month_ends(adj.index); R = period_returns(adj, me)
el, v60 = eligibility(me, C, V, lag=1); U = top_liq(el, v60, 300)
pw = pool_ew(R, U); nxt = dict(zip(me[:-1], me[1:]))
pw.index = pd.PeriodIndex([pd.Period(nxt[t], 'M') for t in pw.index]); pw = pw.drop(pd.Period('2024-02', 'M'), errors='ignore')
sp = (pw - mr.reindex(pw.index)).dropna()
out['D_poolEW_Feb'] = {}
for nm, (a, b) in PER.items():
    s = pcut(pw, a, b); m50 = pcut(mr, a, b); ss = pcut(sp, a, b)
    f, o_ = s[s.index.month == 2], s[s.index.month != 2]
    blk = {'n': int(len(f)), 'mean_pct': round(f.mean() * 100, 2), 'other_pct': round(o_.mean() * 100, 2), 'hit': round(float((f > 0).mean()), 2),
           't_vs0': tstat(f), 'welch_vs_other': welch(f, o_),
           '0050_Feb_mean_pct': round(m50[m50.index.month == 2].mean() * 100, 2),
           '0050_Feb_welch_vs_other': welch(m50[m50.index.month == 2], m50[m50.index.month != 2]),
           'spread_EW_minus_0050_Feb_pct': round(ss[ss.index.month == 2].mean() * 100, 2),
           'spread_Feb_t_vs0': tstat(ss[ss.index.month == 2]), 'spread_Feb_welch_vs_other': welch(ss[ss.index.month == 2], ss[ss.index.month != 2]),
           'by_year_pct': {int(p.year): {'EW': round(float(f[p]) * 100, 1), '0050': round(float(mr.get(p, np.nan)) * 100, 1)} for p in f.index},
           'other_months_t_vs_other': {m: welch(s[s.index.month == m], s[s.index.month != m]) for m in range(1, 13)}}
    blk['count_abs_t_ge2'] = int(sum(abs(v) >= 2 for v in blk['other_months_t_vs_other'].values()))
    out['D_poolEW_Feb'][nm] = blk
# 合併兩期（驗證期是否同向，只作參考）
out['D_note'] = '驗證期 2024-02 因面板 2024-01-02～02-14 缺資料而剔除（n=6）；2019–2023 面板只有存活者。'

print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
(OUTDIR / 'recheck_r1_0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
