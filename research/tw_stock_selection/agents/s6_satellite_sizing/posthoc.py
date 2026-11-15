#!/usr/bin/env python3
"""S6 事後分析（看過 results.json 之後才加；不影響分類、主格與取捨表）。python3 -I -B posthoc.py → results_posthoc.json
H1 底倉被賣去買股：依金額門檻（≥ 淨值 0.1%、≥ 1%）重數；底倉占淨值 < 10%、< 1% 的交易日比例（「底倉被用光」有多常見）。
H2 名次加權的安慰劑：權重反過來（L_MA 名次 1～5 ×1.0、6～10 ×1.5），衛星 70%，W10 與 WN 各一跑（嚴成本）。
H3 逐年贏 0050 的年數；2020 年換成 0050 當年報酬後的 2019 起年化（看單一年份的貢獻）。
H4 位移 +5 日逐年：WN_70、RWN_70 與 W10_70 的差在哪幾年消失。
H5 主格 WN_70 的拉回減碼事件與底倉被賣最大的 5 次。"""
import sys, json, time, resource, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s6lib
from s6lib import s5lib, s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
t0 = time.time()
res = json.load(open(H / 'results.json'))
nv = pd.read_csv(H / 'navs.csv.gz', index_col=0, parse_dates=True)
b = nv['0050']
CELLS = list(res['cells'].keys())
D19 = pd.Timestamp(res['meta']['since2019_from']); D0 = pd.Timestamp(res['meta']['start'])
out = {}


def cagr_from_yearly(yr, years, swap=None):
    v = 1.0
    for y in years:
        r = yr[str(y)]['strat'] if swap is None or y != swap else yr[str(y)]['e0050']
        v *= 1 + r
    return v


# H1
h1 = {}
for c in CELLS:
    ex = res['cells'][c]['runs']['strict']['extra']
    n = nv[f'{c}|strict']; B = nv[f'{c}|strict|B']
    first = pd.Timestamp(ex['first_entry'])
    ev = ex['base_sold']['events']
    rel = [e['amt'] / n.loc[pd.Timestamp(e['date'])] for e in ev]
    bf = (B / n).loc[first:]
    h1[c] = dict(n_prereg=len(ev), n_ge_0p1pct_nav=int(sum(r >= 0.001 for r in rel)), n_ge_1pct_nav=int(sum(r >= 0.01 for r in rel)),
                 max_pct_nav=float(max(rel, default=0.0)), frac_days_base_lt10=float((bf < 0.10).mean()), frac_days_base_lt1=float((bf < 0.01).mean()),
                 base_frac_mean=float(bf.mean()), base_frac_median=float(bf.median()))
out['H1_base_sold_material'] = h1

# H2 安慰劑（需要重跑）
P = s6lib.load()
U = stlib.universe(P, 'twse') & ~s4lib.innov_panel(P)[0]
RT = s2lib.rev_tables(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed'):
    if hasattr(P, a):
        delattr(P, a)
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
DE = P.IDX[-1]
S = s4lib.make_signal(P, U, RT, depth=10, mom_lo=0.10, exit='top20', keep_info=True)
h2 = {}
for sz in ('W10', 'WN'):
    nav, lg, st = s6lib.simulate_s6(P, S, U, sat_w=0.70, sizing=sz, rank_w=True, wt=(1.0, 1.5), cost=COSTS['strict'])
    t1 = int(lg['ent_t'].min())
    h2[f'R{sz}_70_reverse'] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE),
                                   avg_stock_frac=float(st['stockfrac'].loc[P.IDX[t1]:].mean()), n_closed=int((lg['why'] != 'open_end').sum()))
    for ref in (sz + '_70', 'R' + sz + '_70'):
        rr = res['cells'][ref]['runs']['strict']
        h2[f'R{sz}_70_reverse'][f'ref_{ref}'] = dict(since2019=rr['windows']['since2019']['cagr'], full=rr['windows']['full']['cagr'], maxdd=rr['windows']['full']['maxdd'],
                                                     avg_stock_frac=rr['extra']['avg_stock_frac_from_first'])
out['H2_rank_weight_placebo'] = h2

# H3
h3 = {}
years19 = list(range(2019, 2027))
for c in CELLS:
    yr = res['cells'][c]['runs']['strict']['yearly']
    d = {k: v['strat'] - v['e0050'] for k, v in yr.items() if k != '2015'}
    days = (DE - D19).days
    v_all = cagr_from_yearly(yr, years19); v_sw = cagr_from_yearly(yr, years19, swap=2020)
    h3[c] = dict(n_years=len(d), n_beat=int(sum(v > 0 for v in d.values())), lost_years=[k for k, v in d.items() if v <= 0],
                 since2019_check=float(v_all ** (365.25 / days) - 1), since2019_with_2020_as_0050=float(v_sw ** (365.25 / days) - 1))
yb = {str(k): v for k, v in res['bench_0050_yearly'].items()}
h3['0050'] = dict(since2019_with_2020_as_0050=float(np.prod([1 + yb[str(y)] for y in years19]) ** (365.25 / (DE - D19).days) - 1))
out['H3_years'] = h3

# H4
h4 = {}
for c in ('W10_70', 'WN_70', 'RWN_70', 'RW10_70', 'WN_100'):
    y0 = stlib.yearly(nv[f'{c}|strict'].dropna(), b); y5 = stlib.yearly(nv[f'{c}|strict|shift5'].dropna(), b)
    h4[c] = {str(k): dict(shift0=y0[k]['strat'], shift5=y5[k]['strat']) for k in y0 if k >= 2016}
out['H4_shift5_yearly'] = h4

# H5
ex = res['cells']['WN_70']['runs']['strict']['extra']
evs = sorted(ex['base_sold']['events'], key=lambda e: -e['amt'])[:5]
out['H5_main_base_sold_top5'] = [dict(e, pct_nav=float(e['amt'] / nv['WN_70|strict'].loc[pd.Timestamp(e['date'])])) for e in evs]
nav, lg, st = s6lib.simulate_s6(P, S, U, sat_w=0.70, sizing='WN', cost=COSTS['strict'])
assert abs(stlib.cagr(nav, D19, DE) - res['cells']['WN_70']['runs']['strict']['windows']['since2019']['cagr']) < 1e-9
# 拉回日：重算衛星占比 > 80% 的換股執行日（用收盤 B 與淨值近似，只列日期）
exd = sorted(int(x['k']) + 1 for x in S['info'])
satr = 1 - st['B'] / nav
out['H5_main_exec_days_sat_gt80_close_prev'] = [str(P.IDX[t].date()) for t in exd if t - 1 >= P.t0 and satr.iloc[t - 1 - P.t0] > 0.80][:60]
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'H4_shift5_yearly'}, ensure_ascii=False, indent=0)[:6000])
for c, d in h4.items():
    print(c, {y: (round(v['shift0'] * 100, 1), round(v['shift5'] * 100, 1)) for y, v in d.items()})
