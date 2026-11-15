#!/usr/bin/env python3
"""S7 事後分析（看過 results.json 之後才加；不影響分類、主格與取捨表）。python3 -I -B posthoc.py → results_posthoc.json
Q1 底倉占淨值：< 10%、< 20%、< 1% 的交易日比例，平均、中位數（第一筆進場日起）。
Q2 衛星現金不夠：縮小／跳過的次數、少買金額占目標的比例、「實質縮小」（買到 < 目標 90%）、逐年次數、受影響的換股月數。
Q3 滾動 12 個月最多落後 0050 幾個百分點（重列）。
Q4 逐年報酬、逐年贏 0050 的年數；2020 換成 0050 當年報酬後的 2019 起年化。
Q5 20 條路徑逐條相減：鎖定 − 不鎖定（WN_70L − WN_70、W10_70L − W10_70）。
Q6 最大回撤的解剖：高點日的個股部位與底倉占比、回撤期間平均個股部位。
Q7 執行延遲 × 5 種資金（60、80、100、150、200 萬）：主要 3 格 d＝0／1／2 的 2019 起中位數；WN_70L 延遲的逐年報酬。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7lib
from s7lib import s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
t0 = time.time()
res = json.load(open(H / 'results.json'))
nv = pd.read_csv(H / 'navs.csv.gz', index_col=0, parse_dates=True)
b = nv['0050']
CELLS = list(res['cells'].keys())
D19 = pd.Timestamp(res['meta']['since2019_from']); D0 = pd.Timestamp(res['meta']['start']); DE = pd.Timestamp(res['meta']['end'])
out = {}

# Q1
q1 = {}
for c in CELLS:
    ex = res['cells'][c]['runs']['strict']['extra']
    first = pd.Timestamp(ex['first_entry'])
    bf = (nv[f'{c}|strict|B'] / nv[f'{c}|strict']).loc[first:]
    q1[c] = dict(frac_days_lt10=float((bf < 0.10).mean()), frac_days_lt20=float((bf < 0.20).mean()), frac_days_lt1=float((bf < 0.01).mean()),
                 mean=float(bf.mean()), median=float(bf.median()), min=float(bf.min()),
                 since2019_frac_days_lt10=float((bf.loc[D19:] < 0.10).mean()))
out['Q1_base_frac'] = q1

# Q2
q2 = {}
for c in CELLS:
    ev = pd.DataFrame(res['cells'][c]['runs']['strict']['extra']['short_events'])
    if not len(ev):
        q2[c] = dict(n_shrunk=0, n_skip=0); continue
    ev['date'] = pd.to_datetime(ev['date']); ev['yr'] = ev['date'].dt.year
    sh = ev[ev['kind'] == 'shrunk']; sk = ev[ev['kind'] == 'skip']
    fr = (sh['bought'] / sh['target'])
    q2[c] = dict(n_shrunk=int(len(sh)), n_skip=int(len(sk)), n_shrunk_material_lt90=int((fr < 0.9).sum()),
                 shrunk_bought_over_target_median=float(fr.median()) if len(fr) else None,
                 shortfall_amt_total=float((sh['target'] - sh['bought']).sum()), skip_target_total=float(sk['target'].sum()),
                 shortfall_pct_nav_sum=float(((sh['target'] - sh['bought']) / sh['nav']).sum()), skip_target_pct_nav_sum=float((sk['target'] / sk['nav']).sum()),
                 n_exec_days_affected=int(ev['date'].nunique()),
                 by_year={int(y): dict(shrunk=int((g['kind'] == 'shrunk').sum()), skip=int((g['kind'] == 'skip').sum())) for y, g in ev.groupby('yr')})
out['Q2_short_cash'] = q2

# Q3
out['Q3_rolling12'] = {c: res['cells'][c]['runs']['strict']['rolling12'] for c in CELLS}


# Q4
def cagr_from_yearly(yr, years, swap=None):
    v = 1.0
    for y in years:
        v *= 1 + (yr[str(y)]['e0050'] if swap == y else yr[str(y)]['strat'])
    return v


years19 = list(range(2019, 2027)); days = (DE - D19).days
q4 = {}
for c in CELLS:
    yr = res['cells'][c]['runs']['strict']['yearly']
    d = {k: v['strat'] - v['e0050'] for k, v in yr.items() if k != '2015'}
    q4[c] = dict(yearly={k: v['strat'] for k, v in yr.items()}, n_years=len(d), n_beat=int(sum(v > 0 for v in d.values())), lost_years=[k for k, v in d.items() if v <= 0],
                 since2019_check=float(cagr_from_yearly(yr, years19) ** (365.25 / days) - 1),
                 since2019_with_2020_as_0050=float(cagr_from_yearly(yr, years19, swap=2020) ** (365.25 / days) - 1))
yb = {str(k): v for k, v in res['bench_0050_yearly'].items()}
q4['0050'] = dict(yearly=yb, since2019_with_2020_as_0050=float(np.prod([1 + yb[str(y)] for y in years19]) ** (365.25 / days) - 1))
for lk, ul in (('WN_70L', 'WN_70'), ('W10_70L', 'W10_70'), ('WN_80L', 'WN_70')):
    a = q4[lk]['yearly']; bb = q4[ul]['yearly']
    q4[f'{lk}_minus_{ul}_yearly'] = {k: a[k] - bb[k] for k in a if k != '2015'}
out['Q4_years'] = q4

# Q5
q5 = {}
for lk, ul in (('WN_70L', 'WN_70'), ('W10_70L', 'W10_70'), ('WN_80L', 'WN_70')):
    pa = res['cells'][lk]['paths20']['values']; pb = res['cells'][ul]['paths20']['values']
    dd_ = np.array([(pa[k] - pb[k]) * 100 for k in pa])
    q5[f'{lk}_minus_{ul}'] = dict(min=float(dd_.min()), median=float(np.median(dd_)), max=float(dd_.max()), n_pos=int((dd_ > 0).sum()), n=int(len(dd_)),
                                  values={k: round((pa[k] - pb[k]) * 100, 2) for k in pa})
out['Q5_paths_paired'] = q5

# Q6
q6 = {}
for c in CELLS:
    dd = res['cells'][c]['runs']['strict']['dd_dates']
    pk, tr = pd.Timestamp(dd['peak']), pd.Timestamp(dd['trough'])
    sf = nv[f'{c}|strict|stockfrac']; bf = nv[f'{c}|strict|B'] / nv[f'{c}|strict']
    q6[c] = dict(peak=dd['peak'], trough=dd['trough'], maxdd=dd['maxdd'], stockfrac_at_peak=float(sf.loc[pk]), base_frac_at_peak=float(bf.loc[pk]),
                 stockfrac_mean_in_dd=float(sf.loc[pk:tr].mean()), base_frac_mean_in_dd=float(bf.loc[pk:tr].mean()),
                 e0050_same_window=float(b.loc[tr] / b.loc[pk] - 1))
out['Q6_maxdd_anatomy'] = q6

# Q7（需要重跑）
P = s7lib.load()
U = stlib.universe(P, 'twse') & ~s4lib.innov_panel(P)[0]
RT = s2lib.rev_tables(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed'):
    if hasattr(P, a):
        delattr(P, a)
U_ALL = np.ones((P.T, P.N), bool)
S0 = s4lib.make_signal(P, U, RT, depth=10, mom_lo=0.10, exit='top20', keep_info=True)
SPEC = {c['id']: c for c in json.load(open(H / 'PREREG.json'))['grid']['cells']}
q7 = {}
bench19 = res['bench_0050']['since2019']['cagr']
for d in (0, 1, 2):
    Sd = s7lib.delay_signal(P, S0, d)
    for c in ('WN_70L', 'W10_70L', 'WN_70'):
        sp = SPEC[c]
        vals = {}
        for cap in (6e5, 8e5, 1e6, 1.5e6, 2e6):
            nav = s7lib.simulate_s7(P, Sd, U_ALL, sat_w=sp['sat_w'], sizing=sp['sizing'], lock=sp['lock'], cost=COSTS['strict'], capital=cap)[0]
            vals[f'{int(cap/1e4)}萬'] = stlib.cagr(nav, D19, DE)
        if d == 0:
            assert abs(vals['100萬'] - res['cells'][c]['runs']['strict']['windows']['since2019']['cagr']) < 1e-9
        v = np.array(list(vals.values()))
        q7.setdefault(c, {})[str(d)] = dict(values=vals, min=float(v.min()), median=float(np.median(v)), max=float(v.max()), n_above_0050=int((v > bench19).sum()))
    del Sd
for c in q7:
    for d in ('1', '2'):
        q7[c][d]['median_cost_pp'] = (q7[c][d]['median'] - q7[c]['0']['median']) * 100
        q7[c][d]['paired_cost_pp'] = {k: (q7[c][d]['values'][k] - q7[c]['0']['values'][k]) * 100 for k in q7[c]['0']['values']}
out['Q7_delay_capitals'] = q7
q7y = {}
for c in ('WN_70L', 'W10_70L', 'WN_70'):
    y0 = stlib.yearly(nv[f'{c}|strict'].dropna(), b)
    q7y[c] = {}
    for d in (1, 2):
        yd = stlib.yearly(nv[f'{c}|strict|delay{d}'].dropna(), b)
        q7y[c][str(d)] = {str(k): (yd[k]['strat'] - y0[k]['strat']) * 100 for k in y0 if k >= 2016}
    q7y[c]['dd_dates_delay'] = {str(d): s7lib.dd_dates(nv[f'{c}|strict|delay{d}'].dropna()) for d in (1, 2)}
out['Q7_delay_yearly_pp'] = q7y
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ('Q3_rolling12',)}, ensure_ascii=False, default=str)[:20000])
