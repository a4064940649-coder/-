#!/usr/bin/env python3
"""S8 事後分析（看過 results.json 之後才跑；不影響分類、主格、判定與取捨表）。python3 -I -B posthoc.py → results_posthoc.json
Q1 逐年報酬、贏 0050 年數；2020 換成 0050 後的 2019 起年化；主格 − WN_70L、主格 − W10_70L、WN_60L − WN_70L 逐年差。
Q2 20 條路徑逐條相減：主格對 WN_70L、主格對 W10_70L（另列 WN_60L、WN_70L_C08 對 WN_70L）：2019 起報酬差與全期最大回撤差，正的條數。
Q3 最大回撤解剖：高點日／回撤期間的個股部位、底倉占比、衛星閒置、持股檔數；同期 0050；回撤期間各部位的貢獻。
Q4 單檔上限：觸發次數、逐年、依名單檔數 n 分、省下金額；省下的錢去哪了（個股部位、衛星閒置、底倉占比對 WN_70L 的差；跳過次數）。
Q5 滾動 12 個月最多落後 0050（重列）。
Q6 延遲 × 5 種資金（60、80、100、150、200 萬）：主格、W10_70L 的 d＝0／1／2 2019 起中位數；主格延遲的逐年損失。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s8lib
from s8lib import s4lib, s2lib, stlib, COSTS

H = Path(__file__).resolve().parent
t0 = time.time()
res = json.load(open(H / 'results.json'))
nv = pd.read_csv(H / 'navs.csv.gz', index_col=0, parse_dates=True)
tr = pd.read_csv(H / 'trades.csv.gz', parse_dates=['ent_date', 'exit_date'])
b = nv['0050']
CELLS = list(res['cells'].keys())
MAIN = 'WN_70L_C12'
D19 = pd.Timestamp(res['meta']['since2019_from']); D0 = pd.Timestamp(res['meta']['start']); DE = pd.Timestamp(res['meta']['end'])
out = {}


# Q1
def cagr_from_yearly(yr, years, swap=None):
    v = 1.0
    for y in years:
        v *= 1 + (yr[str(y)]['e0050'] if swap == y else yr[str(y)]['strat'])
    return v


years19 = list(range(2019, 2027)); days = (DE - D19).days
q1 = {}
for c in CELLS:
    yr = res['cells'][c]['runs']['strict']['yearly']
    d = {k: v['strat'] - v['e0050'] for k, v in yr.items() if k != '2015'}
    q1[c] = dict(yearly={k: v['strat'] for k, v in yr.items()}, n_years=len(d), n_beat=int(sum(v > 0 for v in d.values())), lost_years=[k for k, v in d.items() if v <= 0],
                 since2019_with_2020_as_0050=float(cagr_from_yearly(yr, years19, swap=2020) ** (365.25 / days) - 1))
yb = {str(k): v for k, v in res['bench_0050_yearly'].items()}
q1['0050'] = dict(yearly=yb, since2019_with_2020_as_0050=float(np.prod([1 + yb[str(y)] for y in years19]) ** (365.25 / days) - 1))
for a_, b_ in ((MAIN, 'WN_70L'), (MAIN, 'W10_70L'), ('WN_60L', 'WN_70L'), ('WN_70L_C08', 'WN_70L')):
    ya = q1[a_]['yearly']; yb_ = q1[b_]['yearly']
    dd_ = {k: ya[k] - yb_[k] for k in ya if k != '2015'}
    q1[f'{a_}_minus_{b_}_yearly'] = dict(diff=dd_, n_pos=int(sum(v > 0 for v in dd_.values())), n=len(dd_))
out['Q1_years'] = q1

# Q2
q2 = {}
for a_, b_ in ((MAIN, 'WN_70L'), (MAIN, 'W10_70L'), ('WN_60L', 'WN_70L'), ('WN_70L_C08', 'WN_70L'), ('WN_70L_C10', 'WN_70L'), ('WN_60L_C10', 'WN_70L'), ('WN_60L', MAIN)):
    pa = res['cells'][a_]['paths20']; pb = res['cells'][b_]['paths20']
    rd = np.array([(pa['values'][k] - pb['values'][k]) * 100 for k in pa['values']])
    dd_ = np.array([(pa['maxdd'][k] - pb['maxdd'][k]) * 100 for k in pa['values']])   # 正＝a 的回撤較淺
    q2[f'{a_}_minus_{b_}'] = dict(ret_pp=dict(min=float(rd.min()), median=float(np.median(rd)), max=float(rd.max()), n_pos=int((rd > 0).sum())),
                                  maxdd_pp_positive_means_shallower=dict(min=float(dd_.min()), median=float(np.median(dd_)), max=float(dd_.max()), n_pos=int((dd_ > 0).sum())),
                                  n=int(len(rd)), values={k: dict(ret=round(float(r_), 2), dd=round(float(x_), 2)) for k, r_, x_ in zip(pa['values'], rd, dd_)})
out['Q2_paths_paired'] = q2
out['Q2_paths_maxdd_by_cell'] = {c: dict(min=res['cells'][c]['paths20']['maxdd_min'], median=res['cells'][c]['paths20']['maxdd_median'], max=res['cells'][c]['paths20']['maxdd_max'],
                                          n_within_36=res['cells'][c]['paths20']['n_dd_within_36'], values=res['cells'][c]['paths20']['maxdd']) for c in CELLS}


# Q3
def npos_series(c):
    t = tr[tr['run'] == f'{c}|strict']
    idx = nv.index
    cnt = np.zeros(len(idx))
    for e, x in zip(t['ent_date'], t['exit_date']):
        i0 = idx.searchsorted(e); i1 = idx.searchsorted(x)
        cnt[i0:i1] += 1
    return pd.Series(cnt, index=idx)


q3 = {}
for c in CELLS:
    dd = res['cells'][c]['runs']['strict']['dd_dates']
    pk, trd = pd.Timestamp(dd['peak']), pd.Timestamp(dd['trough'])
    nav = nv[f'{c}|strict']; sf = nv[f'{c}|strict|stockfrac']; bf = nv[f'{c}|strict|B'] / nav; si = 1 - sf - bf
    npz = npos_series(c)
    # 回撤期間各部位的日貢獻（前一日權重 × 當日報酬；底倉與衛星閒置都視為 0050）
    r = nav.pct_change(); r50 = b.pct_change()
    w_s, w_b = sf.shift(1), bf.shift(1)
    contrib_50 = ((1 - w_s) * r50).loc[pk:trd].iloc[1:]
    contrib_stock = (r - (1 - w_s) * r50).loc[pk:trd].iloc[1:]
    q3[c] = dict(peak=dd['peak'], trough=dd['trough'], maxdd=dd['maxdd'], stockfrac_at_peak=float(sf.loc[pk]), base_frac_at_peak=float(bf.loc[pk]), sat_idle_at_peak=float(si.loc[pk]),
                 npos_at_peak=float(npz.loc[pk]), stockfrac_mean_in_dd=float(sf.loc[pk:trd].mean()), base_frac_mean_in_dd=float(bf.loc[pk:trd].mean()),
                 sat_idle_mean_in_dd=float(si.loc[pk:trd].mean()), npos_mean_in_dd=float(npz.loc[pk:trd].mean()),
                 per_position_frac_mean_in_dd=float((sf / npz.replace(0, np.nan)).loc[pk:trd].mean()),
                 e0050_same_window=float(b.loc[trd] / b.loc[pk] - 1),
                 contrib_sum_stock=float(contrib_stock.sum()), contrib_sum_0050_parts=float(contrib_50.sum()),
                 stockfrac_mean_full=float(sf.mean()), sat_idle_mean_full=float(si.mean()), base_frac_mean_full=float(bf.mean()))
    # 回撤期間持股（高點日）
    t = tr[(tr['run'] == f'{c}|strict') & (tr['ent_date'] <= pk) & (tr['exit_date'] > pk)]
    q3[c]['held_at_peak'] = t['code'].astype(str).tolist()
out['Q3_maxdd_anatomy'] = q3

# Q4（需要重跑主格取得上限觸發與名單檔數）
P = s8lib.load()
U = stlib.universe(P, 'twse') & ~s4lib.innov_panel(P)[0]
RT = s2lib.rev_tables(P)
for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'AO', 'listed'):
    if hasattr(P, a):
        delattr(P, a)
U_ALL = np.ones((P.T, P.N), bool)
S0 = s4lib.make_signal(P, U, RT, depth=10, mom_lo=0.10, exit='top20', keep_info=True)
SPEC = {c['id']: c for c in json.load(open(H / 'PREREG.json'))['grid']['cells']}
ntgt = {int(x['k']): int(x['n_tgt']) for x in S0['info']}


def run(c, S=S0, univ=U, cap=1e6):
    sp = SPEC[c]
    return s8lib.simulate_s8(P, S, univ, sat_w=sp['sat_w'], sizing=sp['sizing'], lock=sp['lock'], cap=sp['cap'], cost=COSTS['strict'], capital=cap)


q4 = {}
ref_ex = res['cells']['WN_70L']['runs']['strict']['extra']
for c in ('WN_70L_C12', 'WN_70L_C10', 'WN_70L_C08', 'WN_60L_C10'):
    nav, lg, st = run(c)
    assert abs(nav.iloc[-1] - res['cells'][c]['runs']['strict']['final_nav']) < 1e-3
    ce = pd.DataFrame(st['cap_events'], columns=['t', 'j', 'u', 'capv', 'bought'])
    ce['n'] = [ntgt[int(t) - 1] for t in ce['t']]
    ce['yr'] = P.IDX[ce['t'].to_numpy()].year
    ce['nav'] = nav.reindex(P.IDX[ce['t'].to_numpy()]).to_numpy()
    ex_days = sorted(set(int(x['k']) + 1 for x in S0['info'] if int(x['k']) + 1 < P.T))
    buy_days = sorted(set(int(x[0]) for x in st['buy_events']))
    exm = res['cells'][c]['runs']['strict']['extra']
    q4[c] = dict(n_cap=int(len(ce)), n_cap_days=int(ce['t'].nunique()), n_exec_days=len(ex_days), n_buy_days=len(buy_days), n_buys=len(st['buy_events']),
                 share_of_buys_capped=float(len(ce) / max(len(st['buy_events']), 1)),
                 by_n={int(k): int(v) for k, v in ce.groupby('n').size().items()},
                 by_year={int(k): int(v) for k, v in ce.groupby('yr').size().items()},
                 saved_amt=float((ce['u'] - ce['capv']).sum()), saved_pct_nav_mean=float(((ce['u'] - ce['capv']) / ce['nav']).mean()),
                 saved_pct_nav_per_cap_day=float(((ce['u'] - ce['capv']) / ce['nav']).groupby(ce['t']).sum().mean()),
                 wn_target_over_cap_median=float((ce['u'] / ce['capv']).median()),
                 where_it_went_vs_WN_70L=dict(
                     avg_stock_frac_diff_pp=(exm['avg_stock_frac_from_first'] - ref_ex['avg_stock_frac_from_first']) * 100,
                     sat_idle_diff_pp=(exm['sat_idle_frac']['mean'] - ref_ex['sat_idle_frac']['mean']) * 100,
                     base_frac_diff_pp=(exm['base_frac']['mean'] - ref_ex['base_frac']['mean']) * 100,
                     avg_npos=exm['avg_npos_from_first'], avg_npos_WN_70L=ref_ex['avg_npos_from_first'],
                     skip_cash=exm['skip_cash'], skip_cash_WN_70L=ref_ex['skip_cash'], shrunk=exm['shrunk'], shrunk_WN_70L=ref_ex['shrunk'],
                     n_buys=len(st['buy_events']), n_buys_WN_70L=ref_ex['cap_summary']['n_buys'],
                     avg_per_position_frac=exm['avg_per_position_frac'], avg_per_position_frac_WN_70L=ref_ex['avg_per_position_frac']))
out['Q4_cap'] = q4

# Q5
out['Q5_rolling12'] = {c: res['cells'][c]['runs']['strict']['rolling12'] for c in CELLS}

# Q6
q6 = {}
bench19 = res['bench_0050']['since2019']['cagr']
for d in (0, 1, 2):
    Sd = s8lib.delay_signal(P, S0, d)
    for c in (MAIN, 'W10_70L'):
        vals = {}; dds = {}
        for cap in (6e5, 8e5, 1e6, 1.5e6, 2e6):
            nav = run(c, Sd, U_ALL, cap)[0]
            vals[f'{int(cap/1e4)}萬'] = stlib.cagr(nav, D19, DE); dds[f'{int(cap/1e4)}萬'] = stlib.maxdd(nav, D0, DE)
        if d == 0:
            assert abs(vals['100萬'] - res['cells'][c]['runs']['strict']['windows']['since2019']['cagr']) < 1e-9
        v = np.array(list(vals.values()))
        q6.setdefault(c, {})[str(d)] = dict(values=vals, maxdd=dds, min=float(v.min()), median=float(np.median(v)), max=float(v.max()), n_above_0050=int((v > bench19).sum()),
                                           maxdd_median=float(np.median(list(dds.values()))))
    del Sd
for c in q6:
    for d in ('1', '2'):
        q6[c][d]['median_cost_pp'] = (q6[c][d]['median'] - q6[c]['0']['median']) * 100
out['Q6_delay_capitals'] = q6
y0 = stlib.yearly(nv[f'{MAIN}|strict'].dropna(), b)
out['Q6_delay_yearly_pp'] = {str(d): {str(k): (stlib.yearly(nv[f'{MAIN}|strict|delay{d}'].dropna(), b)[k]['strat'] - y0[k]['strat']) * 100 for k in y0 if k >= 2016} for d in (1, 2)}
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ('Q5_rolling12', 'Q2_paths_maxdd_by_cell')}, ensure_ascii=False, default=str)[:30000])
