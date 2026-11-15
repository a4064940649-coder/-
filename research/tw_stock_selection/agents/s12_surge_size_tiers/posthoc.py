#!/usr/bin/env python3
"""S12 事後描述（PREREG posthoc_planned；不影響判定）：只讀 results.json、trades.csv.gz、navs.csv.gz → results_posthoc.json。python3 -I -B posthoc.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
R = json.loads((H / 'results.json').read_text())
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
nv = pd.read_csv(H / 'navs.csv.gz', parse_dates=['date']).set_index('date')
TN = {0: '大', 1: '中', 2: '小'}


def tstats(x):
    c = x[x['why'] != 'open_end']
    if len(c) == 0:
        return dict(n=0)
    w = c[c['ret'] > 0]['ret']; l = c[c['ret'] <= 0]['ret']; p = len(w) / len(c)
    return dict(n=int(len(c)), win_rate=float(p), avg_win=float(w.mean()) if len(w) else 0.0, avg_loss=float(-l.mean()) if len(l) else 0.0,
                expectancy=float(p * (w.mean() if len(w) else 0) - (1 - p) * (-l.mean() if len(l) else 0)), mean_ret=float(c['ret'].mean()),
                median_ret=float(c['ret'].median()), avg_hold=float(c['hold_days'].mean()), pnl_sum=float(x['pnl'].sum()))


out = {'note': '事後描述，PREREG posthoc_planned；不影響判定。交易＝100 萬主跑；pnl 單位元。'}
per = {}
for run, g in tr.groupby('run'):
    d = dict(all=tstats(g), r2019_2023=tstats(g[(g['ent_date'] >= '2019-01-01') & (g['ent_date'] < '2024-01-01')]), r2024=tstats(g[g['ent_date'] >= '2024-01-01']))
    d['by_tier_at_entry'] = {TN[t]: tstats(gg) for t, gg in g.groupby('tier_mc')}
    if run == 'Q55':
        d['by_sleeve'] = {s: tstats(gg) for s, gg in g.groupby('sleeve')}
    tot = g['pnl'].sum()
    d['pnl_share_by_tier'] = {TN[t]: float(gg['pnl'].sum() / tot) for t, gg in g.groupby('tier_mc')}
    for nm, (a, b) in {'2019_2023': ('2019-01-01', '2024-01-01'), '2024': ('2024-01-01', '2100-01-01')}.items():
        gg = g[(g['ent_date'] >= a) & (g['ent_date'] < b)]
        t2 = gg['pnl'].sum()
        d[f'pnl_share_by_tier_{nm}'] = {TN[t]: float(x['pnl'].sum() / t2) for t, x in gg.groupby('tier_mc')} if t2 != 0 else {}
    top = g.sort_values('pnl', ascending=False).head(5)
    d['top5'] = [dict(code=r.code, sleeve=r.sleeve, tier=TN[int(r.tier_mc)], ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), ret=float(r.ret), pnl=float(r.pnl),
                      mc_e8=float(r.mc_e8), why=r.why) for r in top.itertuples()]
    d['top5_share_of_pnl_sum'] = float(top['pnl'].sum() / tot) if tot else None
    w = g['hold_days'].clip(lower=1)
    mc = g['mc_e8']
    o = np.argsort(mc.to_numpy()); cw = np.cumsum(w.to_numpy()[o]) / w.sum()
    d['holding_mc_e8'] = dict(weighted_mean=float((mc * w).sum() / w.sum()), weighted_median=float(mc.to_numpy()[o][np.searchsorted(cw, 0.5)]),
                              entry_median=float(mc.median()), entry_min=float(mc.min()), entry_max=float(mc.max()))
    per[run] = d
out['per_run'] = per
out['maxdd_dates'] = {c: dict(main=R['cells'][c]['dd_dates'], paths=sorted(set(tuple(v['dd_dates']) for v in R['cells'][c]['paths'].values()))) for c in R['cells']}
out['yearly'] = {c: {y: round(v['strat'] * 100, 2) for y, v in R['cells'][c]['yearly'].items()} for c in R['cells']}
out['yearly']['0050'] = {y: round(v['e0050'] * 100, 2) for y, v in R['cells']['CUR']['yearly'].items()}
out['Q55_L_book_share'] = R['cells']['Q55'].get('L_book_share')
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
for run, d in per.items():
    print(run, {k: (round(v['expectancy'], 4), round(v['win_rate'], 3), v['n']) for k, v in d['by_tier_at_entry'].items() if v.get('n')}, 'pnl 層占比', {k: round(v, 3) for k, v in d['pnl_share_by_tier'].items()},
          '持股市值(億，加權中位數)', round(d['holding_mc_e8']['weighted_median']))
    print('   top5', [(x['code'], x['tier'], round(x['ret'], 2)) for x in d['top5']], 'all', round(d['all']['expectancy'], 4), round(d['all']['win_rate'], 3), d['all']['n'])
    print('   2019-23', {k: round(v, 3) for k, v in d['pnl_share_by_tier_2019_2023'].items()}, '2024+', {k: round(v, 3) for k, v in d['pnl_share_by_tier_2024'].items()})
print(json.dumps(out['maxdd_dates'], ensure_ascii=False)[:2000])
print(json.dumps(out['yearly'], ensure_ascii=False))

# ---------------------------------------------------------------- 拆解的另一種分組（事後）：選股按實際權重（每檔 1/10，空位放同層等權）
ew = pd.DataFrame(R['ew_monthly_shift0']); ew['date'] = pd.to_datetime(ew['date'])
IDX = nv.index


def ann(r, d0, d1):
    y = (d1 - d0).days / 365.25
    return float(np.prod(1 + np.asarray(r)) ** (1 / y) - 1)


PER = {'full': ('2000-01-01', '2100-01-01'), 'since2019': ('2019-01-01', '2100-01-01'), 'y2019_2023': ('2019-01-01', '2023-12-31'), 'since2024': ('2024-01-01', '2100-01-01')}
regroup = {}
for cell, tk in (('CUR', 'CUR'), ('TL', 'L'), ('TM', 'M'), ('TS', 'S'), ('TL_ADV', 'L_ADV')):
    n = ew[f'n_{tk}'].clip(upper=10) / 10
    w10ew = n * ew[f'list_{tk}'] + (1 - n) * ew[f'ew_{tk}']
    dcmp = R['cells'][cell]['decomposition']
    regroup[cell] = {}
    for p, (a, b) in PER.items():
        m = ((ew['date'] >= a) & (ew['date'] <= b)).to_numpy()
        sub = ew[m]; d0 = sub['date'].iloc[0]
        e_end = int(sub['e2'].iloc[-1])
        d1 = pd.Timestamp(dcmp[p]['end'])
        a_ew = ann(0.3 * sub['r50'] + 0.7 * sub[f'ew_{tk}'], d0, d1)
        a_w10ew = ann(0.3 * sub['r50'] + 0.7 * w10ew[m], d0, d1)
        a_w10 = ann(0.3 * sub['r50'] + 0.7 * sub[f'w10_{tk}'], d0, d1)
        dd = dcmp[p]
        regroup[cell][p] = dict(tier_lag=dd['tier_lag'], selection_at_actual_weight=(a_w10ew - a_ew) * 100, empty_slots_0050_vs_tier=(a_w10 - a_w10ew) * 100,
                                exec_rest=dd['exec_rest'], total=dd['total'], e0050=dd['e0050'] * 100, actual=dd['actual'] * 100)
out['decomposition_regroup_posthoc'] = regroup
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
for c, v in regroup.items():
    print('拆解(事後分組)', c, {p: {k: round(x, 2) for k, x in d.items()} for p, d in v.items()})
