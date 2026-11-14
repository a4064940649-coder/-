#!/usr/bin/env python3
"""S2 事後描述（看過報酬後才加，不改分類、不新增回測）：只讀 results.json、trades.csv.gz、navs.csv.gz → results_posthoc.json。python3 -I -B posthoc.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
r = json.load(open(H / 'results.json'))
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str, 'ind2': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
out = {}


def st(x):
    if not len(x):
        return dict(n=0)
    w = x[x['ret'] > 0]['ret']; l = x[x['ret'] <= 0]['ret']; p = len(w) / len(x)
    return dict(n=int(len(x)), mean=float(x['ret'].mean()), median=float(x['ret'].median()), win=float(p), avg_win=float(w.mean()) if len(w) else 0.0, avg_loss=float(-l.mean()) if len(l) else 0.0,
                share_ge30=float((x['ret'] >= 0.30).mean()), share_le_m15=float((x['ret'] <= -0.15).mean()))


b = tr[(tr['run'] == 'B0|cash|strict') & (tr['why'] != 'open_end')].copy()
b19 = b[b['ent_date'] >= '2019-01-01']; b24 = b[b['ent_date'] >= '2024-01-01']
# 1. B0 交易依「訊號日是否在季線上」
out['b0_by_ma60'] = {nm: {str(k): st(x[x['above_ma60'] == k]) for k in (True, False)} for nm, x in (('all', b), ('since2019', b19), ('since2024', b24))}
# 2. 依訊號日前 20 日漲幅三等分（切點用全部已平倉交易，樣本內）
q = b['pre20'].quantile([1 / 3, 2 / 3]).tolist(); out['pre20_cuts'] = q
grp = lambda x: np.where(x['pre20'] <= q[0], 'low', np.where(x['pre20'] <= q[1], 'mid', 'high'))
out['b0_by_pre20'] = {nm: {k: st(x[grp(x) == k]) for k in ('low', 'mid', 'high')} for nm, x in (('all', b), ('since2019', b19), ('h1_to_2021_02', b[b['ent_date'] < '2021-02-19']), ('h2_from_2021_02', b[b['ent_date'] >= '2021-02-19']))}
# 3. 依股價、成交值
out['b0_by_price'] = {k: st(x) for k, x in (('lt20', b[b['close_sig'] < 20]), ('20_50', b[(b['close_sig'] >= 20) & (b['close_sig'] < 50)]), ('50_100', b[(b['close_sig'] >= 50) & (b['close_sig'] < 100)]), ('ge100', b[b['close_sig'] >= 100]))}
out['b0_by_v60'] = {k: st(x) for k, x in (('lt_0.5e8', b[b['v60'] < 5e7]), ('0.5_2e8', b[(b['v60'] >= 5e7) & (b['v60'] < 2e8)]), ('2_10e8', b[(b['v60'] >= 2e8) & (b['v60'] < 1e9)]), ('ge_10e8', b[b['v60'] >= 1e9]))}
out['b0_by_nearhi'] = {k: st(x) for k, x in (('lt0.8', b[b['near_hi240'] < 0.8]), ('0.8_0.9', b[(b['near_hi240'] >= 0.8) & (b['near_hi240'] < 0.9)]), ('ge0.9', b[b['near_hi240'] >= 0.9]))}
out['b0_by_hold'] = {k: st(x) for k, x in (('le23', b[b['hold_days'] <= 23]), ('24_46', b[(b['hold_days'] > 23) & (b['hold_days'] <= 46)]), ('gt46', b[b['hold_days'] > 46]))}
out['b0_by_gap'] = {k: st(x) for k, x in (('gap_le0', b[b['gap_open'] <= 0]), ('gap_0_2', b[(b['gap_open'] > 0) & (b['gap_open'] <= 0.02)]), ('gap_gt2', b[b['gap_open'] > 0.02]))}
# 4. 名單：第 10 名門檻、通過檔數、和 B0 的重疊
T = r['targets']
out['targets'] = {}
b0 = {x['sig_date']: set(x['targets']) for x in T['B0']}
for g in T:
    n10 = [x['scores'][-1] for x in T[g] if len(x['targets']) >= (20 if g == 'D5' else 10)]
    out['targets'][g] = dict(median_last_score=float(np.median(n10)), q25=float(np.percentile(n10, 25)), q75=float(np.percentile(n10, 75)), median_n_ok=float(np.median([x['n_ok'] for x in T[g]])),
                             min_n_ok=int(min(x['n_ok'] for x in T[g])), months_lt_full=int(sum(len(x['targets']) < (20 if g == 'D5' else 10) for x in T[g])),
                             mean_overlap_with_B0=float(np.mean([len(set(x['targets']) & b0[x['sig_date']]) for x in T[g]])))
# 5. 打平 0050 的額外滑價（事後，線性內插／外推：用 light=0、strict=0.3%、stress05=0.5% 三點的後兩點斜率）
out['breakeven_xslip'] = {}
for g, R in r['grid'].items():
    for wn in ('since2019', 'full'):
        c0, c3, c5 = (R['runs'][f'cash|{k}']['windows'][wn]['cagr'] for k in ('light', 'strict', 'stress05'))
        e = R['runs']['cash|light']['windows'][wn]['cagr_0050']
        slope = (c5 - c3) / 0.2
        out['breakeven_xslip'].setdefault(g, {})[wn] = dict(light=c0, strict=c3, stress05=c5, per_0p1pct=slope, breakeven_pct=float(0.5 + (c5 - e) / (-slope)) if slope < 0 else None)
# 6. 逐年：各格（嚴、空位現金）減 B0
out['yearly_minus_B0'] = {g: {y: R['runs']['cash|strict']['yearly'][y]['strat'] - r['grid']['B0']['runs']['cash|strict']['yearly'][y]['strat'] for y in R['runs']['cash|strict']['yearly']} for g, R in r['grid'].items() if g != 'B0'}
out['years_beat_0050_strict'] = {g: [y for y, v in R['runs']['cash|strict']['yearly'].items() if int(y) >= 2016 and v['strat'] > v['e0050']] for g, R in r['grid'].items()}
# 7. D1 與 B0 的交易差異（嚴、空位現金）
d = tr[(tr['run'] == 'D1|cash|strict') & (tr['why'] != 'open_end')]
kb = set(zip(b['code'], b['ent_date'])); kd = set(zip(d['code'], d['ent_date']))
out['d1_vs_b0_trades'] = dict(common=len(kb & kd), only_b0=st(b[[k not in kd for k in zip(b['code'], b['ent_date'])]]), only_d1=st(d[[k not in kb for k in zip(d['code'], d['ent_date'])]]))
# 8. 月超額（對 0050）t：各格嚴、空位現金
nv = pd.read_csv(H / 'navs.csv.gz', index_col=0, parse_dates=True)
m = nv.groupby(nv.index.to_period('M')).last().pct_change().dropna()
out['monthly_excess_vs_0050'] = {}
for g in r['grid']:
    for a in ('2016-03', '2019-01', '2024-01'):
        x = (m[f'{g}|cash|strict'] - m['0050'])[m.index >= pd.Period(a, 'M')]
        out['monthly_excess_vs_0050'].setdefault(g, {})[a] = dict(n=int(len(x)), mean=float(x.mean()), t=float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), win=float((x > 0).mean()))
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
f = lambda d: {k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in d.items()}
for k in ('b0_by_ma60',):
    for a, v in out[k].items():
        for c, x in v.items(): print(k, a, c, f(x))
for k in ('b0_by_pre20',):
    for a, v in out[k].items():
        for c, x in v.items(): print(k, a, c, f(x))
for k in ('b0_by_price', 'b0_by_v60', 'b0_by_nearhi', 'b0_by_hold', 'b0_by_gap'):
    for c, x in out[k].items(): print(k, c, f(x))
print('cuts', out['pre20_cuts'])
for g, v in out['targets'].items(): print(g, {k: round(x, 3) for k, x in v.items()})
for g, v in out['breakeven_xslip'].items(): print(g, {w: (round(x['per_0p1pct'] * 10, 2), round(x['breakeven_pct'], 3)) for w, x in v.items()})
for g, v in out['yearly_minus_B0'].items(): print(g, {y: round(x * 100, 1) for y, x in v.items()})
print(out['years_beat_0050_strict'])
print({k: (f(v) if isinstance(v, dict) else v) for k, v in out['d1_vs_b0_trades'].items()})
for g, v in out['monthly_excess_vs_0050'].items(): print(g, {a: (round(x['mean'] * 100, 2), round(x['t'], 2), round(x['win'], 2)) for a, x in v.items()})
