#!/usr/bin/env python3
"""S11 事後（未登記判定，只描述）：大贏家重疊、持股重疊率、Q 每筆期望值與勝率、逐年、各段回撤 20 條路徑中位數。
只讀 results.json、trades.csv.gz、navs.csv.gz。python3 -I -B posthoc.py → results_posthoc.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
R = json.loads((H / 'results.json').read_text())
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
nav = pd.read_csv(H / 'navs.csv.gz', parse_dates=['date']).set_index('date')
out = {}


def overl(a, b):
    return (a.ent_date < b.exit_date) and (b.ent_date < a.exit_date)


# 1. 大贏家重疊：E0 與 Q 主跑各自最賺 10 筆
cl = tr[tr['why'] != 'open_end']
top = {r: tr[tr['run'] == r].sort_values('pnl', ascending=False).head(10) for r in ('E0', 'Q', 'QF')}
rows = []
for r in top['E0'].itertuples():
    q = tr[(tr['run'] == 'Q') & (tr['code'] == r.code)]
    qf = tr[(tr['run'] == 'QF') & (tr['code'] == r.code)]
    rows.append(dict(code=r.code, ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), ret=round(float(r.ret), 4), pnl=round(float(r.pnl)),
                     Q_overlap=[(str(x.ent_date.date()), str(x.exit_date.date()), round(float(x.ret), 4)) for x in q.itertuples() if overl(x, r)],
                     QF_overlap=[(str(x.ent_date.date()), str(x.exit_date.date()), round(float(x.ret), 4)) for x in qf.itertuples() if overl(x, r)]))
out['E0_top10_in_Q'] = rows
out['E0_top10_caught_by_Q'] = int(sum(1 for x in rows if x['Q_overlap']))
out['E0_top10_caught_by_QF'] = int(sum(1 for x in rows if x['QF_overlap']))
rows = []
for r in top['Q'].itertuples():
    e = tr[(tr['run'] == 'E0') & (tr['code'] == r.code)]
    rows.append(dict(code=r.code, ent=str(r.ent_date.date()), exit=str(r.exit_date.date()), ret=round(float(r.ret), 4), pnl=round(float(r.pnl)),
                     E0_overlap=[(str(x.ent_date.date()), str(x.exit_date.date()), round(float(x.ret), 4)) for x in e.itertuples() if overl(x, r)]))
out['Q_top10_in_E0'] = rows
out['Q_top10_caught_by_E0'] = int(sum(1 for x in rows if x['E0_overlap']))
out['top10_codes_shared_any_time'] = sorted(set(top['E0']['code']) & set(top['Q']['code']))

# 2. 持股重疊率（C 主跑：兩半同時持有）
days = nav.index


def held(df):
    H_ = {}
    for x in df.itertuples():
        for d in days[(days >= x.ent_date) & (days < x.exit_date)]:
            H_.setdefault(d, set()).add(x.code)
    return H_


hE = held(tr[(tr['run'] == 'C') & (tr['sleeve'] == 'E0')]); hQ = held(tr[(tr['run'] == 'C') & (tr['sleeve'] == 'Q')])
ov = []; ovE = []
for d in days:
    a, b = hE.get(d, set()), hQ.get(d, set())
    if b:
        ov.append(len(a & b) / len(b))
    if a:
        ovE.append(len(a & b) / len(a))
both = [d for d in days if hE.get(d) and hQ.get(d) and (hE[d] & hQ[d])]
out['holding_overlap'] = dict(share_of_Q_holdings_also_in_E0=float(np.mean(ov)), share_of_E0_holdings_also_in_Q=float(np.mean(ovE)),
                              days_with_any_common=len(both), days_total=len(days))
# 交易層級：Q 的交易有幾筆和 E0 同股票、持有期重疊
qt = tr[(tr['run'] == 'C') & (tr['sleeve'] == 'Q')]; et = tr[(tr['run'] == 'C') & (tr['sleeve'] == 'E0')]
n_ov = sum(1 for x in qt.itertuples() if any(overl(x, y) for y in et[et['code'] == x.code].itertuples()))
out['Q_trades_overlapping_E0_same_stock'] = dict(n=int(n_ov), of=int(len(qt)))

# 3. 每筆期望值、勝率（主跑），並分 2019～2023／2024 起
def ts(x):
    x = x[x['why'] != 'open_end']
    w = x[x['ret'] > 0]['ret']; l = x[x['ret'] <= 0]['ret']; n = len(x)
    if n == 0:
        return dict(n=0)
    p = len(w) / n
    return dict(n=int(n), win_rate=round(p, 4), avg_win=round(float(w.mean()), 4), avg_loss=round(float(-l.mean()), 4) if len(l) else 0.0,
                expectancy=round(float(p * w.mean() - (1 - p) * (-l.mean() if len(l) else 0)), 4), median_ret=round(float(x['ret'].median()), 4),
                avg_hold=round(float(x['hold_days'].mean()), 1), best=round(float(x['ret'].max()), 4), worst=round(float(x['ret'].min()), 4))


out['trade_stats'] = {r: {'all': ts(tr[tr['run'] == r]), '2015_2018': ts(tr[(tr['run'] == r) & (tr['ent_date'] < '2019-01-01')]),
                          '2019_2023': ts(tr[(tr['run'] == r) & (tr['ent_date'] >= '2019-01-01') & (tr['ent_date'] < '2024-01-01')]),
                          '2024_': ts(tr[(tr['run'] == r) & (tr['ent_date'] >= '2024-01-01')])} for r in ('E0', 'Q', 'QF', 'QR')}

# 4. 各段回撤：20 條路徑中位數（results.json 已有），整理成表；逐年
out['seg_dd_path_median'] = {c: R['cells'][c]['paths20']['seg_median'] for c in R['cells']}
out['seg_dd_main'] = {c: R['cells'][c]['seg_dd'] for c in R['cells']} | {'0050': R['bench_0050']['seg_dd']}
out['yearly_main'] = {c: {y: round(v['strat'] * 100, 2) for y, v in R['cells'][c]['yearly'].items()} for c in R['cells']}
out['yearly_main']['0050'] = {y: round(v['e0050'] * 100, 2) for y, v in R['cells']['E0']['yearly'].items()}
# 5. Q 的最賺 10 筆列表
out['Q_top10'] = [dict(code=x.code, ent=str(x.ent_date.date()), exit=str(x.exit_date.date()), ret=round(float(x.ret), 4), pnl=round(float(x.pnl))) for x in top['Q'].itertuples()]
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k not in ('yearly_main',)}, ensure_ascii=False, indent=1, default=str)[:9000])
