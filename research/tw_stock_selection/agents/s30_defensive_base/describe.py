#!/usr/bin/env python3
"""S30 防禦組合單獨描述（FROZEN 之後才跑；不進判定）：LOWB、LBQ、HDLV、MIX、0050。
期間：2016-01-29 → 2026-10-22（全期）、訓練窗、驗證窗；年化、年化波動、最大回撤、對 0050 日報酬 CAPM β、四段空頭跌幅、逐年、每季換手與成本。
輸出 results_baskets.json、basket_navs.csv.gz。python3 -I -B describe.py"""
import sys, json, hashlib, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s30lib
from s30lib import s17lib, s4lib, stlib

H = Path(__file__).resolve().parent
assert (H / 'FROZEN.json').exists(), '先跑 train.py'
fz = (H / 'FROZEN.json').read_bytes()
assert hashlib.sha256(fz).hexdigest() == (H / 'FROZEN.sha256').read_text().split()[0]
t_all = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s30lib.load()
INNOV, _ = s4lib.innov_panel(P)
s17lib.e50_fix_suspension(P)
Hd, info, FR = s30lib.holdings(P, INNOV)
IDXS = {b: s30lib.basket_index(P, Hd[b], FR) for b in s30lib.BASKETS}
IDXS['MIX'] = s30lib.mix_index(P, IDXS['LBQ'])
bench = pd.Series(P.E50, index=P.IDX)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
A, TB, DE = lastb('2016-02-01'), lastb('2022-01-01'), P.IDX[-1]
WIN = {'full_2016': (A, DE), 'train': (A, TB), 'valid': (TB, DE), 'since2024': (lastb('2024-01-01'), DE)}
out = {'windows': {k: [str(a.date()), str(b.date())] for k, (a, b) in WIN.items()}, 'baskets': {}}
series = {'E50': bench}
for b, D in IDXS.items():
    series[b] = pd.Series(D['Dc'], index=P.IDX)
for nm, s in series.items():
    o = {}
    for k, (a, b) in WIN.items():
        o[k] = s30lib.describe_index(s.to_numpy(), bench, a, b)
    o['yearly'] = {int(y): float(v) for y, v in (lambda x: (x.groupby(x.index.year).last() / x.groupby(x.index.year).last().shift(1) - 1).dropna())(s.loc[lastb('2016-01-01'):]).items()}
    if nm in s30lib.BASKETS:
        tr = IDXS[nm]['turns']
        o['turnover'] = dict(first_switch_cost=float(tr[0]['cost']), avg_quarter_oneway_turnover=float(np.mean([x['turnover'] for x in tr[1:]])),
                             avg_quarter_cost=float(np.mean([x['cost'] for x in tr[1:]])), annual_cost_drag=float(4 * np.mean([x['cost'] for x in tr[1:]])))
    if nm == 'MIX':
        o['turnover'] = dict(total_rebalance_cost=IDXS['MIX']['turns'][0]['cost'], n_rebalances=IDXS['MIX']['turns'][0]['n'])
    out['baskets'][nm] = o
out['ex_ante'] = info[-1]
df = pd.DataFrame({k: v.loc[P.IDX[P.t0]:] for k, v in series.items()})
df.index.name = 'date'
df.round(6).to_csv(H / 'basket_navs.csv.gz')
out['meta'] = dict(runtime_s=round(time.time() - t_all, 1), maxrss_mb=rss())
(H / 'results_baskets.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
for nm, o in out['baskets'].items():
    print(nm, {k: dict(cagr=round(o[k]['cagr'] * 100, 2), vol=round(o[k]['vol'] * 100, 2), maxdd=round(o[k]['maxdd'] * 100, 2), beta=round(o[k]['capm']['beta'], 3),
                       seg={a: round(v * 100, 1) for a, v in o[k]['seg'].items() if np.isfinite(v)}) for k in WIN}, o.get('turnover'))
    print('   yearly', {y: round(v * 100, 1) for y, v in o['yearly'].items()})
