#!/usr/bin/env python3
"""事後資料核對（不影響判定）：抽樣事件原始價、極端值、事件日期集中度、主版剔除最大日期後的 t。"""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'w3_limitup_pullback'))
import lim as Q  # noqa: E402
from btlib import tstat  # noqa: E402
ev = pd.read_csv(HERE / 'events.csv.gz', parse_dates=['d0', 'entry'], dtype={'code': str})
IDX = Q.IDX
res = {}
m = ev[ev.ver == 'MAIN_H5']
rng = np.random.default_rng(0)
samp = []
for _, e in m.iloc[rng.choice(len(m), 6, replace=False)].iterrows():
    t0 = int(e.t0); c = e.code
    sl = slice(t0 - 1, t0 + 3)
    samp.append({'code': c, 'd0': str(e.d0.date()), 'dates': [str(d.date()) for d in IDX[sl]],
                 'close_raw': Q.CR[c].iloc[sl].round(2).tolist(), 'high': Q.HI[c].iloc[sl].round(2).tolist(), 'open': Q.OP[c].iloc[sl].round(2).tolist(),
                 'limup_d0': round(float(Q.LIMUP[t0, Q.cols.index(c)]), 2), 'raw_pct': round(e.raw * 100, 2), 'stop': bool(e.stop), 'hold': int(e.hold)})
res['MAIN_H5_samples'] = samp
for v in ['MAIN_H5', 'AGGR_B']:
    x = ev[ev.ver == v]
    res[f'{v}_top_dates'] = {str(k.date()): int(n) for k, n in x['d0'].value_counts().head(6).items()}
    res[f'{v}_raw_extremes_pct'] = [round(x['raw'].min() * 100, 2), round(x['raw'].max() * 100, 2)]
    for nm, a, b, ex in [('select', '2005-01-01', '2019-04-30', False), ('valid_ex_src3', '2019-05-01', '2026-10-31', True)]:
        y = x[(x.d0 >= a) & (x.d0 <= b)]
        if ex: y = y[~y.src3]
        top = y['d0'].value_counts().index[0]
        z = y[y.d0 != top]
        g = z.groupby(z['entry'].dt.to_period('M'))['ab'].mean()
        res[f'{v}_{nm}_ex_top_date'] = {'dropped': str(top.date()), 'n': int(len(z)), 'mean_ab_net_pct': round(z['ab'].mean() * 100, 3), 't_month': tstat(g)}
        g2 = y.groupby(y['entry'].dt.to_period('M'))['ab'].mean()
        res[f'{v}_{nm}_winsor_1_99'] = {'mean_ab_net_pct': round(y['ab'].clip(*y['ab'].quantile([.01, .99])).mean() * 100, 3),
                                        't_month_median_month': tstat(y.groupby(y['entry'].dt.to_period('M'))['ab'].median())}
print(json.dumps(res, ensure_ascii=False, indent=1))
f = HERE / 'results.json'; d = json.loads(f.read_text(encoding='utf-8')); d['post_checks'] = res
f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
