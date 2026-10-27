#!/usr/bin/env python3
"""事後資料核對：收盤漲停偵測率（src 3 期間 vs 官方期間）、事件日期集中度、極端值。不影響判定。"""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'w3_limitup_pullback'))
import lim as Q  # noqa: E402
IDX = Q.IDX
src = pd.read_parquet(Q.P / 'src.parquet').reindex(index=IDX, columns=Q.cols)
lu = pd.Series(Q.CLOSE_LU.sum(1), index=IDX)
cc = pd.DataFrame(Q.CC, index=IDX)
big = pd.Series(((cc.to_numpy() >= np.where(IDX < pd.Timestamp('2015-06-01'), 0.065, 0.095)[:, None]) & np.isfinite(cc.to_numpy())).sum(1), index=IDX)
seg = pd.Series(np.select([(IDX >= '2024-02-15') & (IDX <= '2025-09-23'), IDX >= '2025-09-24'], ['src3', 'src4'], 'src12'), index=IDX)
out = {}
for y in range(2019, 2027):
    for s in ('src12', 'src3', 'src4'):
        m = (IDX.year == y) & (seg == s).to_numpy()
        if m.sum() == 0: continue
        out[f'{y}_{s}'] = {'days': int(m.sum()), 'close_LU_per_day': round(float(lu[m].mean()), 2), 'cc_ge_9.5pct_per_day': round(float(big[m].mean()), 2),
                           'LU_over_big': round(float(lu[m].sum() / max(big[m].sum(), 1)), 3)}
ev = pd.read_csv(HERE / 'events_PRIMARY.csv.gz', parse_dates=['d0'])
c = ev[ev.ver == 'C']; vc = c['d0'].value_counts()
x = c[c.d0 != pd.Timestamp('2025-04-10')]
gm = x.groupby(x['entry'].str[:7])['ab'].mean()
from btlib import tstat  # noqa: E402
v = x[x.d0 >= '2019-05-01']; gv = v.groupby(v['entry'].str[:7])['ab'].mean()
res = {'limit_up_detection_by_segment': out, 'PRIMARY_C_top_dates': {str(k.date()): int(n) for k, n in vc.head(8).items()},
       'PRIMARY_C_valid_ex_2025-04-10': {'n': int(len(v)), 'mean_ab_net_pct': round(v['ab'].mean() * 100, 3), 't_month': tstat(gv)},
       'PRIMARY_C_raw_extremes_pct': [round(c['raw'].min() * 100, 2), round(c['raw'].max() * 100, 2)]}
print(json.dumps(res, ensure_ascii=False, indent=1))
f = HERE / 'results.json'; d = json.loads(f.read_text(encoding='utf-8')); d['post_checks'] = res
f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
