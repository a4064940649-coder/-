#!/usr/bin/env python3
"""事後資料核對：收盤跌停偵測（各來源段）、事件日期集中度、極端值、剔除集中日後結果。不影響判定。"""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'w3_limitup_pullback'))
import lim as Q  # noqa: E402
from btlib import tstat  # noqa: E402
IDX = Q.IDX
with np.errstate(invalid='ignore'):
    xdn = Q.REF * (1 - Q.lim)
    tkd = Q.tick_of(np.nan_to_num(xdn, nan=0.0), (IDX < pd.Timestamp('2005-03-01'))[:, None])
    LIMDN = np.ceil(xdn / tkd - 1e-6) * tkd
    CLD = (Q.CR.to_numpy() <= LIMDN + 0.5 * tkd) & Q.FLAGOK.to_numpy()
    r = Q.CR.to_numpy() / Q.REF - 1
thr = np.where(IDX < pd.Timestamp('2015-06-01'), -0.065, -0.095)[:, None]
big = (r <= thr) & np.isfinite(r) & Q.FLAGOK.to_numpy()
seg = np.select([(IDX >= '2024-02-15') & (IDX <= '2025-09-23'), IDX >= '2025-09-24', IDX < pd.Timestamp('2015-06-01')], ['src3', 'src4', 'src12_7pct'], 'src12_10pct')
det = {}
for s in ('src12_7pct', 'src12_10pct', 'src3', 'src4'):
    m = seg == s
    det[s] = {'days': int(m.sum()), 'close_LD_per_day': round(float(CLD[m].sum() / m.sum()), 2), 'ret_le_thr_per_day': round(float(big[m].sum() / m.sum()), 2),
              'LD_over_big': round(float(CLD[m].sum() / max(big[m].sum(), 1)), 3),
              'LD_with_ret_gt_-6pct_share': round(float((CLD[m] & (r[m] > -0.06)).sum() / max(CLD[m].sum(), 1)), 4)}
out = {'limit_down_detection_by_segment': det}
ev = pd.read_csv(HERE / 'events_PRIMARY.csv.gz', parse_dates=['d0', 'entry'])
for v in 'ABC':
    c = ev[ev.ver == v]
    vc = c['d0'].value_counts()
    o = {'top_d0_dates': {str(k.date()): int(n) for k, n in vc.head(10).items()},
         'share_events_on_top10_dates': round(float(vc.head(10).sum() / len(c)), 3),
         'raw_extremes_pct': [round(c['raw'].min() * 100, 2), round(c['raw'].max() * 100, 2)]}
    # 每日最多 1 筆權重：先按 Day0 日期平均，再按月份平均（去除崩盤日群聚）
    for wn, a, b in (('select', '2005-01-01', '2019-04-30'), ('valid', '2019-05-01', '2026-10-31')):
        x = c[(c.d0 >= a) & (c.d0 <= b)]
        dm = x.groupby('d0')['ab'].mean()
        o[f'{wn}_date_then_month_t'] = tstat(dm.groupby(dm.index.to_period('M')).mean())
        o[f'{wn}_date_mean_ab_pct'] = round(float(dm.mean() * 100), 3)
        y = x[x['d0'].map(vc) < 20]
        gy = y.groupby(y['entry'].dt.to_period('M'))['ab'].mean()
        o[f'{wn}_ex_dates_ge20_events'] = {'n': int(len(y)), 'mean_ab_net_pct': round(float(y['ab'].mean() * 100), 3), 't_month': tstat(gy)}
    out[f'PRIMARY_{v}'] = o
print(json.dumps(out, ensure_ascii=False, indent=1))
f = HERE / 'results.json'; d = json.loads(f.read_text(encoding='utf-8')); d['post_checks'] = out
f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
