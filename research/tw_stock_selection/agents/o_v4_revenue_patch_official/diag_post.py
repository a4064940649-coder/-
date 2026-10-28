#!/usr/bin/env python3
"""事後診斷（PREREG 之後；規格不變，只報告）：
(1) R_FIX 新聞水準格的「最早見報日」是否早於使用它的訊號日（point-in-time）；
(2) AB（mc40）的 V4_RDEL（＝把唯一的修補入選股月拿掉後的 V4；用來量 R_FIX 在 AB 下的貢獻）。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v4lib as L
K, V = L.K, L.V
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); assert 'PREREG' in res and 'RESULTS' in res
M = pd.read_parquet(HERE / 'raw_news_mentions.parquet')
cells = pd.read_csv(HERE / 'rfix_cells.csv', dtype={'code': str})
first = M.groupby(['code', 'ym']).date.min()
rows = []
for c, y in zip(cells.code, cells.ym):
    for lv in (y, str(pd.Period(y, 'M') - 12)):
        if (c, lv) in first.index:
            p = pd.Period(lv, 'M') + 1
            sig = [d for d in K.rebalance_days('2005-01-01') if pd.Period(d, 'M') == p]
            fd = pd.Timestamp(first[(c, lv)])
            rows.append(dict(code=c, yoy_month=y, level_month=lv, first_news=str(fd.date()), first_signal_using=str(sig[0].date()) if sig else '',
                             late=bool(sig and fd > sig[0])))
P = pd.DataFrame(rows).drop_duplicates(['code', 'level_month'])
out = {'n_news_levels_in_RFIX': int(len(P)), 'n_first_news_after_signal': int(P.late.sum()), 'late_list': P[P.late].to_dict(orient='records')}
XR = L.setup_px()
YRD = L.load_rdel(); Y5 = L.y5_orig(); DMT = L.disp_mask_fn('trade')
rets = []; ewr = []
t0 = t1 = None
for seed in range(40):
    S = L.V4(YRD, Y5, DMT, unavail=V.make_unavail('mc', seed))
    ta, tb, sd = S.bounds(); pl = S.plan(10, 'official')
    nav, _ = L.sim_plan(pl, ta, tb, 'normal', XR); rets.append(nav.pct_change())
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, ta, tb); ewr.append(ew.pct_change())
mn = lambda x: (1 + pd.concat(x, axis=1).mean(axis=1).fillna(0)).cumprod()
nav, ew = mn(rets), mn(ewr)
out['AB_V4_RDEL_normal'] = {k: L.seg(nav, ew, sd, a, b) for k, (a, b) in L.PER.items()}
x = L.excess(nav, K.E50, sd, '2006-04-01', '2026-10-12'); xs = x.sort_values(ascending=False)
out['AB_V4_RDEL_drop_best1_t'] = L.tstat(x.drop(xs.index[:1]))
(HERE / 'diag_post.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k != 'AB_V4_RDEL_normal'}, ensure_ascii=False, indent=1, default=float))
print({k: (round(v['CAGR'], 2), round(v['t_0050'], 3)) for k, v in out['AB_V4_RDEL_normal'].items()})
