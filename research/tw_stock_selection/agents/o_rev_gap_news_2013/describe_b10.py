#!/usr/bin/env python3
"""描述（gate.py 之後；不看報酬）：B10 實際入選的 133 個缺口股月中，有代理者的營收年增率百分位（B10 假設為 1.0）。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rglib as G
K = G.K
b = G.base(); S = G.mk()
T = pd.read_csv(HERE / 'data' / 'proxy_table_gap_cells.csv', dtype={'code': str})
T = T[T.in_B10]
out = {}
for ver in ('N1', 'NQ', 'Q'):
    rows = []
    for s, c, t, v in zip(T.sig, T.code, T.type, T['proxy_' + ver]):
        if not np.isfinite(v): continue
        d = pd.Timestamp(s); cs = S.cross_section(d); y3 = S.y3(d)
        el = cs['ok_other'] & np.isfinite(y3)
        tv = y3[el]
        rows.append((s, c, t, float((tv < v).mean() + 0.5 * (tv == v).mean())))
    R = pd.DataFrame(rows, columns=['sig', 'code', 'type', 'pct'])
    out[ver] = {'n_covered': int(len(R)), 'pct_median': float(R.pct.median()) if len(R) else None,
                'share_pct_ge_0.9': float((R.pct >= 0.9).mean()) if len(R) else None, 'share_pct_lt_0.5': float((R.pct < 0.5).mean()) if len(R) else None,
                'by_type_median': R.groupby('type').pct.median().to_dict(), 'by_type_n': R.groupby('type').size().to_dict()}
F = HERE / 'results.json'; res = json.loads(F.read_text(encoding='utf-8'))
res['DESCRIBE_B10_PICKS_PROXY_PCT'] = out
F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
