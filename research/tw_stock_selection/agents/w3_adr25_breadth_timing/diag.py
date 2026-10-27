#!/usr/bin/env python3
"""事後檢查（不參與判定）：開盤代理品質、引擎一致性（一直持有＝0050 買進持有）、immediate 與 cross 進場日差異。"""
import json, runpy
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
g = runpy.run_path(str(H / 'run.py'), run_name='diag')   # 重跑 run.py（同結果）並取得函式
T, C50, O50, TRAD, NT = g['T'], g['C50'], g['O50'], g['TRAD'], g['NT']
out = {}
gp = pd.Series(O50 / np.r_[np.nan, C50[:-1]] - 1, index=T)[TRAD]
out['open_gap_abs_median_pct'] = round(float(gp.abs().median() * 100), 3)
out['open_gap_by_year_std_pct'] = {int(y): round(float(s.std() * 100), 3) for y, s in gp.groupby(gp.index.year)}
# 引擎：第 1 日進場、永不出場（gross）→ 應等於 0050 收對收（除第一天的開→收）
i0 = int(np.where(TRAD & (T >= pd.Timestamp('2005-01-03')))[0][0])
r, held = g['nav_series']([dict(s=i0 - 1, e=i0, x=NT - 1, reason='trunc', trunc=True)], 'gross')
r50 = g['R50D']
d = (r - r50).iloc[i0 + 1:]
out['engine_always_in_max_abs_diff'] = float(d.abs().max())
# immediate vs cross 進場日
for N, L, Hh in ((25, 70, 120),):
    E1, X1 = g['make_signals'](N, L, Hh, 'cross'); E2, X2 = g['make_signals'](N, L, Hh, 'immediate')
    t1 = g['simulate'](E1, X1); t2 = g['simulate'](E2, X2)
    out['entries_cross_vs_immediate'] = [[str(T[a['e']].date()), str(T[b['e']].date())] for a, b in zip(t1, t2)]
RJ = json.loads((H / 'results.json').read_text()); RJ['POSTHOC_diag'] = out
(H / 'results.json').write_text(json.dumps(RJ, ensure_ascii=False, indent=1, default=lambda x: None if x != x else x), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False)[:3000])
