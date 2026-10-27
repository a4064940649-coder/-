#!/usr/bin/env python3
"""看過主結果後才加的補充診斷（不影響判定）：SURGE 前 10% 與「月營收年增率」前 10% 的重疊，判斷是不是同一個訊號。
輸出 results_diag.json。執行：python3 -I diag.py（需先跑 run.py）"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
src = (HERE / 'run.py').read_text(encoding='utf-8')
g = {'__file__': str(HERE / 'run.py')}
exec(compile(src.split("# ---- 引擎驗證 ----")[0], 'run_head', 'exec'), g)
REV, SC, mm, liq, Dt, N = g['REV'], g['SC'], g['mm'], g['liq'], g['Dt'], g['N']
YOY = (REV / REV.shift(12) - 1).where(REV.shift(12) > 0)
YOY3 = YOY.rolling(3, min_periods=3).mean()
rows = []
for k, dd in enumerate(Dt[:-1]):
    s = SC['S_main'].reindex([mm[k]]).to_numpy()[0]
    for nm, Y in (('yoy1', YOY), ('yoy3', YOY3)):
        y = Y.reindex([mm[k]]).to_numpy()[0]
        pl = liq[k] & np.isfinite(s) & np.isfinite(y); pp = np.where(pl)[0]
        if len(pp) < 50: continue
        nt = int(round(0.1 * len(pp)))
        ta = set(pp[np.argsort(-s[pp])[:nt]]); tb = set(pp[np.argsort(-y[pp])[:nt]])
        rows.append({'d': dd, 'm': nm, 'share_of_surge_top_in_yoy_top': len(ta & tb) / nt,
                     'spearman': pd.Series(s[pp]).corr(pd.Series(y[pp]), method='spearman')})
R = pd.DataFrame(rows)
out = {}
for nm, x in R.groupby('m'):
    x = x.set_index('d')
    out[nm] = {per: {'share_of_surge_top10_also_in_yoy_top10': round(float(x.loc[a:b].share_of_surge_top_in_yoy_top.mean()), 3),
                     'mean_spearman': round(float(x.loc[a:b].spearman.mean()), 3), 'n_months': int(len(x.loc[a:b]))}
               for per, (a, b) in {'2008-2012': ('2008-01-01', '2013-01-31'), '2016-2019.04': ('2016-02-01', '2019-04-30'), '2019.05-2026': ('2019-05-01', '2026-12-31')}.items()}
(HERE / 'results_diag.json').write_text(json.dumps({'說明': '看過結果後才加，不影響判定', 'overlap_with_yoy': out}, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False))
