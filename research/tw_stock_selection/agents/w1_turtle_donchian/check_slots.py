#!/usr/bin/env python3
"""獨立檢查：用事件表（不加碼）做「先到先得、最多 20 檔」的交易層重建，與組合引擎結果對照（毛報酬）。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M
L = M.L
GOOD = (L.adj.notna().sum(axis=1) >= 300).to_numpy(); gi = np.where(GOOD)[0]
V60g = M.V60.to_numpy()[gi]; CI = {c: j for j, c in enumerate(L.COLS)}
for nm in ('S1', 'S2'):
    tr = pd.read_csv(HERE / f'events_{nm}.csv.gz', dtype={'code': str})
    tr['j'] = tr.code.map(CI); tr['liq'] = V60g[tr.s, tr.j]
    tr = tr.sort_values(['e', 'liq'], ascending=[True, False])
    act = []; sel = []
    for r in tr.itertuples():
        act = [x for x in act if x > r.e]
        if len(act) < 20: act.append(r.x); sel.append(r.Index)
    s = tr.loc[sel]; s = s[(s.d >= '2005-01-01')]
    for a, b in (('2005-01-01', '2019-04-30'), ('2019-05-01', '2026-10-31')):
        x = s[(s.d >= a) & (s.d <= b)]; yrs = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
        print(nm, a, 'n', len(x), 'per yr', round(len(x) / yrs), 'mean gross %', round(x.ret.mean() * 100, 2), 'EW same window %', round(x.ew.mean() * 100, 2),
              'win', round((x.ret > 0).mean(), 3), 'sum(ret)/20 per yr %', round(x.ret.sum() / 20 / yrs * 100, 1), 'sum(ew)/20 per yr %', round(x.ew.sum() / 20 / yrs * 100, 1),
              'all-events abn %', round((tr[(tr.d >= a) & (tr.d <= b)].ret - tr[(tr.d >= a) & (tr.d <= b)].ew).mean() * 100, 2))
