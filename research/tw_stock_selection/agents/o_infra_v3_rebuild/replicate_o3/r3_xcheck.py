#!/usr/bin/env python3
"""用 henry753951/finance 原始檔交叉核對：(a) 與面板重疊日的收盤 (b) 與 delist_prices 的收盤（含 src A 列）。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
R = Path('/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/raw/henry753951_finance/data/stocks')
C0 = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/close_raw.parquet')
dp = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/o_infra_v3_rebuild/delist_prices.parquet')
out = {'panel': [0, 0], 'patch_A': [0, 0], 'patch_B': [0, 0], 'missing_raw': []}
for c in sorted(dp.code.unique()):
    f = R / f'{c}.csv'
    if not f.exists(): out['missing_raw'].append(c); continue
    r = pd.read_csv(f, usecols=['date', 'close', 'Trading_Volume'], parse_dates=['date'])
    r = r[r['Trading_Volume'] > 0].set_index('date')['close']
    p = C0[c].dropna(); com = p.index.intersection(r.index)
    out['panel'][0] += len(com); out['panel'][1] += int((np.abs(p[com] - r[com]) < 1e-6).sum())
    for s in ('A', 'B'):
        q = dp[(dp.code == c) & (dp.src == s)].set_index('date')['close']; com = q.index.intersection(r.index)
        out[f'patch_{s}'][0] += len(com); out[f'patch_{s}'][1] += int((np.abs(q[com] - r[com]) < 1e-6).sum())
print(out)
Path(__file__).with_name('my_xcheck.json').write_text(json.dumps(out))
