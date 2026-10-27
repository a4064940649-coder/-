#!/usr/bin/env python3
"""敏感度：完全不用當沖官方量，只用「原始 5 分 K 去重」修正（截斷日用當日截面中位倍率）。輸出 scratchpad/vf_value_dedup_only.parquet。"""
from pathlib import Path
import pandas as pd
S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
HERE = Path(__file__).resolve().parent
V = pd.read_parquet(S / 'panel/value.parquet'); f = pd.read_parquet(HERE / 'fix_log.parquet')
alt = f['val'] * f['q']
tr = f['k'] <= 1.0001                          # 截斷日（無重複列）
dmed = (f.loc[f['method'] == 'official_daytrade', 'val_fix'] / f.loc[f['method'] == 'official_daytrade', 'val']).groupby(f['date']).median()
alt[tr] = f.loc[tr, 'val'] * f.loc[tr, 'date'].map(dmed)
f['alt'] = alt.fillna(f['val_fix'])
VF = V.copy()
for c, g in f.groupby('code'): VF.loc[g['date'].to_numpy(), c] = g['alt'].to_numpy()
VF.to_parquet(S / 'vf_value_dedup_only.parquet')
print('rows', len(f), 'median alt/official-fix', round(float((f['alt'] / f['val_fix'])[f['method'] == 'official_daytrade'].median()), 4))
