#!/usr/bin/env python3
"""診斷：STD_T10 組合賣出順延次數偏高的原因（事後檢查，不改判定）。python3 -I diag_delay.py"""
from pathlib import Path
src = (Path(__file__).resolve().parent / 'run.py').read_text(encoding='utf-8')
src = src.split('# ---------------- 執行全部變體')[0]
g = {'__file__': str(Path(__file__).resolve().parent / 'run.py'), '__name__': 'diag'}
exec(compile(src, 'run_head', 'exec'), g)
import numpy as np, pandas as pd
df = g['build_events'](False, 10)
df['code'] = [g['cols'][j] for j in df.j]
df['delay'] = df.x - (df.e + 10)
lng = df[(df.why == 'time') & (df.delay > 0)].sort_values('delay', ascending=False)
print('time exits', int((df.why == 'time').sum()), 'delayed', len(lng))
print(lng[['code', 'sig', 'e', 'x', 'delay', 'raw']].head(10).to_string())
for r in lng.head(3).itertuples():
    j = r.j; k = r.e + 10
    print(r.code, g['IDX'][k].date(), 'SELLABLE', g['SELLABLE'][k:k + 5, j], 'LOCK', g['LOCK_DN'][k:k + 5, j], 'AO', g['AOn'][k:k + 5, j], 'AH', g['AH'][k:k + 5, j], 'ACprev', g['AC_prev'][k:k + 5, j])
