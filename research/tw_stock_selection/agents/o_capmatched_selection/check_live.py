#!/usr/bin/env python3
"""凍結後：前瞻腳本（live_tracking/o7_capmatched/o7_paper.py，screener 資料）與回測（cmstrat，raw 面板）在 2026-10-12 訊號日的目標權重一致性。寫 results_live_check.json。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cmlib as L, cmstrat as T
res = {}
for d in [pd.Timestamp('2026-08-11'), pd.Timestamp('2026-09-11'), pd.Timestamp('2026-10-12')]:
  cs = T.cross_section(d)
  js, ws, nb, ns = T.target(cs, json.loads((L.HERE / 'results.json').read_text(encoding='utf-8'))['PREREG']['frozen_version'], set())
  bt = pd.Series(ws, index=[L.COLS[j] for j in js])
  lv = pd.read_csv(L.HERE.parent / 'live_tracking' / 'o7_capmatched' / 'targets.tsv', sep='\t', dtype={'code': str})
  lv = lv[lv.signal == str(d.date())].set_index('code').w
  al = pd.concat([bt.rename('backtest'), lv.rename('live')], axis=1).fillna(0)
  out = {'n_backtest': int(len(bt)), 'n_live': int(len(lv)), 'n_common': int(((al.backtest > 0) & (al.live > 0)).sum()),
         'sum_abs_diff': float((al.backtest - al.live).abs().sum()), 'w2330': [float(al.backtest.get('2330', 0)), float(al.live.get('2330', 0))],
         'only_backtest': sorted(al.index[(al.backtest > 0) & (al.live == 0)]), 'only_live': sorted(al.index[(al.live > 0) & (al.backtest == 0)])}
  res[str(d.date())] = out
(L.HERE / 'results_live_check.json').write_text(L.dumps(res), encoding='utf-8')
print(out)
