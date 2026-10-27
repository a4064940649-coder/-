#!/usr/bin/env python3
"""事後診斷（不改任何規則）：各 β 五分組的毛月報酬、對加權指數（價格）與 0050 的實現 β，檢查 SML 是否平坦。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from btlib import tstat  # noqa: E402
nav = pd.read_parquet(HERE / 'nav.parquet')
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
WIN = res['PREREG']['windows']
RF_M = 1.01 ** (1 / 12) - 1
def mo(x): return (1 + x).groupby(x.index.to_period('M')).prod() - 1
tm = mo(tx.reindex(nav.index).ffill().pct_change().fillna(0)); em = mo(e50.reindex(nav.index).ffill().pct_change().fillna(0))
out = {}
keys = [f'Q{k}_EW_TAIEX' for k in range(1, 6)]
for w, (a, b) in WIN.items():
    o = {}
    for k in keys:
        g = nav[k + '_gross'] if k + '_gross' in nav else nav[k]
        m = mo(g.pct_change().fillna(0)); m = m[(m.index >= pd.Period(a[:7], 'M')) & (m.index <= pd.Period(b[:7], 'M'))]
        bt = np.polyfit(tm.reindex(m.index) - RF_M, m - RF_M, 1)[0]; b5 = np.polyfit(em.reindex(m.index) - RF_M, m - RF_M, 1)[0]
        o[k] = {'gross_mean_pct_m': round(float(m.mean() * 100), 3), 'vol_ann': round(float(m.std() * math.sqrt(12)), 3),
                'sharpe_gross': round(float((m - RF_M).mean() / m.std() * math.sqrt(12)), 3),
                'beta_real_vs_TAIEX_price': round(float(bt), 3), 'beta_real_vs_0050': round(float(b5), 3)}
    g1 = mo(nav['Q1_EW_TAIEX_gross'].pct_change().fillna(0)); g5 = mo(nav['Q5_EW_TAIEX_gross'].pct_change().fillna(0))
    s = (g1 - g5); s = s[(s.index >= pd.Period(a[:7], 'M')) & (s.index <= pd.Period(b[:7], 'M'))]
    o['Q1_minus_Q5_gross_unlevered'] = {'mean_pct_m': round(float(s.mean() * 100), 3), 't': tstat(s)}
    out[w] = o
    print(w, json.dumps(o, ensure_ascii=False))
res['diagnostics_post_hoc'] = out
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
