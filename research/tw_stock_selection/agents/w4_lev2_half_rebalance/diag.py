#!/usr/bin/env python3
"""事後診斷（不參與判定）：00631L 實際日報酬是追 0050 還是追加權指數（台指期）？寫入 results.json['POSTHOC_diagnostics']。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; RAW = Path('/home/user/jonatw/finlab-txd/data/raw')
rd = lambda f, c: pd.read_csv(RAW / f, parse_dates=['date']).set_index('date')[c].sort_index()
E = rd('etf_0050.csv', 'adj'); E.loc['2026-09-20'] = np.nan; L = rd('etf_00631L.csv', 'adj'); T = rd('taiex_twii.csv', 'close')
idx = L.index[L.index >= '2014-11-03']
idx = idx.drop(pd.DatetimeIndex(['2026-03-25', '2026-03-26', '2026-03-27', '2026-03-30', '2026-03-31']), errors='ignore')
full = E.index.union(L.index).union(T.index)
r = pd.DataFrame({'e': E.reindex(full).ffill().pct_change(), 'l': L.reindex(full).ffill().pct_change(), 't': T.reindex(full).ffill().pct_change()}).reindex(idx).dropna()
out = {}
for nm, g in (('2014-11_2019-12', r[r.index < '2020-01-01']), ('2020_2023', r[(r.index >= '2020-01-01') & (r.index < '2024-01-01')]), ('2024_2026-10', r[r.index >= '2024-01-01'])):
    X = np.column_stack([np.ones(len(g)), g.e, g.t]); b = np.linalg.lstsq(X, g.l.to_numpy(), rcond=None)[0]
    res_e = g.l - 2 * g.e; res_t = g.l - 2 * g.t
    out[nm] = {'beta_0050': round(float(b[1]), 3), 'beta_TAIEX': round(float(b[2]), 3), 'TE_vs_2x0050_ann': round(float(res_e.std() * np.sqrt(252)), 4),
               'TE_vs_2xTAIEX_ann': round(float(res_t.std() * np.sqrt(252)), 4),
               'cum_L2': round(float((1 + g.l).prod() - 1), 3), 'cum_2x0050_daily': round(float((1 + 2 * g.e).prod() - 1), 3), 'cum_2xTAIEXprice_daily': round(float((1 + 2 * g.t).prod() - 1), 3)}
RJ = json.loads((H / 'results.json').read_text()); RJ['POSTHOC_diagnostics'] = {'L2_tracking_regression': out,
    'note': '00631L 以台指期為主要曝險工具（臺灣50期貨流動性低），所以實際報酬更接近 2×加權指數（價格）＋現金利息，而不是 2×0050 含息。2024 年起台積電權重拉開 0050 與加權指數的差距，合成版因此高估正2。'}
(H / 'results.json').write_text(json.dumps(RJ, ensure_ascii=False, indent=1, default=float)); print(json.dumps(out, indent=1))
