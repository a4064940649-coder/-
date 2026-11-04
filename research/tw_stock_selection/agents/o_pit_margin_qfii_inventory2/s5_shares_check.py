#!/usr/bin/env python3
"""預測 5 的可做部分（描述）：官方 MI_QFIIS 發行股數（月形成日快照，2014 起）對 o13 TEJ 月底股數錨點的對數差；
2006～2013 沒有 MI_QFIIS，o13 D1 無法重算。存活／下市分開報（TEJ 錨點只有存活股，下市股無錨點）。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, glib
t = pd.read_csv(HERE.parent / 'o_pit_shares_industry_2006_2013/raw/tej_ff5/shares_monthly.csv', dtype=str)
print(t.columns.tolist())
t['me'] = pd.to_datetime(t.month_end); t['sh'] = pd.to_numeric(t.shares_k) * 1000
t = t[t.market == 'TSE']
jq = glib.load_joeyc('qfii-twse')
jq = jq[jq.code.str.fullmatch(r'[1-9]\d{3}')]
jq['me'] = (jq.date.dt.to_period('M') - 1).dt.to_timestamp(how='end').dt.normalize()
x = jq.merge(t[['code', 'me', 'sh']], on=['code', 'me'])
x = x[(x.sh > 0) & (x.issued > 0)]
x['ld'] = np.log(x.issued / x.sh).abs()
L, V, FD, LD = glib.denominator()
dl = set(LD[LD < pd.Timestamp('2026-10-01')].index)
out = {'n_pairs': len(x), 'median_abs_logdiff_pct': round(100 * x.ld.median(), 4), 'p90_abs_logdiff_pct': round(100 * x.ld.quantile(0.9), 3),
       'share_within_0.1pct': round(float((x.ld <= 0.001).mean()), 4),
       'by_year_median_pct': {int(k): round(100 * v, 4) for k, v in x.groupby(x.date.dt.year).ld.median().items()},
       'delisted_pairs': int(x.code.isin(dl).sum()), 'delisted_median_pct': round(100 * x[x.code.isin(dl)].ld.median(), 4) if x.code.isin(dl).any() else None,
       'qfii_delisted_codes_without_TEJ_anchor': int(len(set(jq.code) & dl - set(t.code))),
       'note': '比對＝形成日（約 11 日）官方發行股數 vs TEJ 前月底流通股數；只到 2014 起；o13 D1（2006～2013）無法用 MI_QFIIS 重算'}
json.dump(out, open(HERE / 'data' / 'shares_check.json', 'w'), ensure_ascii=False, indent=1)
print(out)
