#!/usr/bin/env python3
"""依 PREREG verdict_rule 判定；附事後檢查（收盤對收盤週頻獨立重算，不參與判定）。寫入 results.json。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
RJ = json.loads((H / 'results.json').read_text())
g = RJ['results']['grid']['N20_W_filt']['exlib_main']['windows']
chk = {}
for w in ('select_price', 'valid_price'):
    x = g[w]; chk[w] = {'t_vs_EW_5050': x['t_vs_EW_5050'], 'CAGR': x['CAGR'], '0050_CAGR': x['0050_CAGR'],
                        'pass_t': x['t_vs_EW_5050'] >= 2, 'pass_0050': x['CAGR'] >= x['0050_CAGR']}
    chk[w]['pass'] = chk[w]['pass_t'] and chk[w]['pass_0050']
npass = sum(chk[w]['pass'] for w in chk)
RJ['VERDICT'] = {'checks': chk, 'verdict': '可用' if npass == 2 else ('待觀察' if npass == 1 else '不可用'),
                 'grid_any_pass': {k: [v['exlib_main']['windows'][w]['t_vs_EW_5050'] >= 2 and v['exlib_main']['windows'][w]['CAGR'] >= v['exlib_main']['windows'][w]['0050_CAGR'] for w in ('select_price', 'valid_price')] for k, v in RJ['results']['grid'].items()}}
nv = pd.read_parquet(H / 'navs.parquet')
df = pd.DataFrame({'S': nv['SI_gross'], 'E': nv['BM_0050|gross']}).dropna()
wk = df.groupby(df.index.to_period('W-SUN')).tail(1); sig = (df / df.shift(20) - 1).reindex(wk.index); nxt = wk.shift(-1) / wk - 1
st = np.where((sig.S <= 0) & (sig.E <= 0), 'C', np.where(sig.E >= sig.S, 'L', 'S'))
r = pd.Series(np.where(st == 'C', 0, np.where(st == 'L', nxt.E, nxt.S)), index=wk.index).dropna()
cc = {}
for nm, (a, b) in {'select_price': ('2005-01', '2019-04'), 'valid_price': ('2019-05', '2026-10')}.items():
    x = r.loc[a:b]; cc[nm] = round(float((1 + x).prod() ** (52.18 / len(x)) - 1), 4)
RJ['POSTHOC_checks'] = {'close_to_close_weekly_gross_CAGR_no_exec_lag': cc,
                        'note': '週五收盤訊號、週五收盤即換（不可成交的理想版、不扣成本）；與引擎 gross（隔日開盤）select 10.0%／valid 22.8% 一致，差距＝開盤延遲成本。'}
(H / 'results.json').write_text(json.dumps(RJ, ensure_ascii=False, indent=1, default=float))
print(json.dumps(RJ['VERDICT'], ensure_ascii=False, indent=1)); print(cc)
