#!/usr/bin/env python3
"""事後診斷（不影響挑選與判定）：成交時點敏感度、與 combo_search 引擎開關版（換股日訊號、d+1 開盤全賣、回場買新選股）對照、回撤區間。
寫入 results.json 的 POSTHOC_diagnostics。執行：python3 -I -B diag.py（需先跑 run.py）"""
import importlib.util, io, contextlib, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('run', H / 'run.py'); m = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(m)
RES = json.loads((H / 'results.json').read_text(encoding='utf-8'))
W = dict(m.WIN); W['oos_2006-04_2020-12'] = ('2006-04-01', '2020-12-31')
D = {'note': '事後診斷，不用於挑選或判定'}
s = pd.read_parquet(H.parent / 'combo_search' / 'search_nav.parquet')
D['engine_raw_vs_oos_nav_maxabsdiff'] = float(np.abs(s['C1+C2|n30|raw'].reindex(m.IDX) / m.TGT['n30'] - 1).max())
def st(f):
    out = {}
    for w, ab in W.items():
        c, dd, _ = m.cagr_dd(f, m.wmask(*ab)); out[w] = {'CAGR': round(c, 4), 'maxDD': round(dd, 4), 'Calmar': round(c / abs(dd), 3)}
    return out
for tg in ['n30', 'n10']:
    D[f'engine_S1_dplus1_open|{tg}'] = st(s[f'C1+C2|{tg}|sw'].reindex(m.IDX).pct_change().fillna(0).to_numpy() + 1)
    for lag in [1, 2]: D[f'overlay_S1_lag{lag}|{tg}'] = st(m.overlay(tg, 'S1', lag)['f'].to_numpy())
D['oos_2006-04_2020-12_main'] = {f'{tg}|{sw}': m.wstats(tg, sw, '2006-04-01', '2020-12-31') for tg in m.TGT for sw in m.ALL}
for v in D['oos_2006-04_2020-12_main'].values(): v.pop('episodes', None)
tr = {}
for tg in m.TGT:
    for sw in m.SW:
        df = m.overlay(tg, sw, 2); r = m.RET[tg]
        tr[f'{tg}|{sw}'] = {'sum_r_on_exit_trade_days(lag2 bears, lag1 avoids)': round(float(r[df['exit']].sum()), 4),
                            'sum_r_on_entry_trade_days(lag2 misses, lag1 earns)': round(float(r[df['entry']].sum()), 4), 'n_exit': int(df['exit'].sum())}
D['one_day_delay_cost'] = tr
dd = {}
for tg in m.TGT:
    for sw in m.ALL:
        eq = pd.Series(np.cumprod(m.overlay(tg, sw)['f'].to_numpy()), index=m.IDX)
        for w, (a, b) in m.WIN.items():
            e = eq[m.wmask(a, b)]; x = e / e.cummax() - 1; t = x.idxmin(); dd[f'{tg}|{sw}|{w}'] = [str(e[:t].idxmax().date()), str(t.date()), round(float(x.min()), 4)]
D['maxDD_dates'] = dd
RES['POSTHOC_diagnostics'] = D
(H / 'results.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1), encoding='utf-8')
for k in ['engine_S1_dplus1_open|n30', 'overlay_S1_lag1|n30', 'overlay_S1_lag2|n30']: print(k, D[k])
for k, v in tr.items(): print(k, v)
