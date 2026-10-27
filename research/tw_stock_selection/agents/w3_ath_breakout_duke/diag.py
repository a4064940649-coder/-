#!/usr/bin/env python3
"""診斷（看完主結果後補做，不影響判定）：把「flag 3/4/5」與「報酬 >100%」分開剔除，看事件異常報酬靠什麼。"""
import json, sys, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
FL = pd.read_parquet(P / 'flag.parquet'); adj = pd.read_parquet(P / 'adj.parquet')
GOOD = adj.notna().sum(axis=1) >= 300; G = adj.index[GOOD]; FLg = FL.reindex(index=G, columns=adj.columns).to_numpy(); CI = {c: j for j, c in enumerate(adj.columns)}
FEE = 0.001425 * 0.6; b, s = FEE + 0.001, FEE + 0.003 + 0.001
def t(x):
    x = pd.Series(x).dropna(); return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 else None
out = {}
for nm in ('ATH_raw', 'ATH_adj', 'H52_raw', 'H52_adj'):
    tr = pd.read_csv(HERE / f'events_{nm}.csv.gz', dtype={'code': str}, parse_dates=['d'])
    tr['f345'] = [bool(np.isin(FLg[r.s:r.x + 1, CI[r.code]], [3, 4, 5]).any()) for r in tr.itertuples()]
    net = (1 + tr.ret) * (1 - s) / (1 + b) - 1; tr['abn'] = net - tr.ew
    o = {}
    for k, (a, z) in {'select': ('2007-01-01', '2019-04-30'), 'valid': ('2019-05-01', '2026-10-31')}.items():
        m = (tr.d >= a) & (tr.d <= z); x = tr[m]
        def st(y): return {'n': int(len(y)), 'abn_pct': round(float(y.abn.mean() * 100), 2), 't_month': t(y.abn.groupby(y.em).mean())}
        o[k] = {'all': st(x), 'ex_flag345': st(x[~x.f345]), 'ex_ret_gt100': st(x[x.ret <= 1.0]), 'ex_top1pct_abn': st(x[x.abn < x.abn.quantile(0.99)]),
                'median_abn_pct': round(float(x.abn.median() * 100), 2), 'flag345_mean_abn_pct': round(float(x[x.f345].abn.mean() * 100), 2) if x.f345.any() else None,
                'share_abn_from_top20': round(float(x.abn.nlargest(20).sum() / x.abn.sum()), 2) if x.abn.sum() > 0 else None}
    out[nm] = o
(HERE / 'results_diag.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False))
