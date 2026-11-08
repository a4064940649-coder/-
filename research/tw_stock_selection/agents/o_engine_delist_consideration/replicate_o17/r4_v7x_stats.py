#!/usr/bin/env python3
"""o17 獨立複製 r4：V7X 判定欄位、Δ（對 orig）、orig 對 navs_oos_R4 重現、事件計數、年度差。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent.parent
VX = AG / 'o_exec_engine_rewrite_v7x'; sys.path.insert(0, str(VX))
import x_stats as S
df = pd.read_parquet(HERE / 'r3_navs_v7x.parquet')
REF = pd.read_parquet(VX / 'out' / 'navs_oos_R4.parquet')
TIERS = ['100萬', '300萬', '1000萬']; ENG = ['orig', 'main', 'lo', 'hi', 'c50']
FLD = ['CAGR_med', 'exc_med', 't_0050_med', 'exc_p10', 'CAGR_Y19on_med', 'exc_Y19on_med', 't_Y19on_med', 'exc_Y19on_p10']
out = {'orig_vs_R4_maxabs': 0.0}
for c in df.columns:
    if c.endswith('|orig'):
        k = c[:-5]; a = df[c].dropna(); b = REF[k].reindex(a.index)
        out['orig_vs_R4_maxabs'] = max(out['orig_vs_R4_maxabs'], float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))))
TAB = {}; YR = {}
for co in ('normal', 'stress'):
    for t in TIERS:
        for e in ENG:
            per = [S.stats(df[f'B10|J{j}|K5T|{co}|{t}|GROW|{e}'], 'AB_V7_B10', j) for j in range(13)]
            sm = S.jsumm(per); TAB[f'{co}|{t}|{e}'] = {k: sm[k] for k in FLD}
            if co == 'normal' and t == '100萬':
                YR[e] = {j: S.yearly_exc(df[f'B10|J{j}|K5T|{co}|{t}|GROW|{e}'], 'AB_V7_B10', j) for j in range(13)}
out['table'] = TAB
DEL = {}; mx = dict(ann=(0, None), t=(0, None))
for k, v in TAB.items():
    co, t, e = k.split('|')
    if e == 'orig': continue
    o = TAB[f'{co}|{t}|orig']; d = {f: v[f] - o[f] for f in FLD}; DEL[k] = d
    for f, x in d.items():
        kind = 't' if f.startswith('t_') else 'ann'
        if e == 'main' and abs(x) > abs(mx[kind][0]): mx[kind] = (x, f'{k}|{f}')
out['delta'] = DEL; out['max_main'] = mx
out['worst'] = min(((x, f'{k}|{f}') for k, d in DEL.items() for f, x in d.items()), key=lambda z: z[0])
yd = {}
for y in range(2006, 2027):
    v = [YR['main'][j].get(y, np.nan) - YR['orig'][j].get(y, np.nan) for j in range(13)]
    yd[y] = dict(med_of_diff=float(np.nanmedian(v)), mean_diff=float(np.nanmean(v)))
out['yearly_diff_main_100萬_normal'] = yd
EVL = json.loads((HERE / 'r3_evlog.json').read_text())
cnt = {}
for k, L in EVL.items():
    for code, ty in (L.get('main') or []): cnt.setdefault(code, set()).add(ty)
out['settled_codes'] = {c: sorted(v) for c, v in sorted(cnt.items())}
out['n_settled_cons'] = len([c for c, v in cnt.items() if any(not x.startswith('orig') for x in v)])
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
(HERE / 'r4_v7x_stats.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
