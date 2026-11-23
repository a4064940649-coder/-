#!/usr/bin/env python3
"""R97 補充檢查（不跑引擎）：(1) 2015-06-01 前 275 筆名單，用原始前收手算 7% 漲停，對引擎 REF 版的判定；(2) 原始檔沒有的代號（data_panel 補格）成交是否在 7% 帶內；
(3) 2008-10 跌幅減半期間每天跌破 −3.6% 的檔數。python3 -I -B extra_checks.py → extra_checks.json"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; RS = HERE.parent.parent
sys.path.insert(0, str(HERE / '_none'))
def tick(p): return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0
def lim(ref, pct):
    u = ref * (1 + pct); tu = tick(u); u = math.floor(u / tu + 1e-9) * tu
    l = ref * (1 - pct); tl = tick(l); l = math.ceil(l / tl - 1e-9) * tl
    return round(u, 2), round(l, 2)
z = np.load('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s75/cache_cont/panel_cs.npz', allow_pickle=False)
IDX = pd.DatetimeIndex(z['IDX']); COLS = list(z['COLS']); ci = {c: j for j, c in enumerate(COLS)}
cut = int(np.searchsorted(IDX.values, np.datetime64('2015-06-01')))
O, C, H, L, CHG = (z[k][:cut + 1] for k in ('O', 'C', 'H', 'L', 'CHG'))
CF = pd.DataFrame(np.where(C > 0, C, np.nan)).ffill().to_numpy()
out = {}
sigs = pd.read_csv(HERE.parent / 's75_continuous_2008_2026' / 'signals.csv', dtype=str).fillna('')
sigs = sigs[sigs['version'] == 'V_BRIDGE']
di = {str(d.date()): i for i, d in enumerate(IDX)}
n = 0; lock = []; refdiff = []; noopen = 0
for _, r in sigs.iterrows():
    t = di[r['sig_date']] + 1
    if t >= cut: continue
    for c in r['targets'].split():
        j = ci[c]; n += 1; pc = CF[t - 1, j]; o = O[t, j]
        if not (np.isfinite(o) and o > 0): noopen += 1; continue
        u7, l7 = lim(pc, 0.07)
        chg = CHG[t, j] if np.isfinite(CHG[t, j]) else 0.0
        if abs((C[t, j] - chg) - pc) > 1e-6: refdiff.append((c, str(IDX[t].date()), float(pc), float(C[t, j] - chg), float(o), u7))
        if o >= u7 - 1e-9: lock.append((c, str(IDX[t].date()), float(pc), float(o), u7, bool(L[t, j] >= u7 - 1e-9) if np.isfinite(L[t, j]) else None))
out['list_pre_cut_by_raw_prev_close'] = dict(n=n, n_no_open=noopen, n_open_ge_limup7=len(lock), locked=lock, engine_ref_differs_from_prev_close=refdiff)
# (2)
A = json.loads((HERE / 'analysis.json').read_text()); res = []
for c, dt, s in A['fills_scan_lim7']['no_prev']:
    t = di[dt]; j = ci[c]; pc = CF[t - 1, j]; u7, l7 = lim(pc, 0.07); chg = CHG[t, j] if np.isfinite(CHG[t, j]) else 0.0
    res.append(dict(code=c, date=dt, side=s, prev_close=float(pc), open=float(O[t, j]), limup7=u7, limdn7=l7, inside=bool(l7 - 1e-9 <= O[t, j] <= u7 + 1e-9), at_limit=bool(O[t, j] >= u7 - 1e-9 or O[t, j] <= l7 + 1e-9),
                    engine_ref=float(C[t, j] - chg)))
out['fills_from_data_panel_fill'] = dict(n=len(res), n_inside_7pct=int(sum(r['inside'] for r in res)), n_at_limit=int(sum(r['at_limit'] for r in res)), n_ref_ne_prev=int(sum(abs(r['engine_ref'] - r['prev_close']) > 0.011 for r in res)), rows=res)
# (3)
d = pd.read_csv(RS / 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz', dtype={'code': str}, parse_dates=['date'])
d = d[(d['date'] >= '2008-10-01') & (d['date'] <= '2008-11-05') & (d['close'] > 0)].sort_values(['code', 'date'])
d['pc'] = d.groupby('code')['close'].shift(1); d = d[(d['xflag'] == 0) & d['pc'].notna()]
d['rl'] = d['low'] / d['pc'] - 1; d['rc'] = d['close'] / d['pc'] - 1
out['oct2008'] = [(str(k.date()), int(len(g)), int((g['rl'] < -0.036).sum()), int(((g['rc'] <= -0.033) & (g['rc'] >= -0.036)).sum()), int((g['rc'] < -0.066).sum())) for k, g in d.groupby('date')]
(HERE / 'extra_checks.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False)[:5000])
