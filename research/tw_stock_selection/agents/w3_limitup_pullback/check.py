#!/usr/bin/env python3
"""事後核對（不影響判定）：抽樣手算一筆事件、固定持有期毛異常（不含停損停利）、Day0 本身隔日報酬。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402

out = {}
df = pd.read_csv(HERE / 'events_PRIMARY.csv.gz', parse_dates=['d0', 'sig', 'entry'], dtype={'code': str})
chk = []
for seed in (3, 17):
    r = df.sample(1, random_state=seed).iloc[0]
    c = r['code']; j = R.cols.index(c); t0 = R.IDX.get_loc(r['d0']); s = R.IDX.get_loc(r['sig']); e = s + 1
    rows = []
    for t in range(t0 - 1, min(e + 6, R.T)):
        rows.append({'date': str(R.IDX[t].date()), 'open': float(R.OP[c].iloc[t]), 'high': float(R.Q.HI[c].iloc[t]), 'close': float(R.CR[c].iloc[t]),
                     'adj': round(float(R.ADJn[t, j]), 4), 'adj_open': round(float(R.AOn[t, j]), 4), 'vol': float(R.VOLn[t, j]),
                     'limup': round(float(R.Q.LIMUP[t, j]), 3), 'LU': bool(R.LU[t, j])})
    chk.append({'code': c, 'd0': str(r['d0'].date()), 'sig': str(r['sig'].date()), 'entry': str(r['entry'].date()), 'hold': int(r['hold']), 'why': r['why'],
                'raw_csv': round(float(r['raw']), 5), 'vma20_prior_d0': float(R.VMA20P[t0, j]), 'path': rows,
                'raw_manual': round(float(R.PX[e + int(r['hold']), j] / R.AOn[e, j] - 1), 5)})
out['spot_check'] = chk

# 固定持有期（不含停損停利），開盤進→開盤出，對同池等權
AO, PX = R.AOn, R.PX
WIN = [('select', '2005-01-01', '2019-04-30'), ('valid', '2019-05-01', '2026-10-31')]
for h in (1, 3, 5, 10, 20):
    rows = []
    for x in df.itertuples():
        j = R.cols.index(x.code); e = R.IDX.get_loc(x.entry); xx = e + h
        if xx >= R.T: continue
        raw = PX[xx, j] / AO[e, j] - 1; ew = math.expm1(R.LEW[xx] - R.LEW[e])
        rows.append((x.entry, x.sig, raw - ew, (1 + raw) * (1 - R.SELL) / (1 + R.BUY) - 1 - ew))
    f = pd.DataFrame(rows, columns=['entry', 'sig', 'abg', 'abn'])
    for nm, a, b in WIN:
        g = f[(f.sig >= a) & (f.sig <= b)]; gm = g.groupby(g.entry.dt.to_period('M'))
        out[f'fixed_h{h}_{nm}'] = {'n': len(g), 'ab_gross_pct': round(g.abg.mean() * 100, 3), 't_gross': R.tstat(gm.abg.mean()),
                                   'ab_net_pct': round(g.abn.mean() * 100, 3), 't_net': R.tstat(gm.abn.mean())}

# Day0（首板＋爆量）全部、不論有無回檔：Day0 收盤後隔日開盤買、持有 5 日的毛異常（比較基準）
D0 = R.day0_mask(False)
rows = []
for t0, j in np.argwhere(D0):
    e = t0 + 1; xx = e + 5
    if xx >= R.T or not np.isfinite(AO[e, j]): continue
    raw = PX[xx, j] / AO[e, j] - 1; ew = math.expm1(R.LEW[xx] - R.LEW[e])
    rows.append((R.IDX[e], R.IDX[t0], raw - ew, bool(R.OPEN_LU[e, j])))
f = pd.DataFrame(rows, columns=['entry', 'sig', 'abg', 'olu'])
for nm, a, b in WIN:
    g = f[(f.sig >= a) & (f.sig <= b) & ~f.olu]; gm = g.groupby(g.entry.dt.to_period('M'))
    out[f'all_day0_next_open_h5_{nm}'] = {'n': len(g), 'ab_gross_pct': round(g.abg.mean() * 100, 3), 't_gross': R.tstat(gm.abg.mean()),
                                          'share_open_limit_up_day1': round(float(f[(f.sig >= a) & (f.sig <= b)].olu.mean()), 3)}
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
fj = HERE / 'results.json'
d = json.loads(fj.read_text(encoding='utf-8')); d['diagnostics_posthoc'] = out
fj.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
