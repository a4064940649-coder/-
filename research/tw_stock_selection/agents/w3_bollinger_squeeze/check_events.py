#!/usr/bin/env python3
"""事後診斷（不參與判定）：用 run.py 產出的事件進場點，改看固定持有 5/10/20/40 日（e 開盤 → e+h 開盤）的毛異常報酬（對同池等權），
區分「訊號本身沒資訊」與「出場規則（跌破 MB）傷害」。也比較 MAIN 與無壓縮對照 BB_BREAK。結果寫入 results.json 的 posthoc_diag。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
IDX, COLS, N = L.IDX, L.COLS, L.N
GOOD = (L.adj.notna().sum(axis=1) >= 300).to_numpy(); gi = np.where(GOOD)[0]; G = IDX[gi]; T = len(G)
AOg = L.AO[gi]; ADJg = L.ADJ_RAW[gi]; OPOK = np.isfinite(L.OPEN_RAW[gi]) & np.isfinite(ADJg)
Cg = L.C.loc[GOOD].to_numpy(); V60g = M.V60.to_numpy()[gi]
common = np.array([not c.startswith('91') for c in COLS])
with np.errstate(invalid='ignore'):
    POOL = (V60g >= 1e7) & (Cg >= 10) & common[None, :] & np.isfinite(ADJg)
CI = {c: j for j, c in enumerate(COLS)}
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
out = {}
for nm in ('MAIN', 'a_NOVOL', 'b_PCT5', 'c_TTM', 'd_HEADFAKE'):
    ev = pd.read_csv(HERE / f'events_{nm}.csv.gz', dtype={'code': str})
    o = {}
    for h in (1, 5, 10, 20, 40):
        rows = []
        for r in ev.itertuples():
            e = r.e; x = e + h
            if x >= T: continue
            j = CI[r.code]
            pl = np.flatnonzero(POOL[r.s] & OPOK[r.s + 1])
            bench = np.nanmean(AOg[x, pl] / AOg[e, pl] - 1)
            rows.append((pd.Timestamp(r.d), G[e].to_period('M'), AOg[x, j] / AOg[e, j] - 1 - bench))
        df = pd.DataFrame(rows, columns=['d', 'em', 'abn'])
        for lab, a, b in (('select', '2005-01-01', '2019-04-30'), ('valid', '2019-05-01', '2026-10-31')):
            m = (df.d >= a) & (df.d <= b); g = df[m].groupby('em').abn.mean()
            o[f'h{h}_{lab}'] = {'n': int(m.sum()), 'mean_gross_abn_pct': round(float(df[m].abn.mean() * 100), 2), 't_month': L.tstat(g)}
    out[nm] = o
    print(nm, {k: (v['mean_gross_abn_pct'], v['t_month']) for k, v in o.items()}, flush=True)
res['posthoc_diag_fixed_horizon_gross'] = out
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
