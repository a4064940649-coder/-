#!/usr/bin/env python3
"""Q1（差中差）、Q2（前 k 產業內強勢股對產業市值加權）——自定口徑：靜態權重、毛報酬（零成本），
窗＝e 開盤 → 下一訊號日收盤（同紙上引擎的新持股段）。每產業各算後再對 k 個產業等權平均。"""
import sys, pickle, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as R
OUT = Path(__file__).resolve().parent / 'out'

def wret(idx, w, e, t1):
    if len(idx) == 0: return np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        base = np.where(np.isfinite(R.AOPEN[e, idx]), R.AOPEN[e, idx], R.AFF[e, idx]); G = R.AFF[t1, idx] / base
    G = np.where(np.isfinite(G), G, 1.0); w = np.asarray(w, float); return float((w / w.sum() * G).sum() - 1)

def main(j):
    D = pickle.load(open(OUT / f'plans_J{j}.pkl', 'rb')); days = D['days']; tend = len(R.IDX) - 1
    rows = []
    for k, (t, P) in enumerate(D['plans']):
        info = D['diag'][k]; t1 = days[k + 1] if k + 1 < len(days) else tend; e = t + 1
        d = R.IDX[t]; U, mc = R.universe(t); ind = R.ind_cached(d); fin = R.finmask(ind)
        sc, rv = R.scores(t, U, fin); o = R.order_key(sc, rv, U)
        rank = {U[i]: r for r, i in enumerate(o)}
        sel = info['sel']; order = R.SECT.get(str(d.date()), [])
        bot = [s for s in order[::-1] if s not in sel][:3]
        row = {'d': d}
        for n in (1, 2, 3):
            picks = set(R.CI[c] for c in info[f'IND_n{n}'])
            ex = []
            for s in sel:
                mem = [u for u in U if ind[u] == s]
                st = [u for u in mem if u in picks]
                if not mem or not st: continue
                ex.append((wret(st, mc[st], e, t1) - wret(mem, mc[mem], e, t1), wret(st, np.ones(len(st)), e, t1) - wret(mem, mc[mem], e, t1)))
            row[f'Q2_n{n}'] = np.mean([a for a, b in ex]) if ex else np.nan
            row[f'Q2eq_n{n}'] = np.mean([b for a, b in ex]) if ex else np.nan
        def spread(secs):
            v = []
            for s in secs:
                mem = [u for u in U if ind[u] == s and u in rank]
                if len(mem) < 2: continue
                mem.sort(key=lambda u: rank[u]); h = len(mem) // 2
                top = mem[:h]; bt = mem[-h:]
                v.append(wret(top, np.ones(len(top)), e, t1) - wret(bt, np.ones(len(bt)), e, t1))
            return np.mean(v) if v else np.nan
        row['Q1_top'] = spread(sel); row['Q1_bot'] = spread(bot); row['Q1'] = row['Q1_top'] - row['Q1_bot']
        rows.append(row)
    df = pd.DataFrame(rows).set_index('d'); df.to_parquet(OUT / f'q_J{j}.parquet'); return df

if __name__ == '__main__':
    for j in [int(x) for x in sys.argv[1:]]:
        main(j); print(j, flush=True)
