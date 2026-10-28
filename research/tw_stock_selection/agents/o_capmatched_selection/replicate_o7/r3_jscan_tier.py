#!/usr/bin/env python3
"""換股日 j＝0～12 掃描；P1 層內選股力。"""
import sys, json, math, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L, rstrat as S
from r1_main import XS, DATES

res = {}
# ---------------- j-scan ----------------
jp = L.OUT / 'xs_j.pkl'
XJ = pickle.load(open(jp, 'rb')) if jp.exists() else {}
rows = []
for j in range(13):
    GI = L.IDX[L.GOOD.to_numpy()]; GP = {d: k for k, d in enumerate(GI)}
    G = L.GOOD.to_numpy()
    def sh(d):
        k = GP[d] + j
        while not G[L.POS[GI[k]] + 1]: k += 1   # e 必須是正常交易日（2018-01-13 只有 46 檔）
        return GI[k]
    dj = pd.DatetimeIndex([sh(d) for d in DATES if GP[d] + j + 2 < len(GI)])
    lab = dict(zip(dj, DATES))
    W = {}; held = set()
    for d in dj[:-1]:
        if j == 0: x = XS[d]
        else:
            if d not in XJ: XJ[d] = S.XS(d)
            x = XJ[d]
        w, _, _ = S.f1_weights(x, held); W[d] = w; held = set(np.where(w > 0)[0])
    nv, _ = S.simulate(W, 'normal')
    wr = S.win_returns(nv, dj)
    b = L.E50.reindex(dj); bw = (b.shift(-1) / b - 1).iloc[:-1]
    ex = (wr - bw).dropna(); ex.index = [lab[d] for d in ex.index]
    r = {'j': j}
    for p in ('SEL', 'VAL', 'POST2019'):
        x = S.sel(ex, p); r[p + '_ex'] = x.mean() * 100; r[p + '_t'] = S.tstat(x)
    rows.append(r); print(r, flush=True)
pickle.dump(XJ, open(jp, 'wb'))
df = pd.DataFrame(rows).set_index('j')
res['jscan'] = df.round(3).to_dict(orient='index')
res['jscan_summary'] = {
    'VAL_t_median': df.VAL_t.median(), 'VAL_t_p10': df.VAL_t.quantile(0.1),
    'VAL_ex_median': df.VAL_ex.median(), 'VAL_ex_p10': df.VAL_ex.quantile(0.1),
    'POST_t_median': df.POST2019_t.median(), 'SEL_t_median': df.SEL_t.median(), 'SEL_t_max': df.SEL_t.max(),
    'SEL_t_rank_of_j0': int((df.SEL_t > df.SEL_t.loc[0]).sum()) + 1}

# ---------------- P1 層內選股力 ----------------
b0 = S.bench_win(DATES)
def tier_port(lo, hi, top_w, bench_w):
    WT, WB = {}, {}
    for d in DATES[:-1]:
        x = XS[d]
        m = x.el & (x.rank >= lo) & (x.rank <= hi)
        idx = np.where(m)[0]
        wb = np.zeros(L.NC); wt = np.zeros(L.NC)
        if len(idx) == 0:
            WT[d] = wb; WB[d] = wb; continue
        wb[idx] = x.cap[idx] if bench_w == 'cap' else 1.0; wb /= wb.sum()
        o = [j for j in x.order if m[j]]
        k = max(1, int(round(0.3 * len(o)))); top = np.array(o[:k])
        wt[top] = x.cap[top] if top_w == 'cap' else (np.sqrt(x.cap[top]) if top_w == 'sqrt' else 1.0); wt /= wt.sum()
        WT[d] = wt; WB[d] = wb
    nt, _ = S.simulate(WT, 'normal'); nb, _ = S.simulate(WB, 'normal')
    d_ = (S.win_returns(nt, DATES) - S.win_returns(nb, DATES)).dropna()
    return {p: (round(S.sel(d_, p).mean() * 100, 3), round(S.tstat(S.sel(d_, p)), 2)) for p in ('SEL', 'VAL', 'PRE')}
res['P1'] = {}
for lo, hi in ((51, 300), (11, 50), (2, 10)):
    for tw in ('eq', 'cap'):
        k = f'{lo}-{hi}|top30_{tw}_minus_capw'
        res['P1'][k] = tier_port(lo, hi, tw, 'cap'); print(k, res['P1'][k], flush=True)
json.dump(res, open(L.OUT / 'r3_jscan_tier.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res['jscan_summary'], indent=1, default=float))
