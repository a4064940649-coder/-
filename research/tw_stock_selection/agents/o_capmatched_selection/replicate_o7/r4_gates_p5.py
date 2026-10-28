#!/usr/bin/env python3
"""閘門 (i) 規則重建一致率、(iii) 複製追蹤誤差（日曆月）；P5 去台積電。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L, rstrat as S
from r1_main import XS, DATES, run_f1
res = {}
# (i) 2019 年後每季審核：以官方前一期名單＋規則，比對官方新名單
segs = [(pd.Timestamp('2015-09-18'), L.FIRST_BEFORE)] + L.SEG   # (eff, members_after)
def official_after(eff):
    cur = L.FIRST_BEFORE
    for e, m in L.SEG:
        if e <= eff: cur = m
    return cur
def official_before(eff):
    cur = L.FIRST_BEFORE
    for e, m in L.SEG:
        if e < eff: cur = m
    return cur
agree = []
for y in range(2019, 2027):
    for mo in (3, 6, 9, 12):
        first = pd.Timestamp(y, mo, 1)
        fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7) + pd.Timedelta(days=14)
        # 若官方有該季的 eff（可能不是第三個星期五）就用官方日期
        offs = [e for e, _ in L.SEG if e.year == y and e.month == mo]
        eff = offs[0] if offs else fri
        if eff > pd.Timestamp('2026-10-22'): continue
        prev = {L.CI[c] for c in official_before(eff) if c in L.CI}
        i = L.IDX.searchsorted(eff - pd.Timedelta(days=25), side='right') - 1
        cap, r = L.cap_rank(i)
        mem = {j for j in prev if r[j] <= 60}
        mem |= {j for j in np.argsort(r) if r[j] <= 40}
        if len(mem) < 50:
            for j in np.argsort(r):
                if len(mem) >= 50: break
                mem.add(j)
        elif len(mem) > 50: mem = set(sorted(mem, key=lambda j: r[j])[:50])
        newo = {L.CI[c] for c in official_after(eff) if c in L.CI}
        agree.append((str(eff.date()), len(mem & newo) / 50))
res['gate_i'] = dict(n=len(agree), min=min(a for _, a in agree), mean=float(np.mean([a for _, a in agree])), low=[x for x in agree if x[1] < 0.98])
# (iii) REP 對 0050 日曆月追蹤誤差
navs = pd.read_parquet(L.OUT / 'navs_main.parquet')
rep = navs['REP|normal'].dropna()
m = rep.resample('ME').last(); b = L.E50.reindex(rep.index).resample('ME').last()
ex = (m.pct_change() - b.pct_change()).dropna()
res['gate_iii_calmonth_TE'] = {'2019-2022': ex['2019':'2022'].std() * math.sqrt(12) * 100, '2023-2026': ex['2023':'2026'].std() * math.sqrt(12) * 100}
res['gate_iii_calmonth_monthly_sd'] = {'2019-2022': ex['2019':'2022'].std() * 100, '2023-2026': ex['2023':'2026'].std() * 100}
# P5：去台積電
W, _ = run_f1(); tj = L.CI['2330']
def drop(w):
    w = w.copy(); w[tj] = 0; return w / w.sum()
Fx = {d: drop(W[d]) for d in DATES[:-1]}; Rx = {d: drop(XS[d].R) for d in DATES[:-1]}
nf, _ = S.simulate(Fx, 'normal'); nr, _ = S.simulate(Rx, 'normal')
d_ = (S.win_returns(nf, DATES) - S.win_returns(nr, DATES)).dropna()
res['P5_exTSMC_vs_exTSMC_rep'] = {p: (round(S.sel(d_, p).mean() * 100, 3), round(S.tstat(S.sel(d_, p)), 2)) for p in ('VAL', 'POST2019')}
# 台積電主動權重貢獻：恆為 0 → 0
json.dump(res, open(L.OUT / 'r4_gates_p5.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
