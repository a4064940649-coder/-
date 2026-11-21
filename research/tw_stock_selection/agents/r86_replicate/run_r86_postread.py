#!/usr/bin/env python3
"""R86 讀完原程式之後的對照（只用來解釋差異；判定用的是 run_r86.py／run_r86_extra.py 的數字）。
 1. 原程式把「應收股利」也算進換股日的總資產（影響每檔買多少）、權息合併事件的現金當 0；照這兩點改，看三情境標準路徑是否對上。
 2. 原程式的「第一年」＝2018～2025 每年最後一個交易日全現金新開戶、零股全期可買、看之後一年的報酬；低消影響＝低消 0 元 − 低消 20 元的第一年報酬差。
用法：python3 -I -B run_r86_postread.py → r86_postread.json"""
import sys, json, inspect
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r86lib as L
P, U, S, avA = L.load()
c = L.stlib.cagr
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]; DE = P.IDX[-1]
c19 = lambda nav: c(nav, D19, DE) * 100
SG = {s: L.s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=s) for s in range(4)}
DVz = L.dividends(P, 'zero'); DVe = L.dividends(P, 'est')
PR, THR, DLY = 0.0211, 20000.0, 20
SC = {'小資': dict(cap=5e5, tau=0.05, disc=0.6), '一般': dict(cap=1e6, tau=0.12, disc=0.28), '高所得': dict(cap=5e6, tau=0.28, disc=0.28)}
R = {}
# 1. 應收股利算進換股日總資產
src = inspect.getsource(L.NS['simulate_r86']) if False else None
import r86lib
s0 = inspect.getsource(L.s10lib.simulate_s10)
a = "        return cash + v50 + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())\n"
assert s0.count(a) == 1
orig_get = inspect.getsource
inspect.getsource = lambda f: s0.replace(a, a.replace("return cash + v50 +", "return sum(x[1] for x in pend) + cash + v50 +"))
NS2 = L._patched_sim(); inspect.getsource = orig_get
NS1 = L.NS
out = {}
for name, sc in SC.items():
    full = dict(tau=sc['tau'], prem=PR, thr=THR, delay=DLY)
    row = {}
    for lab, ns, dv in (('mine_est', NS1, DVe), ('mine_zero', NS1, DVz), ('recv_in_equity_zero', NS2, DVz), ('recv_in_equity_est', NS2, DVe)):
        L.NS = ns
        row[lab] = [c19(L.sim(P, SG[s], U, dv, cap=sc['cap'], disc=sc['disc'], **full)[0]) for s in range(4)]
    L.NS = NS1
    out[name] = row
R['scenario_after_by_shift'] = out
# 2. 第一年
odd0 = L.cslib.ODD_DATE; L.cslib.ODD_DATE = pd.Timestamp('2015-01-01')
T_full, t0_full = P.T, P.t0
fy = {}
for cap in (3e5, 5e5, 1e6):
    rows = []
    for y in range(2018, 2026):
        ts = int(np.searchsorted(P.IDX.values, np.datetime64(f'{y + 1}-01-01'))) - 1
        if ts + 252 >= T_full:
            continue
        for s in range(4):
            P.t0 = ts; P.T = ts + 253
            n1 = L.sim(P, SG[s], U, DVe, cap=cap)[0]; n0 = L.sim(P, SG[s], U, DVe, cap=cap, minfee=0.0)[0]
            rows.append((y, s, n1.iloc[-1] / cap - 1, n0.iloc[-1] / cap - 1))
    P.t0, P.T = t0_full, T_full
    d = pd.DataFrame(rows, columns=['y', 's', 'r20', 'r0'])
    fy[f'{int(cap/1e4)}萬'] = dict(minfee0_minus_minfee20_pp=float((d['r0'] - d['r20']).mean() * 100), base_first_year_mean_pct=float(d['r20'].mean() * 100), n=int(len(d)))
L.cslib.ODD_DATE = odd0
R['first_year_orig_def'] = fy
json.dump(R, open(HERE / 'r86_postread.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=0))
