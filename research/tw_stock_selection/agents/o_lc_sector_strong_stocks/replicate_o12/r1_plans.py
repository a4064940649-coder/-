#!/usr/bin/env python3
"""每個 j 建立各組合在每個訊號日的目標權重（稀疏 dict），存 pickle。"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as R

K = 3; SBUF = 2
NS = (1, 2, 3); WTS = ('cap', 'eq'); CAPS = (0.15, 0.25)
import os
OUT = Path(__file__).resolve().parent / (os.environ.get('POUT') or ('out_sigex' if os.environ.get('SIGEX', '0') == '1' else 'out')); OUT.mkdir(exist_ok=True)

def build(j):
    days = R.sig_days(j)
    held_sec = []
    held_ind = {n: set() for n in NS}; held_ctrl = {n: set() for n in NS}
    plans = []; diag = []
    for t in days:
        d = R.IDX[t]; ds = str(d.date())
        U, mc = R.universe(t)
        ind = R.ind_cached(d); fin = R.finmask(ind)
        sc, rv = R.scores(t, U, fin)
        o = R.order_key(sc, rv, U)            # 位置（U 內索引），依分數排序
        rank_all = {U[i]: r for r, i in enumerate(o)}
        mcU = mc[U]
        P = {'B': dict(zip(U, mcU / mcU.sum()))}
        # 產業選擇（題 1 排名＋緩衝 k+2）
        order = R.SECT.get(ds)
        if order is None:
            diag.append((ds, 'no_order')); order = []
        keep = [s for s in held_sec if s in order[:K + SBUF]]
        sel = list(keep)
        for s in order:
            if len(sel) >= K: break
            if s not in sel: sel.append(s)
        sel = sel[:K] if len(keep) <= K else keep
        held_sec = sel
        sec_of = {U[i]: ind[U[i]] for i in range(len(U))}
        # IND_ONLY：前 k 產業的宇宙成員市值加權
        mem = [s for s in U if sec_of[s] in sel]
        if mem:
            w = mc[mem]; P['IND_ONLY'] = dict(zip(mem, w / w.sum()))
        else:
            P['IND_ONLY'] = {}
        info = {'d': ds, 'sel': sel, 'nmem': {s: sum(1 for x in mem if sec_of[x] == s) for s in sel}}
        for n in NS:
            # IND 個股選擇
            picks = []
            for s in sel:
                ranked = [U[i] for i in o if sec_of[U[i]] == s]   # 產業內依分數
                pos = {c: r for r, c in enumerate(ranked)}
                kp = [c for c in ranked if c in held_ind[n] and pos[c] <= n]   # 名次 ≤ n+1（0 起算 ≤ n）
                ch = list(kp)
                for c in ranked:
                    if len(ch) >= n: break
                    if c not in ch: ch.append(c)
                picks += ch
            held_ind[n] = set(picks)
            # CTRL
            ranked = [U[i] for i in o]; pos = {c: r for r, c in enumerate(ranked)}
            kp = [c for c in ranked if c in held_ctrl[n] and pos[c] <= K * (n + 1) - 1]
            ch = list(kp)
            for c in ranked:
                if len(ch) >= K * n: break
                if c not in ch: ch.append(c)
            held_ctrl[n] = set(ch)
            for kind, pk in (('IND', picks), ('CTRL', ch)):
                for wt in WTS:
                    for cap in CAPS:
                        if not pk:
                            P[f'{kind}_n{n}_{wt}_c{int(cap*100)}'] = {}; continue
                        base = mc[pk] if wt == 'cap' else np.ones(len(pk))
                        w = R.cap_weights(base.astype(float), cap)
                        P[f'{kind}_n{n}_{wt}_c{int(cap*100)}'] = dict(zip(pk, w))
            info[f'IND_n{n}'] = [R.COLS[c] for c in picks]; info[f'CTRL_n{n}'] = [R.COLS[c] for c in ch]
        # Q2：前 k 產業內每產業前 n 名（市值加權、不設上限）
        plans.append((t, P)); diag.append(info)
    return days, plans, diag

if __name__ == '__main__':
    js = [int(x) for x in sys.argv[1:]] or list(range(13))
    for j in js:
        days, plans, diag = build(j)
        pickle.dump(dict(days=days, plans=plans, diag=diag), open(OUT / f'plans_J{j}.pkl', 'wb'))
        print(j, len(days), str(R.IDX[days[0]].date()), str(R.IDX[days[-1]].date()), flush=True)
