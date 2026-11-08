#!/usr/bin/env python3
"""獨立複製之三：V7X consideration 自寫（不 import 研究員 cz_v7x／cz_engine）。只讀 import o_exec_engine_rewrite_v7x 的 x_data／x_exec／x_stats。
做法：RSim 繼承 x_exec.Sim，自己重寫 run() 的下市處理（依 SPEC 文字），事件表用 rp_events.csv；
orig 不重跑，直接用 navs_oos_R4.parquet（研究中心現行正式）。
python3 -I -B rp_v7x.py → rp_v7x.json（判定欄）"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
V7X = AG / 'o_exec_engine_rewrite_v7x'
sys.path.insert(0, str(V7X)); import x_data as D, x_exec as X, x_stats as S; sys.path.pop(0)
DB = 0.057
VR = json.loads((V7X / 'results.json').read_text())
CFG0 = dict(VR['PREREG']['orig_cfg']); CFG0.update(dict(f1=True, f2=True, f3=0.5, f4=True))
T = pd.read_csv(HERE / 'rp_events.csv', dtype={'code': str, 'successor': str})
OVR = {}
TAG = sys.argv[1] if len(sys.argv) > 1 else 'asis'
if TAG == 'aligned':   # 只為了檢查引擎一致性：把複製員與研究員分類不同的事件對齊到研究員分類
    OVR = json.loads((HERE / 'adjudication.json').read_text(encoding='utf-8'))['align_to_researcher']


def rules(bound, c50):
    dl = {'main': 0.0, 'lo': -DB, 'hi': DB}[bound]; out = {}
    for r in T.itertuples():
        j = D.CI.get(r.code)
        if j is None: continue
        typ = OVR.get(r.code, r.type)
        ltU = pd.Timestamp(r.last_trade); term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) and r.terminate_date else ltU + pd.Timedelta(days=14)
        w = np.flatnonzero(D.TR[:, j]); w = w[D.IDX[w] <= max(term, ltU)]; lt = int(w[-1])
        lc = float(D.C[lt, j]); ti = int(np.searchsorted(D.DATES, np.datetime64(term))); pay = min(ti + 20, len(D.IDX) - 1)
        sj = D.CI.get(r.successor) if isinstance(r.successor, str) and r.successor else None
        s = None
        if sj is not None:
            ww = np.flatnonzero(D.TR[:, sj] & np.isfinite(D.O[:, sj])); ww = ww[ww >= ti]; s = int(ww[0]) if len(ww) else None
        P = []
        if D.IDX[lt] < ltU - pd.Timedelta(days=7): P.append(('sellfix', lc, lt + 1, None, 0))
        elif typ == 'cash': P.append(('cash', r.cash, pay, None, 0))
        elif typ in ('swap', 'conversion', 'mixed') and s is not None:
            P.append(('stock', lc, s, sj, r.ratio))
            if typ == 'mixed': P.append(('cash', r.cash, pay, None, 0))
            if typ == 'mixed' and np.isfinite(r.pref_ratio): P.append(('sellfix', r.pref_ratio * r.pref_value, s, None, 0))
        elif typ == 'swap_ur' and s is not None and np.isfinite(D.C[lt, sj]): P.append(('stock', lc, s, sj, lc / D.C[lt, sj] * (1 + dl)))
        elif typ in ('swap_ur', 'acq_ur'): P.append(('cash', lc * (1 + dl), pay, None, 0))
        elif typ == 'unknown': P.append(('zero', lc, lt + 1, None, 0) if bound == 'lo' else ('cash', lc, pay, None, 0))
        out[j] = dict(at=lt + 1, P=P, c50=c50)
    return out


class RSim(X.Sim):
    def __init__(self, pl, cfg, RU):
        super().__init__(pl, cfg); self.RU = RU
        self.at = {}
        for j, ru in RU.items(): self.at.setdefault(ru['at'], []).append(j)

    def run(self, t1=None):
        c = self.c; t0 = self.t0; t1 = len(D.IDX) - 1 if t1 is None else t1
        self.S = np.zeros(D.N); self.cash = c['tier'] * D.cpi_scale(2006); self.cash_late = 0.0
        self.pend = {}; mp = []; Q = []
        unit = 1.0; prev = self.cash; out = np.empty(t1 - t0 + 1); out[0] = 1.0
        for t in range(t0 + 1, t1 + 1):
            g50 = D.E50[t] / D.E50[t - 1] if np.isfinite(D.E50[t - 1]) and np.isfinite(D.E50[t]) else 1.0
            for m in mp:
                if not (t in self.reb and m[1] < t): m[0] *= g50
            H = np.flatnonzero(self.S > 0)
            if len(H): self.S[H] *= D.ADJ[t, H]
            if self.cash_late: self.cash += self.cash_late; self.cash_late = 0.0
            tr = D.TR[t] & np.isfinite(D.O[t]); rb = t in self.reb
            for j in self.at.get(t, []):
                if self.S[j] > 0:
                    n = self.S[j]; ru = self.RU[j]; self.S[j] = 0.0
                    if j in self.pend: del self.pend[j]
                    for kind, ps, rel, sj, ratio in ru['P']:
                        Q.append([kind, n * ps, max(rel, t), sj, n * ratio, ru['c50']])
            for q in Q:
                if q[0] == 'cash' and q[5] and t <= q[2]: q[1] *= g50
            nq = []
            for q in Q:
                if t >= q[2]:
                    if q[0] == 'cash': self.cash += q[1]; continue
                    if q[0] == 'stock': self.S[q[3]] += q[4]; continue
                    if rb: self.cash += 0.0 if q[0] == 'zero' else q[1] * (1 - X.TAX - self.feerate(t)); continue
                nq.append(q)
            Q = nq
            qv = lambda: sum(q[1] for q in Q)
            if rb:
                pk = self.reb[t]; self.pend = {}
                H = np.flatnonzero(self.S > 0)
                for j in H[~tr[H]]:
                    if j in self.RU: continue
                    if j in D.MCASH:
                        fc, di, pi, uu = D.MCASH[j]
                        if di <= t < uu:
                            mp.append([self.S[j] * D.CF[t - 1, j] * D.XR[j][0] * g50, pi]); self.S[j] = 0.0
                        continue
                    if j in D.XR and D.XR[j][1] <= t < D.XR[j][2]:
                        self.cash += self.S[j] * D.CF[t - 1, j] * D.XR[j][0]; self.S[j] = 0.0
                k2 = []
                for m in mp:
                    if m[1] < t: self.cash += m[0]
                    else: k2.append(m)
                mp = k2
                H = np.flatnonzero(self.S > 0); a = H[tr[H]]; b = H[~tr[H]]
                pre = self.cash + (self.S[a] * D.O[t, a]).sum() + (self.S[b] * D.CF[t - 1, b]).sum() + sum(m[0] for m in mp) + qv()
                unit *= pre / prev
                fr = self.S[a] - np.floor(self.S[a] + 1e-9); fr = np.where(fr > 0, fr, 0.0)
                self.cash += (fr * D.O[t, a]).sum(); self.S[a] -= fr
                self.rebalance(t, pk, tr)
                H = np.flatnonzero(self.S > 0)
                eq = self.cash + self.cash_late + (self.S[H] * D.CF[t, H]).sum() + sum(m[0] for m in mp) + qv()
                unit *= eq / pre
            else:
                if self.pend: self.deferred(t, tr); self.acc['deferred_days'] += 1
                H = np.flatnonzero(self.S > 0)
                eq = self.cash + self.cash_late + (self.S[H] * D.CF[t, H]).sum() + sum(m[0] for m in mp) + qv()
                unit *= eq / prev
            prev = eq; out[t - t0] = unit
        return pd.Series(out, index=D.IDX[t0:t1 + 1])


RUS = {'cons_main': rules('main', False), 'cons_lo': rules('lo', False), 'cons_hi': rules('hi', False), 'cons_c50': rules('main', True)}
if TAG == 'aligned': RUS = {k: v for k, v in RUS.items() if k in ('cons_lo', 'cons_hi')}
TI = {'100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
CELLS = [(c, t) for c in ('normal', 'stress') for t in TI]


def job(a):
    key, pl = a; o = {}
    for c, t in CELLS:
        cfg = dict(tier=TI[t], mode='GROW', model='K5T', cost=c); cfg.update(CFG0)
        for eg, RU in RUS.items(): o[(c, t, eg)] = RSim(pl, cfg, RU).run()
    return key, o


if __name__ == '__main__':
    t0 = time.time(); tag = TAG
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    acc = {}
    with Pool(4) as pool:
        for n, (key, o) in enumerate(pool.imap_unordered(job, sorted(P.items()), chunksize=1)):
            for ck, nv in o.items(): acc.setdefault((key[0],) + ck, []).append((key[1], nv))
            if n % 40 == 0: print(n, round(time.time() - t0), flush=True)
    R4 = pd.read_parquet(V7X / 'out' / 'navs_oos_R4.parquet')
    TAB = {}
    for c, t in CELLS:
        per0 = [S.stats(R4[f'B10|J{j}|K5T|{c}|{t}|GROW'], 'AB_V7_B10', j) for j in range(13)]
        TAB[f'{c}|{t}|orig'] = S.jsumm(per0)
        for eg in RUS:
            per = [S.stats(S.avg_nav([nv for s, nv in sorted(acc[(j, c, t, eg)], key=lambda x: x[0])]), 'AB_V7_B10', j) for j in range(13)]
            TAB[f'{c}|{t}|{eg}'] = S.jsumm(per)
    F = ['CAGR_med', 'exc_med', 't_0050_med', 'exc_p10', 'CAGR_Y19on_med', 'exc_Y19on_med', 't_Y19on_med', 'exc_Y19on_p10']
    out = {k: {f: v[f] for f in F} for k, v in TAB.items()}
    (HERE / f'rp_v7x_{tag}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('done', round(time.time() - t0), 's')
