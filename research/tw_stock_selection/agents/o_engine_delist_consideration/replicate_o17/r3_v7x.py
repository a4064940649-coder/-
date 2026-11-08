#!/usr/bin/env python3
"""o17 獨立複製 r3：V7X（AB_V7_B10、K5T、GROW、R4）orig 與 consideration（main/lo/hi/c50）。
只讀 import V7X 共用引擎（o_exec_engine_rewrite_v7x/x_*.py，不改），自己寫 consideration 子類別（不讀研究員程式）。"""
import sys, json, math, time, hashlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; T = HERE.parent; AG = T.parent
VX = AG / 'o_exec_engine_rewrite_v7x'
sys.path.insert(0, str(VX))
import x_data as D, x_exec as X, x_stats as S
EXTRA = X.EXTRA; TAX = X.TAX
EV = pd.read_csv(T / 'events_v2.csv', dtype=str)
assert hashlib.sha256((T / 'events_v2.csv').read_bytes()).hexdigest().startswith('66e80725')
D_ = 0.057
fl = lambda x: float(x) if isinstance(x, str) and x not in ('', 'nan') else 0.0
EVENTS = {}
for _, r in EV.iterrows():
    c = r.code
    if c not in D.CI or not isinstance(r.last_trade, str): continue
    j = D.CI[c]
    term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) else pd.Timestamp(r.last_trade) + pd.Timedelta(days=14)
    tr = np.where(D.TR[:, j] & (D.IDX <= term))[0]
    if not len(tr): continue
    lt = int(tr[-1])
    if lt >= len(D.IDX) - 2: continue
    gap = (pd.Timestamp(r.last_trade) - D.IDX[lt]).days >= 7
    pay = min(int(np.searchsorted(D.IDX, term, side='right')) + 19, len(D.IDX) - 1)
    sj = D.CI.get(r.successor) if isinstance(r.successor, str) else None
    s = int(np.searchsorted(D.IDX, pd.Timestamp(r.succ_first_trade))) if isinstance(r.succ_first_trade, str) else None
    EVENTS[j] = dict(code=c, type=r.type, lt=lt, gap=gap, pay=pay, cash=fl(r.cash), ratio=fl(r.ratio), pref=fl(r.pref_ratio) * fl(r.pref_value),
                     sj=sj, s=s, lc=float(D.C[lt, j]))
TRIG = {}
for j, ev in EVENTS.items(): TRIG.setdefault(ev['lt'] + 1, []).append(j)
UNKNOWN_EXTRA = set()

class SimC(X.Sim):
    """variant：main/lo/hi/c50。"""
    def run(self, t1=None):
        c = self.c; t0 = self.t0; var = c['variant']
        dl = {'main': 0.0, 'c50': 0.0, 'lo': -D_, 'hi': D_}[var]
        sellc = lambda t: self.feerate(t) + TAX + self.extra
        t1 = len(D.IDX) - 1 if t1 is None else t1
        self.S = np.zeros(D.N); self.cash = c['tier'] * D.cpi_scale(2006); self.cash_late = 0.0
        self.pend = {}; mpend = []
        claims = []; swaps = []; prefs = []; wo = []; stale = set(); self.evlog = []
        side = lambda: sum(x[0] for x in claims) + sum(x['v'] for x in swaps) + sum(x[0] for x in prefs) + sum(wo)
        unit = 1.0; eq_prev = self.cash
        out = np.empty(t1 - t0 + 1); out[0] = 1.0
        for t in range(t0 + 1, t1 + 1):
            r50 = D.E50[t] / D.E50[t - 1] if np.isfinite(D.E50[t - 1]) and np.isfinite(D.E50[t]) else 1.0
            for m in mpend:
                if not (t in self.reb and m[1] < t): m[0] *= r50
            H = np.flatnonzero(self.S > 0)
            if len(H): self.S[H] *= D.ADJ[t, H]
            if self.cash_late:
                self.cash += self.cash_late; self.cash_late = 0.0
            # ---- consideration：lt 的下一個交易日觸發
            for j in TRIG.get(t, []):
                if self.S[j] <= 0: continue
                ev = EVENTS[j]; sh = self.S[j]; v = sh * ev['lc']; ty = ev['type']
                self.pend.pop(j, None)
                if ev['gap']:
                    stale.add(j); self.evlog.append((ev['code'], 'price_gap')); continue
                self.S[j] = 0.0; self.evlog.append((ev['code'], ty))
                if ty == 'unknown' or ev['code'] in UNKNOWN_EXTRA:
                    if var == 'lo': wo.append(v)
                    else: claims.append([v, ev['pay']])
                elif ty == 'distress':
                    pass
                elif ty == 'acq_ur' or (ty == 'swap_ur' and (ev['sj'] is None or ev['s'] is None)):
                    claims.append([v * (1 + dl), ev['pay']])
                elif ty == 'swap_ur':
                    ratio = ev['lc'] / D.C[ev['lt'], ev['sj']] * (1 + dl)
                    swaps.append(dict(u=sh * ratio, sj=ev['sj'], s=ev['s'], v=v, pref=0.0))
                else:
                    if ev['cash'] > 0: claims.append([sh * ev['cash'], ev['pay']])
                    if ev['ratio'] > 0:
                        swaps.append(dict(u=sh * ev['ratio'], sj=ev['sj'], s=ev['s'], v=v - sh * ev['cash'], pref=sh * ev['pref']))
            if var == 'c50':
                for x in claims: x[0] *= r50
            for x in [x for x in claims if t >= x[1]]:
                self.cash += x[0]; claims.remove(x)
            for x in [x for x in swaps if t >= x['s']]:
                self.S[x['sj']] += x['u']; swaps.remove(x)
                if x['pref'] > 0: prefs.append([x['pref'], t])
            tr = D.TR[t] & np.isfinite(D.O[t])
            if t in self.reb:
                pk = self.reb[t]
                self.pend = {}
                for j in list(stale):
                    if self.S[j] > 0: self.cash += self.S[j] * D.CF[t - 1, j] * (1 - sellc(t)); self.S[j] = 0.0
                    stale.discard(j)
                for x in [x for x in prefs if x[1] < t]:
                    self.cash += x[0] * (1 - sellc(t)); prefs.remove(x)
                wo.clear()
                H = np.flatnonzero(self.S > 0)
                frz = H[~tr[H]]
                for j in frz:
                    if j in D.MCASH:
                        fc, di, pi, uu = D.MCASH[j]
                        if di <= t < uu:
                            mpend.append([self.S[j] * D.CF[t - 1, j] * D.XR[j][0] * r50, pi]); self.S[j] = 0.0
                            self.acc['n_merger_delay'] += 1; self.evlog.append((D.COLS[j], 'orig_MCASH'))
                        continue
                    if j in D.XR and D.XR[j][1] <= t < D.XR[j][2]:
                        self.cash += self.S[j] * D.CF[t - 1, j] * D.XR[j][0]; self.S[j] = 0.0; self.evlog.append((D.COLS[j], 'orig_XR'))
                keep = []
                for m in mpend:
                    if m[1] < t: self.cash += m[0]
                    else: keep.append(m)
                mpend = keep
                H = np.flatnonzero(self.S > 0); trH = H[tr[H]]; frH = H[~tr[H]]
                mval = sum(m[0] for m in mpend) + side()
                eq_pre = self.cash + (self.S[trH] * D.O[t, trH]).sum() + (self.S[frH] * D.CF[t - 1, frH]).sum() + mval
                unit *= eq_pre / eq_prev
                flow = 0.0
                fr = self.S[trH] - np.floor(self.S[trH] + 1e-9)
                fr = np.where(fr > 0, fr, 0.0)
                self.cash += (fr * D.O[t, trH]).sum(); self.S[trH] -= fr
                self.rebalance(t, pk, tr)
                H = np.flatnonzero(self.S > 0)
                eq = self.cash + self.cash_late + (self.S[H] * D.CF[t, H]).sum() + sum(m[0] for m in mpend) + side()
                unit *= eq / (eq_pre + flow)
            else:
                if self.pend:
                    self.deferred(t, tr); self.acc['deferred_days'] += 1
                H = np.flatnonzero(self.S > 0)
                eq = self.cash + self.cash_late + (self.S[H] * D.CF[t, H]).sum() + sum(m[0] for m in mpend) + side()
                unit *= eq / eq_prev
            eq_prev = eq
            out[t - t0] = unit
        return pd.Series(out, index=D.IDX[t0:t1 + 1])

TIERS = {'100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
CELLS = [(c, t) for c in ('normal', 'stress') for t in TIERS]
ENG = ['orig', 'main', 'lo', 'hi', 'c50']
BASE = dict(mode='GROW', model='K5T', stress_odd_extra=True, rank_right=True, f1=True, f2=True, f3=0.5, f4=True)

def job(args):
    key, pl = args; out = {}; logs = {}
    for c, t in CELLS:
        for e in ENG:
            cfg = dict(BASE, tier=TIERS[t], cost=c)
            if e == 'orig': nav = X.Sim(pl, cfg).run(); lg = None
            else:
                s = SimC(pl, dict(cfg, variant=e)); nav = s.run(); lg = s.evlog
            out[(c, t, e)] = nav
            if c == 'normal' and t == '100萬' and e in ('main', 'lo'): logs[e] = lg
    return key, out, logs

if __name__ == '__main__':
    t0 = time.time()
    P = D.plans('AB_V7_B10', js=list(range(13)), seeds=list(range(20)))
    jobs = sorted(P.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    navs = {}; EVL = {}
    with Pool(4) as pool:
        for key, out, logs in pool.imap_unordered(job, jobs, chunksize=2):
            for ck, n in out.items(): navs.setdefault((key[0],) + ck, []).append(n)
            EVL[f'{key[0]}|{key[1]}'] = logs
    cols = {}
    for (j, c, t, e), L in navs.items():
        cols[f'B10|J{j}|K5T|{c}|{t}|GROW|{e}'] = S.avg_nav(L)
    df = pd.DataFrame(cols).sort_index(axis=1)
    df.to_parquet(HERE / 'r3_navs_v7x.parquet')
    (HERE / 'r3_evlog.json').write_text(json.dumps(EVL, ensure_ascii=False), encoding='utf-8')
    print('done', round(time.time() - t0), 's', flush=True)
