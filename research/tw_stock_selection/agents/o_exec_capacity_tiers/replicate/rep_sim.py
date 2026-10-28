#!/usr/bin/env python3
"""獨立複製：只依 PREREG.frozen_spec 重寫成交模擬（不 import ecore.py／run_oos.py），
跑 300 萬與 30 萬（GROW、手續費 A，40 seeds），和 navs_main.parquet 逐日比對（門檻：最大相對差 < 1e-6）。
寫法刻意不同：持股用 dict、逐檔物件化成本、自己從 data/ 重建 ADJ／GAP／OA／特徵矩陣。
用法：python3 -I -B replicate/rep_sim.py
"""
import json, math, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
res = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in res and hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256']

# ---------- 資料 ----------
def rd(n): return pd.read_parquet(DATA / n)
po = rd('px_open.parquet'); dates = po.index; codes = list(po.columns)
OPEN = po.values; CLOSE = rd('px_close.parquet').values; GROW = rd('px_grow.parquet').values
TRD = rd('px_traded.parquet').values > 0; PREV = rd('prevclose.parquet').values
LASTC = pd.DataFrame(CLOSE).ffill().values
ADJF = np.ones_like(CLOSE)
ok = TRD & np.isfinite(GROW * PREV / CLOSE)
ADJF[ok] = (GROW * PREV / CLOSE)[ok]
fd = pd.read_parquet(DATA / 'feat_daily.parquet')
cpos = {c: i for i, c in enumerate(codes)}; dpos = {d: i for i, d in enumerate(dates)}
V60 = np.full(CLOSE.shape, np.nan); SIG = np.full(CLOSE.shape, np.nan); RNK = np.full(CLOSE.shape, np.nan)
ii = fd['date'].map(dpos).to_numpy(); jj = fd['code'].map(cpos).to_numpy()
V60[ii, jj] = fd['V60'].to_numpy(); SIG[ii, jj] = fd['sig60'].to_numpy(); RNK[ii, jj] = fd['vrank'].to_numpy()
xr = json.loads((DATA / 'xrules.json').read_text(encoding='utf-8'))
XR = {cpos[c]: (v[0], dates.searchsorted(pd.Timestamp(v[1])), dates.searchsorted(pd.Timestamp(v[3])) if v[3] else 10 ** 9) for c, v in xr.items()}
ms = json.loads((DATA / 'model_spec.json').read_text(encoding='utf-8'))
cpi = {int(k): v for k, v in json.loads((ROOT / 'raw' / 'cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100'].items()}
plans = pd.read_csv(DATA / 'plans_ab.csv', dtype={'picks': str})
p4 = pd.read_csv(DATA / 'plan_v4.csv', dtype={'picks': str})
T0 = dates.get_loc(pd.Timestamp(p4.signal.iloc[0])); T1 = dates.get_loc(pd.Timestamp('2026-10-12'))
CUT = pd.Timestamp('2020-10-26')
RATE, MINF = 0.001425 * 0.28, 20.0

def gap(t, j):
    if TRD[t, j] and np.isfinite(OPEN[t, j]): return OPEN[t, j] / PREV[t, j] - 1
    return float('nan')

def oa_value(t, j, ctx):
    g = gap(t, j); g = 0.0 if not np.isfinite(g) else g
    x = [1.0, math.log(max(RNK[t - 1, j], 1)) if np.isfinite(RNK[t - 1, j]) else float('nan'), math.log(PREV[t, j]),
         math.log(max(SIG[t - 1, j], 1e-4)) if np.isfinite(SIG[t - 1, j]) else float('nan'), abs(g), 1.0 if ctx == 'pick' else 0.0, 1.0 if ctx == 'drop' else 0.0]
    z = sum(b * v for b, v in zip(ms['oa']['coef'], x))
    v = V60[t - 1, j]
    r = ms['oa']['kappa'] * math.exp(z) if np.isfinite(z) else float('nan')
    return r * v if (np.isfinite(v) and np.isfinite(r)) else float('nan')

def drift(g):
    e = ms['drift']['edges']; k = sum(1 for x in e if g >= x)
    return ms['drift']['vals'][k]

def oddc(p, rk, side):
    pe, re_ = ms['odd']['price_edges'], ms['odd']['rank_edges']
    a = sum(1 for x in pe if p >= x); rr = rk if np.isfinite(rk) else 1e9; b = sum(1 for x in re_ if rr >= x)
    return (ms['odd']['buy'] if side > 0 else ms['odd']['sell'])[a][b]

def fee(v): return max(MINF, RATE * v) if v > 0 else 0.0

class Leg:
    """單檔某日的成本計算（整股＋零股）。"""
    def __init__(self, t, j, ctx):
        self.t, self.j = t, j
        self.oa = oa_value(t, j, ctx); self.sg = SIG[t - 1, j]
        self.o, self.p = OPEN[t, j], PREV[t, j]
    def imp(self, lots):
        if np.isfinite(self.oa) and self.oa > 0 and np.isfinite(self.sg):
            return 0.001 + 0.5 * self.sg * math.sqrt(lots * 1000.0 * self.p / self.oa)
        return 0.001
    def oddbase(self):
        t, j = self.t, self.j
        if dates[t] < CUT: return CLOSE[t, j]
        return OPEN[t, j] * (1.0 + drift(gap(t, j)))
    def cash_need(self, d):
        if d == 0: return 0.0
        s = 1 if d > 0 else -1; a = abs(d); L, q = divmod(a, 1000); tot = 0.0
        if L:
            v = L * 1000.0 * self.o * (1 + s * self.imp(L)); tot += s * v + fee(v) + (0.003 * v if s < 0 else 0.0)
        if q:
            c = oddc(self.p, RNK[self.t - 1, self.j], s); v = q * self.oddbase() * (1 + s * c)
            tot += s * v + fee(v) + (0.003 * v if s < 0 else 0.0)
        return tot

def lot_trade(t, j, L, s, ctx):
    leg = Leg(t, j, ctx)
    n = L
    if np.isfinite(leg.oa):
        n = min(L, max(1, int(math.floor(0.2 * leg.oa / (1000.0 * leg.p)))))
    v = n * 1000.0 * OPEN[t, j] * (1 + s * leg.imp(n))
    return n, (-(v + fee(v)) if s > 0 else v - fee(v) - 0.003 * v)

def odd_trade(t, j, q, s):
    leg = Leg(t, j, None)
    v = q * leg.oddbase() * (1 + s * oddc(leg.p, RNK[t - 1, j], s))
    return -(v + fee(v)) if s > 0 else v - fee(v) - 0.003 * v

def target_shares(T, p):
    L = int(math.floor(T / (1000.0 * p) + 1e-12)); q = int(math.floor((T - L * 1000.0 * p) / p + 1e-9))
    return L * 1000 + max(q, 0)

def run(seed, tier):
    g = plans[plans.seed == seed]
    reb = {int(e): [cpos[c] for c in pk.split(',')] for e, pk in zip(g.exec_idx, g.picks)}
    hold = {}                       # j → 股數
    cash = tier * cpi[2006] / cpi[2026]
    prev_eq = cash; unit = 1.0; out = [1.0]; pend = {}
    for t in range(T0 + 1, T1 + 1):
        for j in list(hold): hold[j] *= ADJF[t, j]
        if t in reb:
            pk = reb[t]; pend = {}
            tradable = lambda j: TRD[t, j] and np.isfinite(OPEN[t, j])
            for j in list(hold):
                if hold[j] > 0 and not tradable(j) and j in XR and XR[j][1] <= t < XR[j][2]:
                    cash += hold[j] * LASTC[t - 1, j] * XR[j][0]; del hold[j]
            for j in list(hold):
                if hold[j] > 0 and tradable(j):
                    fr = hold[j] - math.floor(hold[j] + 1e-9)
                    if fr > 0: cash += fr * OPEN[t, j]; hold[j] -= fr
            cur = {j: int(round(s)) for j, s in hold.items() if s > 0 and tradable(j)}
            legs = {j: Leg(t, j, 'pick' if j in pk else 'drop') for j in set(cur) | set(pk)}
            memo = {}
            def nd(j, d):
                if (j, d) not in memo: memo[(j, d)] = legs[j].cash_need(d)
                return memo[(j, d)]
            def plan_d(T):
                d = {}
                for j in set(cur) | set(pk):
                    tg = target_shares(T, PREV[t, j]) if j in pk else 0
                    if tg - cur.get(j, 0) != 0: d[j] = tg - cur.get(j, 0)
                return d
            need = lambda T: sum(nd(j, d) for j, d in plan_d(T).items())
            hi = (sum(hold[j] * OPEN[t, j] for j in cur) + cash) / len(pk); lo = 0.0
            if need(hi) <= cash: lo = hi
            else:
                for _ in range(40):
                    m = (lo + hi) / 2
                    if need(m) <= cash: lo = m
                    else: hi = m
            for j, d in sorted(plan_d(lo).items(), key=lambda kv: kv[1]):
                s = 1 if d > 0 else -1; L, q = divmod(abs(d), 1000)
                if s > 0 and nd(j, d) > cash + 1e-9:
                    pend[j] = pend.get(j, 0) + d; continue
                if q:
                    cash += odd_trade(t, j, q, s); hold[j] = hold.get(j, 0) + s * q
                if L:
                    n, dc = lot_trade(t, j, L, s, 'pick' if j in pk else 'drop'); cash += dc; hold[j] = hold.get(j, 0) + s * n * 1000
                    if n < L: pend[j] = pend.get(j, 0) + s * (L - n) * 1000
        elif pend:
            np_ = {}
            for j, d in sorted(pend.items(), key=lambda kv: kv[1]):
                if not (TRD[t, j] and np.isfinite(OPEN[t, j])): np_[j] = d; continue
                s = 1 if d > 0 else -1; a = abs(d)
                if s < 0: a = min(a, int(round(hold.get(j, 0))))
                L, q = divmod(a, 1000)
                if s > 0:
                    u = OPEN[t, j] * 1000.0 * 1.05
                    L = min(L, int(cash // u)) if u > 0 else 0
                    if q and Leg(t, j, None).cash_need(q) > cash - L * u: q = 0
                if q: cash += odd_trade(t, j, q, s); hold[j] = hold.get(j, 0) + s * q
                n = 0
                if L:
                    n, dc = lot_trade(t, j, L, s, None); cash += dc; hold[j] = hold.get(j, 0) + s * n * 1000
                rem = (abs(d) if s > 0 else a) - q - n * 1000
                if rem > 0: np_[j] = s * rem
            pend = np_
        hold = {j: s for j, s in hold.items() if s != 0}
        eq = cash + sum(s * LASTC[t, j] for j, s in hold.items())
        unit *= eq / prev_eq; prev_eq = eq; out.append(unit)
    return pd.Series(out, index=dates[T0:T1 + 1])

if __name__ == '__main__':
    NM = pd.read_parquet(ROOT / 'navs_main.parquet')
    rep = {}; worst = 0.0
    seeds = range(40) if len(sys.argv) < 2 else [int(x) for x in sys.argv[1].split(',')]
    for tier, name in ((3e6, '300萬'), (3e5, '30萬')):
        for s in seeds:
            nv = run(s, tier)
            ref = NM[f'{name}|GROW|A|s{s:02d}']
            d = float((nv / ref - 1).abs().max()); worst = max(worst, d)
            rep[f'{name}|s{s:02d}'] = d
            print(name, s, d, flush=True)
    out = {'max_rel_diff_all': worst, 'pass_1e-6': worst < 1e-6, 'per_run': rep}
    (ROOT / 'replicate' / 'rep_result.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print('worst', worst, 'pass', worst < 1e-6)
