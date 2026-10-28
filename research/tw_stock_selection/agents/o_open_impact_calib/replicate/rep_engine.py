#!/usr/bin/env python3
"""獨立複製（容量表主欄）：只依 frozen_spec.txt 第三～五節與 o5 凍結規格的帳戶規則，自寫成本引擎（不 import ecore6／ecore_o5），
跑 AB、κ 點、四項摩擦＋Z(1.0)、4 級距 × {GROW, FIX} × 40 seeds，報每格 40 seed 平均淨值的年化與對 0050 t，
並與研究員 RESULTS 的 AB|M3V_point 主欄比對（年化差 ≤0.15pp、t 差 ≤0.05）。必須在 PREREG 之後執行。
注意：複製者與研究員為同一代理（無法真正獨立），見 FINDINGS 揭露。
用法：python3 -I -B replicate/rep_engine.py
"""
import json, math, hashlib, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
R = Path(__file__).resolve().parent; T = R.parent; DT = T / 'data'
res = json.loads((T / 'results.json').read_text(encoding='utf-8'))
assert 'PREREG' in res and hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256']

MC = json.load(open(DT / 'model_calib.json'))
CPI = {int(k): v for k, v in json.load(open(T / 'raw' / 'cpi_tw.json'))['index_2006_100'].items()}
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}

class P:
    pass
def load():
    p = P()
    O = pd.read_parquet(DT / 'px_open.parquet'); p.idx = O.index; p.codes = list(O.columns); ci = {c: i for i, c in enumerate(p.codes)}
    p.O = O.values; p.C = pd.read_parquet(DT / 'px_close.parquet').values; p.G = pd.read_parquet(DT / 'px_grow.parquet').values
    p.TR = pd.read_parquet(DT / 'px_traded.parquet').values > 0; p.PC = pd.read_parquet(DT / 'prevclose.parquet').values
    p.CF = pd.DataFrame(p.C).ffill().values
    F = pd.read_parquet(DT / 'feat_daily.parquet')
    p.V60 = F.pivot(index='date', columns='code', values='V60').reindex(index=p.idx, columns=p.codes).values
    p.SIG = F.pivot(index='date', columns='code', values='sig60').reindex(index=p.idx, columns=p.codes).values
    p.RK = F.pivot(index='date', columns='code', values='vrank').reindex(index=p.idx, columns=p.codes).values
    xr = json.load(open(DT / 'xrules.json'))
    p.X = {ci[c]: (v[0], p.idx.searchsorted(pd.Timestamp(v[1])), p.idx.searchsorted(pd.Timestamp(v[3])) if v[3] else 10 ** 9) for c, v in xr.items()}
    pl = pd.read_csv(DT / 'plans_ab.csv', dtype={'picks': str})
    p.plans = {s: {int(e): [ci[c] for c in k.split(',')] for e, k in zip(g.exec_idx, g.picks)} for s, g in pl.groupby('seed')}
    p4 = pd.read_csv(DT / 'plan_v4.csv')
    p.t0 = p.idx.get_loc(pd.Timestamp(p4.signal.iloc[0])); p.t1 = p.idx.get_loc(pd.Timestamp('2026-10-12'))
    p.sig = [pd.Timestamp(x) for x in p4.signal] + [pd.Timestamp('2026-10-12')]
    with np.errstate(all='ignore'):
        a = np.where(p.TR, p.G * p.PC / p.C, 1.0)
        p.ADJ = np.where(np.isfinite(a), a, 1.0)
        p.GAP = np.where(p.TR & np.isfinite(p.O), p.O / p.PC - 1, np.nan)
        ref = np.where(p.TR & np.isfinite(p.G) & (p.G > 0), p.C / p.G, p.PC)
    lim = np.where(p.idx >= pd.Timestamp('2015-06-01'), 0.10, 0.07)[:, None]
    def tk(x): return np.select([x < 10, x < 50, x < 100, x < 500, x < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
    with np.errstate(all='ignore'):
        u = ref * (1 + lim); d = ref * (1 - lim)
        tu = tk(np.nan_to_num(u, nan=1.0)); td = tk(np.nan_to_num(d, nan=1.0))
        p.UP = np.floor(u / tu + 1e-6) * tu; p.DN = np.ceil(d / td - 1e-6) * td
    p.E50 = pd.read_parquet(DT / 'bench.parquet')['E50']
    return p

IM = MC['impact']; DL, BB, BASE = IM['point']['delta'], IM['point']['b'], IM['base']
OAc = np.array(MC['oa']['coef']); OAs = MC['oa']['smear']
DE, DV = np.array(MC['drift']['edges']), np.array(MC['drift']['vals'])
PE, RE = np.array(MC['odd']['price_edges']), np.array(MC['odd']['rank_edges'])
OB, OS = np.array(MC['odd']['buy']), np.array(MC['odd']['sell'])
ZS = {int(k): v for k, v in MC['zcap']['share_by_dec'].items()}; ZD = MC['zcap']['share_default']
ODDDAY, FEECUT = pd.Timestamp('2020-10-26'), pd.Timestamp('2015-01-01')

def run(p, seed, tier, ver):
    plan = p.plans[seed]; N = len(p.codes)
    S = np.zeros(N); E0 = tier * CPI[2006] / CPI[2026]; cash = E0; peq = E0; unit = 1.0
    nav = [1.0]; pend = {}
    for t in range(p.t0 + 1, p.t1 + 1):
        day = p.idx[t]
        rate = 0.001425 * (0.6 if day < FEECUT else 0.28)
        def fee(v): return 0.0 if v <= 0 else max(20.0, rate * v)
        def oddpx(j):
            if day < ODDDAY: return p.C[t, j]
            g = p.GAP[t, j] if np.isfinite(p.GAP[t, j]) else 0.0
            return p.O[t, j] * (1 + DV[np.searchsorted(DE, g, side='right')])
        def oddc(j, sd):
            i = np.searchsorted(PE, p.PC[t, j], side='right'); r = p.RK[t - 1, j]
            k = np.searchsorted(RE, r if np.isfinite(r) else 1e9, side='right')
            return OB[i, k] if sd > 0 else max(OS[i, k], 0.0)
        def imp(j, L):
            s, v = p.SIG[t - 1, j], p.V60[t - 1, j]
            if not np.isfinite(s) or not (np.isfinite(v) and v > 0): return BASE
            return BASE + BB * s * (L * 1000 * p.PC[t, j] / v) ** DL
        def oahat(j, cx):
            v, s = p.V60[t - 1, j], p.SIG[t - 1, j]
            if not (np.isfinite(v) and np.isfinite(s)): return np.nan
            r = p.RK[t - 1, j]; r = 300.0 if not np.isfinite(r) else r
            x = np.array([1.0, np.log(max(r, 1.0)), np.log(p.PC[t, j]), np.log(max(s, 1e-4)), cx == 'pick', cx == 'drop'], float)
            return OAs * np.exp(OAc @ x) * v
        def zcap(j, px):
            if day < ODDDAY: return None
            v = p.V60[t - 1, j]
            if not np.isfinite(v): return None
            r = p.RK[t - 1, j]; d = min(max(int(math.ceil(r / 30)) if np.isfinite(r) else 10, 1), 10)
            return int(math.floor(ZS.get(d, ZD) * v / px))
        def blk(j, sd):
            return p.O[t, j] >= p.UP[t, j] - 1e-6 if sd > 0 else p.O[t, j] <= p.DN[t, j] + 1e-6
        def lots_cash(j, L, sd, cx):
            n = L; oa = oahat(j, cx)
            if np.isfinite(oa): n = min(n, max(1, int(math.floor(0.2 * oa / (1000 * p.PC[t, j])))))
            v = n * 1000 * p.O[t, j] * (1 + sd * imp(j, n))
            return n, (-(v + fee(v)) if sd > 0 else v - fee(v) - 0.003 * v)
        def odd_cash(j, q, sd, carry):
            px = oddpx(j); z = zcap(j, px); n = q if z is None else min(q, max(z, 0))
            if n < q: carry[j] = carry.get(j, 0) + sd * (q - n)
            if n <= 0: return 0, 0.0
            v = n * px * (1 + sd * oddc(j, sd))
            return n, (-(v + fee(v)) if sd > 0 else v - fee(v) - 0.003 * v)
        S = S * p.ADJ[t]
        flow = 0.0; epre = None
        if t in plan:
            pk = plan[t]; pend = {}
            tr = p.TR[t] & np.isfinite(p.O[t])
            fz = (S > 0) & ~tr
            for j in np.where(fz)[0]:
                if j in p.X and p.X[j][1] <= t < p.X[j][2]:
                    cash += S[j] * p.CF[t - 1, j] * p.X[j][0]; S[j] = 0.0; fz[j] = False
            if ver == 'FIX':
                epre = float((np.where(tr, S, 0) * np.where(tr, p.O[t], 0)).sum() + (S[fz] * p.CF[t - 1, fz]).sum() + cash)
                tg = tier * CPI[min(max(day.year, 2006), 2026)] / CPI[2026]; sc = tg / epre
                S = S * sc; cash *= sc; flow = tg - epre
            fr = np.clip(np.where(tr & (S > 0), S - np.floor(S + 1e-9), 0.0), 0, None)
            cash += float((fr * np.where(tr, p.O[t], 0)).sum()); S = S - fr
            cur = {int(j): int(round(S[j])) for j in np.where(tr & ~fz & (S > 0))[0]}
            V = sum(S[j] * p.O[t, j] for j in cur) + cash; k = len(pk); pks = set(pk)
            def tgt(T):
                d = {}
                for j in set(cur) | pks:
                    if j in pks:
                        pc = p.PC[t, j]; L = int(math.floor(T / (1000 * pc) + 1e-12)); q = int(math.floor((T - L * 1000 * pc) / pc + 1e-9)); x = L * 1000 + max(q, 0)
                    else: x = 0
                    if x != cur.get(j, 0): d[j] = x - cur.get(j, 0)
                return d
            memo = {}
            def cst(j, dd):
                if (j, dd) in memo: return memo[(j, dd)]
                sd = 1 if dd > 0 else -1; a = abs(dd); L, q = a // 1000, a % 1000; c = 0.0
                if L:
                    v = L * 1000 * p.O[t, j] * (1 + sd * imp(j, L)); c += sd * v + fee(v) + (0.003 * v if sd < 0 else 0)
                if q:
                    v = q * oddpx(j) * (1 + sd * oddc(j, sd)); c += sd * v + fee(v) + (0.003 * v if sd < 0 else 0)
                memo[(j, dd)] = c; return c
            need = lambda T: sum(cst(j, dd) for j, dd in tgt(T).items())
            hi = V / k * 0.98; lo = 0.0
            if need(hi) <= cash: lo = hi
            else:
                for _ in range(40):
                    m = (lo + hi) / 2
                    if need(m) <= cash: lo = m
                    else: hi = m
            for j, dd in sorted(tgt(lo).items(), key=lambda z: z[1]):
                sd = 1 if dd > 0 else -1; a = abs(dd)
                if blk(j, sd) or (sd > 0 and cst(j, dd) > cash + 1e-9):
                    pend[j] = pend.get(j, 0) + dd; continue
                L, q = a // 1000, a % 1000
                if q:
                    n, cc = odd_cash(j, q, sd, pend); cash += cc; S[j] += sd * n
                if L:
                    n, cc = lots_cash(j, L, sd, 'pick' if j in pks else 'drop'); cash += cc; S[j] += sd * n * 1000
                    if n < L: pend[j] = pend.get(j, 0) + sd * (L - n) * 1000
        elif pend:
            new = {}
            for j, dd in sorted(pend.items(), key=lambda z: z[1]):
                if not (p.TR[t, j] and np.isfinite(p.O[t, j])) or blk(j, 1 if dd > 0 else -1): new[j] = dd; continue
                sd = 1 if dd > 0 else -1; a = abs(dd) if sd > 0 else min(abs(dd), int(round(S[j])))
                L, q = a // 1000, a % 1000; got = 0
                if sd > 0:
                    u = p.O[t, j] * 1050.0; L = min(L, int(cash // u)) if u > 0 else 0
                    if q:
                        v = q * oddpx(j) * (1 + oddc(j, 1))
                        if v + fee(v) > cash - L * u: q = 0
                if q:
                    n, cc = odd_cash(j, q, sd, {}); cash += cc; S[j] += sd * n; got += n
                if L:
                    n, cc = lots_cash(j, L, sd, None); cash += cc; S[j] += sd * n * 1000; got += n * 1000
                rem = (abs(dd) if sd > 0 else a) - got
                if rem > 0: new[j] = sd * rem
            pend = new
        nz = S != 0
        eq = cash + float((S[nz] * p.CF[t, nz]).sum())
        unit *= (eq / peq) if epre is None else (epre / peq) * (eq / (epre + flow))
        nav.append(unit); peq = eq
    return pd.Series(nav, index=p.idx[p.t0:p.t1 + 1])

_G = {}
def _init(): _G['p'] = load()
def job(a):
    ti, ve, s = a
    return a, run(_G['p'], s, TIERS[ti], ve).values

def tstat(x):
    x = np.asarray(x); return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
if __name__ == '__main__':
    T0 = time.time()
    jobs = [(ti, ve, s) for ti in TIERS for ve in ('GROW', 'FIX') for s in range(40)]
    with ProcessPoolExecutor(4, initializer=_init) as ex: out = dict(ex.map(job, jobs, chunksize=4))
    p = load(); idx = p.idx[p.t0:p.t1 + 1]
    rr = res.get('RESULTS', {}).get('cells', {})
    rep = {}
    for ti in TIERS:
        for ve in ('GROW', 'FIX'):
            r = pd.DataFrame({s: pd.Series(out[(ti, ve, s)], index=idx).pct_change() for s in range(40)}).mean(axis=1).fillna(0)
            nav = (1 + r).cumprod()
            yrs = (nav.index[-1] - nav.index[0]).days / 365.25
            cg = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) * 100 - 100
            sd = [d for d in p.sig if d in nav.index]
            a, b = nav.reindex(sd), p.E50.reindex(sd)
            x = ((a.shift(-1) / a) - (b.shift(-1) / b)).iloc[:-1].dropna()
            mine = rr.get(f'AB|M3V_point|{ti}|{ve}', {})
            rep[f'{ti}|{ve}'] = dict(CAGR=float(cg), t=float(tstat(x)), researcher_CAGR=mine.get('CAGR'), researcher_t=mine.get('t'),
                                     d_CAGR=(float(cg) - mine['CAGR']) if mine else None, d_t=(float(tstat(x)) - mine['t']) if mine else None)
    ok = all(v['d_CAGR'] is not None and abs(v['d_CAGR']) <= 0.15 and abs(v['d_t']) <= 0.05 for v in rep.values())
    json.dump(dict(cells=rep, pass_=ok, runtime_s=time.time() - T0), open(R / 'rep_engine.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1), ok)
