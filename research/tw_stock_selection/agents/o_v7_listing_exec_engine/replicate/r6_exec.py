#!/usr/bin/env python3
"""獨立複製（可執行表）：只依 frozen_spec.txt 第三節與 model_spec7.json，自寫逐日模擬器（不 import ecore7／run_exec7／v4lib），
價格、成交值、σ60、V60 名次、處置名單、併購資料一律取自 o4 複製員的 rcore.py（面板路徑換成本題複製員自建的 V7 面板）。
名單：研究員 data/plans_j7.csv（B10，j＝0～12、seeds 0～19；名單本身的正確性由官方日淨值複製驗證，j 名單沒有獨立重產——限制）。
格：B10 × j0～12 × seeds 0～19 × 4 級距 × {FIX, GROW} × {K5T, TICK}，一般成本。輸出 replicate/rep_exec.json（各格 j 中位數年化）。"""
import sys, types, json, math, time
from pathlib import Path
import numpy as np, pandas as pd
import multiprocessing as mp
H = Path(__file__).resolve().parent; TOP = H.parent; AG = TOP.parent
RC = AG / 'o_v4_revenue_patch_official' / 'replicate_o4' / 'rcore.py'
src = RC.read_text(encoding='utf-8')
OLD = "PNL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')"
src = src.replace(OLD, f"PNL = Path({str(H / 'data')!r})")
K = types.ModuleType('rcore_v7x'); K.__file__ = str(RC); exec(compile(src, str(RC), 'exec'), K.__dict__)
SPEC = json.loads((TOP / 'data' / 'model_spec7.json').read_text(encoding='utf-8'))
CPI = {int(k): v for k, v in json.loads((TOP / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100'].items()}
IDX = K.IDX; N = K.N; O = K.On; Cc = K.Cn; G = K.GROW; TR = K.TRADED; PC = K.PREVC; VAL = K.Vn
CF = pd.DataFrame(Cc).ffill().to_numpy()
gg = np.where(K.GOOD)[0]
SIG = pd.DataFrame(K.Rn[gg], index=IDX[gg]).rolling(60, min_periods=40).std().reindex(IDX).ffill().to_numpy()
V60 = K.V60
base = np.isfinite(V60) & (V60 >= 1e7) & np.isfinite(Cc) & (Cc >= 10)
RK = np.full(V60.shape, np.nan)
for t in range(len(IDX)):
    b = np.where(base[t])[0]
    if len(b):
        o = b[np.argsort(-V60[t, b], kind='stable')]; RK[t, o] = np.arange(1, len(o) + 1)
DISP = np.zeros((len(IDX), N), bool)
for j, s_, e_ in zip(K.DISP_J, K.DISP_S, K.DISP_E):
    a = np.searchsorted(IDX.values, s_); b = np.searchsorted(IDX.values, e_, side='right'); DISP[a:b, j] = True
E50 = K.E50
SH = {int(k): float(v) for k, v in SPEC['oa_share_by_dec'].items()}
ZS = {int(k): float(v) for k, v in SPEC['zcap']['share_by_dec'].items()}
DE, DV = np.array(SPEC['drift']['edges']), np.array(SPEC['drift']['vals'])
PE, RE = np.array(SPEC['odd']['price_edges']), np.array(SPEC['odd']['rank_edges']); OB, OS = np.array(SPEC['odd']['buy']), np.array(SPEC['odd']['sell'])
RATE, MINF, TAX = 0.001425 * 0.28, 20.0, 0.003
def tick(p): return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0
def dec(r): return 10 if not np.isfinite(r) else min(max(int(math.ceil(r / 30)), 1), 10)
def fee(v, t): return 0.0 if v <= 0 else max(MINF, (0.001425 * 0.6 if IDX[t] < pd.Timestamp('2015-01-01') else RATE) * v)
# 漲跌停
ref = np.where(TR & (G > 0), Cc / G, PC); lim = np.where((IDX >= pd.Timestamp('2015-06-01'))[:, None], 0.10, 0.07)
vt = np.vectorize(tick)
with np.errstate(invalid='ignore'):
    u = ref * (1 + lim); d = ref * (1 - lim)
    LU = np.floor(u / vt(np.nan_to_num(u, nan=1)) + 1e-6) * vt(np.nan_to_num(u, nan=1)); LD = np.ceil(d / vt(np.nan_to_num(d, nan=1)) - 1e-6) * vt(np.nan_to_num(d, nan=1))
# 現金對價併購：下市日、撥款＝下市日＋20 交易日；其餘死亡股用 rcore.x_action
CASHX = {}
for r in K.MC.itertuples():
    if r.code in K.CI and r.type == 'cash' and isinstance(r.cash_over_last_close, str) and r.cash_over_last_close:
        td = int(np.searchsorted(IDX.values, np.datetime64(pd.Timestamp(r.delist_date)))); CASHX[K.CI[r.code]] = (td, min(td + 20, len(IDX) - 1), float(r.cash_over_last_close), pd.Timestamp(r.last_trade))

def unit_cost(t, j, side, Q, model):
    o = O[t, j]; tk = tick(o) / o
    if model == 'TICK': return tk
    oa = SH[dec(RK[t - 1, j])] * (VAL[t, j] if np.isfinite(VAL[t, j]) and VAL[t, j] > 0 else V60[t - 1, j])
    k5 = 0.001 + (0.5 * SIG[t - 1, j] * math.sqrt(Q / oa) if (np.isfinite(oa) and oa > 0 and np.isfinite(SIG[t - 1, j]) and Q > 0) else 0.0)
    return max(k5, tk)
def oa_of(t, j): return SH[dec(RK[t - 1, j])] * (VAL[t, j] if np.isfinite(VAL[t, j]) and VAL[t, j] > 0 else V60[t - 1, j])
def odd_px(t, j):
    if IDX[t] < pd.Timestamp('2020-10-26'): return Cc[t, j]
    g = O[t, j] / PC[t, j] - 1 if np.isfinite(PC[t, j]) else 0.0
    return O[t, j] * (1 + DV[int(np.searchsorted(DE, g, side='right'))])
def odd_c(t, j, side):
    i = int(np.searchsorted(PE, PC[t, j], side='right')); r = RK[t - 1, j]; k = int(np.searchsorted(RE, r if np.isfinite(r) else 1e9, side='right'))
    c = float((OB if side > 0 else OS)[i, k]); return max(c, 0.0) if side < 0 else c
def trade(t, j, sh, side, model, cash, pend, budget=None):
    """執行 sh 股（side）；回傳 (成交股數, 現金變動)。"""
    if sh <= 0: return 0, 0.0
    if side < 0 and DISP[t, j]:
        c = Cc[t, j] if np.isfinite(Cc[t, j]) else CF[t, j]; px = max(c - tick(c), 0.01); v = sh * px
        return sh, v - fee(v, t) - TAX * v
    lots, odd = divmod(int(sh), 1000); done = 0; dc = 0.0
    if odd:
        bp = odd_px(t, j); n = odd
        if IDX[t] >= pd.Timestamp('2020-10-26') and np.isfinite(V60[t - 1, j]):
            capz = int(math.floor(ZS.get(dec(RK[t - 1, j]), 0.0002) * V60[t - 1, j] / bp)); n = min(n, max(capz, 0))
        if n > 0:
            px = bp * (1 + side * odd_c(t, j, side)); v = n * px
            dc += (v - fee(v, t) - TAX * v) if side < 0 else -(v + fee(v, t)); done += n
        if n < odd: pend[j] = pend.get(j, 0) + side * (odd - n)
    if lots:
        oa = oa_of(t, j); mx = max(1, int(math.floor(0.2 * oa / (1000.0 * PC[t, j])))) if np.isfinite(oa) else lots
        n = min(lots, mx)
        Q = n * 1000.0 * PC[t, j]; im = unit_cost(t, j, side, Q, model); px = O[t, j] * (1 + side * im); v = n * 1000.0 * px
        dc += (v - fee(v, t) - TAX * v) if side < 0 else -(v + fee(v, t)); done += n * 1000
        if n < lots: pend[j] = pend.get(j, 0) + side * (lots - n) * 1000
    return done, dc

def est_cost(t, j, delta, model):
    side = 1 if delta > 0 else -1; a = abs(delta); lots, odd = divmod(a, 1000); tot = 0.0
    if side < 0 and DISP[t, j]:
        c = Cc[t, j] if np.isfinite(Cc[t, j]) else CF[t, j]; px = max(c - tick(c), 0.01); v = a * px; return -(v - fee(v, t) - TAX * v)
    if lots:
        Q = lots * 1000.0 * PC[t, j]; v = lots * 1000.0 * O[t, j] * (1 + side * unit_cost(t, j, side, Q, model))
        tot += (v + fee(v, t)) if side > 0 else -(v - fee(v, t) - TAX * v)
    if odd:
        v = odd * odd_px(t, j) * (1 + side * odd_c(t, j, side)); tot += (v + fee(v, t)) if side > 0 else -(v - fee(v, t) - TAX * v)
    return tot

def sim(plan, t0, t1, tier, ver, model):
    reb = {p[0]: p[1] for p in plan}
    sc_ = lambda ts: CPI[min(max(ts.year, 2006), 2026)] / CPI[2026]
    S = np.zeros(N); cash = tier * sc_(IDX[t0]); rec = []; pend = {}
    nav = [1.0]; prev = cash; u = 1.0
    for t in range(t0 + 1, t1 + 1):
        with np.errstate(invalid='ignore', divide='ignore'):
            f = np.where(TR[t], G[t] * PC[t] / Cc[t], 1.0)
        S = S * np.where(np.isfinite(f), f, 1.0)
        flow = 0.0; pre = None
        if t in reb:
            pend = {}
            ok = TR[t] & np.isfinite(O[t])
            frz = (S > 0) & ~ok
            for j in np.where(frz)[0]:
                if j in CASHX and CASHX[j][0] <= t:
                    td, tp, mult, lt = CASHX[j]; rec.append([S[j] * CF[t - 1, j] * mult * E50[td] / E50[t - 1], td, tp]); S[j] = 0; frz[j] = False; continue
                a = K.x_action(j, t)
                if a is not None and a[0] == 'cash' and j not in CASHX: cash += S[j] * CF[t - 1, j] * a[1]; S[j] = 0; frz[j] = False
                elif a is not None and a[0] == 'zero': S[j] = 0; frz[j] = False
            keep = []
            for r in rec:
                if t >= r[2]: cash += r[0] * E50[t - 1] / E50[r[1]]
                else: keep.append(r)
            rec = keep
            if ver == 'FIX':
                pre = cash + float((np.where(ok, S, 0) * np.where(ok, O[t], 0)).sum()) + float((S[frz] * CF[t - 1, frz]).sum()) + sum(r[0] * E50[t - 1] / E50[r[1]] for r in rec)
                tg = tier * sc_(IDX[t]); k = tg / pre; S *= k; cash *= k; flow = tg - pre
                for r in rec: r[0] *= k
            fr = np.where(ok & (S > 0), S - np.floor(S + 1e-9), 0.0); cash += float((fr * np.where(ok, O[t], 0)).sum()); S = S - fr
            picks = reb[t]; cur = {j: int(round(S[j])) for j in np.where(ok & ~frz & (S > 0))[0]}
            Vt = cash + sum(S[j] * O[t, j] for j in cur)
            def want(T):
                dd = {}
                for j in set(cur) | set(picks):
                    tgt = 0
                    if j in picks and np.isfinite(PC[t, j]) and PC[t, j] > 0:
                        L_ = int(math.floor(T / (1000 * PC[t, j]) + 1e-12)); q = int(math.floor((T - L_ * 1000 * PC[t, j]) / PC[t, j] + 1e-9)); tgt = L_ * 1000 + max(q, 0)
                    if tgt != cur.get(j, 0): dd[j] = tgt - cur.get(j, 0)
                return dd
            hi = Vt / len(picks) * 0.98; lo = 0.0
            if sum(est_cost(t, j, d_, model) for j, d_ in want(hi).items()) > cash:
                for _ in range(40):
                    m_ = (lo + hi) / 2
                    if sum(est_cost(t, j, d_, model) for j, d_ in want(m_).items()) <= cash: lo = m_
                    else: hi = m_
                T = lo
            else: T = hi
            for j, d_ in sorted(want(T).items(), key=lambda x: x[1]):
                side = 1 if d_ > 0 else -1
                if not (side < 0 and DISP[t, j]) and ((side > 0 and O[t, j] >= LU[t, j] - 1e-6) or (side < 0 and O[t, j] <= LD[t, j] + 1e-6)):
                    pend[j] = pend.get(j, 0) + d_; continue
                if side > 0 and est_cost(t, j, d_, model) > cash + 1e-9: pend[j] = pend.get(j, 0) + d_; continue
                n, dc = trade(t, j, abs(d_), side, model, cash, pend); cash += dc; S[j] += side * n
        elif pend:
            newp = {}
            for j, d_ in sorted(pend.items(), key=lambda x: x[1]):
                side = 1 if d_ > 0 else -1
                if not (TR[t, j] and np.isfinite(O[t, j])) or (not (side < 0 and DISP[t, j]) and ((side > 0 and O[t, j] >= LU[t, j] - 1e-6) or (side < 0 and O[t, j] <= LD[t, j] + 1e-6))):
                    newp[j] = d_; continue
                a = abs(d_) if side > 0 else min(abs(d_), int(round(S[j])))
                if side > 0:
                    lots, odd = divmod(a, 1000); unit = O[t, j] * 1050.0; lots = min(lots, int(cash // unit)) if unit > 0 else 0
                    if odd and est_cost(t, j, odd, model) > cash - lots * unit: odd = 0
                    a2 = lots * 1000 + odd
                else: a2 = a
                tmp = {}
                n, dc = trade(t, j, a2, side, model, cash, tmp); cash += dc; S[j] += side * n
                rem = abs(d_) - n if side > 0 else a - n
                if rem > 0: newp[j] = side * rem
            pend = newp
        nz = S != 0
        eq = cash + float((S[nz] * CF[t, nz]).sum()) + sum(r[0] * E50[t] / E50[r[1]] for r in rec)
        u = u * (eq / prev) if pre is None else u * (pre / prev) * (eq / (pre + flow))
        nav.append(u); prev = eq
    return pd.Series(nav, index=IDX[t0:t1 + 1])

PJ = pd.read_csv(TOP / 'data' / 'plans_j7.csv', dtype={'picks': str}); PJ = PJ[(PJ.strat == 'AB_V7_B10') & (PJ.seed < 20)]
JEND = json.loads((TOP / 'data' / 'jscan_end_days.json').read_text(encoding='utf-8'))
PL = {}; BD = {}; GR = {}
for (j, s), g in PJ.groupby(['j', 'seed']):
    PL[(j, s)] = [(int(e), [K.CI[c] for c in pk.split(',')]) for e, pk in zip(g.exec_idx, g.picks)]
    de = pd.Timestamp(JEND[f'AB_V7_B10|J{j}']); BD[j] = (int(K.TPOS[pd.Timestamp(g.signal.iloc[0])]), int(K.TPOS[de])); GR[j] = [pd.Timestamp(x) for x in g.signal] + [de]
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
def job(a):
    j, s, tier, ver, model = a; t0, t1 = BD[j]
    return a, sim(PL[(j, s)], t0, t1, TIERS[tier], ver, model).pct_change().to_numpy()
def cagr(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25; return float(nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) * 100 - 100
if __name__ == '__main__':
    T0 = time.time()
    jobs = [(j, s, ti, ve, m) for j in range(13) for s in range(20) for ti in TIERS for ve in ('FIX', 'GROW') for m in ('K5T', 'TICK')]
    if len(sys.argv) > 1 and sys.argv[1] == 'dry': jobs = jobs[:8]
    acc = {}
    with mp.get_context('fork').Pool(4) as P:
        for i, (a, r) in enumerate(P.imap_unordered(job, jobs, chunksize=4)):
            acc.setdefault((a[0], a[2], a[3], a[4]), []).append(r)
            if i % 500 == 0: print(i, len(jobs), round(time.time() - T0), flush=True)
    out = {}
    for (j, ti, ve, m), rs in acc.items():
        t0, t1 = BD[j]; nav = pd.Series((1 + np.nan_to_num(np.mean(rs, axis=0), nan=0)).cumprod(), index=IDX[t0:t1 + 1])
        out.setdefault(f'{m}|{ti}|{ve}', {})[f'J{j}'] = {'CAGR': cagr(nav), 'n': len(rs)}
    summ = {k: {'CAGR_median': float(np.median([x['CAGR'] for x in v.values()])), 'by_j': v} for k, v in out.items()}
    (H / 'rep_exec.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print({k: round(v['CAGR_median'], 2) for k, v in summ.items()}, round(time.time() - T0))
