#!/usr/bin/env python3
"""獨立複製（第 2 部分）：只依 PREREG 規格文字重寫 F1_K5_N20_sqrt 名單、成交引擎、j 日、隨機分數與統計。
不 import pcore／cmstrat；cmlib（o7）只當原始資料載入器（面板、52 週比、60 日均量、營收已公告值、處置、金融／KY／DR、臺灣50 成分、漲停判定、exfix）。
股數：replicate/shares_c.parquet（r_shares.py 自建，嚴格公告落後版）或 o7 shares_est（引擎核對用）。
python3 -I -B r_main.py <shares: c|est> <period: MAIN|GATE> [delist: 1|0]"""
import sys, json, math, time
from pathlib import Path
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
O7 = AG / 'o_capmatched_selection'
SH, PER = sys.argv[1], sys.argv[2]
DEL = (sys.argv[3] == '1') if len(sys.argv) > 3 else (SH == 'c')
sys.path.insert(0, str(O7)); import cmlib as L; sys.path.pop(0)   # noqa: E402（資料載入器）
IDX, COLS, N = L.IDX, list(L.COLS), L.N
# ---- 下市股補價（自己寫）----
if DEL:
    dp = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet'); dp['date'] = pd.to_datetime(dp['date'])
    ci = {c: j for j, c in enumerate(COLS)}; ti = {d: i for i, d in enumerate(IDX)}
    Cn, On, Vn, Rn = L.Cn.copy(), L.On.copy(), L.Vn.copy(), L.Rn.copy()
    for r in dp.itertuples(index=False):
        if r.code not in ci or r.date not in ti: continue
        i, j = ti[r.date], ci[r.code]
        if np.isfinite(Cn[i, j]): continue
        Cn[i, j] = r.close; On[i, j] = r.open; Vn[i, j] = r.value; Rn[i, j] = r.ret
    L.Cn, L.On, L.Vn, L.Rn = Cn, On, Vn, Rn
    L.TRADED = np.isfinite(Cn); L.GROW = np.where(L.TRADED, 1 + np.nan_to_num(Rn, nan=0.0), 1.0)
    L.LAST_TRADE = np.where(L.TRADED.any(axis=0), len(IDX) - 1 - np.argmax(L.TRADED[::-1], axis=0), -1)
    L.CFF = pd.DataFrame(Cn).ffill().to_numpy(); L.PREVC = pd.DataFrame(Cn).ffill().shift(1).to_numpy()
    L.NOBS = L.TRADED.sum(axis=1); L.GOOD = L.NOBS >= 300; L._GX.clear(); L._RS.clear()
S52, V60, _ = L.rolling_signals()
G = L.grow_exfix()
TR = L.TRADED; Cn, On = L.Cn, L.On
if SH == 'c':
    SHR = pd.read_parquet(HERE / 'shares_c.parquet').reindex(index=IDX, columns=COLS).to_numpy()
else:
    SHR = pd.read_parquet(O7 / 'data' / 'shares_est.parquet').reindex(index=IDX, columns=COLS).to_numpy()
if DEL:
    LT = pd.read_parquet(AG / 'o_v7_listing_exec_engine' / 'data' / 'listed_twse.parquet').reindex(columns=COLS).astype(float)
    LT = LT.reindex(IDX.union(LT.index)).sort_index().ffill().reindex(IDX).fillna(1.0).to_numpy() > 0.5
    SHR = np.where(LT, SHR, np.nan)
PHI = pd.read_parquet(O7 / 'data' / 'float_phi.parquet')['phi'].reindex(COLS).to_numpy()
J2330 = COLS.index('2330')
BUY, SELL = 0.001425 * 0.6 + 0.001, 0.001425 * 0.6 + 0.003 + 0.001

# ---- 日期（規格第三節）----
TX = pd.DatetimeIndex(sorted(set(pd.read_csv(O7 / 'raw' / 'bench' / 'taiex_twii.csv', parse_dates=['date'])['date'])))
def dl(m):
    base = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12')}.get(str(m), pd.Timestamp(m.year, m.month, 10))
    k = TX.searchsorted(base); return k
GOODI = set(np.where(L.GOOD)[0])
def jday(m, j):
    k = dl(m) + j
    if k >= len(TX): return None
    d = TX[k]
    if d not in IDX: return None
    t = IDX.get_loc(d)
    if t not in GOODI or (t + 1) not in GOODI: return None
    return d
def jset(j, m0, m1, me):
    ds = [jday(m, j) for m in pd.period_range(m0, m1, freq='M')]; ds = [d for d in ds if d is not None]
    de = jday(pd.Period(me, 'M'), j); de = IDX[-1] if de is None else de
    return [d for d in ds if d < de], de
def offset(m0, m1, me):
    out = []
    for m in pd.period_range(m0, me, freq='M'):
        x = IDX[(IDX >= m.start_time + pd.Timedelta(days=10)) & (IDX <= m.end_time)]
        if len(x): out.append(x[0])
    de = out[-1] if out[-1].to_period('M') == pd.Period(me, 'M') else IDX[-1]
    return [d for d in out if d < de], de
if PER == 'MAIN':
    SETS = {'OFF': offset('2019-05', '2026-09', '2026-10')}
    for j in range(13): SETS[f'J{j}'] = jset(j, '2019-05', '2026-09', '2026-10')
else:   # GATE：o7 原口徑 2019-01～2026-10-12 官方日
    SETS = {'OFF': offset('2019-01', '2026-09', '2026-10')}

# ---- 橫斷面 ----
CSC = {}
def xsec(d):
    if d in CSC: return CSC[d]
    t = IDX.get_loc(d); e = t + 1
    mc = SHR[t] * L.CFF[t]
    live = TR[max(t - 5, 0):t + 1].any(axis=0)
    cu = np.isfinite(mc) & (mc > 0) & live & ~L.DRMASK
    rank = np.full(N, 10 ** 9); o = np.argsort(-np.where(cu, mc, -1.0), kind='stable'); rank[o] = np.arange(1, N + 1); rank[~cu] = 10 ** 9
    mem = L.tw50_at(IDX[e]) or []
    js = np.array([COLS.index(c) for c in mem if c in COLS], int)
    fa = mc[js] * PHI[js]; ok = np.isfinite(fa) & (fa > 0); js = js[ok]
    R = np.zeros(N); R[js] = fa[ok] / fa[ok].sum()
    y3 = L.y3_ab(d)
    el = (cu & (rank <= 300) & ~L.FINMASK & ~L.FORMASK & np.isfinite(V60[t]) & (V60[t] >= 1e7) & np.isfinite(Cn[t]) & (Cn[t] >= 10)
          & np.isfinite(S52[t]) & np.isfinite(y3) & ~L.disp_mask(d) & np.isfinite(On[e]) & TR[e])
    p1 = pd.Series(np.where(el, S52[t], np.nan)).rank(pct=True).to_numpy(); p2 = pd.Series(np.where(el, y3, np.nan)).rank(pct=True).to_numpy()
    sc = np.where(el, 0.5 * p1 + 0.5 * p2, np.nan)
    blk = L.limit_up_open(e)
    CSC[d] = dict(t=t, e=e, mc=mc, rank=rank, R=R, mem=js, el=el, sc=sc, y3=y3, blk=blk)
    return CSC[d]

def f1(cs, score, held):
    """規格第一節 F1：H＝前 5 大成分 ∪ 金融成分（權重 R）；其他成分 R>5pp 保留 R−5pp；餘額給 rank>5、不在 H 的合格股分數前 20，∝√市值，單檔主動 ≤5pp，觸頂加檔；開盤漲停未持有跳過。"""
    R, mc = cs['R'], cs['mc']
    mem = sorted(cs['mem'], key=lambda j: -mc[j])
    H = set(mem[:5]) | {j for j in mem if L.FINMASK[j]}
    w = {j: R[j] for j in H}
    for j in mem:
        if j not in H and R[j] > 0.05: w[j] = R[j] - 0.05
    budget = 1 - sum(w.values())
    ok = cs['el'] & (cs['rank'] > 5) & np.isfinite(score)
    idx = np.where(ok)[0]; idx = idx[~np.isin(idx, list(H))]
    order = idx[np.lexsort((idx, -np.nan_to_num(cs['y3'][idx], nan=-1e18), -np.round(score[idx], 10)))]
    avail = [j for j in order if not (cs['blk'][j] and j not in held)]
    resid = {j: (R[j] - w.get(j, 0.0)) if j in w else R[j] for j in avail}
    k = min(20, len(avail))
    while True:
        pick = avail[:k]
        b = np.sqrt(np.array([mc[j] for j in pick])); cap = np.array([max(resid[j], 0.0) + 0.05 for j in pick])
        x = budget * b / b.sum(); fixed = np.zeros(len(pick), bool)
        for _ in range(len(pick) + 1):
            ov = (x > cap + 1e-12) & ~fixed
            if not ov.any(): break
            fixed |= ov; x[fixed] = cap[fixed]
            fr = ~fixed
            if not fr.any(): break
            x[fr] = (budget - x[fixed].sum()) * b[fr] / b[fr].sum()
        if x.sum() >= budget - 1e-9 or k >= len(avail): break
        k += 1
    for j, v in zip(pick, x): w[j] = w.get(j, 0.0) + v
    js = np.array([j for j in w if w[j] > 1e-9], int); ws = np.array([w[j] for j in js]); return js, ws / ws.sum()

def engine(plan, sdays, de):
    """自寫引擎：收盤→開盤 G×O/C；開盤無成交者凍結；之後不再成交者以賣出成本轉現金；成本以固定點解；開盤→收盤 C/O；其餘日 ×G。"""
    t0 = IDX.get_loc(sdays[0]); t1 = IDX.get_loc(de)
    q = {IDX.get_loc(d) for d in sdays} | {t1}
    h = np.zeros(N); cash = 1.0; nav = {}
    for t in range(t0 + 1, t1 + 1):
        if t in plan:
            js, ws = plan[t]
            tr = TR[t] & np.isfinite(On[t])
            h = np.where(tr, h * G[t] * On[t] / np.where(tr, Cn[t], 1.0), h)
            fz = (h > 0) & ~tr
            dead = fz & (L.LAST_TRADE < t)
            cash += (h[dead] * (1 - SELL)).sum(); h[dead] = 0.0; fz &= ~dead
            cur = np.where(fz, 0.0, h); V = cur.sum() + cash
            w = np.zeros(N); w[js] = ws; w[fz] = 0.0; w = w / w.sum()
            Tv = V
            for _ in range(200):
                tg = w * Tv; c = BUY * np.clip(tg - cur, 0, None).sum() + SELL * np.clip(cur - tg, 0, None).sum()
                Tn = V - c
                if abs(Tn - Tv) < 1e-15: break
                Tv = Tn
            h = np.where(fz, h, w * Tv * np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)); cash = 0.0
        else:
            h = h * G[t]
        if t in q: nav[t] = h.sum() + cash
    nav[t0] = 1.0
    v = np.array([nav[IDX.get_loc(d)] for d in sdays] + [nav[t1]])
    return v[1:] / v[:-1] - 1

def run(key, seed):
    sd, de = SETS[key]; plan = {}; held = set()
    rg = None if seed is None else np.random.default_rng(seed)
    for d in sd:
        cs = xsec(d)
        sc = cs['sc'] if rg is None else np.where(cs['el'], rg.random(N), np.nan)
        js, ws = f1(cs, sc, held); plan[cs['e']] = (js, ws); held = set(js.tolist())
    return engine(plan, sd, de)
NSEED = 50 if PER == 'GATE' else 1000
SEED0 = 5000 if PER == 'GATE' else 20000
for k, (sd, de) in SETS.items():
    for d in sd: xsec(d)
def job(a):
    k, s0, s1 = a
    return k, s0, np.array([run(k, SEED0 + s) for s in range(s0, s1)])
def tstat(x): x = np.asarray(x); return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
if __name__ == '__main__':
    t0 = time.time()
    jobs = [(k, s, min(s + 125, NSEED)) for k in SETS for s in range(0, NSEED, 125)]
    with Pool(4) as p: res = p.map(job, jobs)
    E50 = L.E50.to_numpy()
    out = {}
    for k, (sd, de) in SETS.items():
        ra = run(k, None); Rr = np.vstack([x[2] for x in sorted([x for x in res if x[0] == k], key=lambda x: x[1])])
        ti = [IDX.get_loc(d) for d in sd] + [IDX.get_loc(de)]; r5 = E50[ti][1:] / E50[ti][:-1] - 1
        x = ra - Rr.mean(axis=0)
        out[k] = dict(n=len(x), mean_pp=float(x.mean() * 100), t=tstat(x), vs0050_t=tstat(ra - r5), perm_p=float(np.mean((Rr - r5).mean(axis=1) >= (ra - r5).mean())))
    jk = [k for k in out if k.startswith('J')]
    summ = dict(shares=SH, period=PER, delist=DEL, nseed=NSEED, t_off=out['OFF']['t'], mean_pp_off=out['OFF']['mean_pp'], by_set=out, runtime_s=time.time() - t0)
    if jk:
        ts = [out[k]['t'] for k in jk]
        summ.update(t_jmed=float(np.median(ts)), t_p10=float(np.percentile(ts, 10)), mean_pp_jmed=float(np.median([out[k]['mean_pp'] for k in jk])),
                    vs0050_t_jmed=float(np.median([out[k]['vs0050_t'] for k in jk])))
    (HERE / f'r_out_{SH}_{PER}_{int(DEL)}.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in summ.items() if k != 'by_set'}, ensure_ascii=False))
