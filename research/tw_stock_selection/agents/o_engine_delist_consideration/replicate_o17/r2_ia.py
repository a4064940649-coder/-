#!/usr/bin/env python3
"""o17 獨立複製 r2：C0／U_opt／U_opt_F2 名單（沿用 integrate_o16 的名單建構，只讀 import FM/flib），
自己寫 consideration 結算引擎（不讀研究員程式），算 integrate_o16 窗口口徑的判定欄位。"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; T = HERE.parent; AG = T.parent
FM = AG / 'o_finmask_delisted_fin_erratum'
sys.path.insert(0, str(FM)); import fclass as FCL, flib as FL; sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_forward_sequential_v2')); import fslib as F; sys.path.pop(0)
V, K = FL.V, FL.K; COLS = FL.COLS; CI = {c: j for j, c in enumerate(COLS)}; N = len(COLS)
IDX = K.IDX; LO, HI = FL.E.LO, FL.E.HI
SPEC = json.loads((FM / 'ERRATUM_SPEC_FM.json').read_text(encoding='utf-8'))
cls = SPEC['classifiers']; CL = FCL.Classifier(cls['keywords'], cls['f2_manual_exclusion'], cls['name_exclusion'])
C0M = FL.C0MASK.copy(); F2V = np.array([CL.f2(c) for c in COLS])
MASKF = {'C0': lambda d: C0M, 'F2': lambda d: F2V if LO <= pd.Timestamp(d) <= HI else C0M}
VARS = {'C0': ('C0', 'base'), 'U_opt': ('C0', 'opt'), 'U_opt_F2': ('F2', 'opt')}
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
E = F.e50(); E50 = E.reindex(IDX.union(E.index)).ffill().reindex(IDX).to_numpy()
G50 = np.ones(len(IDX)); G50[1:] = E50[1:] / E50[:-1]
TRADED, Cn, On, GROW = K.TRADED, K.Cn, K.On, K.GROW

# ------------------------------------------------ 事件表（凍結 v2）
EV = pd.read_csv(T / 'events_v2.csv', dtype=str)
assert hashlib.sha256((T / 'events_v2.csv').read_bytes()).hexdigest().startswith('66e80725')
D_ = 0.057
f = lambda x: float(x) if isinstance(x, str) and x not in ('', 'nan') else 0.0
def tpos(day):   # 第一個 ≥ day 的面板交易日
    return int(IDX.searchsorted(pd.Timestamp(day)))
EVENTS = {}
for _, r in EV.iterrows():
    c = r.code
    if c not in CI or not isinstance(r.last_trade, str): continue
    j = CI[c]
    term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) else pd.Timestamp(r.last_trade)
    # 本引擎面板 lt：終止日以前最後一筆成交（代碼重用以終止日切開）
    tr = np.where(TRADED[:, j] & (IDX <= term))[0]
    if not len(tr): continue
    lt = int(tr[-1])
    gap = (pd.Timestamp(r.last_trade) - IDX[lt]).days >= 7
    pay = int(IDX.searchsorted(term, side='right')) + 19
    sj = CI.get(r.successor) if isinstance(r.successor, str) else None
    s = tpos(r.succ_first_trade) if isinstance(r.succ_first_trade, str) else None
    EVENTS[j] = dict(code=c, type=r.type, lt=lt, gap=gap, pay=pay, cash=f(r.cash), ratio=f(r.ratio), pref=f(r.pref_ratio) * f(r.pref_value),
                     sj=sj, s=s, lc=Cn[lt, j])

def simulate(plan, buy, sell, t0, t1, mode='freeze', variant='main', unknown_codes=()):
    """mode：freeze/nofreeze/dzero（＝ia_core）或 cons（variant main/lo/hi/c50）。"""
    if mode in ('freeze', 'nofreeze', 'dzero'):
        kw = dict(freeze=mode != 'nofreeze', delist_zero=mode == 'dzero')
        return K.simulate(plan, buy, sell, t0, t1, **kw)
    dl = {'main': 0.0, 'c50': 0.0, 'lo': -D_, 'hi': D_}[variant]
    reb = {p[0]: p[1] for p in plan}
    val = np.zeros(N); cash = 1.0
    stale = np.zeros(N, bool)   # price_gap：面板最後收盤固定，下一換股日一般賣出成本出清
    claims = []                 # 現金請求權 dict(v, pay)
    swaps = []                  # 換股 dict(units, sj, s, v_fix, pref)
    prefs = []                  # 特別股固定市值（s 之後下一換股日以一般賣出成本出清）
    wo = []                     # unknown 下界：下一換股日沖銷
    nav = np.empty(t1 - t0 + 1); log = []
    trig = {}
    for j, ev in EVENTS.items(): trig.setdefault(ev['lt'] + 1, []).append(j)
    for t in range(t0, t1 + 1):
        isreb = t in reb
        if t > t0 and not isreb: val = val * GROW[t]
        for j in trig.get(t, []):
            ev = EVENTS[j]; v = val[j]
            if v <= 1e-15: continue
            assert not TRADED[t, j] or ev['gap'] is False
            sh = v / ev['lc']; ty = ev['type']
            if ev['gap']:
                stale[j] = True; log.append((ev['code'], 'price_gap', float(v))); continue
            val[j] = 0.0
            if ty == 'unknown' or ev['code'] in unknown_codes:
                if variant == 'lo': wo.append(v)
                else: claims.append(dict(v=v, pay=ev['pay']))
                log.append((ev['code'], 'unknown', float(v))); continue
            if ty == 'distress':
                log.append((ev['code'], 'distress', 0.0)); continue
            if ty == 'acq_ur' or (ty == 'swap_ur' and (ev['sj'] is None or ev['s'] is None)):
                claims.append(dict(v=v * (1 + dl), pay=ev['pay'])); log.append((ev['code'], 'acq_ur', float(v))); continue
            if ty == 'swap_ur':
                ratio = ev['lc'] / Cn[ev['lt'], ev['sj']] * (1 + dl)
                swaps.append(dict(units=sh * ratio, sj=ev['sj'], s=ev['s'], v_fix=v, pref=0.0)); log.append((ev['code'], 'swap_ur', float(v))); continue
            # 已核實
            if ev['cash'] > 0: claims.append(dict(v=sh * ev['cash'], pay=ev['pay']))
            if ev['ratio'] > 0:
                swaps.append(dict(units=sh * ev['ratio'], sj=ev['sj'], s=ev['s'], v_fix=v - sh * ev['cash'], pref=sh * ev['pref']))
            log.append((ev['code'], ty, float(v)))
        for cl in claims:
            if variant == 'c50' and t <= cl['pay']: cl['v'] *= G50[t]
        for cl in [c for c in claims if t >= c['pay']]:
            cash += cl['v']; claims.remove(cl)
        conv = [x for x in swaps if t >= x['s']]
        for sw in conv: swaps.remove(sw)
        if not isreb:
            for sw in conv:
                val[sw['sj']] += sw['units'] * Cn[t, sw['sj']]
                if sw['pref'] > 0: prefs.append(dict(v=sw['pref'], s=t))
        else:
            pk = reb[t]
            tr = TRADED[t] & np.isfinite(On[t])
            gap = np.where(tr, GROW[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            for sw in conv:
                h[sw['sj']] += sw['units'] * On[t, sw['sj']]
                if sw['pref'] > 0: prefs.append(dict(v=sw['pref'], s=t))
            frz = (h > 0) & ~tr & ~stale
            htr = np.where(frz, 0.0, h)
            pv = sum(p_['v'] for p_ in prefs if p_['s'] < t); prefs = [p_ for p_ in prefs if p_['s'] >= t]
            Vt = htr.sum() + cash + pv * (1 - sell)
            k = len(pk)
            def cost_of(Tt):
                tgt = np.zeros(N); tgt[pk] = Tt
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo_, hi_ = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo_ + hi_)
                if mid * k + cost_of(mid) > Vt: hi_ = mid
                else: lo_ = mid
            newval = np.where(frz, h, 0.0); newval[pk] += lo_
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            cash = 0.0; stale[:] = False; wo.clear()
        side = sum(c['v'] for c in claims) + sum(x['v_fix'] for x in swaps) + sum(wo) + sum(p_['v'] for p_ in prefs)
        nav[t - t0] = val.sum() + cash + side
    return pd.Series(nav, index=IDX[t0:t1 + 1]), {'log': log}

# ------------------------------------------------ 名單與統計（integrate_o16 窗口口徑）
PL, BND = {}, {}
for k, (mk, mode) in VARS.items():
    S = FL.VM(MASKF[mk], mode, 'SURV'); PL[k] = [(p[0], p[1]) for p in S.plan(10, 'official')]; BND[k] = S.bounds()
SD = pd.DatetimeIndex(BND['C0'][2])
def win_v2(nav):
    pts = list(SD[SD < V.END1012]) + [pd.Timestamp(V.END1012)]
    rows = [(pd.Period(a, 'M'), F._at(nav, b) / F._at(nav, a) - 1, F._at(E, b) / F._at(E, a) - 1, a.year) for a, b in zip(pts[:-1], pts[1:])]
    D = pd.DataFrame(rows, columns=['per', 'r', 'r50', 'yr']).set_index('per')
    return D[~D.index.duplicated(keep='last')].dropna()
def tt(x):
    x = np.asarray(x, float); s = x.std(ddof=1); return float(x.mean() / s * math.sqrt(len(x)))
def ann(r): r = np.asarray(r, float); return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
SEG = {'S1': ('2006-01', '2012-12'), 'S2': ('2013-01', '2018-12'), 'S3': ('2019-01', None)}
def judge(nav):
    W = win_v2(nav); r, b, y = W.r.to_numpy(), W.r50.to_numpy(), W.yr.to_numpy()
    o = dict(ann=ann(r), t=tt(r - b), n=len(r))
    for s_, (a, z) in SEG.items():
        m = np.ones(len(W), bool)
        if a: m &= W.index >= pd.Period(a, 'M')
        if z: m &= W.index <= pd.Period(z, 'M')
        o[s_] = ann(r[m]) - ann(b[m])
    sy = pd.Series(r - b).groupby(y).sum(); b3 = list(sy.sort_values(ascending=False).index[:3])
    m = ~np.isin(y, b3); o['t_drop3'] = tt(r[m] - b[m])
    return o, W.r

if __name__ == '__main__':
    RES = {}; NAVS = {}
    OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')
    for k in VARS:
        t0, t1, _ = BND[k]
        for c in ('normal', 'stress'):
            bb, ss = COST[c]
            for eng in ['freeze', 'nofreeze', 'dzero', 'cons_main', 'cons_lo', 'cons_hi', 'cons_c50']:
                if c == 'stress' and eng not in ('freeze', 'cons_main'): continue
                if eng.startswith('cons'): nav, lg = simulate(PL[k], bb, ss, t0, t1, 'cons', eng[5:])
                else: nav, lg = simulate(PL[k], bb, ss, t0, t1, eng)
                o, wr = judge(nav); NAVS[(k, c, eng)] = nav
                if eng.startswith('cons'): o['events'] = lg['log']
                else: o['n_frozen_ev'] = len(lg['frozen']); o['frozen_codes'] = sorted({x for _, cs, _ in lg['frozen'] for x in cs})
                RES[f'{k}|{c}|{eng}'] = o
                print(k, c, eng, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in o.items() if kk not in ('events', 'frozen_codes')}, flush=True)
            if k == 'C0':
                ref = OFF[f'EXFIN|n10|{c}'].dropna()
                RES[f'C0|{c}|maxdiff_main_v2'] = float(np.nanmax(np.abs(NAVS[('C0', c, 'freeze')].reindex(ref.index).to_numpy() - ref.to_numpy())))
                RES[f'C0|{c}|cons_vs_freeze_maxdiff'] = float(np.max(np.abs(NAVS[('C0', c, 'cons_main')].to_numpy() - NAVS[('C0', c, 'freeze')].to_numpy())))
    (HERE / 'r2_ia.json').write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    pd.DataFrame({'|'.join(k): v for k, v in NAVS.items()}).to_parquet(HERE / 'r2_navs.parquet')
