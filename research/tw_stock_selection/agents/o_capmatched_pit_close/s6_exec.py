#!/usr/bin/env python3
"""G1 可執行版（PREREG 後）：F1_K5_N20_sqrt 在 PIT 版本 (a)/(b) 的實際名單，用 K5T 可執行引擎（照 o_0050_tilt_pit/s7_exec.py 的 run_exec 逐行移植：
每邊 max(0.1%＋0.5·σ60·√(Q/OA), 一跳/開盤價)，參與上限 20%×OA；零股價差表＋開盤跳空漂移；2020-10-26 前零股以收盤價成交並標示不可執行；
手續費 0.1425%×0.28、最低 20 元；證交稅 0.3%；FIX 縮放成本；0.25% 不交易帶）。另加壓力：K5T＋每邊 0.5% 額外滑價。
級距：100 萬／300 萬／1,000 萬，GROW（獲利滾入）與 FIX。期間 MAIN（2019-05～2026-10），13 個 j。對 0050 含息。
python3 -I -B s6_exec.py <config>"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from multiprocessing import Pool
import pcore as P
PRE = json.loads((P.DATA / 'PREREG.json').read_text(encoding='utf-8'))
for f in ('s6_exec.py', 'pcore.py'): assert P.sha(HERE / f) == PRE['code_sha256'][f], f
CFG = sys.argv[1]
T = P.setup(CFG); L = P.L
V7 = P.AG / 'o_v7_listing_exec_engine'
SPEC = json.loads((V7 / 'data' / 'model_spec7.json').read_text(encoding='utf-8'))
CPI = {int(k): v for k, v in json.loads((V7 / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100'].items()}
def cpi_scale(ts): return CPI[min(max(ts.year, 2006), 2026)] / CPI[2026]
RATE = 0.001425 * 0.28; MINFEE = 20.0; TAX = 0.003
ODD_INTRADAY = pd.Timestamp('2020-10-26')
SHARE = {int(k): float(v) for k, v in SPEC['oa_share_by_dec'].items()}
DR_E, DR_V = np.array(SPEC['drift']['edges']), np.array(SPEC['drift']['vals'])
OPE, ORE = np.array(SPEC['odd']['price_edges']), np.array(SPEC['odd']['rank_edges'])
OB, OS = np.array(SPEC['odd']['buy']), np.array(SPEC['odd']['sell'])
BAND = 0.0025
V60, SIG = T.V60, T.SIG
with np.errstate(invalid='ignore'):
    VR = pd.DataFrame(V60).rank(axis=1, ascending=False).to_numpy()
GX, CP = P.GX, P.CP
def tickv(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def f_c2o(t, e):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(L.TRADED[e] & np.isfinite(L.On[e]) & (L.On[e] > 0), GX[e] * L.On[e] / L.Cn[e], np.nan)
def f_o2c(e, d2):
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.where(L.TRADED[e] & np.isfinite(L.On[e]) & (L.On[e] > 0), L.Cn[e] / L.On[e], np.nan)
    return f * (CP[d2] / CP[e])
def f_c2c(a, b): return CP[b] / CP[a]

def run_exec(days, plans, tier, mode, extra=0.0):
    ti = [L.IDX.get_loc(d) for d in days]
    v = np.zeros(L.N); cash = tier * cpi_scale(days[0]); out = []
    acc = dict(fee=0., tax=0., impact=0., odd=0., odd_val=0., lot_val=0., cap_hit=0, orders=0, flow_cost=0., pre_odd_val=0., extra=0.)
    for k in range(len(ti) - 1):
        t, t2 = ti[k], ti[k + 1]; e = t + 1; d = days[k]
        V0 = v.sum() + cash
        fo = f_c2o(t, e); tr = np.isfinite(fo)
        dead = (v > 0) & (L.LAST_TRADE < e)
        cash += (v[dead] * (1 - RATE - TAX - 0.001 - extra)).sum(); v[dead] = 0.0
        v = np.where(tr, v * np.nan_to_num(fo, nan=1.0), v)
        frz = (v > 0) & ~tr
        Eq = v.sum() + cash; flow_cost = 0.0; r_on = Eq / V0
        if mode == 'FIX':
            tgtE = tier * cpi_scale(L.IDX[e]); fl = tgtE - Eq
            flow_cost = abs(fl) * ((RATE + TAX + 0.001) if fl < 0 else (RATE + 0.001))
            sc = tgtE / Eq; v = v * sc; cash = cash * sc; Eq = tgtE; acc['flow_cost'] += flow_cost
        idx, w = plans[d]
        wv = np.zeros(L.N); np.add.at(wv, idx, w); wv[frz] = 0.0; wv[~tr] = 0.0; wv = wv / wv.sum()
        Etr = Eq - v[frz].sum()
        O = L.On[e]; cur_sh = np.where(tr, v / np.where(tr, O, 1.0), 0.0)
        tgt_sh = np.where(tr, np.round(wv * Etr * 0.995 / np.where(tr, O, 1.0)), 0.0)
        dsh = tgt_sh - cur_sh
        small = (np.abs(dsh * np.nan_to_num(O)) < BAND * Eq) & (tgt_sh > 0)
        dsh[small] = 0.0
        dsh = np.where(tr, np.round(dsh), 0.0)
        cash_delta = 0.0
        for j in np.where(dsh != 0)[0]:
            q = dsh[j]; side = 1 if q > 0 else -1; a = abs(q); lots, odd = int(a // 1000), a % 1000
            o = O[j]; pc = L.PREVC[e, j] if np.isfinite(L.PREVC[e, j]) else o
            vday = L.Vn[e, j] if np.isfinite(L.Vn[e, j]) and L.Vn[e, j] > 0 else V60[t, j]
            dec = int(min(max(math.ceil(VR[t, j] / 30), 1), 10)) if np.isfinite(VR[t, j]) else 10
            oa = SHARE.get(dec, SHARE[10]) * vday if np.isfinite(vday) else np.nan
            done = 0.0
            if lots:
                if np.isfinite(oa):
                    mx = max(1, int(math.floor(0.2 * oa / (1000 * pc))))
                    if lots > mx: lots = mx; acc['cap_hit'] += 1
                Q = lots * 1000 * pc
                sg = SIG[t, j]
                k5 = 0.001 + (0.5 * sg * math.sqrt(Q / oa) if (np.isfinite(sg) and np.isfinite(oa) and oa > 0) else 0.0)
                im = max(k5, float(tickv(o)) / o) + extra
                px = o * (1 + side * im); val = lots * 1000 * px
                fee = max(MINFEE, RATE * val); tax = TAX * val if side < 0 else 0.0
                cash_delta += -side * val - fee - tax; acc['fee'] += fee; acc['tax'] += tax; acc['impact'] += lots * 1000 * o * im; acc['lot_val'] += lots * 1000 * o; acc['orders'] += 1
                done += side * lots * 1000
            if odd:
                if L.IDX[e] < ODD_INTRADAY:
                    base = L.Cn[e, j]; acc['pre_odd_val'] += odd * base
                else:
                    g = o / pc - 1 if pc > 0 else 0.0
                    base = o * (1 + float(DR_V[int(np.searchsorted(DR_E, g, side='right'))]))
                pi = int(np.searchsorted(OPE, pc, side='right')); ri = int(np.searchsorted(ORE, VR[t, j] if np.isfinite(VR[t, j]) else 1e9, side='right'))
                c = float((OB if side > 0 else OS)[pi, ri]) + extra
                px = base * (1 + side * c); val = odd * px
                fee = max(MINFEE, RATE * val); tax = TAX * val if side < 0 else 0.0
                cash_delta += -side * val - fee - tax; acc['fee'] += fee; acc['tax'] += tax; acc['odd'] += odd * base * c; acc['odd_val'] += odd * base; acc['orders'] += 1
                done += side * odd
            cur_sh[j] += done
        cash = cash + cash_delta
        f2 = np.nan_to_num(f_o2c(e, t2), nan=1.0); fcc = f_c2c(t, t2)
        v = np.where(tr, cur_sh * np.where(tr, O, 0.0) * f2, np.where(frz, v * fcc, 0.0))
        V1 = v.sum() + cash
        out.append(V1 / V0 - 1 if mode == 'GROW' else r_on * (V1 - flow_cost) / Eq - 1)
    return np.array(out), acc

DS = P.daysets('MAIN')
PLANS = {}
for k, (sd, de) in DS.items():
    _, pl = P.run_scores(sd, de, None, want_plan=True)
    PLANS[k] = ({d: (js, ws) for d, js, ws in pl}, list(sd) + [de], P.bench_win(sd, de))
TIERS = [('100萬', 1e6, 'GROW'), ('300萬', 3e6, 'GROW'), ('1000萬', 1e7, 'GROW'), ('100萬', 1e6, 'FIX'), ('300萬', 3e6, 'FIX'), ('1000萬', 1e7, 'FIX')]
def job(a):
    k, tn, tier, mode, extra = a
    pl, days, b = PLANS[k]
    r, acc = run_exec(days, pl, tier, mode, extra)
    x = r - b
    return k, tn, mode, extra, float(x.mean() * 100), P.tstat(x), 100 * (acc['fee'] + acc['tax'] + acc['impact'] + acc['odd'] + acc['flow_cost']) / (tier * len(x)), \
        acc['odd_val'] / max(acc['odd_val'] + acc['lot_val'], 1), acc['pre_odd_val'] / max(acc['odd_val'] + acc['lot_val'], 1)
if __name__ == '__main__':
    jobs = [(k, tn, tier, mode, ex) for k in DS for tn, tier, mode in TIERS for ex in (0.0, 0.005)]
    with Pool(4) as pool: res = pool.map(job, jobs, chunksize=4)
    OUTD = {}
    for tn, tier, mode in TIERS:
        for ex in (0.0, 0.005):
            rows = [r for r in res if r[1] == tn and r[2] == mode and r[3] == ex]
            jr = [r for r in rows if r[0] != 'OFF']; off = [r for r in rows if r[0] == 'OFF'][0]
            key = f'{tn}|{mode}|' + ('K5T' if ex == 0 else 'K5T+0.5%')
            OUTD[key] = dict(vs0050_mean_pp_jmed=float(np.median([r[4] for r in jr])), vs0050_mean_pp_p10=float(np.percentile([r[4] for r in jr], 10)),
                             vs0050_t_jmed=float(np.median([r[5] for r in jr])), vs0050_t_p10=float(np.percentile([r[5] for r in jr], 10)),
                             vs0050_t_off=off[5], vs0050_mean_pp_off=off[4],
                             cost_pp_per_month_jmed=float(np.median([r[6] for r in jr])), odd_share_jmed=float(np.median([r[7] for r in jr])),
                             pre2020_10_odd_share_jmed=float(np.median([r[8] for r in jr])))
            print(CFG, key, {kk: round(vv, 3) for kk, vv in OUTD[key].items()}, flush=True)
    (P.OUT / f'exec_{CFG}.json').write_text(P.dumps(OUTD), encoding='utf-8')
