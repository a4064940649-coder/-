#!/usr/bin/env python3
"""反方審查（成本與可成交）o8：不改研究員檔案，只讀。python3 -I -B review_exec_o8.py
檢查：
 (1) 開盤漲停買不到／跌停賣不掉：s7_exec 未處理。變體 LIM：開盤價 ≥ 漲停（或 ≤ 跌停）時該檔買單（賣單）整筆不成交，保留現有持股。
 (2) 處置股（disposal_hist_rejudge.csv，s0 有抓但 s7 沒用）：變體 DISP：處置期間（start～end）該檔不交易（分盤撮合／預收款券都視為當日不調整）。
 (3) 2020-10-26 前零股以收盤價成交屬不可執行段：只看 2020-11 以後的 t 值。
 (4) 事件次數：漲跌停／處置命中的訂單數與金額占比。
引擎：逐行複製 s7_exec.run_exec（K5T＋零股＋最低手續費 20 元＋FIX），只在下單迴圈加 LIM／DISP 判斷。"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import tcore as T, tlib as L

SPEC = json.loads((L.AG / 'o_v7_listing_exec_engine' / 'data' / 'model_spec7.json').read_text(encoding='utf-8'))
CPI = {int(k): v for k, v in json.loads((L.AG / 'o_v7_listing_exec_engine' / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100'].items()}
def cpi_scale(ts): return CPI[min(max(ts.year, 2006), 2026)] / CPI[2026]
RATE = 0.001425 * 0.28; MINFEE = 20.0; TAX = 0.003
ODD_INTRADAY = pd.Timestamp('2020-10-26')
SHARE = {int(k): float(v) for k, v in SPEC['oa_share_by_dec'].items()}
DR_E, DR_V = np.array(SPEC['drift']['edges']), np.array(SPEC['drift']['vals'])
OPE, ORE = np.array(SPEC['odd']['price_edges']), np.array(SPEC['odd']['rank_edges'])
OB, OS = np.array(SPEC['odd']['buy']), np.array(SPEC['odd']['sell'])
BAND = 0.0025
V60 = T.V60; SIG = T.SIG
with np.errstate(invalid='ignore'):
    VR = pd.DataFrame(V60).rank(axis=1, ascending=False).to_numpy()
def tickv(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)

# 漲跌停（開盤）：2015-06-01 起 10%，之前 7%；以跳動單位向內取整；除權息／事件日（flag 1、2）不判
def lim_pct(ts): return 0.10 if ts >= pd.Timestamp('2015-06-01') else 0.07
def limit_prices(pc, ts):
    lp = lim_pct(ts)
    up = pc * (1 + lp); dn = pc * (1 - lp)
    tu = float(tickv(up)); td = float(tickv(dn))
    return math.floor(up / tu + 1e-9) * tu, math.ceil(dn / td - 1e-9) * td

# 處置矩陣
DP = pd.read_csv(L.RAW / 'agents' / 'disposal_hist_rejudge.csv', dtype={'code': str})
DISP = np.zeros((len(L.IDX), L.N), bool); DREG = {}
for r in DP.itertuples():
    if r.code not in L.CI: continue
    try: a, b = pd.Timestamp(r.start), pd.Timestamp(r.end)
    except Exception: continue
    i0, i1 = L.IDX.searchsorted(a), L.IDX.searchsorted(b, side='right')
    DISP[i0:i1, L.CI[r.code]] = True
    for i in range(i0, i1): DREG[(i, L.CI[r.code])] = r.regime

def run_exec(days, plans, tier, mode, lim=False, disp=False, ev=None):
    ti = [L.IDX.get_loc(d) for d in days]
    v = np.zeros(L.N); cash = tier * cpi_scale(days[0]); out = []
    for k in range(len(ti) - 1):
        t, t2 = ti[k], ti[k + 1]; e = t + 1; d = days[k]
        V0 = v.sum() + cash
        fo = T.factor_close_to_open(t, e); tr = np.isfinite(fo)
        dead = (v > 0) & (L.LAST_TRADE < e)
        cash += (v[dead] * (1 - RATE - TAX - 0.001)).sum(); v[dead] = 0.0
        v = np.where(tr, v * np.nan_to_num(fo, nan=1.0), v)
        frz = (v > 0) & ~tr
        Eq = v.sum() + cash; flow_cost = 0.0; r_on = Eq / V0
        if mode == 'FIX':
            tgtE = tier * cpi_scale(L.IDX[e]); fl = tgtE - Eq
            flow_cost = abs(fl) * ((RATE + TAX + 0.001) if fl < 0 else (RATE + 0.001))
            sc = tgtE / Eq; v = v * sc; cash = cash * sc; Eq = tgtE
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
        ex_day = np.nan_to_num(L.Fn[e], nan=0)
        for j in np.where(dsh != 0)[0]:
            q = dsh[j]; side = 1 if q > 0 else -1; a = abs(q); lots, odd = int(a // 1000), a % 1000
            o = O[j]; pc = L.PREVC[e, j] if np.isfinite(L.PREVC[e, j]) else o
            notional = a * o
            hit_lim = False
            if ex_day[j] not in (1, 2) and np.isfinite(pc) and pc > 0:
                up, dn = limit_prices(pc, L.IDX[e])
                hit_lim = (side > 0 and o >= up - 1e-9) or (side < 0 and o <= dn + 1e-9)
            hit_disp = bool(DISP[e, j])
            if ev is not None:
                ev['n'] += 1; ev['val'] += notional
                if hit_lim: ev['lim_n'] += 1; ev['lim_val'] += notional; ev['lim_list'].append((str(L.IDX[e].date()), L.COLS[j], side))
                if hit_disp: ev['disp_n'] += 1; ev['disp_val'] += notional; ev['disp_list'].append((str(L.IDX[e].date()), L.COLS[j], side, DREG.get((e, j))))
            if (lim and hit_lim) or (disp and hit_disp):
                continue   # 不成交：保留現有持股（剩餘現金閒置）
            vday = L.Vn[e, j] if np.isfinite(L.Vn[e, j]) and L.Vn[e, j] > 0 else V60[t, j]
            dec = int(min(max(math.ceil(VR[t, j] / 30), 1), 10)) if np.isfinite(VR[t, j]) else 10
            oa = SHARE.get(dec, SHARE[10]) * vday if np.isfinite(vday) else np.nan
            done = 0.0
            if lots:
                if np.isfinite(oa):
                    mx = max(1, int(math.floor(0.2 * oa / (1000 * pc))))
                    if lots > mx: lots = mx
                Q = lots * 1000 * pc
                sg = SIG[t, j]
                k5 = 0.001 + (0.5 * sg * math.sqrt(Q / oa) if (np.isfinite(sg) and np.isfinite(oa) and oa > 0) else 0.0)
                im = max(k5, float(tickv(o)) / o)
                px = o * (1 + side * im); val = lots * 1000 * px
                fee = max(MINFEE, RATE * val); tax = TAX * val if side < 0 else 0.0
                cash_delta += -side * val - fee - tax
                done += side * lots * 1000
            if odd:
                if L.IDX[e] < ODD_INTRADAY:
                    base = L.Cn[e, j]
                else:
                    g = o / pc - 1 if pc > 0 else 0.0
                    base = o * (1 + float(DR_V[int(np.searchsorted(DR_E, g, side='right'))]))
                pi = int(np.searchsorted(OPE, pc, side='right')); ri = int(np.searchsorted(ORE, VR[t, j] if np.isfinite(VR[t, j]) else 1e9, side='right'))
                c = float((OB if side > 0 else OS)[pi, ri])
                px = base * (1 + side * c); val = odd * px
                fee = max(MINFEE, RATE * val); tax = TAX * val if side < 0 else 0.0
                cash_delta += -side * val - fee - tax
                done += side * odd
            cur_sh[j] += done
        cash = cash + cash_delta
        f2 = np.nan_to_num(T.factor_open_to_close(e, t2), nan=1.0); fcc = T.factor_close_to_close(t, t2)
        v = np.where(tr, cur_sh * np.where(tr, O, 0.0) * f2, np.where(frz, v * fcc, 0.0))
        V1 = v.sum() + cash
        out.append(V1 / V0 - 1 if mode == 'GROW' else r_on * (V1 - flow_cost) / Eq - 1)
    return pd.Series(out, index=pd.DatetimeIndex(days[:-1]))

TIERS = [('30萬', 3e5, 'FIX'), ('100萬', 1e6, 'FIX'), ('1000萬', 1e7, 'FIX'), ('1000萬', 1e7, 'GROW')]
OUT = {}; EV = {}
POST = pd.Timestamp('2020-11-01')
for vname in ('T1', 'T2'):
    a, z = '2019-01-01', '2026-10-31'
    ev = dict(n=0, val=0.0, lim_n=0, lim_val=0.0, disp_n=0, disp_val=0.0, lim_list=[], disp_list=[])
    pre_cache = {}
    for j in range(13):
        ds = T.sig_days(j, a, z)
        pre = T.precompute(ds, vname); PT = {}
        for d, U, wb, tz in pre:
            wt = wb.copy()
            for nm, (idx, zz) in tz.items(): wt[idx] = T.tilt(wb[idx], zz, 1.5, False)
            PT[d] = (U, wt)
        pre_cache[j] = (ds, PT)
    for tn, tier, mode in TIERS:
        res = {}
        for var, (lim, disp) in {'base': (False, False), 'LIM': (True, False), 'LIM+DISP': (True, True)}.items():
            per = []
            for j in range(13):
                ds, PT = pre_cache[j]
                r = run_exec(ds, PT, tier, mode, lim, disp, ev=(ev if (var == 'base' and tn == '100萬') else None))
                b = T.bench_win(ds); x = (r - b).dropna()
                xp = x[x.index >= POST]
                per.append((T.stats(x)['mean_pp'], T.stats(x)['t'], T.stats(xp)['mean_pp'], T.stats(xp)['t'], T.cagr(r)))
            P = np.array(per)
            res[var] = {'vs0050_pp_jmed': float(np.median(P[:, 0])), 't_jmed': float(np.median(P[:, 1])),
                        'post2020_11_pp_jmed': float(np.median(P[:, 2])), 'post2020_11_t_jmed': float(np.median(P[:, 3])),
                        'post2020_11_t_max': float(np.max(P[:, 3])), 'CAGR_jmed': float(np.median(P[:, 4]))}
        OUT[f'{vname}|{tn}|{mode}'] = res
        print(vname, tn, mode, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}, flush=True)
    EV[vname] = {'orders_13j_100萬': ev['n'], 'lim_orders': ev['lim_n'], 'lim_val_share': ev['lim_val'] / max(ev['val'], 1),
                 'disp_orders': ev['disp_n'], 'disp_val_share': ev['disp_val'] / max(ev['val'], 1),
                 'lim_examples': sorted(set(ev['lim_list']))[:30], 'disp_examples': sorted(set(ev['disp_list']))[:40]}
    print(vname, {k: v for k, v in EV[vname].items()}, flush=True)
# 處置事件中屬宇宙內（T2 2019 後任一 PIT 前 100 名或 0050 成分）的檔數
(HERE / 'data').exists()
print(json.dumps({'cells': OUT}, ensure_ascii=False)[:200])
