#!/usr/bin/env python3
"""可執行版（K5T）：逐行移植 o_0050_tilt_pit/s7_exec.py 的 run_exec（V7 model_spec7：每邊 max(0.1%＋0.5·σ60·√(Q/OA), 一跳/開盤價)，
參與上限 20%×OA；零股價差表＋開盤跳空漂移；2020-10-26 前零股以收盤價（盤後零股）成交並標示；手續費 0.1425%×0.28、每筆最低 20 元；
證交稅 0.3%；FIX 每月縮放回固定金額（CPI），縮放付費稅與 0.1% 衝擊；0.25% 不交易帶），另加：
  - 壓力成本：整股與零股每邊再加 0.5%；
  - 處置期賣出（V7X 口徑）：e 日在官方處置期內的賣出改以收盤價減一跳成交（disposal_hist_rejudge in_panel＋disposal_5y 補缺）。
級距 30 萬／100 萬／300 萬／1,000 萬，FIX 與 GROW。"""
import json, math
import numpy as np, pandas as pd
import lc_lib as X
L, TC = X.L, X.TC
AG = X.AG
SPEC = json.loads((AG / 'o_v7_listing_exec_engine' / 'data' / 'model_spec7.json').read_text(encoding='utf-8'))
CPI = {int(k): v for k, v in json.loads((AG / 'o_v7_listing_exec_engine' / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100'].items()}
def cpi_scale(ts): return CPI[min(max(ts.year, 2006), 2026)] / CPI[2026]
RATE = 0.001425 * 0.28; MINFEE = 20.0; TAX = 0.003
ODD_INTRADAY = pd.Timestamp('2020-10-26')
SHARE = {int(k): float(v) for k, v in SPEC['oa_share_by_dec'].items()}
DR_E, DR_V = np.array(SPEC['drift']['edges']), np.array(SPEC['drift']['vals'])
OPE, ORE = np.array(SPEC['odd']['price_edges']), np.array(SPEC['odd']['rank_edges'])
OB, OS = np.array(SPEC['odd']['buy']), np.array(SPEC['odd']['sell'])
BAND = 0.0025
V60 = TC.V60; SIG = TC.SIG
with np.errstate(invalid='ignore'):
    VR = pd.DataFrame(V60).rank(axis=1, ascending=False).to_numpy()
def tickv(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}

def _disp():
    A = pd.read_csv(AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
    A = A[A['in_panel'] == True]
    B = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    ka = set(zip(A.code, A.start)); B = B[[(c, s) not in ka for c, s in zip(B.code, B.start)]]
    D = pd.concat([A[['code', 'start', 'end']], B[['code', 'start', 'end']]]).dropna()
    D = D[D.code.isin(X.CI)]
    M = np.zeros((len(X.IDX), X.N), bool); dt = X.IDX.values
    for c, s, e in zip(D.code, D.start, D.end):
        a, b = np.searchsorted(dt, np.datetime64(s)), np.searchsorted(dt, np.datetime64(e), side='right'); M[a:b, X.CI[c]] = True
    return M
DISP = _disp()

def run_exec(days, plans, tier, mode, extra=0.0):
    """plans：list（每個訊號日一筆 (idx, w)）。回傳 (每窗報酬 ndarray, 會計)。"""
    ti = [X.IDX.get_loc(d) for d in days]
    v = np.zeros(X.N); cash = tier * cpi_scale(days[0]); out = []
    acc = dict(fee=0., tax=0., impact=0., odd=0., odd_val=0., lot_val=0., cap_hit=0, orders=0, flow_cost=0., pre_odd_val=0., disp_sells=0, extra=0.)
    for k in range(len(ti) - 1):
        t, t2 = ti[k], ti[k + 1]; e = t + 1
        V0 = v.sum() + cash
        fo = TC.factor_close_to_open(t, e); tr = np.isfinite(fo)
        dead = (v > 0) & (L.LAST_TRADE < e)
        cash += (v[dead] * (1 - RATE - TAX - 0.001 - extra)).sum(); v[dead] = 0.0
        v = np.where(tr, v * np.nan_to_num(fo, nan=1.0), v)
        frz = (v > 0) & ~tr
        Eq = v.sum() + cash; flow_cost = 0.0; r_on = Eq / V0
        if mode == 'FIX':
            tgtE = tier * cpi_scale(X.IDX[e]); fl = tgtE - Eq
            flow_cost = abs(fl) * ((RATE + TAX + 0.001) if fl < 0 else (RATE + 0.001))
            sc = tgtE / Eq; v = v * sc; cash = cash * sc; Eq = tgtE; acc['flow_cost'] += flow_cost
        idx, w = plans[k]
        wv = np.zeros(X.N); np.add.at(wv, idx, w); wv[frz] = 0.0; wv[~tr] = 0.0; sw_ = wv.sum(); wv = wv / sw_ if sw_ > 0 else wv
        Etr = Eq - v[frz].sum()
        O = L.On[e]; cur_sh = np.where(tr, v / np.where(tr, O, 1.0), 0.0)
        tgt_sh = np.where(tr, np.round(wv * Etr * 0.995 / np.where(tr, O, 1.0)), 0.0)
        dsh = tgt_sh - cur_sh
        small = (np.abs(dsh * np.nan_to_num(O)) < BAND * Eq) & (tgt_sh > 0)
        dsh[small] = 0.0
        dsh = np.where(tr, np.round(dsh), 0.0)
        cash_delta = 0.0; cash_close = 0.0
        for j in np.where(dsh != 0)[0]:
            q = dsh[j]; side = 1 if q > 0 else -1; a = abs(q); lots, odd = int(a // 1000), a % 1000
            o = O[j]; pc = L.PREVC[e, j] if np.isfinite(L.PREVC[e, j]) else o
            if side < 0 and DISP[e, j] and np.isfinite(L.Cn[e, j]):
                px = L.Cn[e, j] - float(tickv(L.Cn[e, j])); n = min(a, max(cur_sh[j], 0.0))
                for m in (lots * 1000, odd):
                    if m > 0:
                        val = m * px * (1 - extra); fee = max(MINFEE, RATE * val); tax = TAX * val
                        cash_close += val - fee - tax; acc['fee'] += fee; acc['tax'] += tax; acc['orders'] += 1
                acc['disp_sells'] += 1; cur_sh[j] -= n
                continue
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
                if X.IDX[e] < ODD_INTRADAY:
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
        f2 = np.nan_to_num(TC.factor_open_to_close(e, t2), nan=1.0); fcc = TC.factor_close_to_close(t, t2)
        fcl = np.nan_to_num(TC.factor_close_to_close(e, t2), nan=1.0)
        v = np.where(tr, cur_sh * np.where(tr, O, 0.0) * f2, np.where(frz, v * fcc, 0.0))
        cash = cash + cash_close     # 處置期收盤賣出款（e 收盤入帳，之後以現金計）
        V1 = v.sum() + cash
        out.append(V1 / V0 - 1 if mode == 'GROW' else r_on * (V1 - flow_cost) / Eq - 1)
    return np.array(out), acc

def cost_pp(acc, tier, n): return 100 * (acc['fee'] + acc['tax'] + acc['impact'] + acc['odd'] + acc['flow_cost']) / (tier * n)
