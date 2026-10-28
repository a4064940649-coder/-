#!/usr/bin/env python3
"""可執行版引擎（帳戶金額、逐筆成本）。兩個衝擊界：
  'o5'  ：o_exec_capacity_tiers 凍結的 v4 執行量模型（raw/exec/model_spec.json）：
          OA（開盤集合競價成交值）＝V60_{t−1}×κ×exp(b·x)，x＝[1, ln 名次, ln 前收, ln σ60, |跳空|, pick, drop]；
          單邊衝擊＝0.1%＋0.5σ√(Q/OA)（Q＝該筆成交值）。不設 20% 參與上限、不做順延（簡化；o5 有）。
  'tick'：單邊一個跳動單位（tick(前收)/前收），下界模型。
共同：手續費 A（0.1425%×0.28，每筆最低 20 元）、證交稅 0.3%（賣）；零股：成交值不到一張的部分加 o5 零股價差表（買邊，賣邊≤0 取 0）；
股數連續（不取整，簡化）。級距（2026 幣值）× CPI[年]/CPI[2026]。
GROW：起始＝級距×CPI 換算，之後隨淨值成長；FIX：每個換股日開盤前把帳戶縮放回級距×當年 CPI（時間加權報酬）。
0050 核心部位（組合用）：成本＝手續費 A（最低 20）＋賣出稅 0.1%＋單邊一個跳動。"""
import json, math
import numpy as np, pandas as pd
import cmlib as L

IDX, N = L.IDX, L.N
SPEC = json.loads((L.RAW / 'exec' / 'model_spec.json').read_text(encoding='utf-8'))
CPI = {int(k): v for k, v in json.loads((L.RAW / 'exec' / 'cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100'].items()}
B = np.array(SPEC['oa']['coef']); KAPPA = SPEC['oa']['kappa']
ODD_PE = np.array(SPEC['odd']['price_edges']); ODD_RE = np.array(SPEC['odd']['rank_edges']); ODD_BUY = np.array(SPEC['odd']['buy'])
FEE_A = 0.001425 * 0.28; MINFEE = 20.0; TAX = 0.003
S52, V60, SIG = L.rolling_signals()
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}

def cpi_scale(ts): return CPI[min(max(ts.year, 2006), 2026)] / CPI[2026]

_VR = {}
def vrank(t):
    if t not in _VR:
        v = V60[t]; base = np.isfinite(v) & (v >= 1e7) & (L.Cn[t] >= 10)
        r = np.full(N, np.nan); o = np.argsort(-np.where(base, v, -np.inf)); r[o[:base.sum()]] = np.arange(1, base.sum() + 1)
        _VR[t] = r
    return _VR[t]

def unit_cost(t, js, Q, side, model, pick, drop):
    """每筆成本率（不含手續費最低額與稅）：衝擊＋零股價差。js、Q（元，>0）、side（+1 買／−1 賣）陣列。"""
    pc = L.PREVC[t, js]; rk = vrank(t - 1)[js]; sg = SIG[t - 1, js]
    gap = np.where(np.isfinite(L.On[t, js]) & (pc > 0), L.On[t, js] / pc - 1, 0.0)
    if model == 'o5':
        x = np.vstack([np.ones(len(js)), np.log(np.clip(np.nan_to_num(rk, nan=1e4), 1, None)), np.log(pc), np.log(np.clip(np.nan_to_num(sg, nan=0.02), 1e-4, None)),
                       np.abs(gap), pick.astype(float), drop.astype(float)])
        oa = np.nan_to_num(V60[t - 1, js], nan=1e7) * KAPPA * np.exp(B @ x)
        imp = 0.001 + 0.5 * np.nan_to_num(sg, nan=0.02) * np.sqrt(Q / np.clip(oa, 1.0, None))
    elif model == 'tick':
        imp = L.tick(pc) / pc
    else:
        raise ValueError(model)
    lot = 1000 * pc
    oddpart = np.where(Q < lot, 1.0, (Q % lot) / Q)
    i = np.searchsorted(ODD_PE, pc, side='right'); jr = np.searchsorted(ODD_RE, np.nan_to_num(rk, nan=1e9), side='right')
    odd = np.where(side > 0, ODD_BUY[i, jr], 0.0) * oddpart
    return imp + odd

def simulate_exec(plan, t0, t1, tier, version='GROW', model='o5', etf_weight=0.0, G=None):
    G = L.grow_exfix() if G is None else G
    """plan＝{exec_t:(idx, w)}（策略部位內部權重）；etf_weight＝0050 核心比重（組合）；回傳淨值（單位淨值，時間加權）。"""
    pos = np.zeros(N)            # 各股市值（元）
    etf = 0.0; cash = tier * cpi_scale(IDX[t0]); unit = 1.0
    nav = np.empty(t1 - t0 + 1); acct = cash; prev_eq = cash; costs = 0.0
    E = L.E50.to_numpy()
    for t in range(t0, t1 + 1):
        if t in plan and t > t0:
            # 開盤前：持股 × G×O/C（前收→開盤）
            tr = L.TRADED[t] & np.isfinite(L.On[t])
            gap = np.where(tr, G[t] * L.On[t] / np.where(tr, L.Cn[t], 1.0), 1.0)
            pos = pos * gap
            etf = etf * (E[t] / E[t - 1])          # 0050 用收盤到收盤（近似：開盤成交價＝前收→收盤的連續）
            eq_pre = pos.sum() + etf + cash
            eq_open = eq_pre
            if version == 'FIX':
                tgt_eq = tier * cpi_scale(IDX[t]); sc = tgt_eq / eq_pre
                pos *= sc; etf *= sc; cash *= sc; eq_pre = tgt_eq
            js, w = plan[t]
            frz = (pos > 0) & ~tr
            dead = frz & (L.LAST_TRADE < t)          # 下市／併購：最後收盤價值扣賣出稅費轉現金（PREREG 修正 v2）
            if dead.any():
                cash += pos[dead].sum() * (1 - TAX - FEE_A); pos = np.where(dead, 0.0, pos); frz &= ~dead
            wv = np.zeros(N); np.add.at(wv, np.asarray(js, int), np.asarray(w, float)); wv[frz] = 0
            wv = wv / wv.sum() if wv.sum() > 0 else wv
            pick = wv > 0; drop = (pos > 0) & ~pick & tr
            def total_cost(Ttot):
                tgt_s = wv * Ttot * (1 - etf_weight); tgt_e = Ttot * etf_weight
                d = tgt_s - np.where(frz, 0.0, pos)
                k = np.where(np.abs(d) > 1e-6 * max(Ttot, 1))[0]
                c = 0.0
                if len(k):
                    Q = np.abs(d[k]); sd = np.sign(d[k])
                    c += (unit_cost(t, k, Q, sd, model, pick[k], drop[k]) * Q).sum()
                    c += np.maximum(MINFEE, FEE_A * Q).sum() + TAX * Q[sd < 0].sum()
                de = tgt_e - etf
                if abs(de) > 1e-6 * max(Ttot, 1):
                    pe = E[t - 1] if np.isfinite(E[t - 1]) else 100.0
                    c += max(MINFEE, FEE_A * abs(de)) + (0.001 * abs(de) if de < 0 else 0.0) + abs(de) * 0.0005
                return c
            Vt = pos[~frz].sum() + etf + cash
            lo, hi = 0.0, Vt
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if mid + total_cost(mid) > Vt: hi = mid
                else: lo = mid
            c = total_cost(lo); costs += c
            newpos = np.where(frz, pos, wv * lo * (1 - etf_weight)); etf = lo * etf_weight; cash = Vt - lo - c
            cash = max(cash, 0.0)
            # 開盤→收盤
            oc = np.where(tr, L.Cn[t] / np.where(tr, L.On[t], 1.0), 1.0)
            pos = np.where(frz, newpos, newpos * oc)
            eq = pos.sum() + etf + cash
            unit *= (eq_open / prev_eq) * (eq / eq_pre)          # GROW 時 eq_pre＝eq_open，等於 eq/prev_eq
            prev_eq = eq
        elif t > t0:
            pos = pos * G[t]; etf = etf * (E[t] / E[t - 1])
            eq = pos.sum() + etf + cash
            unit *= eq / prev_eq; prev_eq = eq
        nav[t - t0] = unit
    return pd.Series(nav, index=IDX[t0:t1 + 1]), {'cost_total': costs}
