#!/usr/bin/env python3
"""事後修正（PREREG 之後發現的實作錯誤，見 results.json POSTHOC_FIX）：
core.sleeve 的 FIX 只把每槽金額固定在 tier/10，帳戶權益卻隨 0050 長大，使處置部位占比隨時間萎縮到約 1～4%，
和 V7X 的 FIX（帳戶固定規模、每期縮放）語意不符。本檔提供 sleeve_fix：每月初把帳戶縮放回 tier（不計成本），其他邏輯逐字沿用 core.sleeve。
不修改 core.py（PREREG 雜湊不變）。"""
import sys, math
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from core import *  # noqa
from core import IDX, T, LA, E50, C, V, R, TR, LOGCO, LASTV, ETF_BUY, ETF_SELL, TAX, qref, side_cost, cpi_factor, limit_rate


def sleeve_fix(trades, tier_2026=1e6, mode='FIX', K=10, cap=0.2, stress=False, t_start='2006-04-01', t_end=None, idle='0050', min_fill=0.0, defer=True, buffer_slots=0):
    """trades：build() 的表（不需 exec 欄）。回傳 (日 NAV Series, 統計 dict, 逐筆成交 DataFrame)。
    【事後修正版】FIX：帳戶權益在每月第一個交易日開盤前（以前一日收盤值）縮放回 tier×CPI(年)/CPI(2026)（不計成本，同 V7X 原設定的 FIX 縮放），
    每槽目標＝tier×CPI/K；NAV 以縮放前後的日報酬連乘。GROW 與 core.sleeve 完全相同。
    預收款券：買進資金在委託日前一交易日收盤從 0050 贖回（委託日當天現金不計息）。出場超過量上限的部分順延。"""
    t0 = int(IDX.searchsorted(pd.Timestamp(t_start))); t1 = T - 1 if t_end is None else int(IDX.searchsorted(pd.Timestamp(t_end), 'right') - 1)
    ev = {}
    for r in trades.itertuples():
        if t0 + 1 <= r.i0 <= t1: ev.setdefault(r.i0 - (1 if r.ek == 'C' else 1), []).append(r)
    # 委託日前一日（i0−1）先保留現金；C1 的委託日＝i0；O2／C2 委託日＝s_i+1＝i0
    cap0 = tier_2026 * cpi_factor(t0)
    u50 = cap0 * (1 - ETF_BUY) / E50[t0 - 1]; cash = 0.0
    pos = {}   # key -> dict
    resv = []  # 已保留現金、待進場：(r, amt)
    nav = np.empty(t1 - t0 + 1); inv = np.empty(t1 - t0 + 1)
    navi = 1.0; prevE = cap0
    fills = []; stat = dict(entered=0, no_slot=0, held=0, capped=0, zero_cap=0, deferred_sell_days=0)

    def equity(t):
        pv = 0.0
        for p in pos.values():
            pv += p['units'] * math.exp(LA[t, p['j']] - LA[p['i0'], p['j']]) * p['px0']
        return cash + u50 * E50[t] + pv, pv

    for t in range(t0, t1 + 1):
        # 1) 進場（今天收盤或開盤）——用昨日保留的現金
        newresv = []
        for r, amt in resv:
            if r.i0 != t: newresv.append((r, amt)); continue
            j = r.j
            Q0 = qref(t, j, r.n_auct, r.s_i <= t <= r.e_i)
            q = min(amt, cap * Q0) if Q0 > 0 else 0.0
            if q < amt: stat['capped'] += 1
            if q < min_fill * amt: q = 0.0
            if q <= 0: stat['zero_cap'] += 1; u50 += amt * (1 - ETF_BUY) / E50[t]; continue
            cb = side_cost(t, j, q, Q0, +1, stress)
            px0 = C[t, j] if r.ek == 'C' else C[t, j] / math.exp(LOGCO[t, j])
            # 開盤進場：持有單位的價格基準以收盤水準換算
            units_val = q * (1 - cb)
            if r.ek == 'O': units_val *= math.exp(LOGCO[t, j])  # 開盤到收盤的漲跌
            pos[(r.code, r.ordr)] = dict(j=j, i0=t, px0=1.0, units=units_val, r=r, rem_frac=1.0, selling=False, inv=q)
            if amt - q > 0: u50 += (amt - q) * (1 - ETF_BUY) / E50[t]
            stat['entered'] += 1
        resv = newresv
        # 2) 出場
        for key in list(pos.keys()):
            p = pos[key]; r = p['r']; j = p['j']
            if t < r.i1: continue
            lim = min(T - 1, LASTV[j])
            sellable = t >= lim or t == r.i1 or (TR[t, j] and np.isfinite(R[t, j]) and R[t, j] > -(limit_rate(t) - 0.01))
            if not sellable: stat['deferred_sell_days'] += 1; continue
            first = (t == r.i1)
            val = p['units'] * math.exp(LA[t, j] - LA[p['i0'], j])
            if first and r.xk == 'O':
                val = val / math.exp(LOGCO[t, j]); Q1 = 0.05 * V[t, j] / 0.2 if np.isfinite(V[t, j]) else 0.0
            else:
                Q1 = qref(t, j, r.n_auct, r.s_i <= t <= r.e_i)
            sell = val if (t >= lim or not defer) else (min(val, cap * Q1) if Q1 > 0 else 0.0)
            if sell <= 0: stat['deferred_sell_days'] += 1; continue
            c1 = side_cost(t, j, sell, Q1 if Q1 > 0 else sell, -1, stress) + TAX
            proceeds = sell * (1 - c1)
            u50 += proceeds * (1 - ETF_BUY) / E50[t]
            frac = sell / val
            if frac >= 1 - 1e-9:
                fills.append(dict(code=r.code, date=IDX[p['i0']], exit=IDX[t], inv=p['inv'], ret=None))
                pos.pop(key)
            else:
                p['units'] *= (1 - frac); stat['deferred_sell_days'] += 1
        # 3) 今日收盤後：為明天（或今天之後）的委託保留現金
        E, pv = equity(t)
        for r in ev.get(t, []):
            if any(pp['r'].code == r.code for pp in pos.values()) or any(rr.code == r.code for rr, _ in resv):
                stat['held'] += 1; continue
            if len(pos) + len(resv) >= K: stat['no_slot'] += 1; continue
            tgt = (E / K) if mode == 'GROW' else tier_2026 * cpi_factor(t) / K
            avail = u50 * E50[t] * (1 - ETF_SELL)
            amt = min(tgt, avail)
            if amt <= 0: continue
            u50 -= amt / (1 - ETF_SELL) / E50[t]; cash += 0.0
            resv.append((r, amt))
        # 現金緩衝（嚴格預收款券敏感度）：閒置中 buffer_slots 槽留現金（0 報酬），其餘 0050；不計調整成本
        if buffer_slots:
            idle_tot = cash + u50 * E50[t]
            tgt_s = (equity(t)[0] / K) if mode == 'GROW' else tier_2026 * cpi_factor(t) / K
            want = min(buffer_slots * tgt_s, idle_tot)
            cash = want; u50 = (idle_tot - want) / E50[t]
        # 保留中的現金
        E, pv = equity(t)
        E += sum(a for _, a in resv)
        navi *= E / prevE; prevE = E
        nav[t - t0] = navi; inv[t - t0] = pv / E if E > 0 else 0
        if mode == 'FIX' and t < t1 and IDX[t + 1].month != IDX[t].month and E > 0:
            sc = tier_2026 * cpi_factor(t + 1) / E
            u50 *= sc; cash *= sc
            for p in pos.values(): p['units'] *= sc
            resv = [(rr, aa * sc) for rr, aa in resv]
            prevE = E * sc
    s = pd.Series(nav, index=IDX[t0:t1 + 1])
    stat['mean_invested'] = round(float(inv.mean()), 3)
    return s, stat


