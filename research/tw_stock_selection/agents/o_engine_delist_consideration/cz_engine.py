#!/usr/bin/env python3
"""ia_core 相容的下市「收購／換股對價」口徑（consideration）。只 import independent_audit/ia_core.py，不修改它。

simulate(plan, buy, sell, t0, t1, freeze=True, delist_zero=False, ret_override=None, mode=None, settle=None)
  mode=None 或 'freeze'／'nofreeze'／'dzero'：與 ia_core.simulate 完全相同的三種舊口徑（freeze 預設不變）。
  mode='consideration'：settle＝build_settlement(events_df, bound) 的結果（每個下市證券的結算規則）。
結算規則（凍結於 SPEC；此處只是實作）：
  觸發：持有中的證券在其「最後成交日 lt」的下一個交易日轉成「請求權」（claim），原部位歸零（代碼重用時新證券另計）。
  cash：股數 × 每股現金對價，付款日入帳（付款日未知＝終止日＋20 個交易日）；入帳前 0% 現金（cash_0050=True 時改以 0050 計值＝A16 敏感度）；現金對價不課證交稅、不付手續費（假設）。
  swap／conversion：股數 × 換股比例 → 存續股，自存續股首個可交易日 s 起隨其日報酬；s 之前（空窗期）市值固定於最後收盤；
      s 當天若為換股日，以 s 開盤價計入持股；之後由一般換股流程以一般賣出成本出清（若存續股正好入選則照目標權重保留）。
  mixed：現金部分（同 cash）＋股票部分（同 swap）＋特別股部分（以發行價固定市值，s 起的下一個換股日以一般賣出成本出清）。
  swap_ur（併入上市存續公司已核實、比例未核實）：比例＝市場隱含比例（lt 當日 標的收盤 ÷ 存續收盤）×(1+δ)，δ∈{0, −d, +d}。
  acq_ur（被併購已核實、形式或金額未核實）：最後收盤 ×(1+δ) 的固定請求權，終止日＋20 交易日轉現金。
  unknown：上界＝最後收盤固定、終止日＋20 交易日轉現金；下界＝0（lt 後第一個換股日沖銷；沖銷前市值固定於最後收盤，與 dzero 同時點）。
  distress：主口徑 0；對照＝最後收盤（同 unknown 上界）。
  price_gap（本引擎面板在真實最後交易日之前就沒有價格；ia 面板 2019–2023 存活者缺口）：以面板最後收盤固定，下一個換股日以一般賣出成本出清（＝nofreeze 處理）。
  不在事件表的永久停牌：維持 freeze（並記錄）。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'independent_audit'))
import ia_core as K
sys.path.pop(0)

TYPES = ('cash', 'conversion', 'swap', 'mixed', 'swap_ur', 'acq_ur', 'unknown', 'distress')


def _lt_engine(j, term, TR, IDX):
    w = np.flatnonzero(TR[:, j])
    if term is not None: w = w[IDX[w] <= term]
    return int(w[-1]) if len(w) else -1


def build_settlement(E, bound='main', d=0.057, cash_0050=False, TR=None, IDX=None, Cn=None, CI=None, On=None):
    """E：events_v2 DataFrame。bound：'main'｜'lo'｜'hi'（swap_ur／acq_ur 的 δ 與 unknown／distress 的界限）。
    回傳 {欄位 j: [規則 dict]}（同一代碼可有多個證券，依 lt 區分）。"""
    TR = K.TRADED if TR is None else TR; IDX = K.IDX if IDX is None else IDX
    Cn = K.Cn if Cn is None else Cn; CI = K.CI if CI is None else CI
    On = K.On if On is None else On
    out = {}
    dl = {'main': 0.0, 'lo': -d, 'hi': d}[bound]
    for r in E.itertuples():
        j = CI.get(r.code)
        if j is None or not isinstance(r.last_trade, str) or not r.last_trade: continue
        ltU = pd.Timestamp(r.last_trade)
        term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) and r.terminate_date else ltU + pd.Timedelta(days=14)
        lt = _lt_engine(j, max(term, ltU), TR, IDX)
        if lt < 0: continue
        n_tr = len(np.flatnonzero(TR[:, j]))
        if lt + 1 >= len(IDX): continue
        # 這個 lt 之後同代碼若在終止日前又有成交，表示不是這個事件（安全檢查）
        lc = float(Cn[lt, j])
        ti = int(np.searchsorted(IDX.values, np.datetime64(term)))
        pay = min(ti + 20, len(IDX) - 1)
        rule = dict(code=r.code, lt=lt, trig=lt + 1, lc=lc, type=r.type, pay=pay, term=ti, cash_0050=cash_0050, parts=[])
        gap = (IDX[lt] < ltU - pd.Timedelta(days=7))
        if gap:
            rule['type'] = 'price_gap'
            rule['parts'].append(dict(kind='fixed_sell', per_share=lc, rel=lt + 1))
        else:
            t = r.type
            sj = CI.get(r.successor) if isinstance(r.successor, str) and r.successor else None
            sfirst = int(np.searchsorted(IDX.values, np.datetime64(pd.Timestamp(r.succ_first_trade)))) if isinstance(r.succ_first_trade, str) and r.succ_first_trade else None
            if sj is not None and sfirst is not None:
                w = np.flatnonzero(TR[:, sj] & np.isfinite(On[:, sj])); w = w[w >= sfirst]
                sfirst = int(w[0]) if len(w) else None
            if t == 'cash':
                rule['parts'].append(dict(kind='cash', per_share=float(r.cash), rel=pay))
            elif t in ('swap', 'conversion', 'mixed') and sj is not None and sfirst is not None:
                if np.isfinite(r.ratio): rule['parts'].append(dict(kind='stock', succ=sj, ratio=float(r.ratio), rel=sfirst, per_share_window=lc))
                if t == 'mixed' and np.isfinite(r.cash): rule['parts'].append(dict(kind='cash', per_share=float(r.cash), rel=pay))
                if t == 'mixed' and np.isfinite(r.pref_ratio): rule['parts'].append(dict(kind='fixed_sell', per_share=float(r.pref_ratio * r.pref_value), rel=sfirst))
            elif t == 'swap_ur' and sj is not None and sfirst is not None and np.isfinite(Cn[lt, sj]):
                rule['parts'].append(dict(kind='stock', succ=sj, ratio=float(lc / Cn[lt, sj] * (1 + dl)), rel=sfirst, per_share_window=lc))
            elif t in ('acq_ur', 'swap_ur', 'swap', 'conversion', 'mixed'):   # 存續股不在面板 → 價值型處理
                rule['type'] = t if t == 'acq_ur' else t + '->acq_ur'
                rule['parts'].append(dict(kind='cash', per_share=lc * (1 + dl), rel=pay))
            elif t in ('unknown', 'distress'):
                hi = (bound == 'hi') or (bound == 'main' and t == 'unknown')
                if hi: rule['parts'].append(dict(kind='cash', per_share=lc, rel=pay))
                else: rule['parts'].append(dict(kind='fixed_sell', per_share=lc, rel=lt + 1, zero=True))
            else:
                continue
        out.setdefault(j, []).append(rule)
    return out


def simulate(plan, buy, sell, t0, t1, freeze=True, delist_zero=False, ret_override=None, mode=None, settle=None):
    if mode is None:
        mode = 'dzero' if delist_zero else ('freeze' if freeze else 'nofreeze')
    if mode in ('freeze', 'nofreeze', 'dzero'):
        return _sim_old(plan, buy, sell, t0, t1, mode, ret_override)
    assert mode == 'consideration' and settle is not None
    return _sim_cons(plan, buy, sell, t0, t1, settle, ret_override)


def _solve(htr, cash, pk, buy, sell):
    N = len(htr); Vt = htr.sum() + cash; k = len(pk)
    def cost_of(T):
        tgt = np.zeros(N); tgt[pk] = T
        return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
    lo, hi = 0.0, Vt / k
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mid * k + cost_of(mid) > Vt: hi = mid
        else: lo = mid
    return lo, Vt


def _sim_old(plan, buy, sell, t0, t1, mode, ret_override):
    """逐行對照 ia_core.simulate 的三種舊口徑（smoke 測試斷言逐日差＝0）。"""
    G = K.GROW if ret_override is None else ret_override
    N = K.N; reb = {p[0]: p[1] for p in plan}
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1); tov = []; frozen_ev = []
    lt = np.where(K.TRADED.any(axis=0), len(K.IDX) - 1 - np.argmax(K.TRADED[::-1], axis=0), -1)
    for t in range(t0, t1 + 1):
        if t in reb and t > t0 - 1:
            pk = reb[t]
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            gap = np.where(tr, G[t] * K.On[t] / np.where(tr, K.Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            if mode == 'nofreeze': frz[:] = False
            if mode == 'dzero':
                dead = frz & (lt <= t); h[dead] = 0.0; frz &= ~dead
            Vfz = h[frz].sum(); htr = np.where(frz, 0.0, h)
            T, Vt = _solve(htr, cash, pk, buy, sell)
            tgt = np.zeros(N); tgt[pk] = T
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz))
            newval = np.where(frz, h, 0.0); newval[pk] += T
            oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc); cash = 0.0
            if frz.any(): frozen_ev.append((str(K.IDX[t].date()), [K.COLS[j] for j in np.where(frz)[0]], float(Vfz / (Vt + Vfz))))
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=K.IDX[t0:t1 + 1]), {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan, 'frozen': frozen_ev}


def _sim_cons(plan, buy, sell, t0, t1, settle, ret_override):
    G = K.GROW if ret_override is None else ret_override
    N = K.N; reb = {p[0]: p[1] for p in plan}
    E50 = K.E50.to_numpy()
    trig = {}
    for j, rules in settle.items():
        for ru in rules: trig.setdefault(ru['trig'], []).append((j, ru))
    val = np.zeros(N); cash = 0.0 + 1.0
    claims = []   # dict(kind, value, rel, succ?, q?)
    nav = np.empty(t1 - t0 + 1); tov = []; frozen_ev = []; settled = []
    for t in range(t0, t1 + 1):
        # 1) 觸發：lt 後第一個交易日把持有部位轉成請求權
        for j, ru in trig.get(t, []):
            if val[j] <= 0 or t <= t0: continue
            V = val[j]; sh = V / ru['lc']; val[j] = 0.0
            ev = dict(date=str(K.IDX[t].date()), code=ru['code'], type=ru['type'], value_lastclose=float(V), parts=[])
            for p in ru['parts']:
                if p['kind'] == 'cash':
                    claims.append(dict(kind='cash', value=sh * p['per_share'], rel=max(p['rel'], t), c50=ru['cash_0050'], ev=ev))
                elif p['kind'] == 'fixed_sell':
                    claims.append(dict(kind='fixed_sell', value=sh * p['per_share'], rel=max(p['rel'], t), zero=p.get('zero', False), ev=ev))
                elif p['kind'] == 'stock':
                    claims.append(dict(kind='stock', value=sh * p['per_share_window'], rel=max(p['rel'], t), succ=p['succ'], q=sh * p['ratio'], ev=ev))
            settled.append(ev)
        # 2) 0050 計值的現金請求權（A16 敏感度）
        if t > t0:
            r50 = E50[t] / E50[t - 1] if np.isfinite(E50[t]) and np.isfinite(E50[t - 1]) else 1.0
            for c in claims:
                if c['kind'] == 'cash' and c.get('c50') and t <= c['rel']: c['value'] *= r50
        isreb = t in reb
        # 3) 到期：現金入帳；股票請求權在 s 轉成存續股（s 為換股日時以開盤價計入）
        keep = []
        add_open = np.zeros(N)
        for c in claims:
            if t >= c['rel']:
                if c['kind'] == 'cash':
                    cash += c['value']; c['ev']['parts'].append(('cash', str(K.IDX[t].date()), float(c['value'])))
                    continue
                if c['kind'] == 'stock':
                    sj = c['succ']
                    if isreb: add_open[sj] += c['q'] * K.On[t, sj]
                    else: val[sj] += c['q'] * K.Cn[t, sj] / G[t, sj]   # 之後乘上當日 G → 收盤值 q×C
                    c['ev']['parts'].append(('stock', str(K.IDX[t].date()), K.COLS[sj], float(c['q'] * (K.On[t, sj] if isreb else K.Cn[t, sj]))))
                    continue
                if c['kind'] == 'fixed_sell' and isreb:
                    pr = 0.0 if c.get('zero') else c['value'] * (1 - sell)
                    cash += pr; c['ev']['parts'].append(('writeoff' if c.get('zero') else 'fixed_sell', str(K.IDX[t].date()), float(pr)))
                    continue
            keep.append(c)
        claims = keep
        if isreb:
            pk = reb[t]
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            gap = np.where(tr, G[t] * K.On[t] / np.where(tr, K.Cn[t], 1.0), 1.0)
            h = val * gap + add_open
            frz = (h > 0) & ~tr
            Vfz = h[frz].sum(); htr = np.where(frz, 0.0, h)
            Vcl = sum(c['value'] for c in claims)
            T, Vt = _solve(htr, cash, pk, buy, sell)
            tgt = np.zeros(N); tgt[pk] = T
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz + Vcl))
            newval = np.where(frz, h, 0.0); newval[pk] += T
            oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc); cash = 0.0
            if frz.any() or claims:
                frozen_ev.append((str(K.IDX[t].date()), [K.COLS[j] for j in np.where(frz)[0]], float((Vfz + Vcl) / (Vt + Vfz + Vcl))))
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash + sum(c['value'] for c in claims)
    return pd.Series(nav, index=K.IDX[t0:t1 + 1]), {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan, 'frozen': frozen_ev, 'settled': settled}
