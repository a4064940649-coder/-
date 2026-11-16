#!/usr/bin/env python3
"""S37 獨立複製：成交價掛鉤引擎（最小包裝）。
由 s10lib.simulate_s10（共用引擎，只讀不改）逐行改寫，只保留 exec_mode='open'；未讀、未 import s37lib.py／run.py。
新增：
- fill：'open'（＝simulate_s10）、'avgv'、'avg4'、'open05'；defer=True 為 DEFER（開盤漲停買不到者順延一個交易日再試一次）。
- 嘗試紀錄：買進（名額已滿、禁買之外的每一筆價格檢查）、賣出（已標記出場者每一次開盤嘗試）。
- 容量統計：進入成交值上限檢查的買進筆數、被上限縮量筆數、縮量後整筆放棄筆數。
fill='open' 且 defer=False 時，所有浮點運算順序與 simulate_s10 相同（驗收 A1：淨值逐日最大絕對差應為 0）。"""
import math
import numpy as np, pandas as pd


def make_px(P, tol=0.0):
    """回傳均價代理 (a) VWAP（成交金額÷成交股數，不在 最低×(1−tol)～最高×(1＋tol) 之間改 (b)）與 (b) (開＋高＋低＋收)/4，以及 (a) 改用 (b) 的格數。"""
    O, H, L, C = P.O, P.H, P.L, P.C
    with np.errstate(invalid='ignore', divide='ignore'):
        A4 = (O + H + L + C) / 4.0
        V = P.AMT / P.VOL
        bad = ~(np.isfinite(V) & (V >= L * (1 - tol) - 1e-9) & (V <= H * (1 + tol) + 1e-9))
    AV = np.where(bad, A4, V)
    nfb = int((bad & np.isfinite(O) & (O > 0)).sum())
    nfb_t0 = int((bad[P.t0:] & np.isfinite(O[P.t0:]) & (O[P.t0:] > 0)).sum())
    return AV, A4, nfb, nfb_t0


def simulate_x(P, S, univ, s10lib, fill='open', defer=False, AV=None, A4=None, sat_w=1.0, sizing='W10', lock=False, band=0.10, cost=None, block=None,
               capital=1e6, slots=10, cap_pct=0.05, sweep_min=20000.0, min_frac=0.5):
    cslib, stlib, COSTS = s10lib.cslib, s10lib.stlib, s10lib.COSTS
    FEE, TAX, MINFEE, TAX_ETF = s10lib.FEE, s10lib.TAX, s10lib.MINFEE, s10lib.TAX_ETF
    cost = cost or COSTS['strict']; block = block or {}
    cm = cost['cmult']; xs = cost['xslip'] * cm; ox = cost['oddx'] * cm
    sig, rank, X, max_hold = S['sig'], S['rank'], S['X'], S['max_hold']
    sigrows = S['sigrows']
    exec_days = set(int(x['k']) + 1 for x in S['info'])
    ntgt = {int(x['k']): int(x['n_tgt']) for x in S['info']}
    T, t0 = P.T, P.t0
    O, C, R, REF, AMT = P.O, P.CF, P.R, P.REF, P.AMT
    H, L, LIMUP, LIMDN = P.H, P.L, P.LIMUP, P.LIMDN
    fee, tax, minfee = FEE * cm, TAX * cm, MINFEE * cm
    tick = cslib.tick; _fee = stlib._fee; etf_tick = stlib.etf_tick
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))
    cash = float(capital); v50 = 0.0; pos = {}
    B = (1.0 - sat_w) * capital
    nav = np.full(T, np.nan); nav[t0] = capital
    log = []
    ent_att = []; ex_att = []
    st = dict(n_sig=0, n_buy=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, blk_min=0, noslot=0, blk_dropped=0, sell_defer_limdn=0,
              sell_defer_noopen=0, sell_partial_cap=0, n_sell=0, n_delist=0, buy_val=0.0, sell_val=0.0, fee=0.0, tax=0.0, slip=0.0,
              cap_enter=0, cap_dropped=0, n_defer_try=0, n_defer_buy=0, n_defer_blk=0, part=[])

    # ---------------- 成交價掛鉤
    def bpx(t, j):
        """買進：回傳 (成交參考價或 None, 原因)。"""
        o = O[t, j]
        if not (np.isfinite(o) and o > 0):
            return None, 'noopen'
        if fill in ('open', 'open05'):
            if o >= LIMUP[t, j] - 1e-9:
                return None, 'limup'
            return (o if fill == 'open' else min(o * 1.005, LIMUP[t, j])), ''
        if L[t, j] >= LIMUP[t, j] - 1e-9:                     # 一字漲停
            return None, 'limup'
        return (AV[t, j] if fill == 'avgv' else A4[t, j]), ''

    def spx(t, j):
        o = O[t, j]
        if not (np.isfinite(o) and o > 0):
            return None, 'noopen'
        if fill in ('open', 'open05'):
            if o <= LIMDN[t, j] + 1e-9:
                return None, 'limdn'
            return (o if fill == 'open' else max(o * 0.995, LIMDN[t, j])), ''
        if H[t, j] <= LIMDN[t, j] + 1e-9:                     # 一字跌停
            return None, 'limdn'
        return (AV[t, j] if fill == 'avgv' else A4[t, j]), ''

    def banned(j, t):
        for a, b in block.get(j, ()):
            if a <= t <= b:
                return True
        return False

    def e50_net(po):
        nonlocal cash, v50
        sl50 = 0.5 * cm * etf_tick(po) / po
        if cash < -1e-6 and v50 > 0:
            need = -cash
            x = min(v50, (need + minfee) / ((1 - sl50) * (1 - fee - TAX_ETF * cm)))
            for _ in range(3):
                g = x * (1 - sl50); f = max(minfee, g * fee); net = g - f - g * TAX_ETF * cm
                if net >= need or x >= v50:
                    break
                x = min(v50, x + (need - net) * 1.002)
            g = x * (1 - sl50); f = max(minfee, g * fee); net = g - f - g * TAX_ETF * cm
            v50 -= x; cash += net
        elif cash >= sweep_min:
            f = max(minfee, cash * fee)
            val = (cash - f) / (1 + sl50)
            v50 += val; cash = 0.0

    def eq_now(t):
        return cash + v50 + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())

    def rebalance(t):
        nonlocal cash, B
        eq = eq_now(t)
        if eq <= 0:
            return
        sv = eq - B
        dev = sv / eq - sat_w
        if abs(dev) <= band + 1e-12:
            return
        Bt = (1.0 - sat_w) * eq
        if dev < 0:
            B = Bt; return
        idle = cash + v50
        need = Bt - idle
        if need > 1e-6:
            vals = {}; pxs = {}
            for k, p in pos.items():
                if p['flag']:
                    continue
                o, _ = spx(t, k)
                if o is None:
                    continue
                vals[k] = p['v'] * o / REF[t, k]; pxs[k] = o
            tv = sum(vals.values())
            if tv > 0:
                frac_all = min(1.0, need / tv)
                for k, vo in vals.items():
                    p = pos[k]; o = pxs[k]
                    sh = int(math.floor(p['sh'] * frac_all))
                    if t < odd_t:
                        sh = (sh // 1000) * 1000
                    if cap_pct and np.isfinite(AMT[t, k]):
                        mx = int(cap_pct * AMT[t, k] / o)
                        if t < odd_t:
                            mx = (mx // 1000) * 1000
                        sh = min(sh, mx)
                    if sh <= 0 or p['sh'] <= 0:
                        continue
                    frac = sh / p['sh']
                    sl = 0.5 * cm * float(tick(o)) / o + xs
                    if ox > 0 and t >= odd_t:
                        sl += ox * (sh % 1000) / sh
                    gross = vo * frac * (1 - sl)
                    f = _fee(sh, gross / sh, fee, minfee); tx = gross * tax
                    cash += gross - f - tx
                    st['sell_val'] += vo * frac; st['fee'] += f; st['tax'] += tx; st['slip'] += vo * frac * sl
                    p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] -= sh
        B = min(Bt, cash + v50)

    def buys(t, cand, po, rk, is_defer=False):
        nonlocal cash
        free = slots - len(pos)
        blocked = []
        if free <= 0 or not cand:
            return blocked
        order = [cand[i] for i in np.lexsort((np.arange(len(cand)), -np.nan_to_num(rk, nan=-1e18)))]
        rkd = dict(zip(cand, rk))
        if lock:
            sat50 = max(v50 - B, 0.0)
            e50_liq = sat50 * (1 - 0.5 * cm * etf_tick(po) / po) * (1 - fee - TAX_ETF * cm) - (minfee if sat50 > 0 else 0.0)
            cash_s = cash - max(B - v50, 0.0)
        else:
            e50_liq = v50 * (1 - 0.5 * cm * etf_tick(po) / po) * (1 - fee - TAX_ETF * cm) - (minfee if v50 > 0 else 0.0)
            cash_s = cash
        e50_liq = max(e50_liq, 0.0)
        eq = eq_now(t)
        sv = max(eq - B, 0.0)
        k = t - 1
        div = {'W10': 10.0, 'W7': 7.0, 'WN': float(max(ntgt.get(k, 0), 5))}[sizing]
        u = sv / div
        for j in order:
            target = u
            if free <= 0:
                st['noslot'] += 1; continue
            if banned(j, t):
                st['blk_dropped'] += 1; continue
            if is_defer:
                st['n_defer_try'] += 1
            o, why = bpx(t, j)
            if o is None:
                if why == 'noopen':
                    st['blk_noopen'] += 1
                elif is_defer:
                    st['n_defer_blk'] += 1
                else:
                    st['blk_limup'] += 1; blocked.append((j, rkd[j]))
                if not is_defer:
                    ent_att.append((t, j, why))
                continue
            sl = 0.5 * cm * float(tick(o)) / o + xs
            px = o * (1 + sl)
            avail = max(cash_s, 0.0) + e50_liq
            budget = min(target, avail)
            sh = int(budget / (px * (1 + fee) + (o * ox if t >= odd_t else 0.0)))
            if t < odd_t:
                sh = (sh // 1000) * 1000
            res = None
            if sh <= 0:
                st['blk_lot' if budget >= 0.5 * target else 'blk_nocash'] += 1
                res = 'lot' if budget >= 0.5 * target else 'nocash'
            capped = False
            if res is None:
                st['cap_enter'] += 1
                if cap_pct and np.isfinite(AMT[t, j]):
                    mx = int(cap_pct * AMT[t, j] / o)
                    if t < odd_t:
                        mx = (mx // 1000) * 1000
                    if sh > mx:
                        sh = mx; st['blk_cap'] += 1; capped = True
                        if sh <= 0:
                            st['cap_dropped'] += 1; res = 'cap0'
            if res is None:
                f = _fee(sh, px, fee, minfee); oc = (sh % 1000) * o * ox
                while sh > 0 and sh * px + f + oc > avail + 1e-6:
                    sh -= 1000 if t < odd_t else 1
                    f = _fee(sh, px, fee, minfee) if sh > 0 else 0.0; oc = (sh % 1000) * o * ox
                if sh <= 0:
                    st['blk_nocash'] += 1; res = 'nocash'
                    if capped:
                        st['cap_dropped'] += 1
                elif min_frac > 0 and sh * o < min_frac * target:
                    st['blk_min'] += 1; res = 'min'
                    if capped:
                        st['cap_dropped'] += 1
            if not is_defer:
                ent_att.append((t, j, res or 'bought'))
            if res is not None:
                continue
            cst = sh * px + f + oc
            if cash_s < cst:
                e50_liq -= (cst - max(cash_s, 0.0))
            cash -= cst; cash_s -= cst
            st['buy_val'] += sh * o; st['fee'] += f; st['slip'] += sh * o * sl + oc; st['n_buy'] += 1
            if np.isfinite(AMT[t, j]) and AMT[t, j] > 0:
                st['part'].append(sh * o / AMT[t, j])
            if is_defer:
                st['n_defer_buy'] += 1
            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=t - 1 if not is_defer else t - 2, cost=cst, flag=False, why='', proc=0.0, new=True, fx=o)
            free -= 1
        return blocked

    pending = []                                               # DEFER：[(j, 分數)]
    for t in range(t0 + 1, T):
        if v50 > 0:
            v50 *= P.E50O[t] / P.E50[t - 1]
        B *= P.E50O[t] / P.E50[t - 1]
        sold_today = set()
        for j in list(pos):
            p = pos[j]
            if P.MK[t, j] == 0 and not P.MK[t:, j].any():
                cash += p['v']; st['n_delist'] += 1
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + p['v'], 'delist', p['sh0'])); del pos[j]; sold_today.add(j); continue
            if not p['flag']:
                continue
            o, why = spx(t, j)
            if o is None:
                st['sell_defer_noopen' if why == 'noopen' else 'sell_defer_limdn'] += 1
                ex_att.append((t, j, why, p['ft'])); continue
            vopen = p['v'] * o / REF[t, j]
            frac = 1.0
            if cap_pct and np.isfinite(AMT[t, j]) and vopen > cap_pct * AMT[t, j]:
                frac = cap_pct * AMT[t, j] / vopen; st['sell_partial_cap'] += 1
            st['n_sell'] += 1
            ex_att.append((t, j, 'sold' if frac >= 1.0 else 'partial', p['ft']))
            sh = int(round(p['sh'] * frac))
            sl = 0.5 * cm * float(tick(o)) / o + xs
            if ox > 0 and sh > 0 and t >= odd_t:
                sl += ox * (sh % 1000) / sh
            gross = vopen * frac * (1 - sl)
            f = _fee(max(sh, 1), gross / max(sh, 1), fee, minfee); tx = gross * tax
            cash += gross - f - tx; sold_today.add(j)
            st['sell_val'] += vopen * frac; st['fee'] += f; st['tax'] += tx; st['slip'] += vopen * frac * sl
            if frac >= 1.0:
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + gross - f - tx, p['why'], p['sh0'])); del pos[j]
            else:
                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0); p['ft'] = t + 1
        if sat_w < 1.0 and t in exec_days:
            rebalance(t)
        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
        if len(cand):
            held = set(pos) | sold_today
            cand = [j for j in cand if j not in held]
            st['n_sig'] += len(cand)
        else:
            cand = []
        po_o = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
        pend_now = [(j, r) for j, r in pending if j not in pos and j not in sold_today] if defer else []
        pending = []
        if pend_now:                                           # 順延者與當天其他候選依分數一起排序（只試一次）
            dset = set(j for j, _ in pend_now)
            cand2 = [j for j, _ in pend_now] + [j for j in cand if j not in dset]
            rk2 = np.array([r for _, r in pend_now] + list(rank[t - 1, [j for j in cand if j not in dset]]), float)
            if cand:
                raise RuntimeError('順延日與執行日重疊（本題不應發生）')
            buys(t, cand2, po_o, rk2, is_defer=True)
        elif cand:
            blk = buys(t, cand, po_o, rank[t - 1, cand])
            if defer:
                pending = blk
        e50_net(po_o)
        v50 *= P.E50[t] / P.E50O[t]
        B *= P.E50[t] / P.E50O[t]
        for j, p in pos.items():
            if p['new']:
                p['v'] = p['v'] * (C[t, j] / p['fx']); p['new'] = False
            else:
                p['v'] *= (1 + R[t, j])
        tot = 0.0
        is_sig = bool(sigrows[t])
        for j, p in pos.items():
            tot += p['v']
            if not p['flag']:
                if X[t, j]:
                    p['flag'] = True; p['why'] = 'rule' if is_sig else 'ma_daily'; p['ft'] = t + 1
                elif max_hold is not None and t - p['e'] >= max_hold - 1:
                    p['flag'] = True; p['why'] = 'time'; p['ft'] = t + 1
        idle_now = cash + v50
        if B > idle_now:
            B = max(idle_now, 0.0)
        if B < 0:
            B = 0.0
        nav[t] = cash + v50 + tot
    for j, p in pos.items():
        log.append((j, p['sig'], p['e'], T - 1, p['cost'], p['proc'] + p['v'], 'open_end', p['sh0']))
    navs = pd.Series(nav[t0:T], index=P.IDX[t0:T])
    lg = pd.DataFrame(log, columns=['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares'])
    if len(lg):
        lg['code'] = P.COLS[lg['j'].to_numpy()]; lg['ret'] = lg['proceeds'] / lg['cost'] - 1
    st['ent_att'] = ent_att; st['ex_att'] = ex_att
    return navs, lg, st
