#!/usr/bin/env python3
"""S37 共用程式庫：換股日隔天開盤實際買得到嗎？成交價差多少？（描述性；不改策略規則）
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD）、F1、E0、引擎）之上，本檔只新增：
- 開除權息修正：cslib.EXRIGHT_FIX = True（load() 之前）。
- add_avg()：當日均價代理 AVGV（成交金額÷成交股數；不在最低～最高之間或缺值時改用 (開＋高＋低＋收)/4）、AVG4（(開＋高＋低＋收)/4）。
- simulate_x()：s10lib.simulate_s10 的逐行複製（exec_mode='open'），只多：
    fill＝'open'（現行，與 simulate_s10 逐日相同）／'avgv'／'avg4'／'open05'／'defer'（見 SPEC.json part2）；
    嘗試紀錄 st['att_b']（買進：t, j, 結果, 是否被成交值上限縮量, 是否順延單）、st['att_s']（賣出：t, j, 結果）。
  新買進部位當天收盤市值＝股數 × 收盤（p['fp']＝成交參考價；fill='open' 時 fp＝開盤，與原式 v×收盤÷開盤 相同）。
中間檔放 scratchpad/s37/。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S37 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s37')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S37; s17lib.S17 = S37; s10lib.S10 = S37; s9lib.S9 = S37; s8lib.S8 = S37; s7lib.S7 = S37; s6lib.S6 = S37
s5lib.S5 = S37; s4lib.S4 = S37; s3lib.S3 = S37; s2lib.S2 = S37; stlib.S1 = S37; cslib.TMP = S37
cslib.EXRIGHT_FIX = True
FEE, TAX, MINFEE, TAX_ETF = stlib.FEE, stlib.TAX, stlib.MINFEE, stlib.TAX_ETF
FILLS = ('open', 'avgv', 'avg4', 'open05', 'defer')
EPS = 1e-9


def load():
    S37.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix is True, '除權息修正沒有開'
    return P


def add_avg(P):
    O, H, L, C, V, A = P.O, P.H, P.L, P.C, P.VOL, P.AMT
    with np.errstate(invalid='ignore', divide='ignore'):
        a4 = (O + H + L + C) / 4.0
        vw = A / V
        ok = np.isfinite(vw) & (V > 0) & (vw >= L * 0.999) & (vw <= H * 1.001)
    P.AVG4 = a4
    P.AVGV = np.where(ok, vw, a4)
    P.avgv_fallback = int((np.isfinite(a4) & ~ok).sum())
    return P


def simulate_x(P, S, univ, sat_w=1.0, sizing='W10', lock=False, band=0.10, cost=None, block=None, capital=1e6, slots=10, cap_pct=0.05, sweep_min=20000.0, min_frac=0.5,
               fill='open', trail=None, trail_daily=True, requal=None):
    assert fill in FILLS
    cost = cost or COSTS['strict']; block = block or {}
    cm = cost['cmult']; xs = cost['xslip'] * cm; ox = cost['oddx'] * cm
    sig, rank, X, max_hold = S['sig'], S['rank'], S['X'], S['max_hold']
    sigrows = S['sigrows']; RQ = S.get('RQ', {})
    exec_days = set(int(x['k']) + 1 for x in S['info'])
    ntgt = {int(x['k']): int(x['n_tgt']) for x in S['info']}
    assert sizing in ('W10', 'W7', 'WN')
    T, t0 = P.T, P.t0
    O, C, R, REF, AMT = P.O, P.CF, P.R, P.REF, P.AMT
    Hh, Ll = P.H, P.L
    AVG = P.AVGV if fill == 'avgv' else (P.AVG4 if fill == 'avg4' else None)
    fee, tax, minfee = FEE * cm, TAX * cm, MINFEE * cm
    tick = cslib.tick; _fee = stlib._fee; etf_tick = stlib.etf_tick
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))
    cash = float(capital); v50 = 0.0; pos = {}
    B = (1.0 - sat_w) * capital
    nav = np.full(T, np.nan); nav[t0] = capital
    log = []; att_b = []; att_s = []
    pend = {}                                                  # defer：{j: (rank, sig_t)}，只在下一個交易日有效
    st = dict(n_sig=0, n_buy=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, blk_min=0, noslot=0, blk_dropped=0, sell_defer_limdn=0, sell_defer_noopen=0,
              sell_partial_cap=0, n_delist=0, buy_val=0.0, sell_val=0.0, fee=0.0, tax=0.0, slip=0.0, n_defer_try=0, n_defer_buy=0)

    def buy_px(t, j):
        o = O[t, j]
        if not (np.isfinite(o) and o > 0):
            return None, 'noopen'
        if AVG is not None:
            if Ll[t, j] >= P.LIMUP[t, j] - EPS:
                return None, 'limup'
            return float(AVG[t, j]), ''
        if o >= P.LIMUP[t, j] - EPS:
            return None, 'limup'
        if fill == 'open05':
            return float(min(o * 1.005, P.LIMUP[t, j])), ''
        return o, ''

    def sell_px(t, j):
        o = O[t, j]
        if not (np.isfinite(o) and o > 0):
            return None, 'noopen'
        if AVG is not None:
            if Hh[t, j] <= P.LIMDN[t, j] + EPS:
                return None, 'limdn'
            return float(AVG[t, j]), ''
        if o <= P.LIMDN[t, j] + EPS:
            return None, 'limdn'
        if fill == 'open05':
            return float(max(o * 0.995, P.LIMDN[t, j])), ''
        return o, ''

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
            vals = {}
            for k, p in pos.items():
                if p['flag']:
                    continue
                o, why = sell_px(t, k)
                if o is None:
                    continue
                vals[k] = (p['v'] * o / REF[t, k], o)
            tv = sum(v for v, _ in vals.values())
            if tv > 0:
                frac_all = min(1.0, need / tv)
                for k, (vo, o) in vals.items():
                    p = pos[k]
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

    def buys(t, items, po):
        """items：[(j, rank, sig_t, is_defer)]。"""
        nonlocal cash
        free = slots - len(pos)
        if not items:
            return
        rk = np.array([x[1] for x in items], float)
        order = [items[i] for i in np.lexsort((np.arange(len(items)), -np.nan_to_num(rk, nan=-1e18)))]
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
        for j, rkj, sgt, isd in order:
            target = u
            if isd:
                st['n_defer_try'] += 1
            if free <= 0:
                st['noslot'] += 1; att_b.append((t, j, 'noslot', False, isd)); continue
            if banned(j, t):
                st['blk_dropped'] += 1; att_b.append((t, j, 'banned', False, isd)); continue
            o, why = buy_px(t, j)
            if o is None:
                st['blk_' + why] += 1; att_b.append((t, j, why, False, isd))
                if fill == 'defer' and why == 'limup' and not isd:
                    pend[j] = (rkj, sgt)
                continue
            sl = 0.5 * cm * float(tick(o)) / o + xs
            px = o * (1 + sl)
            avail = max(cash_s, 0.0) + e50_liq
            budget = min(target, avail)
            sh = int(budget / (px * (1 + fee) + (o * ox if t >= odd_t else 0.0)))
            if t < odd_t:
                sh = (sh // 1000) * 1000
            if sh <= 0:
                w = 'lot' if budget >= 0.5 * target else 'nocash'
                st['blk_' + w] += 1; att_b.append((t, j, w, False, isd)); continue
            capped = False
            if cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / o)
                if t < odd_t:
                    mx = (mx // 1000) * 1000
                if sh > mx:
                    sh = mx; st['blk_cap'] += 1; capped = True
                    if sh <= 0:
                        att_b.append((t, j, 'cap0', True, isd)); continue
            f = _fee(sh, px, fee, minfee); oc = (sh % 1000) * o * ox
            while sh > 0 and sh * px + f + oc > avail + 1e-6:
                sh -= 1000 if t < odd_t else 1
                f = _fee(sh, px, fee, minfee) if sh > 0 else 0.0; oc = (sh % 1000) * o * ox
            if sh <= 0:
                st['blk_nocash'] += 1; att_b.append((t, j, 'nocash', capped, isd)); continue
            if min_frac > 0 and sh * o < min_frac * target:
                st['blk_min'] += 1; att_b.append((t, j, 'min', capped, isd)); continue
            cst = sh * px + f + oc
            if cash_s < cst:
                e50_liq -= (cst - max(cash_s, 0.0))
            cash -= cst; cash_s -= cst
            st['buy_val'] += sh * o; st['fee'] += f; st['slip'] += sh * o * sl + oc; st['n_buy'] += 1
            if isd:
                st['n_defer_buy'] += 1
            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=sgt, cost=cst, flag=False, why='', proc=0.0, new=True, g=1.0, gpk=-np.inf, nm=0, fp=o)
            att_b.append((t, j, 'bought', capped, isd))
            free -= 1

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
            o, why = sell_px(t, j)
            if o is None:
                st['sell_defer_' + why] += 1; att_s.append((t, j, why)); continue
            vopen = p['v'] * o / REF[t, j]
            frac = 1.0
            if cap_pct and np.isfinite(AMT[t, j]) and vopen > cap_pct * AMT[t, j]:
                frac = cap_pct * AMT[t, j] / vopen; st['sell_partial_cap'] += 1
            sh = int(round(p['sh'] * frac))
            sl = 0.5 * cm * float(tick(o)) / o + xs
            if ox > 0 and sh > 0 and t >= odd_t:
                sl += ox * (sh % 1000) / sh
            gross = vopen * frac * (1 - sl)
            f = _fee(max(sh, 1), gross / max(sh, 1), fee, minfee); tx = gross * tax
            cash += gross - f - tx; sold_today.add(j)
            st['sell_val'] += vopen * frac; st['fee'] += f; st['tax'] += tx; st['slip'] += vopen * frac * sl
            att_s.append((t, j, 'sold' if frac >= 1.0 else 'partial'))
            if frac >= 1.0:
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + gross - f - tx, p['why'], p['sh0'])); del pos[j]
            else:
                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0)
        if sat_w < 1.0 and t in exec_days:
            rebalance(t)
        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
        held = set(pos) | sold_today
        items = []
        if len(cand):
            cand = [j for j in cand if j not in held]
            st['n_sig'] += len(cand)
            items = [(int(j), rank[t - 1, j], t - 1, False) for j in cand]
        if pend:
            have = set(x[0] for x in items)
            items += [(j, r, sg, True) for j, (r, sg) in pend.items() if j not in held and j not in have]
            pend.clear()
        po_o = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
        if items:
            buys(t, items, po_o)
        e50_net(po_o)
        v50 *= P.E50[t] / P.E50O[t]
        B *= P.E50[t] / P.E50O[t]
        for j, p in pos.items():
            if p['new']:
                p['v'] = p['v'] * (C[t, j] / p['fp']); p['new'] = False
                p['g'] = C[t, j] / p['fp']
            else:
                p['v'] *= (1 + R[t, j])
                p['g'] *= (1 + R[t, j])
            if p['g'] > p['gpk']:
                p['gpk'] = p['g']
        tot = 0.0
        is_sig = bool(sigrows[t])
        for j, p in pos.items():
            tot += p['v']
            if is_sig and t > p['e']:
                p['nm'] += 1
            if not p['flag']:
                if X[t, j]:
                    p['flag'] = True; p['why'] = 'rule' if is_sig else 'ma_daily'
                elif max_hold is not None and t - p['e'] >= max_hold - 1:
                    p['flag'] = True; p['why'] = 'time'
                elif trail is not None and (trail_daily or is_sig) and p['g'] <= p['gpk'] * (1.0 - trail):
                    p['flag'] = True; p['why'] = 'trail'
                elif requal is not None and is_sig and p['nm'] >= requal and j not in RQ.get(t, ()):
                    p['flag'] = True; p['why'] = 'requal'
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
        lg['code'] = P.COLS[lg['j'].to_numpy()]; lg['ret'] = lg['proceeds'] / lg['cost'] - 1; lg['pnl'] = lg['proceeds'] - lg['cost']
        lg['hold_days'] = lg['exit_t'] - lg['ent_t']
        lg['sig_date'] = P.IDX[lg['sig_t'].to_numpy()]; lg['ent_date'] = P.IDX[lg['ent_t'].to_numpy()]; lg['exit_date'] = P.IDX[lg['exit_t'].to_numpy()]
    st['att_b'] = pd.DataFrame(att_b, columns=['t', 'j', 'res', 'capped', 'defer'])
    st['att_s'] = pd.DataFrame(att_s, columns=['t', 'j', 'res'])
    return navs, lg, st
