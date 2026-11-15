#!/usr/bin/env python3
"""S7 共用程式庫：底倉鎖定的衛星＋執行延遲。
在 s6lib（不改）之上：資料、訊號、創新板、統計全用 s6lib／s5lib／s4lib；引擎 simulate_s7() 是 s6lib.simulate_s6 的逐行複製（拿掉名次加權），
只多 lock（底倉鎖定：買股只能用衛星可用資金）與「衛星現金不夠而縮小／跳過」的記錄。lock=False 時與 s6lib.simulate_s6(rank_w=False) 逐筆相同。
delay_signal() 做「訊號日不變、進出場延後 d 個交易日」。中間檔放 scratchpad/s7tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S7 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s7tmp')
sys.path.insert(0, str(AG / 's6_satellite_sizing'))
import s6lib                                               # noqa: E402
from s6lib import s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s6lib.S6 = S7; s5lib.S5 = S7; s4lib.S4 = S7; s3lib.S3 = S7; s2lib.S2 = S7; stlib.S1 = S7; cslib.TMP = S7   # 快取改放本題 scratchpad
FEE, TAX, MINFEE, TAX_ETF = stlib.FEE, stlib.TAX, stlib.MINFEE, stlib.TAX_ETF
rolling12, dd_dates, frontier = s6lib.rolling12, s6lib.dd_dates, s6lib.frontier


def load():
    S7.mkdir(parents=True, exist_ok=True)
    return s4lib.load()


def delay_signal(P, S, d):
    """訊號日 k 不變，進出場延後 d 個交易日：把 sig／rank／X／info／pos_ma 從 k 搬到 k＋d（k＋d ≥ T 的月份丟掉）。
    名單已在 k 日套用股池，所以搭配 univ＝全開使用（買進時不再用 k＋d 日的股池重篩）。"""
    T = P.T
    sig = np.zeros_like(S['sig']); X = np.zeros_like(S['X']); rank = np.full_like(S['rank'], np.nan)
    info = []; pos_ma = {}
    for x in S['info']:
        k = int(x['k']); k2 = k + d
        if k2 >= T:
            continue
        sig[k2] = S['sig'][k]; X[k2] = S['X'][k]; rank[k2] = S['rank'][k]
        y = dict(x); y['k'] = k2; y['sig_k'] = k
        info.append(y)
        if k in (S.get('pos_ma') or {}):
            pos_ma[k2] = S['pos_ma'][k]
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=S['max_hold'], info=info, pos_ma=pos_ma, kmonth={})


def simulate_s7(P, S, univ, sat_w=1.0, sizing='W10', lock=False, band=0.10, cost=None, block=None, capital=1e6, slots=10, cap_pct=0.05, sweep_min=20000.0, min_frac=0.5):
    """S7：s6lib.simulate_s6 的逐行複製（拿掉名次加權），多一個 lock：True＝底倉鎖定，買股只能用衛星可用資金
    （max(衛星現金, 0)＋衛星自己的 0050 可變現金額；衛星現金＝現金 − max(B − 0050, 0)，衛星 0050＝max(0050 − B, 0)）。
    另記「衛星現金不夠而縮小／跳過」。lock=False 時與 s6lib.simulate_s6(rank_w=False) 逐筆相同。
    以下為 s6lib 原 docstring：S6：s5lib.simulate_cs 的逐行複製，只改每檔目標金額（sizing＝W10／W7／WN、rank_w 名次加權），另記「底倉真的被賣去買股」。
    以下為 s5lib 原 docstring：s4lib.simulate(idle='e50', e50_fix=True, buy_at='open') 的逐行複製＋底倉簿記。
    B＝底倉金額（跟 0050 漲跌）；衛星資金＝淨值 − B；每檔目標＝衛星資金 ÷ slots；
    換股執行日（訊號日的下一個交易日）開盤、賣完之後、買之前：|衛星÷淨值 − sat_w| > band 就把 B 設回 (1−sat_w)×淨值，
    閒置（現金＋0050）不夠時個股依市值等比例減碼。"""
    cost = cost or COSTS['strict']; block = block or {}
    cm = cost['cmult']; xs = cost['xslip'] * cm; ox = cost['oddx'] * cm
    sig, rank, X, max_hold = S['sig'], S['rank'], S['X'], S['max_hold']
    exec_days = set(int(x['k']) + 1 for x in S['info'])
    ntgt = {int(x['k']): int(x['n_tgt']) for x in S['info']}          # S6：當月名單檔數
    assert sizing in ('W10', 'W7', 'WN')
    T, t0 = P.T, P.t0
    O, C, R, REF, AMT, G, CR = P.O, P.CF, P.R, P.REF, P.AMT, P.G, P.C
    fee, tax, minfee = FEE * cm, TAX * cm, MINFEE * cm
    tick = cslib.tick; _fee = stlib._fee; etf_tick = stlib.etf_tick
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))
    cash = float(capital); v50 = 0.0; pos = {}
    B = (1.0 - sat_w) * capital
    nav = np.full(T, np.nan); nav[t0] = capital
    stockfrac = np.full(T, np.nan); bser = np.full(T, np.nan); bser[t0] = B
    log = []; bsold = []                                   # S6：bsold＝(t, 金額, 比例)
    short_ev = []                                          # S7：(t, j, 目標, 買到金額, 'shrunk'／'skip')
    st = dict(n_sig=0, n_buy=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, blk_min=0, noslot=0, blk_dropped=0, sell_defer_limdn=0, sell_defer_noopen=0,
              sell_partial_cap=0, n_delist=0, buy_val=0.0, sell_val=0.0, fee=0.0, tax=0.0, slip=0.0, e50_cost=0.0, e50_orders=0, npos_sum=0, idle_frac_sum=0.0,
              odd_buys=0, odd_buy_val=0.0, shrunk=0, shrunk_amt=0.0, skip_cash=0, rb_up=0, rb_down=0, rb_trim_days=0, rb_trim_orders=0, rb_trim_val=0.0, rb_trim_short=0, b_clamped=0, b_drop_nonexec=0)

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
            v50 -= x; cash += net; st['e50_cost'] += x - net; st['e50_orders'] += 1
        elif cash >= sweep_min:
            f = max(minfee, cash * fee)
            val = (cash - f) / (1 + sl50)
            st['e50_cost'] += cash - val; st['e50_orders'] += 1
            v50 += val; cash = 0.0

    def eq_open(t):
        return cash + v50 + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())

    def rebalance(t):
        """S5：換股執行日開盤、賣完後、買之前。"""
        nonlocal cash, B
        eq = eq_open(t)
        if eq <= 0:
            return
        sv = eq - B
        dev = sv / eq - sat_w
        if abs(dev) <= band + 1e-12:
            return
        Bt = (1.0 - sat_w) * eq
        if dev < 0:
            st['rb_down'] += 1; B = Bt; return                 # 衛星偏低：底倉劃給衛星（簿記）
        st['rb_up'] += 1
        idle = cash + v50
        need = Bt - idle
        if need > 1e-6:                                        # 閒置不夠：個股等比例減碼
            vals = {}
            for k, p in pos.items():
                if p['flag']:
                    continue
                o = O[t, k]
                if not (np.isfinite(o) and o > 0) or o <= P.LIMDN[t, k] + 1e-9:
                    continue
                vals[k] = p['v'] * o / REF[t, k]
            tv = sum(vals.values())
            if tv > 0:
                st['rb_trim_days'] += 1
                frac_all = min(1.0, need / tv)
                for k, vo in vals.items():
                    p = pos[k]; o = O[t, k]
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
                    st['rb_trim_orders'] += 1; st['rb_trim_val'] += vo * frac
                    p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] -= sh
            if cash + v50 < Bt - 1e-6:
                st['rb_trim_short'] += 1
        B = min(Bt, cash + v50)

    def buys(t, cand, at_close, po):
        nonlocal cash
        free = slots - len(pos)
        if free <= 0 or not cand:
            return
        rk = rank[t - 1, cand]
        order = [cand[i] for i in np.lexsort((np.arange(len(cand)), -np.nan_to_num(rk, nan=-1e18)))]
        if lock:                                               # S7：只能用衛星自己的現金與 0050
            sat50 = max(v50 - B, 0.0)
            e50_liq = sat50 * (1 - 0.5 * cm * etf_tick(po) / po) * (1 - fee - TAX_ETF * cm) - (minfee if sat50 > 0 else 0.0)
            cash_s = cash - max(B - v50, 0.0)
        else:
            e50_liq = v50 * (1 - 0.5 * cm * etf_tick(po) / po) * (1 - fee - TAX_ETF * cm) - (minfee if v50 > 0 else 0.0)
            cash_s = cash
        e50_liq = max(e50_liq, 0.0)
        eq = eq_open(t)
        sv = max(eq - B, 0.0)                                  # 衛星資金（同 S5）
        k = t - 1
        div = {'W10': 10.0, 'W7': 7.0, 'WN': float(max(ntgt.get(k, 0), 5))}[sizing]   # S6：每檔＝衛星 ÷ div
        u = sv / div
        for j in order:
            target = u                                         # 該檔目標金額（S7 不做名次加權）
            if free <= 0:
                st['noslot'] += 1; continue
            if banned(j, t):
                st['blk_dropped'] += 1; continue
            o = O[t, j]
            if not (np.isfinite(o) and o > 0):
                st['blk_noopen'] += 1; continue
            if o >= P.LIMUP[t, j] - 1e-9:
                st['blk_limup'] += 1; continue
            sl = 0.5 * cm * float(tick(o)) / o + xs
            px = o * (1 + sl)
            avail = max(cash_s, 0.0) + e50_liq
            budget = min(target, avail)
            short = budget < target - 1e-9                     # S7：錢不夠（鎖定時＝衛星可用資金不夠）
            sh = int(budget / (px * (1 + fee) + (o * ox if t >= odd_t else 0.0)))
            if t < odd_t:
                sh = (sh // 1000) * 1000
            if sh <= 0:
                st['blk_lot' if budget >= 0.5 * target else 'blk_nocash'] += 1
                if short:
                    st['skip_cash'] += 1; short_ev.append((int(t), int(j), float(target), 0.0, 'skip'))
                continue
            if cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / o)
                if t < odd_t:
                    mx = (mx // 1000) * 1000
                if sh > mx:
                    sh = mx; st['blk_cap'] += 1
                    if sh <= 0:
                        continue
            f = _fee(sh, px, fee, minfee); oc = (sh % 1000) * o * ox
            while sh > 0 and sh * px + f + oc > avail + 1e-6:
                sh -= 1000 if t < odd_t else 1
                f = _fee(sh, px, fee, minfee) if sh > 0 else 0.0; oc = (sh % 1000) * o * ox
            if sh <= 0:
                st['blk_nocash'] += 1
                if short:
                    st['skip_cash'] += 1; short_ev.append((int(t), int(j), float(target), 0.0, 'skip'))
                continue
            if min_frac > 0 and sh * o < min_frac * target:          # 固定規格 2：最小買進金額（對衛星目標）
                st['blk_min'] += 1
                if short:
                    st['skip_cash'] += 1; short_ev.append((int(t), int(j), float(target), 0.0, 'skip'))
                continue
            cst = sh * px + f + oc
            if cash_s < cst:
                e50_liq -= (cst - max(cash_s, 0.0))
            cash -= cst; cash_s -= cst
            if short:
                st['shrunk'] += 1; st['shrunk_amt'] += max(target - sh * o, 0.0); short_ev.append((int(t), int(j), float(target), float(sh * o), 'shrunk'))
            st['buy_val'] += sh * o; st['fee'] += f; st['slip'] += sh * o * sl + oc; st['n_buy'] += 1
            if sh % 1000:
                st['odd_buys'] += 1; st['odd_buy_val'] += (sh % 1000) * o
            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=t - 1, cost=cst, flag=False, why='', proc=0.0, new=True)
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
            o = O[t, j]
            if not (np.isfinite(o) and o > 0):
                st['sell_defer_noopen'] += 1; continue
            if o <= P.LIMDN[t, j] + 1e-9:
                st['sell_defer_limdn'] += 1; continue
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
            if frac >= 1.0:
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + gross - f - tx, p['why'], p['sh0'])); del pos[j]
            else:
                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0)
        if sat_w < 1.0 and t in exec_days:
            rebalance(t)
        b_ref = B if t in exec_days else None                  # S6：拉回後、買進前的 B
        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
        if len(cand):
            held = set(pos) | sold_today
            cand = [j for j in cand if j not in held]
            st['n_sig'] += len(cand)
        else:
            cand = []
        po_o = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
        if cand:
            buys(t, cand, False, po_o)
        e50_net(po_o)
        v50 *= P.E50[t] / P.E50O[t]
        B *= P.E50[t] / P.E50O[t]
        for j, p in pos.items():
            if p['new']:
                p['v'] = p['v'] * (C[t, j] / O[t, j]); p['new'] = False
            else:
                p['v'] *= (1 + R[t, j])
        tot = 0.0
        for j, p in pos.items():
            tot += p['v']
            if not p['flag']:
                if X[t, j]:
                    p['flag'] = True; p['why'] = 'rule'
                elif max_hold is not None and t - p['e'] >= max_hold - 1:
                    p['flag'] = True; p['why'] = 'time'
        idle_now = cash + v50
        b_before = B
        if B > idle_now:
            if B - idle_now > 1e-6:
                st['b_clamped'] += 1
            B = max(idle_now, 0.0)
        if B < 0:
            B = 0.0
        if b_ref is not None and b_ref > 0:                    # S6：底倉真的被賣去買股（換股執行日，B 低於拉回後 B × 0050 開→收 超過 0.01%）
            bref_c = b_ref * P.E50[t] / P.E50O[t]
            if B < bref_c * (1 - 1e-4):
                bsold.append((int(t), float(bref_c - B), float(1 - B / bref_c)))
        elif b_before > 0 and B < b_before * (1 - 1e-4):
            st['b_drop_nonexec'] += 1
        nav[t] = cash + v50 + tot
        stockfrac[t] = tot / nav[t]; bser[t] = B
        st['npos_sum'] += len(pos); st['idle_frac_sum'] += (cash + v50) / nav[t]
    for j, p in pos.items():
        log.append((j, p['sig'], p['e'], T - 1, p['cost'], p['proc'] + p['v'], 'open_end', p['sh0']))
    navs = pd.Series(nav[t0:T], index=P.IDX[t0:T])
    lg = pd.DataFrame(log, columns=['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares'])
    if len(lg):
        lg['code'] = P.COLS[lg['j'].to_numpy()]; lg['ret'] = lg['proceeds'] / lg['cost'] - 1; lg['pnl'] = lg['proceeds'] - lg['cost']
        lg['hold_days'] = lg['exit_t'] - lg['ent_t']
        lg['sig_date'] = P.IDX[lg['sig_t'].to_numpy()]; lg['ent_date'] = P.IDX[lg['ent_t'].to_numpy()]; lg['exit_date'] = P.IDX[lg['exit_t'].to_numpy()]
        lg['market'] = np.where(P.MK[lg['ent_t'].to_numpy(), lg['j'].to_numpy()] == 1, 'twse', 'tpex')
        lg['score'] = rank[lg['sig_t'].to_numpy(), lg['j'].to_numpy()]
    nd = T - 1 - t0
    st['avg_npos'] = st.pop('npos_sum') / nd; st['avg_idle_frac'] = st.pop('idle_frac_sum') / nd
    st['base_sold'] = bsold; st['short_events'] = short_ev
    st['stockfrac'] = pd.Series(stockfrac[t0:T], index=P.IDX[t0:T]); st['B'] = pd.Series(bser[t0:T], index=P.IDX[t0:T])
    return navs, lg, st
