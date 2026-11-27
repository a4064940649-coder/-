#!/usr/bin/env python3
"""S102 模擬器：S100 的 s100sim.simulate_staged（＝s10lib.simulate_s10 的逐行複製＋分批掛鉤；stage=None 時與原引擎逐筆相同，S100 驗收 A1）再複製一份，
另加 U2「破線賣出後站回再買回」掛鉤 rebuy＝dict(r5=bool, RET5=矩陣)。stage 與 rebuy 都是 None 時與 s10lib.simulate_s10 逐日逐筆相同（本題 step0 驗收 A1）。
U2 規則（細節見 PREREG.json）：
  破線：訊號列 k 收盤被出場規則 E0 標記、且當天還原收盤 ≤ 60 日均線（~F_MA60）的持股＝「破線賣出」；照現行次日開盤賣出（跌停／無開盤照引擎遞延）。
  觀察窗：全部賣完那天 → 下一個訊號列 k' 的前一天（收盤看）；每檔每窗只有一次機會。
  觸發：觀察日 d 收盤：仍在股池、還原收盤 > 60 日均線、用最新一期（訊號列 k）的 SURGE 分數與股池，在「d 日季線上」的股票中排前 20 名；r5＝True 時再要求近 5 日漲幅（G[d]/G[d−5]−1）> 0。
  買回：d＋1 日開盤，金額與一切執行細節同新買（buys()：一份＝衛星總值 ÷ 10、開盤漲停不買、5% 成交值、整張／零股、不足目標一半不買、10 檔上限）；
        同日多檔依 SURGE 分數高到低；買不成不遞延。買回後的部位照現行出場規則。"""
import math
import numpy as np, pandas as pd
import s10lib
from s10lib import stlib, cslib, COSTS
FEE, TAX, MINFEE, TAX_ETF = stlib.FEE, stlib.TAX, stlib.MINFEE, stlib.TAX_ETF


def simulate_u(P, S, univ, sat_w=1.0, sizing='W10', lock=False, band=0.10, cost=None, block=None, capital=1e6, slots=10, cap_pct=0.05, sweep_min=20000.0, min_frac=0.5,
                 exec_mode='open', trail=None, trail_daily=True, requal=None, stage=None, rebuy=None):
    """S10：s9lib.simulate_s9 的逐行複製，只用 exec_mode='open'，多出場掛鉤（不開時與 simulate_s9 逐筆相同，驗收 A0／A3）：
    trail＝移動停損門檻（0.15＝回落 15%）：g＝持有期含息累積報酬指數（進場日收盤 g＝收盤 ÷ 開盤，之後 ×(1＋R)），gpk＝進場日收盤起最高 g；
          g ≤ gpk ×（1 − trail）→ 標記 'trail'，次日開盤賣。trail_daily＝True 每日收盤看；False 只在訊號列（S['sigrows']）看。
    requal＝N：進場後經過的訊號列數 ≥ N 時，每個訊號列必須在 S['RQ'][t]（當月買進名單）裡，否則標記 'requal'。
    X 在非訊號列有值（DMA60）時，標記原因寫 'ma_daily'。
    以下是 simulate_s9 原說明：
    'open'：與 s8lib.simulate_s8(cap=None) 逐筆相同（訊號列 t−1 → t 日開盤賣、拉回、買；0050 用開盤價）。
    'close'：訊號列 t−1 → t 日收盤：持股先照 t 日收盤估值，再賣出出場股（收盤價；收盤 ≤ 跌停價或無成交不賣）、拉回、
             依 SURGE 名次買進（收盤價；收盤 ≥ 漲停價或無成交不買）、現金掃 0050（0050 收盤價）。成本、跳動點、5% 成交值上限、
             「股數 × 價格 < 目標一半不買」、零股規則全同 open，只是價格換成收盤。新買進的部位從 t＋1 日起照 R 計報酬。"""
    assert exec_mode == 'open'
    CLOSE = exec_mode == 'close'
    cost = cost or COSTS['strict']; block = block or {}
    cm = cost['cmult']; xs = cost['xslip'] * cm; ox = cost['oddx'] * cm
    sig, rank, X, max_hold = S['sig'], S['rank'], S['X'], S['max_hold']
    sigrows = S['sigrows']; RQ = S.get('RQ', {})
    exec_days = set(int(x['k']) + 1 for x in S['info'])
    ntgt = {int(x['k']): int(x['n_tgt']) for x in S['info']}
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
    log = []; bsold = []; short_ev = []; fills = []; edge = []               # fills：(t, j, 'B'/'S', 成交參考價)，驗收 A3 用
    st = dict(n_sig=0, n_buy=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, blk_min=0, noslot=0, blk_dropped=0, sell_defer_limdn=0, sell_defer_noopen=0,
              sell_partial_cap=0, n_delist=0, buy_val=0.0, sell_val=0.0, fee=0.0, tax=0.0, slip=0.0, e50_cost=0.0, e50_orders=0, npos_sum=0, idle_frac_sum=0.0,
              odd_buys=0, odd_buy_val=0.0, shrunk=0, shrunk_amt=0.0, skip_cash=0, rb_up=0, rb_down=0, rb_trim_days=0, rb_trim_orders=0, rb_trim_val=0.0, rb_trim_short=0, b_clamped=0, b_drop_nonexec=0)

    ST = stage is not None
    f_init = float(stage['init']) if ST else 1.0; f_final = float(stage['final']) if ST else 1.0; n_check = int(stage['check']) if ST else 0
    rand = stage.get('rand') if ST else None
    assert (not ST) or (sizing == 'W10' and not lock and f_final >= f_init > 0 and n_check >= 1)
    full_new = set(); lotlog = []; topev = []
    st.update(top_n=0, top_val=0.0, top_pend=0, top_blk_limup=0, top_blk_noopen=0, top_blk_min=0, top_blk_ban=0, top_room_bind=0, new_full=0)
    RB = rebuy is not None                                     # U2 掛鉤
    rb_r5 = bool(rebuy.get('r5')) if RB else False
    brk = {}; rb_pend = {}; rbev = []; rbflag = []
    st.update(rb_flag=0, rb_win=0, rb_trig=0, rb_buy=0, rb_fail=0, rb_noslot=0)
    if RB:
        SCR, FIN, FMA, G5 = S['score'], S['fin'], P.F_MA60, rebuy['RET5']
        sigidx = np.flatnonzero(sigrows)

        def nxt_sig(t):
            i = int(np.searchsorted(sigidx, t, side='right'))
            return int(sigidx[i]) if i < len(sigidx) else T

        def last_sig(t):
            i = int(np.searchsorted(sigidx, t, side='right')) - 1
            return int(sigidx[i])

        def in_top20(k, t, j):
            ok = FIN[k] & FMA[t]
            if not ok[j]:
                return False
            v = SCR[k][ok]
            if len(v) <= 20:
                return True
            thr = np.partition(v, len(v) - 20)[len(v) - 20]
            return bool(SCR[k][j] >= thr)

    def lot_alloc(p, amount, frac):
        """賣出所得依各批當時股數比例分給各批；各批股數同比例減少。"""
        tot = sum(l[1] for l in p['lots'])
        for l in p['lots']:
            l[3] += amount * (l[1] / tot) if tot > 0 else amount / len(p['lots'])
            l[1] *= (1.0 - frac)

    def lot_emit(j, p, t, why):
        for i, l in enumerate(p['lots']):
            lotlog.append((j, p['sig'], l[0], t, l[2], l[3], why, i, len(p['lots'])))

    def topups(t, js, po):
        """補買：逐行仿 buys()，目標＝(final − init) 份；多一個上限「衛星總值 − 目前持股開盤市值」。每檔只有一次機會。"""
        nonlocal cash
        e50_liq = max(v50 * (1 - 0.5 * cm * etf_tick(po) / po) * (1 - fee - TAX_ETF * cm) - (minfee if v50 > 0 else 0.0), 0.0)
        cash_s = cash
        eq = eq_now(t)
        sv = max(eq - B, 0.0)
        u = sv / 10.0
        room = max(sv - sum(q['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, q in pos.items()), 0.0)
        sc = np.array([rank[pos[j]['sig'], j] for j in js], float)
        order = [js[i] for i in np.lexsort((np.arange(len(js)), -np.nan_to_num(sc, nan=-1e18)))]
        for j in order:
            p = pos[j]; p['stage'] = 2                         # 2＝試過沒補成；補成改 1
            target = (f_final - f_init) * u
            if banned(j, t):
                st['top_blk_ban'] += 1; continue
            o = px_of(t, j)
            if not (np.isfinite(o) and o > 0):
                st['top_blk_noopen'] += 1; continue
            if o >= P.LIMUP[t, j] - 1e-9:
                st['top_blk_limup'] += 1; continue
            sl = 0.5 * cm * float(tick(o)) / o + xs
            px = o * (1 + sl)
            avail = max(cash_s, 0.0) + e50_liq
            budget = min(target, avail, room)
            if room < min(target, avail) - 1e-9:
                st['top_room_bind'] += 1
            sh = int(budget / (px * (1 + fee) + (o * ox if t >= odd_t else 0.0)))
            if t < odd_t:
                sh = (sh // 1000) * 1000
            if sh > 0 and cap_pct and np.isfinite(AMT[t, j]):
                mx = int(cap_pct * AMT[t, j] / o)
                if t < odd_t:
                    mx = (mx // 1000) * 1000
                if sh > mx:
                    sh = mx; st['blk_cap'] += 1
            f = _fee(sh, px, fee, minfee) if sh > 0 else 0.0; oc = (sh % 1000) * o * ox
            while sh > 0 and sh * px + f + oc > avail + 1e-6:
                sh -= 1000 if t < odd_t else 1
                f = _fee(sh, px, fee, minfee) if sh > 0 else 0.0; oc = (sh % 1000) * o * ox
            if sh <= 0 or sh * o < min_frac * target:
                st['top_blk_min'] += 1; continue
            cst = sh * px + f + oc
            if cash_s < cst:
                e50_liq -= (cst - max(cash_s, 0.0))
            cash -= cst; cash_s -= cst; room -= sh * o
            st['buy_val'] += sh * o; st['fee'] += f; st['slip'] += sh * o * sl + oc; st['top_n'] += 1; st['top_val'] += sh * o
            if sh % 1000:
                st['odd_buys'] += 1; st['odd_buy_val'] += (sh % 1000) * o
            gain = p['g'] * (O[t, j] / REF[t, j]) - 1.0        # 原進場開盤 → 補買日開盤的含息累積報酬（未扣成本）
            topev.append((int(t), int(j), int(p['e']), float(sh * o), float(o), float(target), float(gain), float(p['cost'])))
            p['vadd'] = sh * o; p['sh'] += sh; p['sh0'] += sh; p['cost'] += cst; p['stage'] = 1; p['top_t'] = t
            p['lots'].append([t, float(sh), cst, 0.0])
            fills.append((int(t), int(j), 'B', float(o)))

    def px_of(t, k):
        """成交參考價：open＝t 日開盤；close＝t 日收盤（原始價，無成交為 NaN）。"""
        return CR[t, k] if CLOSE else O[t, k]

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

    def eq_now(t):
        if CLOSE:                                              # 收盤：持股已照收盤估值
            return cash + v50 + sum(p['v'] for p in pos.values())
        return cash + v50 + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items()) + sum(p['vadd'] for p in pos.values())

    def val_now(t, k, p):
        """持股在成交時點的市值：open＝前收市值 × 開盤 ÷ 參考價；close＝已照收盤估值。"""
        return p['v'] if CLOSE else p['v'] * O[t, k] / REF[t, k]

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
            st['rb_down'] += 1; B = Bt; return
        st['rb_up'] += 1
        idle = cash + v50
        need = Bt - idle
        if need > 1e-6:
            vals = {}
            for k, p in pos.items():
                if p['flag']:
                    continue
                o = px_of(t, k)
                if not (np.isfinite(o) and o > 0) or o <= P.LIMDN[t, k] + 1e-9:
                    continue
                vals[k] = val_now(t, k, p)
            tv = sum(vals.values())
            if tv > 0:
                st['rb_trim_days'] += 1
                frac_all = min(1.0, need / tv)
                for k, vo in vals.items():
                    p = pos[k]; o = px_of(t, k)
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
                    lot_alloc(p, gross - f - tx, frac)
            if cash + v50 < Bt - 1e-6:
                st['rb_trim_short'] += 1
        B = min(Bt, cash + v50)

    def buys(t, cand, po, order=None, sigr=None):
        nonlocal cash
        free = slots - len(pos)
        if free <= 0 or not cand:
            return
        if order is None:
            rk = rank[t - 1, cand]
            order = [cand[i] for i in np.lexsort((np.arange(len(cand)), -np.nan_to_num(rk, nan=-1e18)))]
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
            isfull = (not ST) or (j in full_new)
            target = u * (f_final if (ST and isfull) else f_init)
            if free <= 0:
                st['noslot'] += 1; continue
            if banned(j, t):
                st['blk_dropped'] += 1; continue
            o = px_of(t, j)
            if not (np.isfinite(o) and o > 0):
                st['blk_noopen'] += 1; continue
            if o >= P.LIMUP[t, j] - 1e-9:
                st['blk_limup'] += 1; continue
            sl = 0.5 * cm * float(tick(o)) / o + xs
            px = o * (1 + sl)
            avail = max(cash_s, 0.0) + e50_liq
            budget = min(target, avail)
            short = budget < target - 1e-9
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
            if 0.40 * target <= sh * o < 0.60 * target:
                edge.append((int(t), int(j), float(sh * o / target), bool(not (min_frac > 0 and sh * o < min_frac * target))))
            if min_frac > 0 and sh * o < min_frac * target:
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
            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=(t - 1 if sigr is None else sigr[j]), cost=cst, flag=False, why='', proc=0.0, new=not CLOSE, g=1.0, gpk=-np.inf, nm=0,
                          stage=1 if isfull else 0, pend=False, vadd=0.0, lots=[[t, float(sh), cst, 0.0]], top_t=-1, cost0=cst)
            if ST and isfull:
                st['new_full'] += 1
            fills.append((int(t), int(j), 'B', float(o)))
            free -= 1

    for t in range(t0 + 1, T):
        if CLOSE:                                              # close：0050、底倉、持股直接走到 t 日收盤
            if v50 > 0:
                v50 *= P.E50[t] / P.E50[t - 1]
            B *= P.E50[t] / P.E50[t - 1]
        else:
            if v50 > 0:
                v50 *= P.E50O[t] / P.E50[t - 1]
            B *= P.E50O[t] / P.E50[t - 1]
        sold_today = set()
        if CLOSE:                                              # 下市日 R＝0（無成交），先估值不影響下市處理
            for j, p in pos.items():
                p['v'] *= (1 + R[t, j])
        for j in list(pos):
            p = pos[j]
            if P.MK[t, j] == 0 and not P.MK[t:, j].any():
                cash += p['v']; st['n_delist'] += 1
                lot_alloc(p, p['v'], 1.0); lot_emit(j, p, t, 'delist')
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + p['v'], 'delist', p['sh0'], p['top_t'], p['cost0'])); del pos[j]; sold_today.add(j); continue
            if not p['flag']:
                continue
            o = px_of(t, j)
            if not (np.isfinite(o) and o > 0):
                st['sell_defer_noopen'] += 1; continue
            if o <= P.LIMDN[t, j] + 1e-9:
                st['sell_defer_limdn'] += 1; continue
            vopen = val_now(t, j, p)
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
            fills.append((int(t), int(j), 'S', float(o)))
            if frac >= 1.0:
                lot_alloc(p, gross - f - tx, 1.0); lot_emit(j, p, t, p['why'])
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + gross - f - tx, p['why'], p['sh0'], p['top_t'], p['cost0'])); del pos[j]
                if RB and p.get('brk_end') is not None and t < p['brk_end']:
                    brk[j] = p['brk_end']; st['rb_win'] += 1
            else:
                lot_alloc(p, gross - f - tx, frac)
                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0)
        if sat_w < 1.0 and t in exec_days:
            rebalance(t)
        b_ref = B if t in exec_days else None
        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
        if len(cand):
            held = set(pos) | sold_today
            cand = [j for j in cand if j not in held]
            st['n_sig'] += len(cand)
        else:
            cand = []
        if CLOSE:
            po_o = P.E50C_raw[t] if np.isfinite(P.E50C_raw[t]) and P.E50C_raw[t] > 0 else 100.0
        else:
            po_o = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
        if ST and t in exec_days:
            pend = sorted(j for j, p in pos.items() if p['pend'])
            for j in pend:
                pos[j]['pend'] = False
            full_new.clear()
            if rand is not None:                               # 隨機對照：K 個補買額度隨機分給（未補過的續抱股 ∪ 當天新買名單）
                K = int(rand['K'].get(int(t), 0))
                pool = [(0, j) for j in sorted(j for j, p in pos.items() if p['stage'] == 0 and not p['flag'])] + [(1, int(j)) for j in cand]
                pend = []
                if K > 0 and pool:
                    for i in rand['rng'].choice(len(pool), size=min(K, len(pool)), replace=False):
                        if pool[i][0] == 0:
                            pend.append(pool[i][1])
                        else:
                            full_new.add(pool[i][1])
                    pend.sort()
            st['top_pend'] += len(pend)
            if pend and f_final > f_init:
                topups(t, pend, po_o)
        if RB and rb_pend:                                     # U2：前一日收盤觸發的買回（非換股日；當天 cand 必為空）
            js = [j for j in sorted(rb_pend, key=lambda j: (-SCR[rb_pend[j]][j], j)) if j not in pos]
            free0 = slots - len(pos)
            if js:
                buys(t, js, po_o, order=js, sigr=rb_pend)
                for i, j in enumerate(js):
                    if j in pos and pos[j]['e'] == t:
                        st['rb_buy'] += 1; rbev.append((int(t), int(j), int(rb_pend[j])))
                    else:
                        st['rb_fail'] += 1
                        if i >= free0:
                            st['rb_noslot'] += 1
            rb_pend = {}
        if cand:
            buys(t, cand, po_o)
        e50_net(po_o)
        if not CLOSE:
            v50 *= P.E50[t] / P.E50O[t]
            B *= P.E50[t] / P.E50O[t]
            for j, p in pos.items():
                if p['new']:
                    p['v'] = p['v'] * (C[t, j] / O[t, j]); p['new'] = False
                    p['g'] = C[t, j] / O[t, j]
                else:
                    p['v'] *= (1 + R[t, j])
                    if p['vadd']:                              # 當天開盤補買的部分：開盤 → 收盤
                        p['v'] += p['vadd'] * (C[t, j] / O[t, j]); p['vadd'] = 0.0
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
                    if RB and is_sig and not FMA[t, j]:
                        p['brk_end'] = nxt_sig(t); st['rb_flag'] += 1; rbflag.append((int(t), int(j)))
                elif max_hold is not None and t - p['e'] >= max_hold - 1:
                    p['flag'] = True; p['why'] = 'time'
                elif trail is not None and (trail_daily or is_sig) and p['g'] <= p['gpk'] * (1.0 - trail):
                    p['flag'] = True; p['why'] = 'trail'
                elif requal is not None and is_sig and p['nm'] >= requal and j not in RQ.get(t, ()):
                    p['flag'] = True; p['why'] = 'requal'
            if ST and rand is None and is_sig and t > p['e'] and not p['flag'] and p['stage'] == 0 and p['nm'] == n_check:
                p['pend'] = True                               # 通過第 n_check 次換股檢查 → 次日開盤補買
        if RB and brk:                                         # U2：收盤檢查觀察窗
            for j in sorted(brk):
                if t >= brk[j]:
                    del brk[j]; continue
                if j in pos:
                    continue
                k = last_sig(t)
                if univ[t, j] and FMA[t, j] and in_top20(k, t, j) and ((not rb_r5) or G5[t, j] > 0):
                    rb_pend[j] = k; st['rb_trig'] += 1; del brk[j]
        idle_now = cash + v50
        b_before = B
        if B > idle_now:
            if B - idle_now > 1e-6:
                st['b_clamped'] += 1
            B = max(idle_now, 0.0)
        if B < 0:
            B = 0.0
        if b_ref is not None and b_ref > 0:
            bref_c = b_ref if CLOSE else b_ref * P.E50[t] / P.E50O[t]
            if B < bref_c * (1 - 1e-4):
                bsold.append((int(t), float(bref_c - B), float(1 - B / bref_c)))
        elif b_before > 0 and B < b_before * (1 - 1e-4):
            st['b_drop_nonexec'] += 1
        nav[t] = cash + v50 + tot
        stockfrac[t] = tot / nav[t]; bser[t] = B
        st['npos_sum'] += len(pos); st['idle_frac_sum'] += (cash + v50) / nav[t]
    for j, p in pos.items():
        lot_alloc(p, p['v'], 1.0); lot_emit(j, p, T - 1, 'open_end')
        log.append((j, p['sig'], p['e'], T - 1, p['cost'], p['proc'] + p['v'], 'open_end', p['sh0'], p['top_t'], p['cost0']))
    navs = pd.Series(nav[t0:T], index=P.IDX[t0:T])
    lg = pd.DataFrame(log, columns=['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares', 'top_t', 'cost0'])
    if len(lg):
        lg['code'] = P.COLS[lg['j'].to_numpy()]; lg['ret'] = lg['proceeds'] / lg['cost'] - 1; lg['pnl'] = lg['proceeds'] - lg['cost']
        lg['hold_days'] = lg['exit_t'] - lg['ent_t']
        lg['sig_date'] = P.IDX[lg['sig_t'].to_numpy()]; lg['ent_date'] = P.IDX[lg['ent_t'].to_numpy()]; lg['exit_date'] = P.IDX[lg['exit_t'].to_numpy()]
        lg['market'] = np.where(P.MK[lg['ent_t'].to_numpy(), lg['j'].to_numpy()] == 1, 'twse', 'tpex')
        lg['score'] = rank[lg['sig_t'].to_numpy(), lg['j'].to_numpy()]
    nd = T - 1 - t0
    st['avg_npos'] = st.pop('npos_sum') / nd; st['avg_idle_frac'] = st.pop('idle_frac_sum') / nd
    st['base_sold'] = bsold; st['short_events'] = short_ev; st['fills'] = fills; st['minfrac_edge'] = edge
    lots = pd.DataFrame(lotlog, columns=['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'lot', 'n_lots'])
    if len(lots):
        lots['code'] = P.COLS[lots['j'].to_numpy()]; lots['ret'] = lots['proceeds'] / lots['cost'] - 1; lots['pnl'] = lots['proceeds'] - lots['cost']
        lots['ent_date'] = P.IDX[lots['ent_t'].to_numpy()]; lots['exit_date'] = P.IDX[lots['exit_t'].to_numpy()]
    st['lots'] = lots
    st['topups'] = pd.DataFrame(topev, columns=['t', 'j', 'ent_t', 'add_val', 'open', 'target', 'gain_since_entry', 'cost_before'])
    st['rebuys'] = pd.DataFrame(rbev, columns=['t', 'j', 'k']); st['rb_flagged'] = pd.DataFrame(rbflag, columns=['t', 'j'])
    if len(lg):
        rbs = set((j, t) for t, j, _ in rbev)
        lg['rb'] = [(int(a), int(b)) in rbs for a, b in zip(lg['j'], lg['ent_t'])]
    st['stockfrac'] = pd.Series(stockfrac[t0:T], index=P.IDX[t0:T]); st['B'] = pd.Series(bser[t0:T], index=P.IDX[t0:T])
    return navs, lg, st
