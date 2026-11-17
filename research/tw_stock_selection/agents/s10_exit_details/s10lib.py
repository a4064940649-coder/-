#!/usr/bin/env python3
"""S10 共用程式庫：出場細節家族（E0 標準＋10 個新變體）。
在 s9lib（不改）之上：資料、訊號日、創新板、統計全用 s9lib／s8lib／s4lib／s2lib／stlib；
- make_signal_s10()：s4lib.make_signal(depth=10, mom_lo=0.10, exit='top20') 的同一套進場，另外一次算出每個出場變體的 X（訊號列的待賣矩陣）與 RQ（當月買進名單，RQ6 用）；
- simulate_s10()：s9lib.simulate_s9 的逐行複製（只保留 exec_mode='open'），多三個出場掛鉤：trail（移動停損門檻）、trail_daily（每日或只在訊號列看）、requal（持有滿 N 個換股月要重新合格）。
  每日季線（DMA60）直接放在 X（每日列都有值），不需要掛鉤。
中間檔放 scratchpad/s10tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S10 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s10tmp')
sys.path.insert(0, str(AG / 's9_signal_timing'))
import s9lib                                               # noqa: E402
from s9lib import s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s9lib.S9 = S10; s8lib.S8 = S10; s7lib.S7 = S10; s6lib.S6 = S10; s5lib.S5 = S10; s4lib.S4 = S10; s3lib.S3 = S10; s2lib.S2 = S10; stlib.S1 = S10; cslib.TMP = S10   # 快取改放本題 scratchpad
FEE, TAX, MINFEE, TAX_ETF = stlib.FEE, stlib.TAX, stlib.MINFEE, stlib.TAX_ETF
dd_dates = s8lib.dd_dates

EXITS = ['E0', 'R10', 'R15', 'R30', 'MA20', 'MANO', 'DMA60', 'TS15', 'TS25', 'TS15M', 'RQ6']
ENGINE = {   # 每個出場用哪個 X、引擎掛鉤
    'E0': dict(X='E0'), 'R10': dict(X='R10'), 'R15': dict(X='R15'), 'R30': dict(X='R30'), 'MA20': dict(X='MA20'), 'MANO': dict(X='MANO'),
    'DMA60': dict(X='DMA60'), 'TS15': dict(X='E0', trail=0.15, trail_daily=True), 'TS25': dict(X='E0', trail=0.25, trail_daily=True),
    'TS15M': dict(X='E0', trail=0.15, trail_daily=False), 'RQ6': dict(X='E0', requal=6)}


def load():
    S10.mkdir(parents=True, exist_ok=True)
    P = s4lib.load()
    P.F_MA20 = P.G > stlib.rmean(P.G, 20)                 # 還原收盤 > 20 日均線
    return P


# ---------------------------------------------------------------- 訊號＋出場家族
def make_signal_s10(P, univ, RT, shift=0, depth=10, mom_lo=0.10, nsel=10):
    """進場同 s4lib.make_signal(depth=10, mom_lo=0.10, exit='top20')（逐行同邏輯）。
    回傳 S（E0 的 sig／rank／X／info，格式同 s4lib）＋ S['XV']＝{出場 id: X 矩陣}、S['RQ']＝{訊號列: 當月買進名單 set}、S['A2_mismatch']（拆法對 L_MA 前 20 名的不同格數）。
    MA20／MANO：續抱＝股池內、SURGE 有值、SURGE ≥ 當月 L_MA 第 20 名分數（L_MA 不到 20 檔時不設名次門檻），MA20 另要還原收盤 > 20 日均線。
    DMA60：E0 的 X 再加上每個交易日 ~F_MA60（還原收盤 ≤ 60 日均線）。"""
    S, av = RT['S'], RT['av']
    sig = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan)
    XV = {k: np.zeros((P.T, P.N), bool) for k in ('E0', 'R10', 'R15', 'R30', 'MA20', 'MANO')}
    info = []; RQ = {}; a2 = 0
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((univ[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)              # cslib.INS15_MASK（預設 False＝原樣）：15 日期限公司訊號日看不到當月營收 → 缺值
        fin = univ[k] & np.isfinite(s)
        ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        r20 = P.RET20[k]
        pool = o_ma[:depth]
        okr = r20[pool] > mom_lo
        tgt = pool[okr][:nsel]
        sig[k, tgt] = True; rank[k, tgt] = s[tgt]
        RQ[k] = set(int(j) for j in tgt)
        for key, n in (('E0', 20), ('R10', 10), ('R15', 15), ('R30', 30)):
            XV[key][k, :] = True; XV[key][k, o_ma[:n]] = False
        thr = s[o_ma[19]] if len(o_ma) >= 20 else -np.inf
        with np.errstate(invalid='ignore'):
            hi = fin & (np.nan_to_num(s, nan=-np.inf) >= thr)
        keep_mano = hi
        keep_ma20 = hi & P.F_MA20[k]
        keep_ma60 = hi & P.F_MA60[k]                       # 驗收 A2：應等於 L_MA 前 20 名
        ref = np.zeros(P.N, bool); ref[o_ma[:20]] = True
        a2 += int((keep_ma60 != ref).sum())
        XV['MANO'][k, :] = ~keep_mano
        XV['MA20'][k, :] = ~keep_ma20
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok_ma.sum()), n_tgt=int(len(tgt)),
                         targets=[str(P.COLS[j]) for j in tgt]))
    sigrows = np.zeros(P.T, bool); sigrows[[x['k'] for x in info]] = True
    XV['DMA60'] = XV['E0'] | ~P.F_MA60                     # 每日：還原收盤 ≤ 60 日均線（訊號列另有名次條件）
    return dict(sig=sig, rank=rank, X=XV['E0'], SD=None, max_hold=None, info=info, XV=XV, RQ=RQ, sigrows=sigrows, A2_mismatch=a2)


def variant_S(S, eid):
    """出場 id → simulate_s10 要的 (S', 引擎參數)。"""
    e = ENGINE[eid]
    S2 = dict(S); S2['X'] = S['XV'][e['X']]
    kw = {k: v for k, v in e.items() if k != 'X'}
    return S2, kw
def simulate_s10(P, S, univ, sat_w=1.0, sizing='W10', lock=False, band=0.10, cost=None, block=None, capital=1e6, slots=10, cap_pct=0.05, sweep_min=20000.0, min_frac=0.5,
                 exec_mode='open', trail=None, trail_daily=True, requal=None):
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
        return cash + v50 + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())

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
            if cash + v50 < Bt - 1e-6:
                st['rb_trim_short'] += 1
        B = min(Bt, cash + v50)

    def buys(t, cand, po):
        nonlocal cash
        free = slots - len(pos)
        if free <= 0 or not cand:
            return
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
            target = u
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
            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=t - 1, cost=cst, flag=False, why='', proc=0.0, new=not CLOSE, g=1.0, gpk=-np.inf, nm=0)
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
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + p['v'], 'delist', p['sh0'])); del pos[j]; sold_today.add(j); continue
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
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + gross - f - tx, p['why'], p['sh0'])); del pos[j]
            else:
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
    st['base_sold'] = bsold; st['short_events'] = short_ev; st['fills'] = fills; st['minfrac_edge'] = edge
    st['stockfrac'] = pd.Series(stockfrac[t0:T], index=P.IDX[t0:T]); st['B'] = pd.Series(bser[t0:T], index=P.IDX[t0:T])
    return navs, lg, st
