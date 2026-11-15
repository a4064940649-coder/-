#!/usr/bin/env python3
"""S9 共用程式庫：訊號日定義家族＋成交時點（次日開盤／訊號日收盤）。
在 s8lib（不改）之上：資料、訊號、創新板、統計全用 s8lib／s7lib／s6lib／s5lib／s4lib；
- signal_rows()／rt_for()：A_D（研究中心慣例）與 B_R（XQ 慣例）的訊號列，以及每個定義用哪個營收月的 SURGE；
- simulate_s9()：s8lib.simulate_s8 的逐行複製（拿掉 cap），多 exec_mode：'open'＝訊號列 t−1、t 日開盤成交（與 s8 相同）；
  'close'＝訊號列 t−1、t 日收盤成交（賣出、拉回、買進、0050 都用 t 日收盤）。
中間檔放 scratchpad/s9tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S9 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s9tmp')
sys.path.insert(0, str(AG / 's8_drawdown_cap'))
import s8lib                                               # noqa: E402
from s8lib import s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s8lib.S8 = S9; s7lib.S7 = S9; s6lib.S6 = S9; s5lib.S5 = S9; s4lib.S4 = S9; s3lib.S3 = S9; s2lib.S2 = S9; stlib.S1 = S9; cslib.TMP = S9   # 快取改放本題 scratchpad
FEE, TAX, MINFEE, TAX_ETF = stlib.FEE, stlib.TAX, stlib.MINFEE, stlib.TAX_ETF
dd_dates = s8lib.dd_dates


def load():
    S9.mkdir(parents=True, exist_ok=True)
    return s4lib.load()


# ---------------------------------------------------------------- 訊號日
def _first_td_on_or_after(P, ts):
    return int(np.searchsorted(P.IDX.values, np.datetime64(ts), side='left'))


def signal_rows(P, months, conv, D=None, R=None):
    """營收月 m → 訊號列 k（面板列號，≥ P.T 表示超出面板）。
    conv='A'：d*＝m＋1 月第 D 日起第一個交易日，k＝d*＋1（D＝10 即 cslib.rev_avail_days）。
    conv='B'：k＝m＋1 月第一個日期 ≥ R 的交易日（XQ RebalDay）。另回傳每月法定期限日列號 dl（m＋1 月 10 日起第一個交易日）。"""
    out, dl = {}, {}
    for m in months:
        mm = (m + 1).to_timestamp()
        dl[m] = _first_td_on_or_after(P, mm + pd.Timedelta(days=9))
        if conv == 'A':
            out[m] = _first_td_on_or_after(P, mm + pd.Timedelta(days=D - 1)) + 1
        else:
            out[m] = _first_td_on_or_after(P, mm + pd.Timedelta(days=R - 1))
    return out, dl


def rt_for(P, RT, d, close=False):
    """定義 d（PREREG 的一列）→ make_signal 用的 RT（S＝每個「營收月標籤 m」實際使用的 SURGE 列；av＝訊號列）。
    營收月標籤 m 一律代表「m＋1 月的換股」；用 M−2 時 S.loc[m] 換成原本的 S.loc[m−1]。
    close=True：訊號列改成 av − 1（名單用前一天收盤資料，成交在 av 那天收盤）。回傳 (RT', 說明 dict)。"""
    S = RT['S']; months = list(S.index)
    if d['conv'] == 'A':
        av, dl = signal_rows(P, months, 'A', D=d['D'])
    else:
        av, dl = signal_rows(P, months, 'B', R=d['R'])
    Sp = S.shift(1)                                         # 上一個營收月（M−2）
    if d['id'] in ('A08', 'A09'):
        Su = Sp.copy(); stale = set(months)
    elif d['id'] == 'B11pes':
        stale = {m for m in months if av[m] <= dl[m]}
        Su = S.copy()
        idx = [m for m in months if m in stale]
        Su.loc[idx] = Sp.loc[idx].to_numpy()
    else:
        Su = S; stale = set()
    early = {m for m in months if av[m] <= dl[m]}           # 訊號列 ≤ 法定期限列的月份（M−1 不確定已全部公布）
    av2 = {m: (k - 1 if close else k) for m, k in av.items()}
    return dict(S=Su, av=av2), dict(stale_months=sorted(str(m) for m in stale), early_months=sorted(str(m) for m in early), av_exec=av, deadline=dl)


# ---------------------------------------------------------------- 引擎
def simulate_s9(P, S, univ, sat_w=1.0, sizing='W10', lock=False, band=0.10, cost=None, block=None, capital=1e6, slots=10, cap_pct=0.05, sweep_min=20000.0, min_frac=0.5,
                exec_mode='open'):
    """S9：s8lib.simulate_s8 的逐行複製（拿掉 cap），多 exec_mode。
    'open'：與 s8lib.simulate_s8(cap=None) 逐筆相同（訊號列 t−1 → t 日開盤賣、拉回、買；0050 用開盤價）。
    'close'：訊號列 t−1 → t 日收盤：持股先照 t 日收盤估值，再賣出出場股（收盤價；收盤 ≤ 跌停價或無成交不賣）、拉回、
             依 SURGE 名次買進（收盤價；收盤 ≥ 漲停價或無成交不買）、現金掃 0050（0050 收盤價）。成本、跳動點、5% 成交值上限、
             「股數 × 價格 < 目標一半不買」、零股規則全同 open，只是價格換成收盤。新買進的部位從 t＋1 日起照 R 計報酬。"""
    assert exec_mode in ('open', 'close')
    CLOSE = exec_mode == 'close'
    cost = cost or COSTS['strict']; block = block or {}
    cm = cost['cmult']; xs = cost['xslip'] * cm; ox = cost['oddx'] * cm
    sig, rank, X, max_hold = S['sig'], S['rank'], S['X'], S['max_hold']
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
    log = []; bsold = []; short_ev = []; fills = []               # fills：(t, j, 'B'/'S', 成交參考價)，驗收 A3 用
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
            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=t - 1, cost=cst, flag=False, why='', proc=0.0, new=not CLOSE)
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
    st['base_sold'] = bsold; st['short_events'] = short_ev; st['fills'] = fills
    st['stockfrac'] = pd.Series(stockfrac[t0:T], index=P.IDX[t0:T]); st['B'] = pd.Series(bser[t0:T], index=P.IDX[t0:T])
    return navs, lg, st
