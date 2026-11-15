#!/usr/bin/env python3
"""S5 共用程式庫：0050 底倉＋F1 規則衛星。
在 s4lib（不改）之上：訊號、創新板、面板全用 s4lib；引擎 simulate_cs() 是 s4lib.simulate 的逐行複製
（固定 idle='e50'、e50_fix=True、min_frac=0.5），只多「底倉金額 B 簿記、每檔目標＝衛星資金÷10、換股日偏離 > band 才拉回」。
sat_w=1 時與 s4lib.simulate(idle='e50') 逐筆相同。中間檔放 scratchpad/s5/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S5 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s5')
sys.path.insert(0, str(AG / 's4_e3_purity'))
import s4lib                                               # noqa: E402
from s4lib import s3lib, s2lib, stlib, cslib, COSTS        # noqa: E402,F401
s4lib.S4 = S5; s3lib.S3 = S5; s2lib.S2 = S5; stlib.S1 = S5; cslib.TMP = S5   # 快取改放本題 scratchpad（不動 s1～s4 的）
FEE, TAX, MINFEE, TAX_ETF = stlib.FEE, stlib.TAX, stlib.MINFEE, stlib.TAX_ETF


def load():
    S5.mkdir(parents=True, exist_ok=True)
    return s4lib.load()


def simulate_cs(P, S, univ, sat_w=1.0, band=0.10, cost=None, block=None, capital=1e6, slots=10, cap_pct=0.05, sweep_min=20000.0, min_frac=0.5):
    """s4lib.simulate(idle='e50', e50_fix=True, buy_at='open') 的逐行複製＋底倉簿記。
    B＝底倉金額（跟 0050 漲跌）；衛星資金＝淨值 − B；每檔目標＝衛星資金 ÷ slots；
    換股執行日（訊號日的下一個交易日）開盤、賣完之後、買之前：|衛星÷淨值 − sat_w| > band 就把 B 設回 (1−sat_w)×淨值，
    閒置（現金＋0050）不夠時個股依市值等比例減碼。"""
    cost = cost or COSTS['strict']; block = block or {}
    cm = cost['cmult']; xs = cost['xslip'] * cm; ox = cost['oddx'] * cm
    sig, rank, X, max_hold = S['sig'], S['rank'], S['X'], S['max_hold']
    exec_days = set(int(x['k']) + 1 for x in S['info'])
    T, t0 = P.T, P.t0
    O, C, R, REF, AMT, G, CR = P.O, P.CF, P.R, P.REF, P.AMT, P.G, P.C
    fee, tax, minfee = FEE * cm, TAX * cm, MINFEE * cm
    tick = cslib.tick; _fee = stlib._fee; etf_tick = stlib.etf_tick
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))
    cash = float(capital); v50 = 0.0; pos = {}
    B = (1.0 - sat_w) * capital
    nav = np.full(T, np.nan); nav[t0] = capital
    stockfrac = np.full(T, np.nan); bser = np.full(T, np.nan); bser[t0] = B
    log = []
    st = dict(n_sig=0, n_buy=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, blk_min=0, noslot=0, blk_dropped=0, sell_defer_limdn=0, sell_defer_noopen=0,
              sell_partial_cap=0, n_delist=0, buy_val=0.0, sell_val=0.0, fee=0.0, tax=0.0, slip=0.0, e50_cost=0.0, e50_orders=0, npos_sum=0, idle_frac_sum=0.0,
              odd_buys=0, odd_buy_val=0.0, rb_up=0, rb_down=0, rb_trim_days=0, rb_trim_orders=0, rb_trim_val=0.0, rb_trim_short=0, b_clamped=0)

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
        e50_liq = v50 * (1 - 0.5 * cm * etf_tick(po) / po) * (1 - fee - TAX_ETF * cm) - (minfee if v50 > 0 else 0.0)
        e50_liq = max(e50_liq, 0.0)
        eq = eq_open(t)
        target = max(eq - B, 0.0) / slots                      # S5：衛星資金 ÷ slots（sat_w=1 時 B=0，同 s4lib）
        for j in order:
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
            avail = max(cash, 0.0) + e50_liq
            budget = min(target, avail)
            sh = int(budget / (px * (1 + fee) + (o * ox if t >= odd_t else 0.0)))
            if t < odd_t:
                sh = (sh // 1000) * 1000
            if sh <= 0:
                st['blk_lot' if budget >= 0.5 * target else 'blk_nocash'] += 1; continue
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
                st['blk_nocash'] += 1; continue
            if min_frac > 0 and sh * o < min_frac * target:          # 固定規格 2：最小買進金額（對衛星目標）
                st['blk_min'] += 1; continue
            cst = sh * px + f + oc
            if cash < cst:
                e50_liq -= (cst - max(cash, 0.0))
            cash -= cst
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
        if B > idle_now:
            if B - idle_now > 1e-6:
                st['b_clamped'] += 1
            B = max(idle_now, 0.0)
        if B < 0:
            B = 0.0
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
    st['stockfrac'] = pd.Series(stockfrac[t0:T], index=P.IDX[t0:T]); st['B'] = pd.Series(bser[t0:T], index=P.IDX[t0:T])
    return navs, lg, st


# ---------------------------------------------------------------- 統計
def ann_vol(nav, a=None, b=None):
    r = nav.loc[a:b].dropna().pct_change().dropna()
    return float(r.std(ddof=1) * math.sqrt(252))


def beta_alpha(nav, bench, a=None, b=None):
    """月報酬 OLS：r = α + β r_0050 + e（不扣無風險利率）。"""
    s = nav.loc[a:b].dropna(); q = bench.reindex(s.index)
    m1 = s.groupby(s.index.to_period('M')).last().pct_change(); m2 = q.groupby(q.index.to_period('M')).last().pct_change()
    d = pd.concat([m1, m2], axis=1).dropna()
    y = d.iloc[:, 0].to_numpy(); x = d.iloc[:, 1].to_numpy(); n = len(y)
    Xm = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    res = y - Xm @ coef
    s2 = float(res @ res) / (n - 2)
    cov = s2 * np.linalg.inv(Xm.T @ Xm)
    se_a, se_b = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
    return dict(n_months=int(n), first=str(d.index[0]), last=str(d.index[-1]), alpha_m=float(coef[0]), alpha_t=float(coef[0] / se_a), alpha_ann12=float(coef[0] * 12),
                beta=float(coef[1]), beta_se=float(se_b), r2=float(1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))))


def vs0050_trades(P, lg, cut='2021-02-19'):
    """已平倉交易對照同一段持有期 0050（還原開盤進、還原開盤出、不扣成本）。"""
    x = lg[lg['why'] != 'open_end'].copy()
    e = x['ent_t'].to_numpy().astype(int); q = x['exit_t'].to_numpy().astype(int)
    x['r0050'] = P.E50O[q] / P.E50O[e] - 1
    x['excess'] = x['ret'] - x['r0050']

    def summ(y):
        if len(y) < 2:
            return dict(n=int(len(y)))
        return dict(n=int(len(y)), ret_mean=float(y['ret'].mean()), r0050_mean=float(y['r0050'].mean()), excess_mean=float(y['excess'].mean()),
                    excess_median=float(y['excess'].median()), excess_t=float(y['excess'].mean() / (y['excess'].std(ddof=1) / math.sqrt(len(y)))),
                    frac_beat=float((y['excess'] > 0).mean()), hold_mean=float(y['hold_days'].mean()))
    c = pd.Timestamp(cut)
    return dict(all=summ(x), half1=summ(x[x['ent_date'] <= c]), half2=summ(x[x['ent_date'] > c])), x
