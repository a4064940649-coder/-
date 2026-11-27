#!/usr/bin/env python3
"""R108 獨立複製 S102 U2（破線賣出後站回再買回）。本檔為獨立撰寫，未閱讀 agents/s102_u1_u2/ 下任何 .py。
規則來源：agents/s102_u1_u2/PREREG.json 的 topics.U2.execution（寫死）。
- load()：正式引擎（EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK、REDUCE_FIX、0050 官方含息 五個開關全開），載入方式照 S100 的 s100lib.load。
- row_scores()：逐訊號列重算 make_signal_s10 的 SURGE（含 15 日期限遮蔽）與股池，供買回觸發排名用；並與 make_signal_s10 的名單互相驗證。
- simulate_u2()：s10lib.simulate_s10（exec_mode='open'）的逐行複製（只刪掉 trail／requal／max_hold／close 分支）＋ U2 買回掛鉤；rebuy=None 時應與原引擎逐筆相同（step0 驗收）。
一律 python3 -I -B；暫存 scratchpad/r108。不改任何既有檔。"""
import sys, gc, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r108/cache')
TMP.mkdir(parents=True, exist_ok=True)
CAPS = (6e5, 8e5, 1e6, 1.5e6, 2e6)
SHIFTS = (0, 1, 2, 3)
SEGMENTS = {'2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}
VALID_FIRST_MONTH = pd.Period('2021-12', 'M')
TARGET = dict(since2019=0.4886246519182116, full=0.3446294016617564, n_closed=381, paths_median=0.4641920784536262)


def setup_official():
    sys.path.insert(0, str(AG / 's19_xq_threshold'))
    import s19lib
    from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib
    for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                      (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP')):
        setattr(mod, attr, TMP)
    return s19lib


def load():
    s19lib = setup_official()
    from s19lib import s4lib, stlib, cslib
    assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False
    assert cslib.REDUCE_FIX is False and cslib.E50_SOURCE is None
    cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
    cslib.REDUCE_FIX = True; cslib.E50_SOURCE = cslib.E50_OFFICIAL_CSV
    P = s19lib.load()
    assert P.exright_fix and P.reduce_fix and P.e50_source == str(cslib.E50_OFFICIAL_CSV)
    s19lib.e50_fix_suspension(P)
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    del INNOV
    S, _ = s19lib.rev_std(P)
    avA = s19lib.a10_rows(P, list(S.index))
    for a in ('H', 'L', 'VOL', 'CHG', 'HF', 'VOLF', 'adjH', 'HI240', 'MA60', 'nbar', 'F_HI', 'IDXC', 'listed', 'NEARHI', 'V60', 'liq', 'has'):
        if hasattr(P, a):
            delattr(P, a)
    gc.collect()
    return P, U, dict(S=S, av=avA)


def row_scores(P, U, RT, S, shift):
    """每個訊號列 k：(s＝遮蔽後 SURGE 分數向量, fin＝股池內且 SURGE 有值)。迴圈條件逐條照 make_signal_s10；並驗證名單與 S['sig'] 相同。"""
    cslib = sys.modules['cslib']
    Sm, av = RT['S'], RT['av']
    out = {}
    for m in Sm.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = Sm.loc[m].to_numpy()
        if int((U[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        s = cslib.ins15_mask_row(P, m, k, s)
        fin = U[k] & np.isfinite(s)
        ok = fin & P.F_MA60[k]
        o = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
        pool = o[:10]; tgt = pool[P.RET20[k][pool] > 0.10][:10]
        assert set(np.flatnonzero(S['sig'][k])) == set(int(j) for j in tgt), ('名單不符', m, k)
        assert (~S['X'][k] == np.isin(np.arange(P.N), o[:20])).all()
        out[int(k)] = (s.copy(), fin.copy())
    assert sorted(out) == sorted(int(x['k']) for x in S['info'])
    return out


def fresh_from(S, first_month=VALID_FIRST_MONTH):
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first_month:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2


def seg_dd(nav):
    o = {}
    for k, (a, b) in SEGMENTS.items():
        s = nav.loc[a:b].dropna()
        o[k] = float((s / s.cummax() - 1).min())
    return o


def simulate_u2(P, S, univ, ROWS=None, rebuy=None, sat_w=0.7, sizing='W10', lock=False, band=0.10, cost=None, block=None, capital=1e6, slots=10,
                cap_pct=0.05, sweep_min=20000.0, min_frac=0.5):
    """rebuy=None → 現行（應與 s10lib.simulate_s10 逐筆相同）；rebuy=dict(r5=False) → U2|any；dict(r5=True) → U2|r5>0。
    U2 規則（PREREG 寫死）：
      破線賣出＝訊號列 k 被 E0 標記（X[k,j]）且 F_MA60[k,j] 為假；照現行次日開盤起賣出。
      觀察窗＝全部賣完那天 s 起，到 k 之後下一個訊號列 k' 的前一天（k'−1）為止；每日收盤檢查（無下一個訊號列則到資料尾）。
      觸發＝檢查日 d：univ[d,j]、F_MA60[d,j]、在「訊號列 k 的股池且 SURGE 有值、且 d 日 F_MA60」的股票中依 k 的 SURGE 排前 20 名；r5 另要 G[d]/G[d−5]−1 > 0。
      買回＝d＋1 日開盤，用引擎 buys()（目標＝衛星÷10、漲停不買、5% 成交值、整張／零股、不足一半不買、10 檔上限、現金＋0050 可變現）；
            同日多檔依 SURGE 高到低；每檔每窗只有一次機會（買不成不重試）。買回部位之後照現行出場。"""
    s10lib, stlib, cslib = sys.modules['s10lib'], sys.modules['stlib'], sys.modules['cslib']
    COSTS = s10lib.COSTS
    FEE, TAX, MINFEE, TAX_ETF = stlib.FEE, stlib.TAX, stlib.MINFEE, stlib.TAX_ETF
    cost = cost or COSTS['strict']; block = block or {}
    cm = cost['cmult']; xs = cost['xslip'] * cm; ox = cost['oddx'] * cm
    sig, rank, X = S['sig'], S['rank'], S['X']
    sigrows = S['sigrows']
    srows = np.flatnonzero(sigrows)
    exec_days = set(int(x['k']) + 1 for x in S['info'])
    T, t0 = P.T, P.t0
    O, C, R, REF, AMT, G = P.O, P.CF, P.R, P.REF, P.AMT, P.G
    FMA = P.F_MA60
    fee, tax, minfee = FEE * cm, TAX * cm, MINFEE * cm
    tick = cslib.tick; _fee = stlib._fee; etf_tick = stlib.etf_tick
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))
    cash = float(capital); v50 = 0.0; pos = {}
    B = (1.0 - sat_w) * capital
    nav = np.full(T, np.nan); nav[t0] = capital
    stockfrac = np.full(T, np.nan)
    log = []
    st = dict(noslot=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, blk_min=0, blk_dropped=0, buy_val=0.0,
              rb_flag=0, rb_win=0, rb_trig=0, rb_buy=0, rb_fail=0)
    watch = {}      # j → dict(k, start, end)
    pend = {}       # t → [(j, k)]
    rbev = []

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
                o = O[t, k]
                if not (np.isfinite(o) and o > 0) or o <= P.LIMDN[t, k] + 1e-9:
                    continue
                vals[k] = p['v'] * O[t, k] / REF[t, k]
            tv = sum(vals.values())
            if tv > 0:
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
                    p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] -= sh
        B = min(Bt, cash + v50)

    def buys(t, cand, po, rk=None, tag=''):
        nonlocal cash
        free = slots - len(pos)
        if free <= 0 or not cand:
            if cand:
                st['noslot'] += len(cand)
            return
        if rk is None:
            rk = rank[t - 1, cand]
        order = [cand[i] for i in np.lexsort((np.arange(len(cand)), -np.nan_to_num(rk, nan=-1e18)))]
        e50_liq = v50 * (1 - 0.5 * cm * etf_tick(po) / po) * (1 - fee - TAX_ETF * cm) - (minfee if v50 > 0 else 0.0)
        cash_s = cash
        e50_liq = max(e50_liq, 0.0)
        eq = eq_now(t)
        sv = max(eq - B, 0.0)
        u = sv / 10.0
        for j in order:
            target = u
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
            sh = int(budget / (px * (1 + fee) + (o * ox if t >= odd_t else 0.0)))
            if t < odd_t:
                sh = (sh // 1000) * 1000
            if sh <= 0:
                st['blk_lot' if budget >= 0.5 * target else 'blk_nocash'] += 1
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
                continue
            if min_frac > 0 and sh * o < min_frac * target:
                st['blk_min'] += 1
                continue
            cst = sh * px + f + oc
            if cash_s < cst:
                e50_liq -= (cst - max(cash_s, 0.0))
            cash -= cst; cash_s -= cst
            st['buy_val'] += sh * o
            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=t - 1, cost=cst, flag=False, why='', proc=0.0, new=True, brk=False, k=-1, tag=tag)
            free -= 1

    def open_window(j, k, t_sale):
        nx = srows[srows > k]
        end = int(nx[0]) - 1 if len(nx) else T - 1
        st['rb_flag'] += 1
        if t_sale <= end:
            watch[j] = dict(k=k, start=t_sale, end=end); st['rb_win'] += 1

    for t in range(t0 + 1, T):
        if v50 > 0:
            v50 *= P.E50O[t] / P.E50[t - 1]
        B *= P.E50O[t] / P.E50[t - 1]
        sold_today = set()
        for j in list(pos):
            p = pos[j]
            if P.MK[t, j] == 0 and not P.MK[t:, j].any():
                cash += p['v']
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + p['v'], 'delist', p['sh0'], p['tag'])); del pos[j]; sold_today.add(j); continue
            if not p['flag']:
                continue
            o = O[t, j]
            if not (np.isfinite(o) and o > 0):
                continue
            if o <= P.LIMDN[t, j] + 1e-9:
                continue
            vopen = p['v'] * O[t, j] / REF[t, j]
            frac = 1.0
            if cap_pct and np.isfinite(AMT[t, j]) and vopen > cap_pct * AMT[t, j]:
                frac = cap_pct * AMT[t, j] / vopen
            sh = int(round(p['sh'] * frac))
            sl = 0.5 * cm * float(tick(o)) / o + xs
            if ox > 0 and sh > 0 and t >= odd_t:
                sl += ox * (sh % 1000) / sh
            gross = vopen * frac * (1 - sl)
            f = _fee(max(sh, 1), gross / max(sh, 1), fee, minfee); tx = gross * tax
            cash += gross - f - tx; sold_today.add(j)
            if frac >= 1.0:
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + gross - f - tx, p['why'], p['sh0'], p['tag'])); del pos[j]
                if rebuy is not None and p['brk']:
                    open_window(j, p['k'], t)
            else:
                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0)
        if sat_w < 1.0 and t in exec_days:
            rebalance(t)
        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
        if len(cand):
            held = set(pos) | sold_today
            cand = [j for j in cand if j not in held]
        else:
            cand = []
        po_o = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
        if cand:
            buys(t, cand, po_o)
        if rebuy is not None and t in pend:
            rc = [(j, k) for j, k in pend.pop(t) if j not in pos and j not in sold_today]
            if rc:
                cj = [j for j, _ in rc]; rk = np.array([ROWS[k][0][j] for j, k in rc])
                buys(t, cj, po_o, rk=rk, tag='rebuy')
                for j, k in rc:
                    ok = j in pos and pos[j]['e'] == t
                    st['rb_buy' if ok else 'rb_fail'] += 1
                    rbev.append(dict(j=int(j), k=int(k), buy_t=int(t), ok=bool(ok)))
        e50_net(po_o)
        v50 *= P.E50[t] / P.E50O[t]
        B *= P.E50[t] / P.E50O[t]
        for j, p in pos.items():
            if p['new']:
                p['v'] = p['v'] * (C[t, j] / O[t, j]); p['new'] = False
            else:
                p['v'] *= (1 + R[t, j])
        tot = 0.0
        is_sig = bool(sigrows[t])
        for j, p in pos.items():
            tot += p['v']
            if not p['flag'] and X[t, j]:
                p['flag'] = True; p['why'] = 'rule' if is_sig else 'ma_daily'
                if is_sig and not FMA[t, j]:
                    p['brk'] = True; p['k'] = t
        # U2：收盤檢查觀察窗
        if rebuy is not None and watch:
            for j in list(watch):
                w = watch[j]
                if t > w['end']:
                    del watch[j]; continue
                if t < w['start']:
                    continue
                if not (univ[t, j] and FMA[t, j]):
                    continue
                s_k, fin_k = ROWS[w['k']]
                ok = fin_k & FMA[t]
                if not ok[j]:
                    continue
                top = np.flatnonzero(ok)[np.argsort(-s_k[ok], kind='stable')][:20]
                if j not in top:
                    continue
                if rebuy.get('r5') and not (G[t - 5, j] > 0 and G[t, j] / G[t - 5, j] - 1 > 0):
                    continue
                st['rb_trig'] += 1
                del watch[j]
                if t + 1 < T:
                    pend.setdefault(t + 1, []).append((j, w['k']))
        nav[t] = cash + v50 + tot
        stockfrac[t] = tot / nav[t]
        idle_now = cash + v50
        if B > idle_now:
            B = max(idle_now, 0.0)
        if B < 0:
            B = 0.0
    for j, p in pos.items():
        log.append((j, p['sig'], p['e'], T - 1, p['cost'], p['proc'] + p['v'], 'open_end', p['sh0'], p['tag']))
    navs = pd.Series(nav[t0:T], index=P.IDX[t0:T])
    lg = pd.DataFrame(log, columns=['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares', 'tag'])
    if len(lg):
        lg['code'] = P.COLS[lg['j'].to_numpy()]; lg['ret'] = lg['proceeds'] / lg['cost'] - 1; lg['pnl'] = lg['proceeds'] - lg['cost']
        lg['sig_date'] = P.IDX[lg['sig_t'].to_numpy()]; lg['ent_date'] = P.IDX[lg['ent_t'].to_numpy()]; lg['exit_date'] = P.IDX[lg['exit_t'].to_numpy()]
    st['stockfrac'] = pd.Series(stockfrac[t0:T], index=P.IDX[t0:T]); st['rb_events'] = rbev
    return navs, lg, st
