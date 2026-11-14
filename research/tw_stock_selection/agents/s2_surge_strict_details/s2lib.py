#!/usr/bin/env python3
"""S2 共用程式庫：SURGE／YoY3 訊號（可加濾網、位移、前 N 檔、固定持有）、擴充引擎（額外滑價、零股加價、收盤買進、整段持有期禁買）、統計與三個診斷。
一律 python3 -I -B。只 import s1_shortterm_leaderboard/stlib.py 與 o_ext_canslim_xs/cslib.py（不改）；中間檔放 scratchpad/s2/。規格見 PREREG.json。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S2 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s2')
sys.path.insert(0, str(AG / 'o_ext_canslim_xs'))
sys.path.insert(0, str(AG / 's1_shortterm_leaderboard'))
import cslib, stlib                                        # noqa: E402
stlib.S1 = S2; cslib.TMP = S2                              # 面板快取放本題的 scratchpad（不動 s1 的）
FEE, TAX, MINFEE, TAX_ETF = stlib.FEE, stlib.TAX, stlib.MINFEE, stlib.TAX_ETF

COSTS = {                                                  # cmult：全部成本項的倍數；xslip：每邊額外滑價；oddx：不足一張部分每邊再加
    'light':     dict(cmult=1.0, xslip=0.0,   oddx=0.0),
    'strict':    dict(cmult=1.0, xslip=0.003, oddx=0.0),
    'oddlot':    dict(cmult=1.0, xslip=0.003, oddx=0.003),
    'stress05':  dict(cmult=1.0, xslip=0.005, oddx=0.0),
    'strict_x2': dict(cmult=2.0, xslip=0.003, oddx=0.0),
    'zero':      dict(cmult=0.0, xslip=0.0,   oddx=0.0),
}


# ---------------------------------------------------------------- 資料
def load():
    P = stlib.load()
    e = pd.read_csv(S2 / 'e50_ohlc.csv', parse_dates=['date']).set_index('date').reindex(P.IDX)
    P.E50C_raw = e['close'].to_numpy()
    P.adjH = P.HF * P.G / P.CF
    P.MA60 = stlib.rmean(P.G, 60)
    P.HI240 = np.where(P.nbar >= 240, stlib.rmax(P.adjH, 240), np.nan)
    P.F_MA60 = P.G > P.MA60                                 # D1：還原收盤 > 60 日均線
    with np.errstate(invalid='ignore', divide='ignore'):
        P.NEARHI = P.G / P.HI240
    P.F_HI = P.NEARHI >= 0.90                               # D2：還原收盤 ÷ 240 日還原最高價 ≥ 0.90
    return P


def rev_tables(P):
    """月營收 → SURGE、yoy、YoY3（index＝月份 Period，columns＝P.COLS）與各月份的訊號日列號。"""
    r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
    r['pri'] = (r['market'] != 'sii').astype(int)
    r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
    Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
    Rv.index = pd.PeriodIndex(Rv.index, freq='M')
    Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')).reindex(columns=P.COLS)
    Rv = Rv.where(Rv > 0)
    D = Rv - Rv.shift(12)
    D[(D.index.year == 2013)] = np.nan
    mu = D.shift(1).rolling(24, min_periods=24).mean()
    sd = D.shift(1).rolling(24, min_periods=24).std(ddof=1)
    S = ((D - mu) / sd.where(sd > 0)).replace([np.inf, -np.inf], np.nan)
    yoy = (Rv / Rv.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
    yoy3 = yoy.rolling(3, min_periods=3).mean()
    av = cslib.rev_avail_days(P, list(S.index), True)
    return dict(S=S, YOY=yoy, YOY3=yoy3, av=av)


def make_signal(P, univ, RT, kind='surge', nsel=10, filt=None, shift=0, max_hold=None):
    """每月一次。kind='surge'：SURGE 最大的 nsel 檔；kind='yoy3'：YoY3 > 20% 中 YoY3 最大的 nsel 檔。
    filt：布林面板（訊號日要成立才列入排名）。shift：訊號日往後移幾個交易日（股池與 filt 在位移後那天判斷）。
    max_hold：固定持有天數（有值時換股日不賣）。有訊號的月份一律以 B0 的條件（未位移訊號日股池內 SURGE 有值 ≥ 50 檔）決定。"""
    S, av = RT['S'], RT['av']
    sig = np.zeros((P.T, P.N), bool); X = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan)
    info = []
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
        if kind == 'surge':
            score = s; ok = univ[k] & np.isfinite(score)
        else:
            score = RT['YOY3'].loc[m].to_numpy(); ok = univ[k] & np.isfinite(score) & (score > 0.20)
        if filt is not None:
            ok = ok & filt[k]
        order = np.flatnonzero(ok)[np.argsort(-score[ok], kind='stable')]
        tgt = order[:nsel]
        sig[k, tgt] = True; rank[k, tgt] = score[tgt]
        if max_hold is None:
            X[k, :] = True; X[k, tgt] = False
        info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ok=int(ok.sum()), targets=[str(P.COLS[j]) for j in tgt],
                         scores=[round(float(score[j]), 4) for j in tgt]))
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=max_hold, info=info)


# ---------------------------------------------------------------- 引擎（stlib.simulate 的擴充）
def simulate(P, S, univ, idle='cash', cost=None, block=None, capital=1e6, slots=10, cap_pct=0.05, sweep_min=20000.0, buy_at='open', e50_fix=False):
    """訊號日 t−1 盤後 → t 成交。賣出一律在 t 開盤；買進在 t 開盤（buy_at='open'）或 t 收盤（'close'）。
    cost：COSTS 的一項。block：{股票欄號: [(a, b), …]}，列號 a ≤ t ≤ b 的日子不得買進。回傳 nav、交易紀錄、統計（格式同 stlib）。
    e50_fix（事後修正，PREREG 沒寫；只影響 idle='e50'）：stlib 在現金轉負（已動用 0050）之後，可用資金＝現金＋0050 可變現額，
    會把已動用的金額扣兩次，導致買不滿。True＝可用資金改為 max(現金, 0)＋0050 剩餘可變現額；False＝與 stlib 完全相同。"""
    cost = cost or COSTS['light']; block = block or {}
    cm = cost['cmult']; xs = cost['xslip'] * cm; ox = cost['oddx'] * cm
    sig, rank, X, max_hold = S['sig'], S['rank'], S['X'], S['max_hold']
    T, t0 = P.T, P.t0
    O, C, R, REF, AMT, G, CR = P.O, P.CF, P.R, P.REF, P.AMT, P.G, P.C
    fee, tax, minfee = FEE * cm, TAX * cm, MINFEE * cm
    tick = cslib.tick; _fee = stlib._fee; etf_tick = stlib.etf_tick
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))
    cash = float(capital); v50 = 0.0; pos = {}
    nav = np.full(T, np.nan); nav[t0] = capital
    log = []
    st = dict(n_sig=0, n_buy=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, noslot=0, blk_dropped=0, sell_defer_limdn=0, sell_defer_noopen=0,
              sell_partial_cap=0, n_delist=0, buy_val=0.0, sell_val=0.0, fee=0.0, tax=0.0, slip=0.0, e50_cost=0.0, e50_orders=0, npos_sum=0, idle_frac_sum=0.0,
              odd_buys=0, odd_buy_val=0.0)

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

    def buys(t, cand, at_close, po):
        nonlocal cash
        free = slots - len(pos)
        if free <= 0 or not cand:
            return
        rk = rank[t - 1, cand]
        order = [cand[i] for i in np.lexsort((np.arange(len(cand)), -np.nan_to_num(rk, nan=-1e18)))]
        e50_liq = v50 * (1 - 0.5 * cm * etf_tick(po) / po) * (1 - fee - TAX_ETF * cm) - (minfee if v50 > 0 else 0.0)
        e50_liq = max(e50_liq, 0.0) if idle == 'e50' else 0.0
        if at_close:
            eq = cash + v50 + sum(p['v'] for p in pos.values())
        else:
            eq = cash + v50 + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())
        target = eq / slots
        for j in order:
            if free <= 0:
                st['noslot'] += 1; continue
            if banned(j, t):
                st['blk_dropped'] += 1; continue
            o = CR[t, j] if at_close else O[t, j]
            if not (np.isfinite(o) and o > 0):
                st['blk_noopen'] += 1; continue
            if o >= P.LIMUP[t, j] - 1e-9:
                st['blk_limup'] += 1; continue
            sl = 0.5 * cm * float(tick(o)) / o + xs
            px = o * (1 + sl)
            avail = (max(cash, 0.0) + e50_liq) if e50_fix else (cash + e50_liq)
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
            cst = sh * px + f + oc
            if cash < cst:
                e50_liq -= (cst - max(cash, 0.0))
            cash -= cst
            st['buy_val'] += sh * o; st['fee'] += f; st['slip'] += sh * o * sl + oc; st['n_buy'] += 1
            if sh % 1000:
                st['odd_buys'] += 1; st['odd_buy_val'] += (sh % 1000) * o
            pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=t - 1, cost=cst, flag=False, why='', proc=0.0, new=not at_close)
            free -= 1

    for t in range(t0 + 1, T):
        if idle == 'e50' and v50 > 0:
            v50 *= P.E50O[t] / P.E50[t - 1]
        sold_today = set()
        # ---- 開盤：賣
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
        # ---- 候選
        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
        if len(cand):
            held = set(pos) | sold_today
            cand = [j for j in cand if j not in held]
            st['n_sig'] += len(cand)
        else:
            cand = []
        po_o = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
        close_day = (buy_at == 'close') and len(cand) > 0
        # ---- 開盤：買
        if buy_at == 'open' and cand:
            buys(t, cand, False, po_o)
        # ---- 0050：開盤一次淨額換手（收盤買進日改在收盤）＋盤中
        if idle == 'e50':
            if not close_day:
                e50_net(po_o)
            v50 *= P.E50[t] / P.E50O[t]
        # ---- 收盤：估值
        for j, p in pos.items():
            if p['new']:
                p['v'] = p['v'] * (C[t, j] / O[t, j]); p['new'] = False
            else:
                p['v'] *= (1 + R[t, j])
        # ---- 收盤：買（D3／D6）
        if close_day:
            po_c = P.E50C_raw[t] if np.isfinite(P.E50C_raw[t]) and P.E50C_raw[t] > 0 else 100.0
            buys(t, cand, True, po_c)
            if idle == 'e50':
                e50_net(po_c)
        # ---- 收盤：出場訊號
        tot = 0.0
        for j, p in pos.items():
            tot += p['v']
            if not p['flag']:
                if X[t, j]:
                    p['flag'] = True; p['why'] = 'rule'
                elif max_hold is not None and t - p['e'] >= max_hold - 1:
                    p['flag'] = True; p['why'] = 'time'
        nav[t] = cash + v50 + tot
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
    return navs, lg, st


def block_from(lg, n):
    """基準跑已平倉交易中損益金額最大的 n 筆 → {j: [(進場列號, 出場列號)]}（整段持有期禁買）。"""
    x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False).head(n)
    out = {}
    for r in x.itertuples():
        out.setdefault(int(r.j), []).append((int(r.ent_t), int(r.exit_t)))
    return out, x


# ---------------------------------------------------------------- 統計
def monthly_ret(nav):
    m = nav.groupby(nav.index.to_period('M')).last()
    return m.pct_change().dropna()


def paired(nav, base, a, seed=20261114, nboot=10000, blk=6):
    """月報酬配對差（nav − base），視窗＝月份 ≥ a。回傳平均、t、區塊 bootstrap 95% 區間。"""
    d = (monthly_ret(nav) - monthly_ret(base)).dropna()
    d = d[d.index >= pd.Period(a, 'M')]
    x = d.to_numpy(); n = len(x)
    mean = float(x.mean()); sd = float(x.std(ddof=1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float('nan')
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / blk))
    starts = rng.integers(0, n, size=(nboot, nb))
    idx = ((starts[:, :, None] + np.arange(blk)[None, None, :]) % n).reshape(nboot, -1)[:, :n]
    bm = x[idx].mean(axis=1)
    return dict(n_months=n, mean=mean, t=t, ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], frac_pos=float((x > 0).mean()))


def univ_ew_nav(P, univ, sig_days):
    """股池等權：每個訊號日 k 以 univ[k] 全部股票等權，k+1 起收盤對收盤含息報酬，月內不再平衡，不扣成本。第一個訊號日前為現金。"""
    nav = np.full(P.T, np.nan); nav[P.t0] = 1.0
    v = 1.0; w = None; ks = set(sig_days)
    for t in range(P.t0 + 1, P.T):
        if w is not None:
            w = w * (1 + P.R[t, mem]); v = float(w.sum())
        nav[t] = v
        if t in ks:
            mem = np.flatnonzero(univ[t]); w = np.full(len(mem), v / len(mem))
    return pd.Series(nav[P.t0:], index=P.IDX[P.t0:])


def trade_paths(P, lg):
    """每筆交易的每日（不含成本）持有報酬：第 1 日＝開盤買到當日收盤（收盤買進的格子不適用）；之後含息日報酬；出場日開盤的隔夜段算在出場那一日。"""
    out = []
    for r in lg.itertuples():
        j, e, x = int(r.j), int(r.ent_t), int(r.exit_t)
        d = [P.CF[e, j] / P.O[e, j] - 1]
        last = x if r.why == 'open_end' else x - 1
        d += [P.R[t, j] for t in range(e + 1, last + 1)]
        if r.why not in ('open_end', 'delist') and np.isfinite(P.O[x, j]) and P.O[x, j] > 0:
            d.append(P.O[x, j] / P.REF[x, j] - 1)
        out.append(np.array(d, float))
    return out


def path_segments(paths, segs=((1, 5), (6, 10), (11, 20), (21, 10**6))):
    res = []
    for a, b in segs:
        rr = []; dsum = 0.0; dn = 0
        for d in paths:
            s = d[a - 1:b]
            if len(s):
                rr.append(float(np.prod(1 + s) - 1)); dsum += float(s.sum()); dn += len(s)
        rr = np.array(rr)
        res.append(dict(seg=f'{a}-{b if b < 10**6 else "end"}', n=int(len(rr)), mean=float(rr.mean()), median=float(np.median(rr)), win=float((rr > 0).mean()),
                        mean_per_day=dsum / dn if dn else float('nan'), trade_days=dn, sum_ret=float(rr.sum())))
    return res
