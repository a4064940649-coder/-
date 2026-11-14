#!/usr/bin/env python3
"""S1 共用程式庫：官方日 K 面板（含 0050 開盤）、三個候選的訊號、統一引擎、統計。
一律 python3 -I -B。只 import o_ext_canslim_xs/cslib.py（不改）；中間檔放 scratchpad/s1/。規格見 PREREG.json。"""
import sys, io, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S1 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s1')
sys.path.insert(0, str(AG / 'o_ext_canslim_xs'))
import cslib                                              # noqa: E402
cslib.TMP = S1                                            # 面板快取改放本題的 scratchpad
REV_CSV = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
START = pd.Timestamp('2015-07-01')
FEE, TAX, MINFEE, TAX_ETF = 0.001425 * 0.28, 0.003, 20.0, 0.001


# ---------------------------------------------------------------- 資料
def build_panel():
    """一次讀完官方日 K：個股面板（格式同 cslib.build_panel）＋ 0050 的開高低收。"""
    out, e50 = S1 / 'panel_cs.npz', S1 / 'e50_ohlc.csv'
    if out.exists() and e50.exists():
        return out
    S1.mkdir(parents=True, exist_ok=True)
    days, etf = [], []
    cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'change']
    for path, fh in cslib._git_blobs('data/universe/daily'):
        try:
            d = pd.read_csv(fh, dtype={'stock_id': str}, usecols=['date', 'stock_id', 'market'] + cols)
        except Exception:
            continue
        if not len(d):
            continue
        dt = d['date'].iloc[0]
        z = d[d['stock_id'] == '0050']
        if len(z):
            etf.append((dt, float(z['open'].iloc[0]), float(z['high'].iloc[0]), float(z['low'].iloc[0]), float(z['close'].iloc[0])))
        d = d[d['stock_id'].str.fullmatch(r'[1-9]\d{3}') & d['market'].isin(['twse', 'tpex'])].drop_duplicates('stock_id', keep='last')
        vals = d[cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        days.append((dt, d['stock_id'].to_numpy(), np.where(d['market'].to_numpy() == 'twse', 1, 2).astype(np.int8), vals))
    days.sort(key=lambda x: x[0])
    codes = np.array(sorted(set(np.concatenate([x[1] for x in days]))))
    ci = {c: j for j, c in enumerate(codes)}
    T, N = len(days), len(codes)
    A = {k: np.full((T, N), np.nan) for k in ['O', 'H', 'L', 'C', 'VOL', 'AMT', 'CHG']}
    MK = np.zeros((T, N), np.int8)
    for t, (dt, cs, mk, vals) in enumerate(days):
        j = np.array([ci[c] for c in cs])
        for k, col in zip(['O', 'H', 'L', 'C', 'VOL', 'AMT', 'CHG'], range(7)):
            A[k][t, j] = vals[:, col]
        MK[t, j] = mk
    idx = pd.to_datetime([x[0] for x in days]).values.astype('datetime64[ns]')
    np.savez_compressed(out, IDX=idx, COLS=codes.astype(str), MK=MK, **A)
    pd.DataFrame(etf, columns=['date', 'open', 'high', 'low', 'close']).sort_values('date').to_csv(e50, index=False)
    return out


def load():
    build_panel()
    P = cslib.Panel()
    e = pd.read_csv(S1 / 'e50_ohlc.csv', parse_dates=['date']).set_index('date').reindex(P.IDX)
    P.E50O_raw = e['open'].to_numpy()
    ratio = (e['open'] / e['close']).to_numpy()
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
    P.E50O = P.E50 * ratio                                 # 0050 還原開盤
    P.e50_open_missing = int((~np.isfinite(e['open'].to_numpy())).sum())
    P.not91 = np.array([not c.startswith('91') for c in P.COLS])
    amt = np.where(P.listed, np.nan_to_num(P.AMT), np.nan)
    P.V60 = pd.DataFrame(amt).rolling(60, min_periods=60).mean().to_numpy()
    P.liq = (P.V60 >= 1e7) & (P.C >= 10) & P.has & P.not91[None, :]
    P.t0 = int(np.searchsorted(P.IDX.values, np.datetime64(START), side='left'))
    return P


def universe(P, which):
    return P.liq & ((P.MK == 1) if which == 'twse' else (P.MK > 0))


def rmean(a, n):
    return pd.DataFrame(a).rolling(n, min_periods=n).mean().to_numpy()


def rmax(a, n):
    return pd.DataFrame(a).rolling(n, min_periods=n).max().to_numpy()


def rmin(a, n):
    return pd.DataFrame(a).rolling(n, min_periods=n).min().to_numpy()


# ---------------------------------------------------------------- 候選訊號（回傳 dict：sig, rank, X, SD, max_hold）
def sig_revsurge(P, univ):
    """C1：月營收驚奇 SURGE，前 10% 中最大的 10 檔，月換股。"""
    r = pd.read_csv(REV_CSV, dtype={'code': str}, parse_dates=['month'])
    r['pri'] = (r['market'] != 'sii').astype(int)
    r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
    Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
    Rv.index = pd.PeriodIndex(Rv.index, freq='M')
    Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')).reindex(columns=P.COLS)
    Rv = Rv.where(Rv > 0)
    D = Rv - Rv.shift(12)
    D[(D.index.year == 2013)] = np.nan                     # IFRS 斷點（照原題）
    mu = D.shift(1).rolling(24, min_periods=24).mean()
    sd = D.shift(1).rolling(24, min_periods=24).std(ddof=1)
    S = ((D - mu) / sd.where(sd > 0)).replace([np.inf, -np.inf], np.nan)
    av = cslib.rev_avail_days(P, list(S.index), True)
    sig = np.zeros((P.T, P.N), bool); X = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan)
    info = []
    for m in S.index:
        k = av[m]
        if k >= P.T or k < 61:
            continue
        s = S.loc[m].to_numpy()
        ok = univ[k] & np.isfinite(s)
        n = int(ok.sum())
        if n < 50:                                          # 2016-01 以前 SURGE 全缺 → 沒有訊號，也不強制出場
            continue
        nsel = max(int(math.ceil(n * 0.10)), 10)
        order = np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]
        top = order[:nsel]; tgt = order[:10]
        sig[k, tgt] = True; rank[k, tgt] = s[tgt]
        X[k, :] = True; X[k, tgt] = False                  # 不在新目標名單的持股隔日開盤賣
        info.append(dict(month=str(m), sig_date=str(P.IDX[k].date()), n_univ=n, n_top10pct=int(nsel), surge_cut_decile=float(s[top[-1]]), surge_10th=float(s[tgt[-1]])))
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info)


def sig_turtle_s2(P, univ):
    """C2：海龜 S2（55 日高突破、20 日低出場、2N 停損、不加碼）。"""
    G = P.G; ok = P.nbar >= 61
    adjH = P.HF * G / P.CF
    hi55 = cslib.shift(rmax(adjH, 55), 1)
    lo20 = cslib.shift(rmin(G, 20), 1)
    dG = np.abs(np.diff(G, axis=0, prepend=np.nan))
    N20 = rmean(dG, 20)
    sig = (G > hi55) & ok & P.has & univ
    X = (G < lo20) & P.has
    return dict(sig=sig, rank=P.V60, X=X, SD=2 * N20, max_hold=None, info=None)


def sig_granville(P, univ):
    """C3：葛蘭碧買點 2/3（上揚季線量縮回測不破），固定持有 20 日。"""
    G = P.G; ok = P.nbar >= 85
    ma = rmean(G, 60)
    c1 = ma > cslib.shift(ma, 20)
    c2 = rmax(G, 20) / ma >= 1.10
    above = (G >= ma)
    c3 = (G / ma - 1 >= 0) & (G / ma - 1 <= 0.03) & (pd.DataFrame(above.astype(float)).rolling(5, min_periods=5).min().to_numpy() == 1)
    c4 = P.VOLF < 0.7 * rmean(P.VOLF, 20)
    c5 = G > cslib.shift(G, 1)
    sig = c1 & c2 & c3 & c4 & c5 & ok & P.has & univ
    return dict(sig=sig, rank=P.V60, X=np.zeros((P.T, P.N), bool), SD=None, max_hold=20, info=None)


CANDS = {'C1_REVSURGE': sig_revsurge, 'C2_TURTLE_S2': sig_turtle_s2, 'C3_GRANVILLE_MA60': sig_granville}


# ---------------------------------------------------------------- 統一引擎
def _fee(shares, px, rate, minfee):
    if rate == 0:
        return 0.0
    rnd = (shares // 1000) * 1000; odd = shares - rnd; f = 0.0
    if rnd > 0:
        f += max(minfee, rnd * px * rate)
    if odd > 0:
        f += max(minfee, odd * px * rate)
    return f


def etf_tick(p):
    return 0.01 if p < 50 else 0.05


def simulate(P, S, univ, idle='cash', cmult=1.0, block=frozenset(), capital=1e6, slots=10, cap_pct=0.05, sweep_min=20000.0):
    """訊號日 t 盤後 → t+1 開盤成交。idle：'cash' 或 'e50'（空位放 0050，開盤價換手）。cmult：成本倍數（手續費率、最低手續費、稅、滑價）。
    block：封鎖的 (訊號日列號, 股票欄號) 集合（拿掉最賺 5 筆用）。回傳 nav（Series）、交易紀錄、統計。"""
    sig, rank, X, SD, max_hold = S['sig'], S['rank'], S['X'], S['SD'], S['max_hold']
    T, t0 = P.T, P.t0
    O, C, R, REF, AMT, G = P.O, P.CF, P.R, P.REF, P.AMT, P.G
    fee, tax, minfee = FEE * cmult, TAX * cmult, MINFEE * cmult
    cash = float(capital); v50 = 0.0; pos = {}
    nav = np.full(T, np.nan); nav[t0] = capital
    log = []
    st = dict(n_sig=0, n_buy=0, blk_limup=0, blk_noopen=0, blk_lot=0, blk_cap=0, blk_nocash=0, noslot=0, blk_dropped=0, sell_defer_limdn=0, sell_defer_noopen=0,
              sell_partial_cap=0, n_delist=0, buy_val=0.0, sell_val=0.0, fee=0.0, tax=0.0, slip=0.0, e50_cost=0.0, e50_orders=0, npos_sum=0, idle_frac_sum=0.0)
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))
    tick = cslib.tick
    for t in range(t0 + 1, T):
        if idle == 'e50' and v50 > 0:
            v50 *= P.E50O[t] / P.E50[t - 1]                 # 隔夜（收盤 → 開盤）
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
            sl = 0.5 * cmult * float(tick(o)) / o
            gross = vopen * frac * (1 - sl)
            sh = int(round(p['sh'] * frac))
            f = _fee(max(sh, 1), gross / max(sh, 1), fee, minfee); tx = gross * tax
            cash += gross - f - tx; sold_today.add(j)
            st['sell_val'] += vopen * frac; st['fee'] += f; st['tax'] += tx; st['slip'] += vopen * frac * sl
            if frac >= 1.0:
                log.append((j, p['sig'], p['e'], t, p['cost'], p['proc'] + gross - f - tx, p['why'], p['sh0'])); del pos[j]
            else:
                p['proc'] += gross - f - tx; p['v'] *= (1 - frac); p['sh'] = max(p['sh'] - sh, 0)
        # ---- 開盤：買
        cand = np.flatnonzero(sig[t - 1] & univ[t - 1])
        if len(cand):
            held = set(pos) | sold_today
            cand = [j for j in cand if j not in held]
            st['n_sig'] += len(cand)
            free = slots - len(pos)
            if free > 0 and cand:
                rk = rank[t - 1, cand]
                order = [cand[i] for i in np.lexsort((np.arange(len(cand)), -np.nan_to_num(rk, nan=-1e18)))]
                e50_liq = v50 * (1 - 0.5 * cmult * etf_tick(P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) else 100.0) / (P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) else 100.0)) * (1 - fee - TAX_ETF * cmult) - (minfee if v50 > 0 else 0.0)
                e50_liq = max(e50_liq, 0.0) if idle == 'e50' else 0.0
                eq = cash + v50 + sum(p['v'] * (O[t, k] / REF[t, k] if np.isfinite(O[t, k]) and O[t, k] > 0 else 1.0) for k, p in pos.items())
                target = eq / slots
                for j in order:
                    if free <= 0:
                        st['noslot'] += 1; continue
                    if (t - 1, j) in block:
                        st['blk_dropped'] += 1; continue
                    o = O[t, j]
                    if not (np.isfinite(o) and o > 0):
                        st['blk_noopen'] += 1; continue
                    if o >= P.LIMUP[t, j] - 1e-9:
                        st['blk_limup'] += 1; continue
                    sl = 0.5 * cmult * float(tick(o)) / o
                    px = o * (1 + sl)
                    avail = cash + e50_liq
                    budget = min(target, avail)
                    sh = int(budget / (px * (1 + fee)))
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
                    f = _fee(sh, px, fee, minfee)
                    while sh > 0 and sh * px + f > avail + 1e-6:
                        sh -= 1000 if t < odd_t else 1
                        f = _fee(sh, px, fee, minfee) if sh > 0 else 0.0
                    if sh <= 0:
                        st['blk_nocash'] += 1; continue
                    cst = sh * px + f
                    if cash < cst:                           # 動用 0050：先記帳，收尾時一次賣
                        e50_liq -= (cst - max(cash, 0.0))
                    cash -= cst
                    st['buy_val'] += sh * o; st['fee'] += f; st['slip'] += sh * o * sl; st['n_buy'] += 1
                    gE = G[t, j] * o / C[t, j]
                    pos[j] = dict(v=sh * o, sh=sh, sh0=sh, e=t, sig=t - 1, cost=cst, flag=False, why='', proc=0.0, new=True,
                                  stopG=(gE - SD[t - 1, j]) if SD is not None and np.isfinite(SD[t - 1, j]) else -np.inf)
                    free -= 1
        # ---- 開盤：0050 一次淨額換手
        if idle == 'e50':
            po = P.E50O_raw[t] if np.isfinite(P.E50O_raw[t]) and P.E50O_raw[t] > 0 else 100.0
            sl50 = 0.5 * cmult * etf_tick(po) / po
            if cash < -1e-6 and v50 > 0:
                need = -cash
                x = min(v50, (need + minfee) / ((1 - sl50) * (1 - fee - TAX_ETF * cmult)))
                for _ in range(3):
                    g = x * (1 - sl50); f = max(minfee, g * fee); net = g - f - g * TAX_ETF * cmult
                    if net >= need or x >= v50:
                        break
                    x = min(v50, x + (need - net) * 1.002)
                g = x * (1 - sl50); f = max(minfee, g * fee); net = g - f - g * TAX_ETF * cmult
                v50 -= x; cash += net; st['e50_cost'] += x - net; st['e50_orders'] += 1
            elif cash >= sweep_min:
                f = max(minfee, cash * fee)
                val = (cash - f) / (1 + sl50)
                st['e50_cost'] += cash - val; st['e50_orders'] += 1
                v50 += val; cash = 0.0
            v50 *= P.E50[t] / P.E50O[t]                    # 盤中（開盤 → 收盤）
        # ---- 收盤：估值與出場訊號
        tot = 0.0
        for j, p in pos.items():
            if p['new']:
                p['v'] = p['v'] * (C[t, j] / O[t, j]); p['new'] = False
            else:
                p['v'] *= (1 + R[t, j])
            tot += p['v']
            if not p['flag']:
                if X[t, j]:
                    p['flag'] = True; p['why'] = 'rule'
                elif P.has[t, j] and G[t, j] < p['stopG']:
                    p['flag'] = True; p['why'] = 'stop'
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
    nd = T - 1 - t0
    st['avg_npos'] = st.pop('npos_sum') / nd; st['avg_idle_frac'] = st.pop('idle_frac_sum') / nd
    return navs, lg, st


# ---------------------------------------------------------------- 統計
def cagr(nav, a=None, b=None):
    s = nav.loc[a:b].dropna()
    if len(s) < 2:
        return float('nan')
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)


def maxdd(nav, a=None, b=None):
    s = nav.loc[a:b].dropna()
    return float((s / s.cummax() - 1).min())


def sharpe(nav, a=None, b=None):
    r = nav.loc[a:b].dropna().pct_change().dropna()
    return float(r.mean() / r.std(ddof=1) * math.sqrt(252)) if len(r) > 2 and r.std(ddof=1) > 0 else float('nan')


def monthly_excess_t(nav, bench, a=None, b=None):
    s = nav.loc[a:b].dropna(); q = bench.reindex(s.index)
    m1 = s.groupby(s.index.to_period('M')).last(); m2 = q.groupby(q.index.to_period('M')).last()
    x = (m1.pct_change() - m2.pct_change()).dropna()
    if len(x) < 3:
        return float('nan'), 0, float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), int(len(x)), float(x.mean())


def window(nav, bench, a, b):
    q = bench.reindex(nav.index)
    t, n, mx = monthly_excess_t(nav, q, a, b)
    c, cb = cagr(nav, a, b), cagr(q, a, b)
    return dict(start=str(nav.loc[a:b].index[0].date()), end=str(nav.loc[a:b].index[-1].date()), cagr=c, cagr_0050=cb, diff=c - cb, maxdd=maxdd(nav, a, b), maxdd_0050=maxdd(q, a, b),
                sharpe=sharpe(nav, a, b), sharpe_0050=sharpe(q, a, b), t_month_excess=t, n_months=n, mean_month_excess=mx)


def yearly(nav, bench):
    q = bench.reindex(nav.index)
    y1 = nav.groupby(nav.index.year).last(); y2 = q.groupby(q.index.year).last()
    f1 = nav.groupby(nav.index.year).first(); f2 = q.groupby(q.index.year).first()
    out = {}
    prev1, prev2 = None, None
    for y in y1.index:
        b1 = prev1 if prev1 is not None else f1[y]; b2 = prev2 if prev2 is not None else f2[y]
        out[int(y)] = dict(strat=float(y1[y] / b1 - 1), e0050=float(y2[y] / b2 - 1))
        prev1, prev2 = y1[y], y2[y]
    return out


def trade_stats(lg, a=None):
    """已平倉交易（不含期末未平倉）；a：只算進場日 ≥ a 的。"""
    x = lg[lg['why'] != 'open_end']
    if a is not None:
        x = x[x['ent_date'] >= pd.Timestamp(a)]
    n = len(x)
    if n == 0:
        return dict(n=0)
    w = x[x['ret'] > 0]['ret']; l = x[x['ret'] <= 0]['ret']
    p = len(w) / n; aw = float(w.mean()) if len(w) else 0.0; al = float(-l.mean()) if len(l) else 0.0
    yrs = (x['ent_date'].max() - x['ent_date'].min()).days / 365.25
    return dict(n=int(n), per_year=float(n / yrs) if yrs > 0 else float('nan'), win_rate=float(p), avg_win=aw, avg_loss=al, expectancy=float(p * aw - (1 - p) * al),
                mean_ret=float(x['ret'].mean()), median_ret=float(x['ret'].median()), avg_hold_days=float(x['hold_days'].mean()), median_hold_days=float(x['hold_days'].median()),
                share_tpex=float((x['market'] == 'tpex').mean()), why=x['why'].value_counts().to_dict())
