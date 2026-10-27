#!/usr/bin/env python3
"""exit_rules 共用：rlib.simulate 的擴充版事件式日頻引擎（子帳戶、停利、停損、移動停損、時間停損、大盤保護、0050 欄），
以及 D 題三條事件型規則（週反轉、高量溢酬、SURGE）的選股重建。
資料、股池、分數、漲停判定、成本、價格陣列全部 import main_refine/rlib.py → revenue_longhistory/lhlib.py（不改原檔）。"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'main_refine'))
sys.dont_write_bytecode = True
import rlib as R  # noqa: E402
L = R.L
from btlib import tstat  # noqa: E402

IDX, COLS, N = L.IDX, L.COLS, L.N
J50 = N
E50f = L.E50.ffill()
AC2 = np.hstack([L.AC, E50f.to_numpy()[:, None]])
AO2 = np.hstack([L.AO, E50f.shift(1).bfill().to_numpy()[:, None]])
OK_OPEN2 = np.hstack([np.isfinite(L.OPEN_RAW) & np.isfinite(L.ADJ_RAW), np.ones((len(IDX), 1), bool)])
ORF2 = np.hstack([np.isfinite(L.OPEN_RAW), np.ones((len(IDX), 1), bool)])
GU2 = np.hstack([L.GAP_UP_OPEN_ERA, np.zeros((len(IDX), 1), bool)])
LIMDN2 = np.hstack([R.LIMDN, np.zeros((len(IDX), 1), bool)])

# ---------------- 大盤保護觸發 ----------------
_tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
TAIEX = _tx.reindex(IDX.union(_tx.index)).ffill().reindex(IDX)
MKT_R1 = TAIEX / TAIEX.shift(1) - 1
MKT_R5 = TAIEX / TAIEX.shift(5) - 1
MKT_TRIG = ((MKT_R1 <= -0.04) | (MKT_R5 <= -0.08)).to_numpy()

RULES = {
    'TP20': dict(tp=0.20), 'TP30': dict(tp=0.30), 'TP50': dict(tp=0.50), 'TP100': dict(tp=1.00),
    'TR10': dict(tr=0.10), 'TR15': dict(tr=0.15), 'TR20': dict(tr=0.20), 'TR25': dict(tr=0.25),
    'TP30_SL15': dict(tp=0.30, sl=0.15), 'TP50_SL20': dict(tp=0.50, sl=0.20),
    'TIME10': dict(time=10), 'MKT': dict(mkt=True),
}


def simulate(items, K, buy, sell, t0, t1, rule=None, mode='cash'):
    """items: list of dict(t=進場日, s=子帳戶, order=排名陣列, n=檔數, backfill=漲停是否遞補, gu_held=已持有者是否豁免漲停, x=子帳戶出場日或 None)。
    rule: dict（tp／sl／tr／time／mkt），None＝無出場規則。mode: 'cash'／'next'。
    回傳 (nav Series（index 從 t0−1，起始 1.0）, info)。K=1、rule=None 或只有 sl 時與 rlib.simulate 相同。"""
    rule = rule or {}
    tp, sl, tr, tm, mk = rule.get('tp'), rule.get('sl'), rule.get('tr'), rule.get('time'), rule.get('mkt', False)
    has_rule = bool(rule)
    acts = {}
    for it in items:
        if t0 <= it['t'] <= t1: acts.setdefault(it['t'], {})[it['s']] = ('enter', it)
    for it in items:
        x = it.get('x')
        if x is not None and t0 <= x <= t1 and it['s'] not in acts.get(x, {}):
            acts.setdefault(x, {})[it['s']] = ('exit', None)
    NN = N + 1
    u = np.zeros((K, NN)); cash = np.full(K, 1.0 / K)
    entry = np.full((K, NN), np.nan); peak = np.full((K, NN), np.nan); hd = np.zeros((K, NN), int); opened = np.full((K, NN), -1)
    last_order = [None] * K; stopped = [set() for _ in range(K)]; pending = [set() for _ in range(K)]
    nav = np.empty(t1 - t0 + 2); nav[0] = 1.0
    trig_t, sell_t, repl_t, holds = [], [], [], []   # holds: (sell_t, days)
    tov = []; n_reb = 0

    def close_pos(s, j, t):
        if opened[s, j] >= 0: holds.append((t, t - opened[s, j]))
        opened[s, j] = -1

    for t in range(t0, t1 + 1):
        A = acts.get(t, {})
        for s in range(K):
            a = A.get(s)
            if a is not None and a[0] == 'enter':
                it = a[1]; order = it['order']; n = it['n']
                heldmask = u[s] > 0
                if len(order) == 0:
                    pk = [J50]
                elif it['backfill']:
                    pk = []
                    for j in order:
                        if GU2[t, j] and not (it['gu_held'] and heldmask[j]): continue
                        pk.append(int(j))
                        if len(pk) == n: break
                else:
                    pk = [int(j) for j in order[:n] if not (GU2[t, j] and not (it['gu_held'] and heldmask[j]))]
                Pz = AO2[t]
                h = np.where(u[s] > 0, u[s] * np.nan_to_num(Pz), 0.0)
                Vt = h.sum() + cash[s]
                if len(pk) == 0:
                    for j in np.where(u[s] > 0)[0]: close_pos(s, j, t)
                    cash[s] = Vt - sell * h.sum(); u[s] = 0; entry[s] = np.nan; peak[s] = np.nan; hd[s] = 0
                    last_order[s] = None; stopped[s] = set(); pending[s] = set(); continue
                pk = np.array(pk, dtype=int)
                tgt = np.zeros(NN); w = 1.0 / len(pk); Vn = Vt
                for _ in range(6):
                    tgt[:] = 0; tgt[pk] = w * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                    Vn = Vt - cost
                tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0); n_reb += 1
                keep = np.zeros(NN, bool)
                for j in pk:
                    keep[j] = (u[s, j] > 0 and j not in pending[s] and np.isfinite(entry[s, j]))
                for j in np.where(u[s] > 0)[0]:
                    if not keep[j]: close_pos(s, j, t)
                ne = np.full(NN, np.nan); npk = np.full(NN, np.nan); nh = np.zeros(NN, int); no = np.full(NN, -1)
                for j in pk:
                    if keep[j]: ne[j], npk[j], nh[j], no[j] = entry[s, j], peak[s, j], hd[s, j], opened[s, j]
                    else: ne[j], npk[j], nh[j], no[j] = AO2[t, j], np.nan, 0, t
                entry[s], peak[s], hd[s], opened[s] = ne, npk, nh, no
                u[s] = 0; u[s][pk] = tgt[pk] / Pz[pk]; cash[s] = 0.0
                last_order[s] = order; stopped[s] = set(); pending[s] = set()
            elif a is not None and a[0] == 'exit':
                h = np.where(u[s] > 0, u[s] * np.nan_to_num(AO2[t]), 0.0)
                for j in np.where(u[s] > 0)[0]: close_pos(s, j, t)
                cash[s] = h.sum() + cash[s] - sell * h.sum(); u[s] = 0; entry[s] = np.nan; peak[s] = np.nan; hd[s] = 0
                last_order[s] = None; stopped[s] = set(); pending[s] = set()
            elif pending[s]:
                done = []
                for j in sorted(pending[s]):
                    if u[s, j] <= 0: done.append(j); continue
                    if not ORF2[t, j] or LIMDN2[t, j]: continue
                    proceeds = u[s, j] * AO2[t, j] * (1 - sell)
                    close_pos(s, j, t)
                    u[s, j] = 0.0; entry[s, j] = np.nan; peak[s, j] = np.nan; hd[s, j] = 0; done.append(j); sell_t.append(t)
                    rep = None
                    if mode == 'next' and last_order[s] is not None:
                        for j2 in last_order[s]:
                            if u[s, j2] > 0 or j2 in stopped[s]: continue
                            if not OK_OPEN2[t, j2] or GU2[t, j2]: continue
                            rep = int(j2); break
                    if rep is not None:
                        u[s, rep] = proceeds / (1 + buy) / AO2[t, rep]
                        entry[s, rep] = AO2[t, rep]; peak[s, rep] = np.nan; hd[s, rep] = 0; opened[s, rep] = t; repl_t.append(t)
                    else:
                        cash[s] += proceeds
                for j in done: pending[s].discard(j)
        nav[t - t0 + 1] = np.nansum(u * AC2[t]) + cash.sum()
        for s in range(K):
            hj = np.where(u[s] > 0)[0]
            if len(hj) == 0: continue
            hd[s, hj] += 1
            peak[s, hj] = np.fmax(peak[s, hj], AC2[t, hj])
            if not has_rule: continue
            hj = hj[hj != J50]
            if len(hj) == 0: continue
            c = AC2[t, hj]; e = entry[s, hj]
            trg = np.zeros(len(hj), bool)
            if tp is not None: trg |= c >= e * (1 + tp)
            if sl is not None: trg |= c <= e * (1 - sl)
            if tr is not None: trg |= c <= peak[s, hj] * (1 - tr)
            if tm is not None: trg |= (hd[s, hj] == tm) & (c < e)
            if mk and MKT_TRIG[t]: trg[:] = True
            for j in hj[trg]:
                if j in pending[s]: continue
                pending[s].add(int(j)); stopped[s].add(int(j)); trig_t.append(t)
    sN = pd.Series(nav, index=IDX[t0 - 1:t1 + 1])
    info = {'n_rebalances': n_reb, 'trig_t': np.array(trig_t, int), 'sell_t': np.array(sell_t, int), 'repl_t': np.array(repl_t, int),
            'holds': np.array(holds, int).reshape(-1, 2), 'turnover_per_rebalance': round(float(np.mean(tov[1:])), 3) if len(tov) > 1 else None}
    return sN, info


def counts(info, a, b):
    """區段內觸發次數、實際賣出、替補、平均持有天數（以賣出日落在區段內的持股計）。"""
    ia = int(np.searchsorted(IDX, pd.Timestamp(a))); ib = int(np.searchsorted(IDX, pd.Timestamp(b), side='right')) - 1
    f = lambda arr: int(((arr >= ia) & (arr <= ib)).sum())
    H = info['holds']; m = (H[:, 0] >= ia) & (H[:, 0] <= ib) if len(H) else np.zeros(0, bool)
    yrs = (IDX[min(ib, len(IDX) - 1)] - IDX[ia]).days / 365.25
    return {'triggers': f(info['trig_t']), 'sells': f(info['sell_t']), 'replacements': f(info['repl_t']),
            'triggers_per_year': round(f(info['trig_t']) / yrs, 1) if yrs > 0 else None,
            'avg_hold_days': round(float(H[m, 1].mean()), 1) if m.sum() else None, 'n_closed_positions': int(m.sum())}


# ---------------- 統計 ----------------
mret, paired, yearly = R.mret, R.paired, R.yearly

def seg_stats(nav, a, b, ew=None):
    o = R.seg_stats(nav, a, b)
    if o is None: return None
    if ew is not None:
        m = mret(nav); m = m[(m.index >= pd.Period(a, 'M')) & (m.index <= pd.Period(b, 'M'))]
        e = mret(ew.reindex(nav.index).ffill()).reindex(m.index); d = (m - e).dropna()
        o['t_vs_EW_m'] = tstat(d); o['excess_vs_EW_pct_m'] = round(float(d.mean() * 100), 3)
        r = ew.reindex(nav.index).ffill().pct_change(); r = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))].dropna()
        prev = nav.index[nav.index.get_loc(r.index[0]) - 1]; span = (r.index[-1] - prev).days / 365.25
        o['EW_CAGR'] = round(float((1 + r).prod() ** (1 / span) - 1), 4)
    e5 = L.E50.reindex(nav.index); r5 = e5.pct_change(); r5 = r5[(r5.index >= pd.Timestamp(a)) & (r5.index <= pd.Timestamp(b))].dropna()
    prev = nav.index[nav.index.get_loc(r5.index[0]) - 1]; span = (r5.index[-1] - prev).days / 365.25
    o['0050_CAGR'] = round(float((1 + r5).prod() ** (1 / span) - 1), 4)
    return o


def bear_windows():
    S = {'2008': ('2008-01-01', '2008-12-31'), '2011': ('2011-01-01', '2011-12-31'), '2015': ('2015-01-01', '2015-12-31'),
         '2018': ('2018-01-01', '2018-12-31'), '2020-03': ('2020-01-01', '2020-04-30'), '2022': ('2022-01-01', '2022-12-31')}
    out = {}
    for k, (a, b) in S.items():
        x = _tx[(_tx.index >= a) & (_tx.index <= b)]; lo = x.idxmin(); hi = x[x.index <= lo].idxmax()
        out[k] = (str(hi.date()), str(lo.date()), round(float(x[lo] / x[hi] - 1), 4))
    return out
BEARS = bear_windows()

def bear_rets(nav):
    o = {}
    for k, (a, b, _) in BEARS.items():
        s = nav.reindex(IDX).ffill()
        ia, ib = pd.Timestamp(a), pd.Timestamp(b)
        if ia < nav.index[0] or ib > nav.index[-1]: o[k] = None; continue
        va = s[s.index <= ia].iloc[-1]; vb = s[s.index <= ib].iloc[-1]
        o[k] = round(float(vb / va - 1) * 100, 1)
    return o


# 引擎檢查用：value.parquet 修正前的原檔（D 三條規則的原資料夾淨值是用它算的）
VALUE_ORIG = L.P / 'value_orig_dupbug.parquet'


# =============== 主策略選股計畫 ===============
def main_items(n):
    Y = L.yoy_long(); S = L.Setup(Y, start='2005-01-01'); F = R.Frames(S.dates, Y)
    pl = F.plan(F.yoy3)
    items = [dict(t=ex, s=0, order=cand, n=n, backfill=True, gu_held=True, x=None) for ex, cand in pl]
    return items, pl


# =============== D1 週反轉（照 w3_weekly_reversal/run.py 重建） ===============
def weekly_reversal_items(vfile=None):
    P = L.P
    D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag']}
    if vfile is not None: D['value'] = pd.read_parquet(vfile).reindex(index=D['adj'].index, columns=D['adj'].columns)
    cols = list(COLS); keep = np.array([not c.startswith('91') for c in cols])
    good = D['adj'].notna().sum(axis=1) >= 300
    G = {k: v.loc[good] for k, v in D.items()}
    IG = G['adj'].index; TG = len(IG)
    F = G['flag']; bad5 = F == 5
    ADJ = G['adj'].mask(bad5); CR = G['close_raw'].mask(bad5); HI = G['high'].mask(bad5); V = G['value']; RT = G['ret'].mask(bad5)
    v0 = V.fillna(0).where(V.notna().cumsum() > 0)
    V60 = v0.rolling(60, min_periods=60).mean()
    R5 = ADJ / ADJ.shift(5) - 1
    lim = np.where(IG < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
    LOCKUP = (RT.to_numpy() >= lim - 0.006) & (CR.to_numpy() >= HI.to_numpy() * 0.999)
    LOCKDN = RT.to_numpy() <= -(lim - 0.006)
    mkt4 = (F == 4).sum(axis=1) >= 50
    FL = (F.isin([3, 4, 5]) & ~((F == 4) & mkt4.to_numpy()[:, None])).astype(float)
    FLW = FL.rolling(6, min_periods=1).max().to_numpy() > 0
    wk = IG.to_period('W-SUN')
    last_of_week = pd.Series(np.arange(TG), index=IG).groupby(wk).max().to_numpy()
    SIG = [i for i in last_of_week if IG[i] >= pd.Timestamp('2005-01-01') and i + 1 < TG]
    OPEN_OK = np.isfinite(G['open'].to_numpy()) & np.isfinite(ADJ.to_numpy())
    LPOS = {d: k for k, d in enumerate(IDX)}
    ADJn, CRn, V60n, R5n = ADJ.to_numpy(), CR.to_numpy(), V60.to_numpy(), R5.to_numpy()
    BASE = (V60n >= 1e7) & (CRn >= 10) & np.isfinite(ADJn) & keep[None, :]
    POOL = BASE & np.isfinite(R5n) & ~LOCKUP & ~LOCKDN & ~FLW
    items = []
    for k, i in enumerate(SIG):
        ex = i + 1; sc = R5n[i].copy()
        ok = POOL[i] & OPEN_OK[ex] & np.isfinite(sc)
        if ok.sum() < 100: continue
        n = int(round(0.1 * ok.sum()))
        order = np.where(ok, sc, np.inf); cand = np.argsort(order, kind='stable')[:int(ok.sum())]
        items.append(dict(t=LPOS[IG[ex]], s=0, order=cand, n=n, backfill=True, gu_held=True, x=None))
    return items, 1


# =============== D2 高量溢酬 GKM 週版（照 w3_high_volume_premium/run.py 重建） ===============
def high_volume_items(vfile=None):
    P = L.P
    D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'value', 'flag']}
    if vfile is not None: D['value'] = pd.read_parquet(vfile).reindex(index=D['adj'].index, columns=D['adj'].columns)
    cols = list(COLS); keep = np.array([not c.startswith('91') for c in cols])
    good = D['adj'].notna().sum(axis=1) >= 300
    G = {k: v.loc[good] for k, v in D.items()}
    IG = G['adj'].index; TG = len(IG)
    F = G['flag']; bad5 = F == 5
    ADJ = G['adj'].mask(bad5); CR = G['close_raw'].mask(bad5); OP = G['open'].mask(bad5); V = G['value']
    mkt4 = (F == 4).sum(axis=1) >= 50
    FLd = (F.isin([3, 4, 5]) & ~((F == 4) & mkt4.to_numpy()[:, None])).to_numpy()
    v0 = V.fillna(0).where(V.notna().cumsum() > 0)
    V60n = v0.rolling(60, min_periods=60).mean().to_numpy()
    VOL = (V / CR).where((V > 0) & (CR > 0)).to_numpy()
    ADJn, CRn, OPn = ADJ.to_numpy(), CR.to_numpy(), OP.to_numpy()
    BASE = (V60n >= 1e7) & (CRn >= 10) & np.isfinite(ADJn) & keep[None, :]
    OPEN_OK = np.isfinite(OPn) & np.isfinite(ADJn)
    LPOS = {d: k for k, d in enumerate(IDX)}
    LI = np.array([LPOS[d] for d in IG])
    GU = L.GAP_UP_OPEN_ERA
    wk = IG.to_period('W-SUN')
    grp = pd.Series(np.arange(TG), index=IG).groupby(wk)
    last_of_week = grp.max().to_numpy(); first_of_week = grp.min().to_numpy()
    NW = len(last_of_week)
    VW = np.full((NW, N), np.nan)
    with np.errstate(invalid='ignore'):
        for k in range(NW):
            a, b = first_of_week[k], last_of_week[k] + 1
            blk = VOL[a:b]; cnt = np.isfinite(blk).sum(0)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = np.nanmean(blk, axis=0)
            m[cnt < 3] = np.nan; m[FLd[a:b].any(0)] = np.nan; VW[k] = m
    look, minn = 49, 40
    HIw = np.zeros((NW, N), bool); NNw = np.zeros((NW, N), int); RELV = np.full((NW, N), np.nan)
    import warnings
    for t in range(look, NW):
        W = VW[t - look:t]; x = VW[t]; ok = np.isfinite(W); n = ok.sum(0)
        below = (ok & (W < x)).sum(0); need = np.ceil(0.9 * n)
        valid = np.isfinite(x) & (n >= minn)
        HIw[t] = valid & (below >= need); NNw[t] = np.where(valid, n, 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            RELV[t] = x / np.nanmean(W, axis=0)
    SIGW = [k for k in range(NW) if IG[last_of_week[k]] >= pd.Timestamp('2005-01-01') and last_of_week[k] + 1 < TG]
    coh = []
    for k in SIGW:
        i = last_of_week[k]; ex = i + 1
        pool = BASE[i] & OPEN_OK[ex] & np.isfinite(VW[k]) & (NNw[k] >= 40)
        if pool.sum() < 100: continue
        cand = pool & HIw[k]
        rv = np.where(pool & np.isfinite(RELV[k]), RELV[k], -np.inf)
        ci = np.where(cand)[0]; oi = np.where(pool & ~cand)[0]
        ci = ci[np.argsort(-rv[ci], kind='stable')]; oi = oi[np.argsort(-rv[oi], kind='stable')]
        coh.append(dict(e=ex, order=np.concatenate([ci, oi]).astype(int), n=len(ci)))
    K, H = 4, 20
    items = []
    for q, c in enumerate(coh):
        nxt = coh[q + K]['e'] if q + K < len(coh) else None
        x = c['e'] + H
        if nxt is not None: x = min(x, nxt)
        x = min(x, TG - 1); roll = (nxt is not None and x == nxt)
        items.append(dict(t=int(LI[c['e']]), s=q % K, order=c['order'], n=c['n'], backfill=False, gu_held=False,
                          x=None if roll else int(LI[x])))
    return items, K


# =============== D3 SURGE（照 w3_revenue_sue/run.py 重建） ===============
def surge_items(vfile=None):
    d = pd.read_csv(L.LONG, dtype={'code': str, 'market': str, 'name': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
    d = d[d['code'].isin(COLS)]
    REV = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
    REV = REV.reindex(pd.period_range(REV.index.min(), REV.index.max(), freq='M')).reindex(columns=COLS)
    REV = REV.where(REV > 0)
    IFRS_T = (REV.index >= pd.Period('2013-01', 'M')) & (REV.index <= pd.Period('2013-12', 'M'))
    D_ = REV - REV.shift(12); D_.loc[IFRS_T] = np.nan
    Xl = D_.shift(1); mu = Xl.rolling(24, min_periods=24).mean(); sd = Xl.rolling(24, min_periods=24).std(ddof=1)
    SC = ((D_ - mu) / sd).where((sd > 0) & np.isfinite(sd) & D_.notna())
    _s = pd.Series(IDX, index=IDX)
    LD = [g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([IDX.year, IDX.month]) if (g.dt.day >= 11).any()]
    LD = [x for x in LD if x >= pd.Timestamp('2005-01-01')]
    Dt = [x for x in LD if (pd.Period(x, 'M') - 1) in REV.index]
    Dt = Dt + [IDX[-1]] if IDX[-1] > Dt[-1] else Dt
    mm = [pd.Period(x, 'M') - 1 for x in Dt]
    common = np.array([not c.startswith('91') for c in COLS])
    _na = L.adj.notna().sum(axis=1); SPARSE = _na.index[(_na / _na.rolling(21, center=True, min_periods=5).median()) < 0.5]
    VV = L.V if vfile is None else pd.read_parquet(vfile).reindex(index=L.V.index, columns=L.V.columns)
    V60 = VV.drop(SPARSE).rolling(60).mean().reindex(IDX).ffill(limit=1)
    v60 = V60.reindex(pd.DatetimeIndex(Dt)); px = L.C.reindex(pd.DatetimeIndex(Dt))
    liq = ((v60 >= 1e7) & (px >= 10)).to_numpy() & common[None, :]
    Sm = SC.reindex(mm).to_numpy()
    items = []; first = None
    for k, dd in enumerate(Dt[:-1]):
        i = L.pos(dd); ex = i + 1
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        pool = liq[k] & np.isfinite(Sm[k]) & trad
        pp = np.where(pool)[0]; n = len(pp)
        if n > 0:
            s = Sm[k][pp]; order = pp[np.lexsort((pp, -s))]; ntop = max(1, int(round(0.10 * n)))
            if first is None: first = i
        else:
            order = np.array([], int); ntop = 0
        if first is None: continue
        items.append(dict(t=ex, s=0, order=order.astype(int), n=ntop, backfill=False, gu_held=True, x=None))
    return items, 1, first
