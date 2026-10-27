#!/usr/bin/env python3
"""漲停次數樂透股（NET20／NET60／MAXC）月換股（M2）＋主策略排除濾網＋營收公告前後拆段。規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
sys.path.insert(0, str(HERE.parent / 'revenue_longhistory'))
import exlib as L  # noqa: E402  成交引擎價格陣列與成本（只讀取）
import lhlib as LH  # noqa: E402  主策略長歷史（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'high', 'value', 'flag', 'ret', 'src']}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if len(c) == 4 and c.isdigit() and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; N = len(cols)
F = D['flag']; ADJ = D['adj']; CR = D['close_raw']; V = D['value']; RET = D['ret']; HI = D['high']; SRC = D['src']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
BASEU = (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()

# ---------------- 觸板 ----------------
LIM = pd.Series(np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10), index=IDX)
Lm = pd.DataFrame(np.repeat(LIM.to_numpy()[:, None], N, 1), index=IDX, columns=cols)
CP = CR.shift(1)
RR = CR / CP - 1
DAYV = CR.notna() & CP.notna() & (F == 0) & ((1 + RET) - (1 + RR)).abs().le(0.002) & RR.abs().le(Lm + 0.005)
UP = DAYV & (RR >= Lm - 0.003)
DN = DAYV & (RR <= -(Lm - 0.003))
def roll_sum(X, w): return X.astype(float).rolling(w, min_periods=1).sum()
nV20, nV60 = roll_sum(DAYV, 20), roll_sum(DAYV, 60)
UP20, DN20, UP60, DN60, UP21 = roll_sum(UP, 20), roll_sum(DN, 20), roll_sum(UP, 60), roll_sum(DN, 60), roll_sum(UP, 21)
NET20 = ((UP20 - DN20) / nV20).where(nV20 >= 15)
NET60 = ((UP60 - DN60) / nV60).where(nV60 >= 45)
RC = RET.where(RET.notna() & ~F.isin([3, 4, 5]) & (RET.abs() <= 0.25))
MX = np.minimum(RC, Lm).where(RC.notna()).rolling(21, min_periods=15).max()
MAXC = MX.where(UP21 < 1, 1 + UP21).where(MX.notna())
UP20v = UP20.where(nV20 >= 1)
SIGS = {'NET20': NET20, 'NET60': NET60, 'MAXC': MAXC}

LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[x] for x in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP_L = (L.AO_raw / L.AC_prev - 1) >= thrL
IPOS = {d: i for i, d in enumerate(IDX)}
_s = pd.Series(IDX, index=IDX)
ME = [d for d in _s.groupby([IDX.year, IDX.month]).last() if pd.Timestamp('2004-12-01') <= d <= pd.Timestamp('2026-09-30')]
D11 = [g[g.dt.day >= 11].iloc[0] for _, g in _s.groupby([IDX.year, IDX.month]) if (g.dt.day >= 11).any()]
D11 = [d for d in D11 if pd.Timestamp('2005-01-01') <= d <= pd.Timestamp('2026-09-30')]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in sorted(set(ME) | set(D11))}

def base_pool(d):
    ex = LPOS[NEXT[d]]
    u = (BASEU.loc[d] & (V60.loc[d] >= 1e7)).to_numpy()
    trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
    return ex, u & trad

def tiebreak(d):
    return np.random.default_rng([20260427, d.toordinal()]).random(N)

def order_groups(s, ok, d):
    """排序股池內由低到高（同值亂數打破）；回傳 (遞增順序的股票索引, 每個位置所屬的組 0..4)。"""
    idx = np.where(ok)[0]; tb = tiebreak(d)[idx]
    o = idx[np.lexsort((tb, s[idx]))]
    g = (np.arange(len(o)) * 5) // len(o)
    return o, g

def build(sig_name, cal, mode, grp=None):
    """mode: 'low'（Q1，由低往高遞補）、'high'（Q5，由高往低遞補）、'fixed'（指定組、不遞補）、'tie_low'/'tie_high'（同值同組診斷、不遞補）。"""
    S = SIGS[sig_name]
    events, pools, info = {}, {}, []
    held = set(); nblk = 0; ncand = 0
    for d in cal:
        ex, pool = base_pool(d)
        s = S.loc[d].to_numpy(); ok = pool & np.isfinite(s)
        if ok.sum() < 50: continue
        blocked = GAP_UP_L[ex, LCOL]
        o, g = order_groups(s, ok, d)
        if mode in ('low', 'high'):
            tgt = int((g == (0 if mode == 'low' else 4)).sum()); seq = o if mode == 'low' else o[::-1]
            pick = []; ncand += tgt
            for j in seq:
                if len(pick) >= tgt: break
                if blocked[j] and LCOL[j] not in held: nblk += 1; continue
                pick.append(LCOL[j])
        else:
            if mode == 'fixed': mem = o[g == grp]
            else:
                pr = pd.Series(s[ok]).rank(method='average', pct=True).to_numpy()
                gg = np.clip(np.floor((pr - 1e-12) * 5).astype(int), 0, 4)
                mem = np.where(ok)[0][gg == (0 if mode == 'tie_low' else 4)]
            ncand += len(mem)
            pick = [LCOL[j] for j in mem if not (blocked[j] and LCOL[j] not in held)]
            nblk += len(mem) - len(pick)
        events[ex] = ('open', np.array(pick, dtype=int)); pools[ex] = ('open', LCOL[ok])
        held = set(pick)
        jj = [np.where(LCOL == c)[0][0] for c in pick]
        info.append((d, int(ok.sum()), len(pick), float(np.nanmedian(s[jj])) if jj else np.nan,
                     float(np.nanmean(UP20.loc[d].to_numpy()[jj])) if jj else np.nan,
                     float(np.nanmean(DN20.loc[d].to_numpy()[jj])) if jj else np.nan,
                     float(np.nanmedian(V60.loc[d].to_numpy()[jj]) / 1e6) if jj else np.nan))
    inf = pd.DataFrame(info, columns=['d', 'pool', 'held', 'sig_med', 'up20_mean', 'dn20_mean', 'v60_med_m']).set_index('d')
    return events, pools, inf, nblk, ncand

def sim(events, buy, sell, t0, t1):
    """exlib.sim_portfolio 同邏輯（允許空持股）。"""
    u = np.zeros(L.N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]
            Pz = L.AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash
            if len(pk) == 0:
                cash = Vt - sell * h.sum(); tov.append(h.sum() / Vt if Vt > 0 else 0); u = np.zeros(L.N)
            else:
                tgt = np.zeros(L.N); w = 1.0 / len(pk); Vn = Vt
                for _ in range(6):
                    tgt[:] = 0; tgt[pk] = w * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                    Vn = Vt - cost
                tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
                u = np.zeros(L.N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
    return pd.Series(nav, index=L.IDX[t0:t1 + 1]), (float(np.mean(tov[1:])) if len(tov) > 1 else float('nan'))

WIN = {'select_2005-01_2019-04': [('2005-01-01', '2019-04-30')],
       'select_7pct_2005-01_2015-05': [('2005-01-01', '2015-05-31')],
       'select_10pct_2015-06_2019-04': [('2015-06-01', '2019-04-30')],
       'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
       'valid_2024-01_2025-12': [('2024-01-01', '2025-12-31')],
       'valid_survivor_2019-05_2023-12': [('2019-05-01', '2023-12-31')],
       'valid_official_signal': [('2019-05-01', '2024-02-29'), ('2025-11-01', '2026-10-31')],
       'valid_2024-01_2026-10': [('2024-01-01', '2026-10-31')]}
SW, VW = 'select_2005-01_2019-04', 'valid_2019-05_2026-10'
SHOW = [SW, 'select_7pct_2005-01_2015-05', 'select_10pct_2015-06_2019-04', VW, 'valid_2024-01_2025-12', 'valid_official_signal']

def mo(x): return (1 + x).groupby(x.index.to_period('M')).prod() - 1

def wmask(idx, segs):
    m = np.zeros(len(idx), bool)
    for a, b in segs: m |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    return m

def span_years(r, segs):
    y = 0.0
    for a, b in segs:
        m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
        if m.sum() == 0: continue
        f = np.argmax(m); l_ = len(m) - 1 - np.argmax(m[::-1])
        y += (r.index[l_] - r.index[max(f - 1, 0)]).days / 365.25
    return y

def pstats(r, bench, ew, segs):
    m = wmask(r.index, segs)
    rr = r[m].fillna(0); bb = bench.reindex(rr.index).fillna(0); ee = ew.reindex(rr.index).fillna(0)
    yrs = span_years(r, segs)
    def st(x):
        eq = (1 + x).cumprod(); return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}
    o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
    o['vol'] = round(float(rr.std() * math.sqrt(252)), 4)
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['n_months'] = int(len(ms))
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['hit_vs_EW_m'] = round(float((ms > me).mean()), 3)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    return o

def paired(ra, rb, segs):
    m = wmask(ra.index, segs)
    x = mo(ra[m].fillna(0)) - mo(rb.reindex(ra.index)[m].fillna(0))
    return {'diff_pct_m': round(float(x.mean() * 100), 3), 't': tstat(x), 'n_months': int(len(x)), 'hit': round(float((x > 0).mean()), 3)}

def passes(o):
    t = o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9
    ok_ew = (o['excess_vs_EW_pct_m'] > 0) and (t >= 2); ok_50 = o['CAGR'] >= o['0050_CAGR']
    return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}

COST = {'base': L.cost_rates(0.001), 'c05': L.cost_rates(0.0), 'stress': L.cost_rates('stress')}

def run_family(cal, specs, t0, t1, b50, tag):
    res, navs, ewcache, infos = {}, {}, {}, {}
    for key, (sig, mode, grp) in specs.items():
        ev, pools, info, nb, nc = build(sig, cal, mode, grp)
        if sig not in ewcache:
            g_, _ = sim(pools, 0.0, 0.0, t0, t1); n_, _ = sim(pools, *COST['base'], t0, t1); ewcache[sig] = (g_, n_)
            navs[f'EW_{sig}'] = g_
        ewg, ewn = ewcache[sig]; ewr = ewg.pct_change()
        nav_g, _ = sim(ev, 0.0, 0.0, t0, t1); navs[key + '_gross'] = nav_g
        o = {'blocked_gap_up': nb, 'members_total': nc, 'blocked_share': round(nb / nc, 4) if nc else None,
             'avg_pool': round(float(info['pool'].mean()), 1), 'avg_held': round(float(info['held'].mean()), 1)}
        if mode in ('low', 'high') or key.endswith('_tie'):
            nav, tov = sim(ev, *COST['base'], t0, t1); nav_s, _ = sim(ev, *COST['stress'], t0, t1); nav_c, _ = sim(ev, *COST['c05'], t0, t1)
            navs[key] = nav; navs[key + '_stress'] = nav_s
            o['turnover_value_per_rebalance'] = round(tov, 3)
            for w, segs in WIN.items():
                o[w] = pstats(nav.pct_change(), b50, ewr, segs)
                ps_ = pstats(nav_s.pct_change(), b50, ewr, segs); pc = pstats(nav_c.pct_change(), b50, ewr, segs); pg = pstats(nav_g.pct_change(), b50, ewr, segs)
                o[w].update({'CAGR_stress': ps_['CAGR'], 't_stress_vs_EW_m': ps_['t_vs_EW_m'], 't_stress_vs_0050_m': ps_['t_vs_0050_m'],
                             'CAGR_c05': pc['CAGR'], 't_c05_vs_EW_m': pc['t_vs_EW_m'],
                             'CAGR_gross': pg['CAGR'], 't_gross_vs_EW_m': pg['t_vs_EW_m'], 'excess_gross_vs_EW_pct_m': pg['excess_vs_EW_pct_m'],
                             'EW_net_CAGR': pstats(ewn.pct_change(), b50, ewr, segs)['CAGR']})
                sub = info[wmask(info.index, segs)]
                o[w].update({'avg_held': round(float(sub['held'].mean()), 1), 'sig_median': round(float(sub['sig_med'].median()), 4),
                             'up20_mean': round(float(sub['up20_mean'].mean()), 3), 'dn20_mean': round(float(sub['dn20_mean'].mean()), 3),
                             'v60_median_NTD_m': round(float(sub['v60_med_m'].median()), 1)})
        else:
            for w, segs in WIN.items():
                pg = pstats(nav_g.pct_change(), b50, ewr, segs)
                o[w] = {'CAGR_gross': pg['CAGR'], 'excess_gross_vs_EW_pct_m': pg['excess_vs_EW_pct_m'], 't_gross_vs_EW_m': pg['t_vs_EW_m']}
        res[key] = o; infos[key] = info
        line = ' | '.join(f"{w[:12]} {o[w].get('CAGR', o[w]['CAGR_gross'])} g{o[w]['CAGR_gross']} EW{o[w].get('EW_CAGR', '')} 50:{o[w].get('0050_CAGR', '')} "
                          f"tEW {o[w].get('t_vs_EW_m', o[w]['t_gross_vs_EW_m'])} t50 {o[w].get('t_vs_0050_m', '')}" for w in SHOW)
        print(f"[{tag}] {key:14s} held {o['avg_held']} blk {o['blocked_share']} tov {o.get('turnover_value_per_rebalance')} || {line}", flush=True)
    return res, navs, infos

def spreads(navs, fams):
    out = {}
    for fam in fams:
        out[fam] = {}
        for w, segs in WIN.items():
            o = {}
            for tg, suf in (('net', ''), ('gross', '_gross'), ('stress', '_stress')):
                h = navs[f'{fam}_Q5' + suf].pct_change(); l_ = navs[f'{fam}_Q1' + suf].pct_change()
                m = wmask(h.index, segs)
                x = mo(h[m].fillna(0)) - mo(l_[m].fillna(0))
                o[f'Q5_minus_Q1_{tg}_pct_m'] = round(float(x.mean() * 100), 3); o[f't_{tg}'] = tstat(x)
            o['n_months'] = int(len(x)); out[fam][w] = o
    return out

def revenue_split(sig_name):
    """持有期拆段：公告前（d+1 開盤 → 當月 10 日前最後交易日收盤）、公告後（該收盤 → 下一換股日開盤）。Q1、Q5（形成時成員，毛）減排序股池平均。"""
    S = SIGS[sig_name]; rows = []
    for k, d in enumerate(ME[:-1]):
        ex0, pool = base_pool(d); ex1 = LPOS[NEXT[ME[k + 1]]]
        s = S.loc[d].to_numpy(); ok = pool & np.isfinite(s)
        if ok.sum() < 50: continue
        hm = L.IDX[ex0]
        cand = [t for t in range(ex0, ex1) if L.IDX[t].year == hm.year and L.IDX[t].month == hm.month and L.IDX[t].day <= 10]
        o, g = order_groups(s, ok, d)
        ac0, ao0, ao1 = L.AO[ex0, LCOL], L.AO[ex0, LCOL], L.AO[ex1, LCOL]
        full = ao1 / ao0 - 1
        r = {'d': d, 'full_Q1': np.nanmean(full[o[g == 0]]) - np.nanmean(full[o]), 'full_Q5': np.nanmean(full[o[g == 4]]) - np.nanmean(full[o])}
        if cand:
            b = cand[-1]; acb = L.AC[b, LCOL]
            pre = acb / ao0 - 1; post = ao1 / acb - 1
            for nm, x in (('pre', pre), ('post', post)):
                pm = np.nanmean(x[o]); r[f'{nm}_Q1'] = np.nanmean(x[o[g == 0]]) - pm; r[f'{nm}_Q5'] = np.nanmean(x[o[g == 4]]) - pm
            r['pre_days'] = b - ex0 + 1; r['post_days'] = ex1 - b - 1
        rows.append(r)
    X = pd.DataFrame(rows).set_index('d')
    for nm in ('full', 'pre', 'post'): X[f'{nm}_Q5mQ1'] = X[f'{nm}_Q5'] - X[f'{nm}_Q1']
    out = {}
    for w, segs in WIN.items():
        x = X[wmask(X.index, segs)]
        o = {'n_months': int(len(x)), 'avg_pre_days': round(float(x['pre_days'].mean()), 1), 'avg_post_days': round(float(x['post_days'].mean()), 1)}
        for c in [c for c in X.columns if c.endswith(('Q1', 'Q5'))]:
            o[c + '_ex_pct'] = round(float(x[c].mean() * 100), 3); o['t_' + c] = tstat(x[c])
        out[w] = o
    return out

def main_overlay(b50):
    """U3：主策略（lhlib，n30）排除 UP20 ≥ k。"""
    S = LH.Setup(LH.yoy_long(), start='2005-01-01')
    t0 = LH.pos(S.dates[0]); t1 = len(LH.IDX) - 1
    U = UP20v.reindex(S.DATES).reindex(columns=LH.COLS)
    miss = [str(d.date()) for d in S.DATES if d not in UP20v.index]
    # NET20 Q5 成員（排序股池，同 prereg 排序規則）
    Q5m = pd.DataFrame(False, index=S.DATES, columns=LH.COLS)
    for d in S.dates[:-1]:
        if d not in IDX: continue
        _, pool = base_pool(d) if d in NEXT else (None, None)
        if pool is None: continue
        s = NET20.loc[d].to_numpy(); ok = pool & np.isfinite(s)
        if ok.sum() < 50: continue
        o, g = order_groups(s, ok, d)
        Q5m.loc[d, [cols[j] for j in o[g == 4]]] = True
    orig = S.inD.copy()
    variants = {'main': None, 'ex_up20_ge2': U >= 2, 'ex_up20_ge1': U >= 1, 'ex_up20_ge3': U >= 3, 'ex_net20_Q5': Q5m}
    navs, res, picks = {}, {}, {}
    ew = None
    for nm, exc in variants.items():
        S.inD = orig if exc is None else (orig | exc.fillna(False).astype(bool))
        ev, pools = S.selection(30)
        if ew is None:
            ew, _ = LH.sim_portfolio(pools, 0.0, 0.0, t0, t1)
        nav, tov = LH.sim_portfolio(ev, *LH.cost_rates(0.001), t0, t1)
        nav_s, _ = LH.sim_portfolio(ev, *LH.cost_rates('stress'), t0, t1)
        navs[nm] = nav; navs[nm + '_stress'] = nav_s; picks[nm] = ev
        res[nm] = {'turnover_value_per_rebalance': round(tov, 3), 'rebalances': len(ev)}
        for w, segs in WIN.items():
            res[nm][w] = pstats(nav.pct_change(), b50.reindex(nav.index), ew.pct_change(), segs)
            res[nm][w]['CAGR_stress'] = pstats(nav_s.pct_change(), b50.reindex(nav.index), ew.pct_change(), segs)['CAGR']
    S.inD = orig
    # 被排除的檔數：原版 30 檔中有幾檔 UP20 ≥ 2
    U2 = (U >= 2).fillna(False)
    exl = []
    for ex, (_, pk) in picks['main'].items():
        dd = LH.IDX[ex - 1]
        if dd in U2.index: exl.append((dd, int(U2.loc[dd].to_numpy()[pk].sum())))
    exl = pd.DataFrame(exl, columns=['d', 'n']).set_index('d')['n']
    pr = {}
    for nm in variants:
        if nm == 'main': continue
        pr[nm] = {}
        for w, segs in WIN.items():
            pr[nm][w] = paired(navs[nm].pct_change(), navs['main'].pct_change(), segs)
            pr[nm][w]['stress'] = paired(navs[nm + '_stress'].pct_change(), navs['main_stress'].pct_change(), segs)
    for w, segs in WIN.items():
        for nm in variants:
            if nm != 'main': pr[nm][w]['excluded_from_main30_per_month'] = round(float(exl[wmask(exl.index, segs)].mean()), 2) if nm == 'ex_up20_ge2' else None
    return {'first_signal': str(S.dates[0].date()), 'dates_missing_in_signal_index': miss, 'portfolios': res, 'paired_vs_main': pr,
            'main30_months_with_any_excluded_share': round(float((exl > 0).mean()), 3)}, navs

def diagnostics():
    pool_any = (BASEU & (V60 >= 1e7))
    out = {}
    by = {}
    for y in range(2005, 2027):
        m = (IDX.year == y)
        pv = pool_any.loc[m] & DAYV.loc[m]
        nd = float(pv.to_numpy().sum())
        by[y] = {'up_per_100_stockdays': round(float((UP.loc[m] & pv).to_numpy().sum()) / nd * 100, 3),
                 'dn_per_100_stockdays': round(float((DN.loc[m] & pv).to_numpy().sum()) / nd * 100, 3),
                 'valid_share': round(nd / float(pool_any.loc[m].to_numpy().sum()), 3)}
    out['hit_freq_by_year_in_pool'] = by
    segs = {'official_2023': ('2023-01-01', '2024-01-31'), 'src3_2024-02-15_2025-09-23': ('2024-02-15', '2025-09-23'), 'official_2025-09-24_': ('2025-09-24', '2026-10-31')}
    sc = {}
    for k, (a, b) in segs.items():
        m = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
        pv = pool_any.loc[m] & DAYV.loc[m]; up = UP.loc[m] & pv
        hiok = (CR.loc[m] >= HI.loc[m] * 0.999)
        sc[k] = {'up_per_100': round(float(up.to_numpy().sum()) / float(pv.to_numpy().sum()) * 100, 3),
                 'dn_per_100': round(float((DN.loc[m] & pv).to_numpy().sum()) / float(pv.to_numpy().sum()) * 100, 3),
                 'up_close_eq_high_share': round(float((up & hiok).to_numpy().sum()) / max(float(up.to_numpy().sum()), 1), 3),
                 'near_miss_9.0to9.7_per_100': round(float((pv & (RR.loc[m] >= LIM[m].to_numpy()[:, None] - 0.01) & (RR.loc[m] < LIM[m].to_numpy()[:, None] - 0.003)).to_numpy().sum()) / float(pv.to_numpy().sum()) * 100, 3)}
    out['src3_vs_official'] = sc
    # 月底截面：NET20 非 0、UP20 ≥1／≥2 比例
    rows = []
    for d in ME:
        _, pool = base_pool(d); s = NET20.loc[d].to_numpy(); ok = pool & np.isfinite(s)
        if ok.sum() < 50: continue
        u = UP20.loc[d].to_numpy()[ok]; m = MAXC.loc[d].to_numpy()[ok]
        rows.append((d, ok.sum(), (s[ok] != 0).mean(), (s[ok] < 0).mean(), (s[ok] > 0).mean(), (u >= 1).mean(), (u >= 2).mean(), np.isfinite(m).mean(), (m >= 1).mean() if np.isfinite(m).any() else np.nan))
    X = pd.DataFrame(rows, columns=['d', 'pool', 'net_nonzero', 'net_neg', 'net_pos', 'up20_ge1', 'up20_ge2', 'maxc_valid', 'maxc_hit']).set_index('d')
    out['cross_section_by_year'] = {int(y): {c: round(float(g[c].mean()), 3) for c in X.columns} for y, g in X.groupby(X.index.year)}
    out['flag0_ret_rr_mismatch_share'] = round(float((((1 + RET) - (1 + RR)).abs() > 0.002)[F == 0].to_numpy().mean()), 5)
    return out

def main():
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'costs_buy_sell': COST, 'first_signal': str(ME[0].date()), 'last_signal': str(ME[-1].date()),
                    'last_day': str(L.IDX[-1].date()), 'n_month_end_signals': len(ME), 'n_d11_signals': len(D11)}}
    out['diagnostics'] = diagnostics(); print('DIAG', json.dumps(out['diagnostics']['src3_vs_official'], ensure_ascii=False), flush=True)
    t0 = LPOS[NEXT[ME[0]]]; t1 = len(L.IDX) - 1
    b50 = L.E50.reindex(L.IDX[t0:t1 + 1]).pct_change()
    specs = {}
    for sg in ('NET20', 'NET60', 'MAXC'):
        specs[f'{sg}_Q1'] = (sg, 'low', None); specs[f'{sg}_Q5'] = (sg, 'high', None)
        for gq in (1, 2, 3): specs[f'{sg}_Q{gq + 1}_diag'] = (sg, 'fixed', gq)
        specs[f'{sg}_Q1_tie'] = (sg, 'tie_low', None); specs[f'{sg}_Q5_tie'] = (sg, 'tie_high', None)
    res, navs, infos = run_family(ME, specs, t0, t1, b50, 'ME')
    out['portfolios'] = res
    out['U2_spread'] = spreads(navs, ['NET20', 'NET60', 'MAXC'])
    for fam, v in out['U2_spread'].items(): print('SPREAD', fam, {w: (v[w]['Q5_minus_Q1_gross_pct_m'], v[w]['t_gross'], v[w]['t_net']) for w in SHOW}, flush=True)
    infos['NET20_Q1'].to_csv(HERE / 'monthly_NET20_Q1.csv'); infos['NET20_Q5'].to_csv(HERE / 'monthly_NET20_Q5.csv')
    # 營收公告前後
    out['revenue_split_holding'] = {sg: revenue_split(sg) for sg in ('NET20', 'MAXC')}
    for sg, v in out['revenue_split_holding'].items():
        for w in SHOW: print('REVSPLIT', sg, w, {k: v[w][k] for k in v[w] if k.startswith('t_') or k.endswith('Q5mQ1_ex_pct')}, flush=True)
    t0b = LPOS[NEXT[D11[0]]]
    b50b = L.E50.reindex(L.IDX[t0b:t1 + 1]).pct_change()
    specs_b = {'NET20_Q1': ('NET20', 'low', None), 'NET20_Q5': ('NET20', 'high', None), 'MAXC_Q1': ('MAXC', 'low', None), 'MAXC_Q5': ('MAXC', 'high', None)}
    res_b, navs_b, _ = run_family(D11, specs_b, t0b, t1, b50b, 'D11')
    out['revenue_split_formation_after_announcement'] = {'portfolios': res_b, 'spread': spreads(navs_b, ['NET20', 'MAXC'])}
    # U3 主策略排除
    b50m = L.E50.pct_change()
    u3, navs_m = main_overlay(b50m)
    out['U3_main_exclusion'] = u3
    for nm, v in u3['paired_vs_main'].items(): print('U3', nm, {w: (v[w]['diff_pct_m'], v[w]['t'], v[w]['stress']['t']) for w in SHOW}, flush=True)
    for nm, v in u3['portfolios'].items(): print('U3P', nm, {w: (v[w]['CAGR'], v[w]['0050_CAGR'], v[w]['EW_CAGR'], v[w]['t_vs_EW_m'], v[w]['t_vs_0050_m']) for w in SHOW}, flush=True)
    # 判定
    p = res['NET20_Q1']
    u1 = {'select': passes(p[SW]), 'valid': passes(p[VW])}
    k1 = u1['select']['pass'] + u1['valid']['pass']
    u1['verdict'] = '可用' if k1 == 2 else ('待觀察' if k1 == 1 else '不可用')
    def p3(w):
        pp = u3['paired_vs_main']['ex_up20_ge2'][w]; po = u3['portfolios']['ex_up20_ge2'][w]
        ok = pp['diff_pct_m'] > 0 and np.nan_to_num(pp['t'], nan=-9) >= 2; ok50 = po['CAGR'] >= po['0050_CAGR']
        return {'paired_t2': bool(ok), 'not_lose_0050': bool(ok50), 'pass': bool(ok and ok50)}
    u3v = {'select': p3(SW), 'valid': p3(VW)}
    k3 = u3v['select']['pass'] + u3v['valid']['pass']
    u3v['verdict'] = '可用' if k3 == 2 else ('待觀察' if k3 == 1 else '不可用')
    rank = {'不可用': 0, '待觀察': 1, '可用': 2}
    overall = max([u1['verdict'], u3v['verdict']], key=lambda x: rank[x])
    out['verdict'] = {'U1_NET20_Q1': u1, 'U3_exclusion_up20_ge2': u3v, 'overall': overall,
                      'variants_pass': {kk: {w: passes(o[w])['pass'] for w in (SW, VW)} for kk, o in res.items() if 'CAGR' in o[SW]}}
    allnav = pd.DataFrame({**navs, **{'D11_' + k: v for k, v in navs_b.items()}, **{'MAIN_' + k: v for k, v in navs_m.items()}})
    allnav.to_parquet(HERE / 'nav.parquet')
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else (int(x) if isinstance(x, np.integer) else str(x))), encoding='utf-8')
    print(json.dumps(out['verdict'], ensure_ascii=False))

if __name__ == '__main__':
    main()
