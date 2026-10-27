#!/usr/bin/env python3
"""多均線趨勢因子（Han–Zhou–Zhu 2016；Liu–Zhou–Zhu 中國版）月換股（M2）。規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成交引擎價格陣列與成本（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'value', 'flag']}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if len(c) == 4 and c.isdigit() and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; N = len(cols); T = len(IDX)
F = D['flag']; ADJ = D['adj']; CR = D['close_raw']; V = D['value']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()

# ---------------- 均線（累積和） ----------------
An = ADJ.to_numpy()
listed = np.cumsum(np.isfinite(An), axis=0) > 0
Af = ADJ.ffill().to_numpy(); Af = np.where(listed, Af, 0.0)
first = np.where(listed.any(0), np.argmax(listed, axis=0), T + 10)
CSP = np.vstack([np.zeros((1, N)), np.cumsum(Af, axis=0)])
VOL = (V / CR).replace([np.inf, -np.inf], np.nan).to_numpy()
VOLf = np.where(listed, np.nan_to_num(VOL, nan=0.0), 0.0)
CSV = np.vstack([np.zeros((1, N)), np.cumsum(VOLf, axis=0)])
LS11 = [3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000]
LS7 = [3, 5, 10, 20, 50, 100, 200]
LS8 = [3, 5, 10, 20, 50, 100, 200, 400]

def ma_at(CS, i, Lw):
    if i + 1 - Lw < 0: return np.full(N, np.nan)
    return (CS[i + 1] - CS[i + 1 - Lw]) / Lw

def features(i):
    """回傳 dict(variant -> (N,K) 矩陣，缺值 NaN)、hist。"""
    hist = i - first + 1
    pt = An[i]; vt = VOL[i]
    okp = np.isfinite(pt) & (pt > 0) & (hist >= 200)
    A = {}
    for Lw in LS11:
        a = ma_at(CSP, i, Lw) / pt
        A[Lw] = np.where(okp & (hist >= Lw), a, np.nan)
    # 替代：缺的長均線用已有真值的最長均線
    sub = {}
    best = A[200].copy()
    for Lw in LS11:
        if Lw <= 200: sub[Lw] = A[Lw]
        else:
            best = np.where(np.isfinite(A[Lw]), A[Lw], best); sub[Lw] = best.copy()
    okv = np.isfinite(vt) & (vt > 0) & okp
    AV = {}
    for Lw in LS8:
        a = ma_at(CSV, i, Lw) / np.where(okv, vt, np.nan)
        AV[Lw] = np.where(okv & (hist >= Lw), a, np.nan)
    AV[400] = np.where(np.isfinite(AV[400]), AV[400], AV[200])
    X = {'TF11': np.column_stack([sub[l] for l in LS11]),
         'S200': np.column_stack([A[l] for l in LS7]),
         'CN': np.column_stack([sub[l] for l in LS8] + [AV[l] for l in LS8])}
    return X, hist

def winsor(X, mask):
    Y = X.copy()
    if mask.sum() < 10: return Y
    lo = np.nanpercentile(X[mask], 1, axis=0); hi = np.nanpercentile(X[mask], 99, axis=0)
    return np.clip(Y, lo, hi)

LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[x] for x in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP_L = (L.AO_raw / L.AC_prev - 1) >= thrL

_s = pd.Series(IDX, index=IDX)
ME = [d for d in _s.groupby([IDX.year, IDX.month]).last() if d <= pd.Timestamp('2026-09-30')]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in ME}
MEI = pd.DatetimeIndex(ME); MPOS = [IDX.get_loc(d) for d in ME]

PM = ADJ.loc[MEI].to_numpy()
RM = np.full_like(PM, np.nan); RM[1:] = PM[1:] / PM[:-1] - 1
bad_ret = np.isfinite(RM) & ((RM >= 2.0) | (RM <= -0.8)); N_BADRET = int(bad_ret.sum()); RM[bad_ret] = np.nan
BP = ((V60.loc[MEI] >= 1e7) & (CR.loc[MEI] >= 10) & ADJ.loc[MEI].notna() & F.loc[MEI].isin([0, 1, 2])).to_numpy()
M_ = len(ME)

# 第一個回歸月：ME[m-1] 位置 ≥ 999（面板已有 1000 個交易日）
m_reg0 = next(m for m in range(1, M_) if MPOS[m - 1] >= 999)
print('first regression month', ME[m_reg0].date(), 'X date', ME[m_reg0 - 1].date(), flush=True)

FEAT = {}
for m in range(m_reg0 - 1, M_):
    FEAT[m] = features(MPOS[m])
VARS = ['TF11', 'S200', 'CN']
BETA = {v: {} for v in VARS}; NREG = {v: {} for v in VARS}
for m in range(m_reg0, M_):
    X0, _ = FEAT[m - 1]
    for v in VARS:
        X = X0[v]; ok = BP[m - 1] & np.isfinite(RM[m]) & np.all(np.isfinite(X), axis=1)
        if ok.sum() < 50: continue
        Xw = winsor(X, ok)[ok]; y = RM[m, ok]
        Z = np.column_stack([np.ones(len(y)), Xw])
        BETA[v][m] = np.linalg.lstsq(Z, y, rcond=None)[0]; NREG[v][m] = int(ok.sum())

SIG, POOL, EXI, AUX = {}, {}, {}, {}
for m in range(m_reg0 + 11, M_):
    d = ME[m]; ex = LPOS[NEXT[d]]
    X0, hist = FEAT[m]
    s = {}
    for v in VARS:
        bs = [BETA[v].get(k) for k in range(m - 11, m + 1)]
        if any(b is None for b in bs): s[v] = np.full(N, np.nan); continue
        bbar = np.mean(bs, axis=0)
        X = X0[v]; okf = BP[m] & np.all(np.isfinite(X), axis=1)
        Xw = winsor(X, okf)
        e = Xw @ bbar[1:]; e[~okf] = np.nan; s[v] = e
    trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
    pool = BP[m] & trad & np.all([np.isfinite(s[v]) for v in VARS], axis=0)
    SIG[d], POOL[d], EXI[d] = s, pool, ex
    mom = np.prod(1 + RM[m - 11:m], axis=0) - 1 if m >= 12 else np.full(N, np.nan)
    AUX[d] = {'r1': RM[m], 'mom12_1': mom, 'hist': hist}
SD = [d for d in ME if d in SIG and POOL[d].sum() >= 50]
print('signals done; first signal', SD[0].date(), 'pools', int(POOL[SD[0]].sum()), int(POOL[SD[-1]].sum()), flush=True)

def build(sig, mode):
    events, pools, info = {}, {}, []; held = set(); nblk = 0; ncand = 0
    for d in SD:
        ex = EXI[d]; ok = POOL[d]; s = SIG[d][sig]
        n = int(math.floor(0.20 * ok.sum())) if mode == 'Q20' else 30
        idx = np.where(ok)[0]; idx = idx[np.argsort(-s[idx], kind='stable')]
        blocked = GAP_UP_L[ex, LCOL]; pick = []
        for j in idx:
            if len(pick) >= n: break
            if blocked[j] and LCOL[j] not in held: nblk += 1; continue
            pick.append(LCOL[j])
        ncand += n
        events[ex] = ('open', np.array(pick, dtype=int)); pools[ex] = ('open', LCOL[ok]); held = set(pick)
        info.append((d, int(ok.sum()), len(pick), float(np.mean(AUX[d]['hist'][ok] >= 1000))))
    return events, pools, pd.DataFrame(info, columns=['d', 'pool', 'held', 'frac_hist1000']).set_index('d'), nblk, ncand

def sim(events, buy, sell, t0, t1):
    """exlib.sim_portfolio 同邏輯。"""
    u = np.zeros(L.N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]
            Pz = L.AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash
            tgt = np.zeros(L.N); w = 1.0 / len(pk); Vn = Vt
            for _ in range(6):
                tgt[:] = 0; tgt[pk] = w * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(L.N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
    return pd.Series(nav, index=L.IDX[t0:t1 + 1]), (float(np.mean(tov[1:])) if len(tov) > 1 else float('nan'))

WIN = {'select_2009-01_2019-04': ('2009-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'),
       'valid_survivor_2019-05_2023-12': ('2019-05-01', '2023-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'),
       'full_2009-01_2026-10': ('2009-01-01', '2026-10-31')}
MAINW = list(WIN)[:3]

def mo(x): return (1 + x).groupby(x.index.to_period('M')).prod() - 1

def pstats(r, bench, ew, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
    rr = r[m].fillna(0); bb = bench.reindex(rr.index).fillna(0); ee = ew.reindex(rr.index).fillna(0)
    yrs = (rr.index[-1] - r.index[max(np.argmax(m) - 1, 0)]).days / 365.25
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

def seg(x, a, b): return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]

def diagnostics():
    rows = []
    for k, d in enumerate(SD[:-1]):
        ex0 = EXI[d]; ex1 = EXI[SD[k + 1]]
        fr = L.AO[ex1, LCOL] / L.AO[ex0, LCOL] - 1
        ok = POOL[d] & np.isfinite(fr)
        if ok.sum() < 50: continue
        f = pd.Series(fr[ok]).rank()
        r = {'d': d}
        for s in VARS:
            x = pd.Series(SIG[d][s][ok]); r['IC_' + s] = x.rank().corr(f)
            q = pd.qcut(x.rank(method='first'), 5, labels=False).to_numpy()
            r['Q5mQ1_' + s] = np.nanmean(fr[ok][q == 4]) - np.nanmean(fr[ok][q == 0])
            r['Q5mAll_' + s] = np.nanmean(fr[ok][q == 4]) - np.nanmean(fr[ok])
            r['Q1mAll_' + s] = np.nanmean(fr[ok][q == 0]) - np.nanmean(fr[ok])
        tf = pd.Series(SIG[d]['TF11'][ok]).rank()
        r['corr_TF11_S200'] = tf.corr(pd.Series(SIG[d]['S200'][ok]).rank())
        r['corr_TF11_CN'] = tf.corr(pd.Series(SIG[d]['CN'][ok]).rank())
        a1 = pd.Series(AUX[d]['r1'][ok]); r['corr_TF11_lastmonth_ret'] = tf.corr(a1.rank())
        a2 = pd.Series(AUX[d]['mom12_1'][ok]); r['corr_TF11_mom12_1'] = tf.corr(a2.rank())
        r['frac_hist1000'] = float(np.mean(AUX[d]['hist'][ok] >= 1000))
        rows.append(r)
    x = pd.DataFrame(rows).set_index('d')
    out = {}
    for w, (a, b) in WIN.items():
        z = seg(x, a, b); o = {'n_months': int(len(z))}
        for c in x.columns:
            if c.startswith('corr') or c.startswith('frac'): o[c] = round(float(z[c].mean()), 3)
            else: o[c + '_mean'] = round(float(z[c].mean() * (100 if c.startswith('Q') else 1)), 4); o['t_' + c] = tstat(z[c])
        out[w] = o
    # β̄ 平均（各期間的回歸係數平均與 t）
    bt = {}
    for v, lab in (('TF11', LS11), ('S200', LS7), ('CN', [f'P{l}' for l in LS8] + [f'V{l}' for l in LS8])):
        B = pd.DataFrame({ME[m]: b for m, b in BETA[v].items()}).T; B.columns = ['b0'] + [str(l) for l in lab]
        bt[v] = {w: {c: [round(float(seg(B, a, b)[c].mean()), 4), tstat(seg(B, a, b)[c])] for c in B.columns}
                 for w, (a, b) in list(WIN.items())[:2]}
        bt[v]['n_reg_mean'] = round(float(np.mean(list(NREG[v].values()))), 1)
    out['beta_mean_t'] = bt
    return out

def main():
    COST = {'base': L.cost_rates(0.001), 'c05': L.cost_rates(0.0), 'stress': L.cost_rates('stress')}
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'costs_buy_sell': COST, 'monthly_returns_set_nan_extreme': N_BADRET,
                    'first_regression_month_end': str(ME[m_reg0].date()), 'first_signal': str(SD[0].date()), 'last_signal': str(SD[-1].date()),
                    'last_day': str(L.IDX[-1].date()), 'n_rebalances': len(SD),
                    'pool_size_mean_by_year': {int(y): round(float(g.mean()), 1) for y, g in pd.Series({d: int(POOL[d].sum()) for d in SD}).groupby(lambda x: x.year)},
                    'frac_pool_hist1000_by_year': {int(y): round(float(g.mean()), 3) for y, g in pd.Series({d: float(np.mean(AUX[d]['hist'][POOL[d]] >= 1000)) for d in SD}).groupby(lambda x: x.year)}}}
    t0 = LPOS[NEXT[SD[0]]]; t1 = len(L.IDX) - 1
    specs = {'TF11_Q20': ('TF11', 'Q20'), 'TF11_N30': ('TF11', 'N30'), 'S200_Q20': ('S200', 'Q20'), 'S200_N30': ('S200', 'N30'),
             'CN_Q20': ('CN', 'Q20'), 'CN_N30': ('CN', 'N30')}
    res = {}; navs = {}; ew = None
    for key, (sig, mode) in specs.items():
        ev, pools, info, nb, nc = build(sig, mode)
        if ew is None:
            ewg, _ = sim(pools, 0.0, 0.0, t0, t1); ewn, _ = sim(pools, *COST['base'], t0, t1); ew = 1
            navs['EW_gross'] = ewg; navs['EW_net'] = ewn
            b50 = L.E50.reindex(ewg.index).pct_change(); ewr = ewg.pct_change(); ewnr = ewn.pct_change()
        nav, tov = sim(ev, *COST['base'], t0, t1)
        nav_s, _ = sim(ev, *COST['stress'], t0, t1); nav_c, _ = sim(ev, *COST['c05'], t0, t1); nav_g, _ = sim(ev, 0.0, 0.0, t0, t1)
        navs[key] = nav; navs[key + '_gross'] = nav_g; navs[key + '_stress'] = nav_s; navs[key + '_c05'] = nav_c
        r = nav.pct_change()
        o = {'blocked_gap_up': nb, 'target_slots_total': nc, 'turnover_value_per_rebalance': round(tov, 3),
             'avg_pool': round(float(info['pool'].mean()), 1), 'avg_held': round(float(info['held'].mean()), 1)}
        for w, (a, b) in WIN.items():
            o[w] = pstats(r, b50, ewr, a, b)
            ps_ = pstats(nav_s.pct_change(), b50, ewr, a, b); pg = pstats(nav_g.pct_change(), b50, ewr, a, b); pc = pstats(nav_c.pct_change(), b50, ewr, a, b)
            o[w]['CAGR_stress'] = ps_['CAGR']; o[w]['t_stress_vs_EW_m'] = ps_['t_vs_EW_m']; o[w]['t_stress_vs_0050_m'] = ps_['t_vs_0050_m']
            o[w]['CAGR_c05'] = pc['CAGR']; o[w]['t_c05_vs_EW_m'] = pc['t_vs_EW_m']; o[w]['t_c05_vs_0050_m'] = pc['t_vs_0050_m']
            o[w]['CAGR_gross'] = pg['CAGR']; o[w]['t_gross_vs_EW_m'] = pg['t_vs_EW_m']; o[w]['excess_gross_vs_EW_pct_m'] = pg['excess_vs_EW_pct_m']
            o[w]['EW_net_CAGR'] = pstats(ewnr, b50, ewr, a, b)['CAGR']
            sub = seg(info, a, b); o[w]['avg_pool'] = round(float(sub['pool'].mean()), 1); o[w]['avg_held'] = round(float(sub['held'].mean()), 1)
        mr = mo(r.dropna()); mew = mo(ewr.dropna())
        o['worst5_months_pct'] = {str(k): round(float(v) * 100, 1) for k, v in mr.nsmallest(5).items()}
        o['worst5_months_EW_pct_same'] = {str(k): round(float(mew.get(k, np.nan)) * 100, 1) for k in mr.nsmallest(5).index}
        res[key] = o
        ps, pv, p2 = (o[w] for w in MAINW)
        print(f"{key:9s} tov {o['turnover_value_per_rebalance']} blk {nb}/{nc} held {o['avg_held']} pool {o['avg_pool']}",
              '| S', ps['CAGR'], 'g', ps['CAGR_gross'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 't50', ps['t_vs_0050_m'], 'DD', ps['maxDD'],
              '| V', pv['CAGR'], 'g', pv['CAGR_gross'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'], 't50', pv['t_vs_0050_m'], 'DD', pv['maxDD'],
              '| 24-25', p2['CAGR'], p2['0050_CAGR'], p2['EW_CAGR'], p2['t_vs_EW_m'], p2['t_vs_0050_m'], flush=True)
        if key == 'TF11_Q20': info.to_csv(HERE / 'monthly_counts_TF11_Q20.csv')
    out['portfolios'] = res
    pair = {}
    for a_, b_ in (('TF11_Q20', 'S200_Q20'), ('CN_Q20', 'TF11_Q20'), ('TF11_N30', 'S200_N30'), ('CN_N30', 'TF11_N30')):
        pair[f'{a_}_minus_{b_}'] = {}
        for w, (a, b) in WIN.items():
            x = mo(seg(navs[a_].pct_change(), a, b).fillna(0)) - mo(seg(navs[b_].pct_change(), a, b).fillna(0))
            pair[f'{a_}_minus_{b_}'][w] = {'diff_net_pct_m': round(float(x.mean() * 100), 3), 't': tstat(x), 'n_months': int(len(x))}
    out['paired'] = pair
    out['diagnostics'] = diagnostics()
    for w in MAINW: print('DIAG', w, json.dumps(out['diagnostics'][w], ensure_ascii=False), flush=True)
    print('BETA', json.dumps(out['diagnostics']['beta_mean_t'], ensure_ascii=False), flush=True)

    def passes(o):
        t = o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9
        ok_ew = (o['excess_vs_EW_pct_m'] > 0) and (t >= 2); ok_50 = o['CAGR'] >= o['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    SW, VW = MAINW[0], MAINW[1]
    vs = {'primary': 'TF11_Q20', 'select': passes(res['TF11_Q20'][SW]), 'valid': passes(res['TF11_Q20'][VW])}
    k = vs['select']['pass'] + vs['valid']['pass']
    vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: {w: passes(o[w])['pass'] for w in MAINW[:2]} for kk, o in res.items()}
    out['verdict'] = vs
    pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
    f = HERE / 'results.json'
    dd = json.loads(f.read_text(encoding='utf-8')); dd.update(out)
    f.write_text(json.dumps(dd, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()
