#!/usr/bin/env python3
"""殘差動能（Blitz–Huij–Martens 2011）月換股（M2）。規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
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
IDX = D['adj'].index; N = len(cols)
F = D['flag']; ADJ = D['adj']; CR = D['close_raw']; V = D['value']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()

LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[x] for x in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP_L = (L.AO_raw / L.AC_prev - 1) >= thrL

_s = pd.Series(IDX, index=IDX)
ME = [d for d in _s.groupby([IDX.year, IDX.month]).last() if d <= pd.Timestamp('2026-09-30')]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in ME}
MEI = pd.DatetimeIndex(ME)

# ---------------- 月報酬與因子 ----------------
PM = ADJ.loc[MEI].to_numpy()
RM = np.full_like(PM, np.nan); RM[1:] = PM[1:] / PM[:-1] - 1
bad_ret = np.isfinite(RM) & ((RM >= 2.0) | (RM <= -0.8)); N_BADRET = int(bad_ret.sum()); RM[bad_ret] = np.nan
BP = ((V60.loc[MEI] >= 1e7) & (CR.loc[MEI] >= 10) & ADJ.loc[MEI].notna() & F.loc[MEI].isin([0, 1, 2])).to_numpy()
V60M = V60.loc[MEI].to_numpy()
M_ = len(ME); MKT = np.full(M_, np.nan); SMB = np.full(M_, np.nan)
for m in range(1, M_):
    mem = BP[m - 1] & np.isfinite(RM[m])
    if mem.sum() >= 50:
        MKT[m] = RM[m, mem].mean()
        v = V60M[m - 1, mem]; med = np.median(v); r = RM[m, mem]
        sm, bg = v <= med, v > med
        if sm.sum() >= 25 and bg.sum() >= 25: SMB[m] = r[sm].mean() - r[bg].mean()

def resid_score(m, use_smb):
    """回傳 (N,) 訊號。回歸窗 R[m-35..m]，訊號 = Σe[m-11..m-1]/sd。"""
    a = max(m - 35, 1)
    Y = RM[a:m + 1]; T = Y.shape[0]
    Xs = [MKT[a:m + 1]] + ([SMB[a:m + 1]] if use_smb else [])
    okx = np.all([np.isfinite(x) for x in Xs], axis=0)
    ok = np.isfinite(Y) & okx[:, None]
    K = len(Xs) + 1
    Z = np.zeros((T, K)); Z[:, 0] = 1
    for k, x in enumerate(Xs): Z[:, k + 1] = np.nan_to_num(x)
    out = np.full(N, np.nan)
    cnt = ok.sum(0)
    cand = np.where(cnt >= 24)[0]
    for j in cand:
        o = ok[:, j]; Zj = Z[o]; yj = Y[o, j]
        b = np.linalg.lstsq(Zj, yj, rcond=None)[0]
        e = np.full(T, np.nan); e[o] = yj - Zj @ b
        f = e[-12:-1]                      # m-11..m-1
        if f.shape[0] < 11 or not np.all(np.isfinite(f)): continue
        sd = f.std(ddof=1)
        if sd > 0: out[j] = f.sum() / sd
    return out

def raw_mom(m):
    if m < 12: return np.full(N, np.nan)
    f = RM[m - 11:m]
    ok = np.all(np.isfinite(f), axis=0)
    return np.where(ok, np.prod(1 + np.nan_to_num(f), axis=0) - 1, np.nan)

print('computing signals...', flush=True)
SIG, POOL, EXI = {}, {}, {}
for m, d in enumerate(ME):
    if d < pd.Timestamp('2005-12-01'): continue
    ex = LPOS[NEXT[d]]
    trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
    s = {'RC': resid_score(m, False), 'RF': resid_score(m, True), 'RAW': raw_mom(m)}
    pool = BP[m] & trad & np.isfinite(s['RC']) & np.isfinite(s['RF']) & np.isfinite(s['RAW'])
    SIG[d], POOL[d], EXI[d] = s, pool, ex
SD = [d for d in ME if d in SIG and POOL[d].sum() >= 50]
print('signals done; first signal date', SD[0].date(), 'pool sizes', int(POOL[SD[0]].sum()), int(POOL[SD[-1]].sum()), flush=True)

def build(sig, mode):
    events, pools, info = {}, {}, []; held = set(); nblk = 0; ncand = 0
    for d in SD:
        ex = EXI[d]; ok = POOL[d]; s = SIG[d][sig]
        n = int(math.floor(0.10 * ok.sum())) if mode == 'D10' else 30
        idx = np.where(ok)[0]; idx = idx[np.argsort(-s[idx], kind='stable')]
        blocked = GAP_UP_L[ex, LCOL]; pick = []
        for j in idx:
            if len(pick) >= n: break
            if blocked[j] and LCOL[j] not in held: nblk += 1; continue
            pick.append(LCOL[j])
        ncand += n
        events[ex] = ('open', np.array(pick, dtype=int)); pools[ex] = ('open', LCOL[ok]); held = set(pick)
        info.append((d, int(ok.sum()), len(pick)))
    return events, pools, pd.DataFrame(info, columns=['d', 'pool', 'held']).set_index('d'), nblk, ncand

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

WIN = {'select_2007-01_2019-04': ('2007-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'), 'select_ext_2006-02_2019-04': ('2006-02-01', '2019-04-30'),
       'valid_survivor_2019-05_2023-12': ('2019-05-01', '2023-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'),
       'full_2007-01_2026-10': ('2007-01-01', '2026-10-31')}
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
        for s in ('RC', 'RF', 'RAW'):
            x = pd.Series(SIG[d][s][ok]); r['IC_' + s] = x.rank().corr(f)
            q = pd.qcut(x.rank(method='first'), 10, labels=False).to_numpy()
            r['D10mD1_' + s] = np.nanmean(fr[ok][q == 9]) - np.nanmean(fr[ok][q == 0])
        r['corr_RC_RAW'] = pd.Series(SIG[d]['RC'][ok]).rank().corr(pd.Series(SIG[d]['RAW'][ok]).rank())
        r['corr_RC_RF'] = pd.Series(SIG[d]['RC'][ok]).rank().corr(pd.Series(SIG[d]['RF'][ok]).rank())
        rows.append(r)
    x = pd.DataFrame(rows).set_index('d')
    out = {}
    for w, (a, b) in WIN.items():
        z = seg(x, a, b); o = {'n_months': int(len(z))}
        for c in x.columns:
            if c.startswith('corr'): o[c] = round(float(z[c].mean()), 3)
            else: o[c + '_mean'] = round(float(z[c].mean() * (100 if c.startswith('D10') else 1)), 4); o['t_' + c] = tstat(z[c])
        out[w] = o
    return out

def main():
    COST = {'base': L.cost_rates(0.001), 'c05': L.cost_rates(0.0), 'stress': L.cost_rates('stress')}
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'costs_buy_sell': COST, 'monthly_returns_set_nan_extreme': N_BADRET,
                    'first_signal': str(SD[0].date()), 'last_signal': str(SD[-1].date()), 'last_day': str(L.IDX[-1].date()), 'n_rebalances': len(SD),
                    'MKT_factor_monthly_mean_pct': round(float(np.nanmean(MKT) * 100), 3), 'SMB_proxy_monthly_mean_pct': round(float(np.nanmean(SMB) * 100), 3),
                    'SMB_proxy_t': tstat(pd.Series(SMB)),
                    'pool_size_mean_by_year': {int(y): round(float(g.mean()), 1) for y, g in pd.Series({d: int(POOL[d].sum()) for d in SD}).groupby(lambda x: x.year)}}}
    t0 = LPOS[NEXT[SD[0]]]; t1 = len(L.IDX) - 1
    specs = {'RC_D10': ('RC', 'D10'), 'RC_N30': ('RC', 'N30'), 'RF_D10': ('RF', 'D10'), 'RF_N30': ('RF', 'N30'), 'RAW_D10': ('RAW', 'D10'), 'RAW_N30': ('RAW', 'N30')}
    res = {}; navs = {}; ew = None
    for key, (sig, mode) in specs.items():
        ev, pools, info, nb, nc = build(sig, mode)
        if ew is None:
            ewg, _ = sim(pools, 0.0, 0.0, t0, t1); ewn, _ = sim(pools, *COST['base'], t0, t1)
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
        print(f"{key:8s} tov {o['turnover_value_per_rebalance']} blk {nb}/{nc} held {o['avg_held']} pool {o['avg_pool']}",
              '| S', ps['CAGR'], 'g', ps['CAGR_gross'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 't50', ps['t_vs_0050_m'], 'DD', ps['maxDD'],
              '| V', pv['CAGR'], 'g', pv['CAGR_gross'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'], 't50', pv['t_vs_0050_m'], 'DD', pv['maxDD'],
              '| 24-25', p2['CAGR'], p2['0050_CAGR'], p2['EW_CAGR'], p2['t_vs_EW_m'], p2['t_vs_0050_m'], flush=True)
        if key == 'RC_D10': info.to_csv(HERE / 'monthly_counts_RC_D10.csv')
    out['portfolios'] = res
    # 配對：殘差 − 原始
    pair = {}
    for a_, b_ in (('RC_D10', 'RAW_D10'), ('RC_N30', 'RAW_N30'), ('RF_D10', 'RAW_D10'), ('RF_D10', 'RC_D10')):
        pair[f'{a_}_minus_{b_}'] = {}
        for w, (a, b) in WIN.items():
            x = mo(seg(navs[a_].pct_change(), a, b).fillna(0)) - mo(seg(navs[b_].pct_change(), a, b).fillna(0))
            pair[f'{a_}_minus_{b_}'][w] = {'diff_net_pct_m': round(float(x.mean() * 100), 3), 't': tstat(x), 'n_months': int(len(x))}
    out['paired'] = pair
    for k, v in pair.items(): print('PAIR', k, {w: (v[w]['diff_net_pct_m'], v[w]['t']) for w in MAINW}, flush=True)
    out['diagnostics'] = diagnostics()
    for w in MAINW: print('DIAG', w, json.dumps(out['diagnostics'][w], ensure_ascii=False), flush=True)

    def passes(o):
        t = o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9
        ok_ew = (o['excess_vs_EW_pct_m'] > 0) and (t >= 2); ok_50 = o['CAGR'] >= o['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    SW, VW = MAINW[0], MAINW[1]
    vs = {'primary': 'RC_D10', 'select': passes(res['RC_D10'][SW]), 'valid': passes(res['RC_D10'][VW])}
    k = vs['select']['pass'] + vs['valid']['pass']
    vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: {w: passes(o[w])['pass'] for w in MAINW[:2]} for kk, o in res.items()}
    out['verdict'] = vs
    pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
    pd.DataFrame({'MKT': MKT, 'SMB_proxy': SMB}, index=MEI).to_csv(HERE / 'factors_monthly.csv')
    f = HERE / 'results.json'
    dd = json.loads(f.read_text(encoding='utf-8')); dd.update(out)
    f.write_text(json.dumps(dd, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()
