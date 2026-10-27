#!/usr/bin/env python3
"""特質波動率（AHXZ IVOL）月換股（M2）＋變體＋低波動／MAX 控制＋Fama–MacBeth。規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成交引擎價格陣列與成本（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'value', 'flag', 'ret']}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if len(c) == 4 and c.isdigit() and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; N = len(cols)
F = D['flag']; ADJ = D['adj']; CR = D['close_raw']; V = D['value']; RET = D['ret']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
BASEU = (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()
RC = RET.where(RET.notna() & ~F.isin([3, 4, 5]) & (RET.abs() <= 0.25)).to_numpy()   # 清理後日報酬

# 產業別（與 momentum_family/mlib.py 相同對應）
PREFIX_IND = {'11': '水泥工業', '12': '食品工業', '13': '塑膠工業', '14': '紡織纖維', '15': '電機機械', '16': '電器電纜',
              '17': '化學工業', '18': '玻璃陶瓷', '19': '造紙工業', '20': '鋼鐵工業', '21': '橡膠工業', '22': '汽車工業',
              '25': '建材營造業', '26': '航運業', '27': '觀光餐旅', '28': '金融保險業', '29': '貿易百貨業', '99': '其他業'}
_sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype={'code': str}).drop_duplicates('code').set_index('code')['industry']
IND = np.array([(_sl[c] if (c in _sl.index and isinstance(_sl[c], str)) else PREFIX_IND.get(c[:2], '未分類')) for c in cols])

LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[x] for x in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP_L = (L.AO_raw / L.AC_prev - 1) >= thrL

_s = pd.Series(IDX, index=IDX)
ME = list(_s.groupby([IDX.year, IDX.month]).last())
ME = [d for d in ME if pd.Timestamp('2004-12-01') <= d <= pd.Timestamp('2026-09-30')]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in ME}
IPOS = {d: i for i, d in enumerate(IDX)}

def base_pool(d):
    ex = LPOS[NEXT[d]]
    u = (BASEU.loc[d] & (V60.loc[d] >= 1e7)).to_numpy()
    trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
    return ex, u & trad

def ols_sd(Y, Xs, minobs):
    """Y (T,n)；Xs: list of (T,n)；含截距；回傳殘差標準差 sqrt(SSR/(n-K))。"""
    ok = np.isfinite(Y)
    for X in Xs: ok &= np.isfinite(X)
    T, n = Y.shape; K = len(Xs) + 1
    Z = np.zeros((T, n, K)); Z[:, :, 0] = 1.0
    for k, X in enumerate(Xs): Z[:, :, k + 1] = np.where(ok, X, 0)
    Z[~ok] = 0; y = np.where(ok, Y, 0)
    XtX = np.einsum('tnk,tnl->nkl', Z, Z); Xty = np.einsum('tnk,tn->nk', Z, y)
    b = np.einsum('nkl,nl->nk', np.linalg.pinv(XtX), Xty)
    e = np.where(ok, y - np.einsum('tnk,nk->tn', Z, b), 0)
    cnt = ok.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        sd = np.sqrt((e ** 2).sum(0) / (cnt - K))
    sd[cnt < minobs] = np.nan
    return sd

def signals(d, pool):
    i = IPOS[d]; out = {k: np.full(N, np.nan) for k in ['IVOL21_MKT', 'IVOL21_IND', 'IVOL63_MKT', 'MAX21', 'TVOL60']}
    pc = np.where(pool)[0]
    if len(pc) < 50: return out
    W63 = RC[max(i - 62, 0):i + 1]
    Rp = W63[:, pc]
    cntm = np.isfinite(Rp).sum(1)
    with np.errstate(invalid='ignore'):
        mkt = np.where(cntm >= 50, np.nanmean(np.where(np.isfinite(Rp), Rp, np.nan), axis=1), np.nan)
    M = mkt[:, None] * np.ones((1, len(pc)))
    # 產業 leave-one-out
    g = IND[pc].copy()
    vc = pd.Series(g).value_counts(); g = np.where(pd.Series(g).map(vc).to_numpy() >= 5, g, '小產業合併')
    INDR = np.full(Rp.shape, np.nan)
    vm = np.isfinite(Rp); Rz = np.where(vm, Rp, 0)
    for gg in np.unique(g):
        m = g == gg
        S = Rz[:, m].sum(1, keepdims=True); C = vm[:, m].sum(1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            loo = (S - Rz[:, m]) / (C - vm[:, m])
        loo[(C - vm[:, m]) < 1] = np.nan
        INDR[:, m] = loo
    R21, M21, I21 = Rp[-21:], M[-21:], INDR[-21:]
    out['IVOL21_MKT'][pc] = ols_sd(R21, [M21], 15)
    out['IVOL21_IND'][pc] = ols_sd(R21, [M21, I21], 15)
    out['IVOL63_MKT'][pc] = ols_sd(Rp, [M], 45) if Rp.shape[0] == 63 else np.nan
    c21 = np.isfinite(R21).sum(0)
    with np.errstate(invalid='ignore'):
        mx = np.nanmax(np.where(np.isfinite(R21), R21, -np.inf), axis=0)
    mx[c21 < 15] = np.nan; out['MAX21'][pc] = mx
    R60 = Rp[-60:]; c60 = np.isfinite(R60).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        tv = np.nanstd(R60, axis=0, ddof=1)
    tv[c60 < 45] = np.nan; out['TVOL60'][pc] = tv
    return out

print('computing signals...', flush=True)
SIG, POOL, EXI = {}, {}, {}
for d in ME:
    ex, pool = base_pool(d); POOL[d] = pool; EXI[d] = ex; SIG[d] = signals(d, pool)
print('done', flush=True)

def quint(s, ok, nq=5):
    g = np.full(N, -1); idx = np.where(ok)[0]
    r = pd.Series(s[idx]).rank(method='first')
    g[idx] = pd.qcut(r, nq, labels=False).to_numpy()
    return g

def spec_buckets(spec, d):
    """回傳 (ok 排序股池, buckets=[(ordered candidate idx, target)] 或 ('fixed', members))。"""
    var, side = spec['sig'], spec['side']
    S = SIG[d]; pool = POOL[d]
    if spec.get('mode') == 'double':
        iv, tv = S['IVOL21_MKT'], S['TVOL60']
        ok = pool & np.isfinite(iv) & np.isfinite(tv)
        if ok.sum() < 50: return ok, ok, None
        tg = quint(tv, ok, 5); bk = []
        for k in range(5):
            okk = ok & (tg == k); gq = quint(iv, okk, 5)
            grp = 0 if side == 'low' else 4
            idx = np.where(okk)[0]; idx = idx[np.argsort(iv[idx] if side == 'low' else -iv[idx], kind='stable')]
            bk.append((idx, int((gq == grp).sum())))
        return ok, ok, bk
    s = S[var]; ok = pool & np.isfinite(s); ewpool = ok
    if ok.sum() < 50: return ok, ewpool, None
    if spec.get('mode') == 'exmax':
        mx = S['MAX21']; ok2 = ok & np.isfinite(mx)
        cut = np.nanpercentile(mx[ok2], 90)
        ok = ok2 & (mx < cut)   # 剔除 MAX 前 10%
    gq = quint(s, ok, 5)
    if 'grp' in spec:
        return ok, ewpool, ('fixed', np.where(gq == spec['grp'])[0])
    grp = 0 if side == 'low' else 4
    idx = np.where(ok)[0]; idx = idx[np.argsort(s[idx] if side == 'low' else -s[idx], kind='stable')]
    return ok, ewpool, [(idx, int((gq == grp).sum()))]

def build(spec):
    events, pools, info = {}, {}, []
    held = set(); nblk = 0; ncand = 0
    for d in ME:
        ex = EXI[d]
        ok, ewpool, bk = spec_buckets(spec, d)
        if bk is None: continue
        blocked = GAP_UP_L[ex, LCOL]
        pick = []
        if isinstance(bk, tuple):
            mem = bk[1]; ncand += len(mem)
            pick = [LCOL[j] for j in mem if not (blocked[j] and LCOL[j] not in held)]
            nblk += len(mem) - len(pick)
        else:
            for idx, tgt in bk:
                ncand += tgt; got = 0
                for j in idx:
                    if got >= tgt: break
                    if blocked[j] and LCOL[j] not in held: nblk += 1; continue
                    pick.append(LCOL[j]); got += 1
        events[ex] = ('open', np.array(pick, dtype=int)); pools[ex] = ('open', LCOL[ewpool])
        held = set(pick)
        jj = np.searchsorted(LCOL, pick) if False else [np.where(LCOL == c)[0][0] for c in pick]
        pv = V60.loc[d].to_numpy()[jj] if pick else np.array([np.nan])
        sv = SIG[d][spec['sig']][jj] if pick else np.array([np.nan])
        info.append((d, int(ok.sum()), len(pick), float(np.nanmedian(pv)), float(np.nanmedian(sv)), float(np.nanmedian(CR.loc[d].to_numpy()[jj])) if pick else np.nan))
    inf = pd.DataFrame(info, columns=['d', 'pool', 'held', 'v60_med', 'sig_med', 'price_med']).set_index('d')
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

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'), 'valid_survivor_2019-05_2023-12': ('2019-05-01', '2023-12-31'),
       'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31')}
MAINW = list(WIN)[:3]

def mo(x): return (1 + x).groupby(x.index.to_period('M')).prod() - 1

def pstats(r, bench, ew, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
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

def pr(x): return pd.Series(x).rank(pct=True).to_numpy() - 0.5

def fama_macbeth():
    rows = []
    for k, d in enumerate(ME[:-1]):
        ex0 = EXI[d]; ex1 = EXI[ME[k + 1]]; S = SIG[d]
        fr = L.AO[ex1, LCOL] / L.AO[ex0, LCOL] - 1
        iv, tv, mx = S['IVOL21_MKT'], S['TVOL60'], S['MAX21']
        ok = POOL[d] & np.isfinite(fr) & np.isfinite(iv) & np.isfinite(tv) & np.isfinite(mx)
        if ok.sum() < 50: continue
        y = np.clip(fr[ok], -0.5, 1.0); ri, rt, rm = pr(iv[ok]), pr(tv[ok]), pr(mx[ok]); one = np.ones(len(y))
        def reg(*xs): return np.linalg.lstsq(np.column_stack([one, *xs]), y, rcond=None)[0][1:]
        b1 = reg(ri); b2 = reg(ri, rt); b3 = reg(ri, rm); b4 = reg(ri, rt, rm); bt = reg(rt); bm = reg(rm)
        rows.append({'d': d, 'iv_only': b1[0], 'iv_c_tv': b2[0], 'tv_c_iv': b2[1], 'iv_c_max': b3[0], 'max_c_iv': b3[1],
                     'iv_all': b4[0], 'tv_all': b4[1], 'max_all': b4[2], 'tv_only': bt[0], 'max_only': bm[0],
                     'corr_iv_tv': np.corrcoef(ri, rt)[0, 1], 'corr_iv_max': np.corrcoef(ri, rm)[0, 1], 'n': int(ok.sum())})
    return pd.DataFrame(rows).set_index('d')

def fm_stats(fm, a, b):
    x = fm[(fm.index >= pd.Timestamp(a)) & (fm.index <= pd.Timestamp(b))]
    o = {'n_months': int(len(x)), 'avg_n': round(float(x['n'].mean()), 1),
         'corr_rank_iv_tv': round(float(x['corr_iv_tv'].mean()), 3), 'corr_rank_iv_max': round(float(x['corr_iv_max'].mean()), 3)}
    for c in ['iv_only', 'iv_c_tv', 'tv_c_iv', 'iv_c_max', 'max_c_iv', 'iv_all', 'tv_all', 'max_all', 'tv_only', 'max_only']:
        o['b_' + c + '_pct_m'] = round(float(x[c].mean() * 100), 3); o['t_' + c] = tstat(x[c])
    return o

def sig_corr():
    rows = []
    for d in ME:
        S = SIG[d]; base = S['IVOL21_MKT']
        r = {'d': d}
        for k in ['IVOL21_IND', 'IVOL63_MKT', 'TVOL60', 'MAX21']:
            ok = POOL[d] & np.isfinite(base) & np.isfinite(S[k])
            if ok.sum() >= 50: r[k] = pd.Series(base[ok]).rank().corr(pd.Series(S[k][ok]).rank())
        rows.append(r)
    x = pd.DataFrame(rows).set_index('d')
    return {k: round(float(x[k].mean()), 3) for k in x.columns}

def main():
    COST = {'base': L.cost_rates(0.001), 'c05': L.cost_rates(0.0), 'stress': L.cost_rates('stress')}
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'costs_buy_sell': COST,
                    'first_rebalance_signal': str(ME[0].date()), 'last_rebalance_signal': str(ME[-1].date()), 'last_day': str(L.IDX[-1].date()),
                    'industry_groups_in_list': int(len(set(IND)))}}
    t0 = LPOS[NEXT[ME[0]]]; t1 = len(L.IDX) - 1
    specs = {
        'MKT21_A_low': {'sig': 'IVOL21_MKT', 'side': 'low'}, 'MKT21_B_high': {'sig': 'IVOL21_MKT', 'side': 'high'},
        'IND21_A_low': {'sig': 'IVOL21_IND', 'side': 'low'}, 'IND21_B_high': {'sig': 'IVOL21_IND', 'side': 'high'},
        'MKT63_A_low': {'sig': 'IVOL63_MKT', 'side': 'low'}, 'MKT63_B_high': {'sig': 'IVOL63_MKT', 'side': 'high'},
        'exMAX10_A_low': {'sig': 'IVOL21_MKT', 'side': 'low', 'mode': 'exmax'}, 'exMAX10_B_high': {'sig': 'IVOL21_MKT', 'side': 'high', 'mode': 'exmax'},
        'inTVOL_A_low': {'sig': 'IVOL21_MKT', 'side': 'low', 'mode': 'double'}, 'inTVOL_B_high': {'sig': 'IVOL21_MKT', 'side': 'high', 'mode': 'double'},
        'MKT21_Q2_diag': {'sig': 'IVOL21_MKT', 'side': 'x', 'grp': 1}, 'MKT21_Q3_diag': {'sig': 'IVOL21_MKT', 'side': 'x', 'grp': 2},
        'MKT21_Q4_diag': {'sig': 'IVOL21_MKT', 'side': 'x', 'grp': 3},
        'TVOL60_Q1_lowvol': {'sig': 'TVOL60', 'side': 'low'}, 'TVOL60_Q5_highvol': {'sig': 'TVOL60', 'side': 'high'},
        'MAX21_Q1_low': {'sig': 'MAX21', 'side': 'low'}, 'MAX21_Q5_high': {'sig': 'MAX21', 'side': 'high'},
    }
    b50 = None; ewcache = {}; res = {}; navs = {}
    for key, spec in specs.items():
        ev, pools, info, nb, nc = build(spec)
        pk = (spec['sig'], spec.get('mode') == 'double')
        if pk not in ewcache:
            g_, _ = sim(pools, 0.0, 0.0, t0, t1); n_, _ = sim(pools, *COST['base'], t0, t1); ewcache[pk] = (g_, n_)
            navs[f'EW_{pk[0]}{"_dbl" if pk[1] else ""}'] = g_
        ewg, ewn = ewcache[pk]
        if b50 is None: b50 = L.E50.reindex(ewg.index).pct_change()
        nav, tov = sim(ev, *COST['base'], t0, t1)
        nav_s, _ = sim(ev, *COST['stress'], t0, t1); nav_c, _ = sim(ev, *COST['c05'], t0, t1); nav_g, _ = sim(ev, 0.0, 0.0, t0, t1)
        navs[key] = nav; navs[key + '_gross'] = nav_g; navs[key + '_stress'] = nav_s
        ewr = ewg.pct_change(); ewnr = ewn.pct_change(); r = nav.pct_change()
        o = {'blocked_gap_up': nb, 'q_members_total': nc, 'blocked_share': round(nb / nc, 4) if nc else None,
             'turnover_value_per_rebalance': round(tov, 3), 'avg_pool': round(float(info['pool'].mean()), 1), 'avg_held': round(float(info['held'].mean()), 1)}
        for w, (a, b) in WIN.items():
            o[w] = pstats(r, b50, ewr, a, b)
            ps_ = pstats(nav_s.pct_change(), b50, ewr, a, b); pg = pstats(nav_g.pct_change(), b50, ewr, a, b); pc = pstats(nav_c.pct_change(), b50, ewr, a, b)
            o[w]['CAGR_stress'] = ps_['CAGR']; o[w]['t_stress_vs_EW_m'] = ps_['t_vs_EW_m']; o[w]['t_stress_vs_0050_m'] = ps_['t_vs_0050_m']
            o[w]['CAGR_c05'] = pc['CAGR']; o[w]['t_c05_vs_EW_m'] = pc['t_vs_EW_m']
            o[w]['CAGR_gross'] = pg['CAGR']; o[w]['t_gross_vs_EW_m'] = pg['t_vs_EW_m']; o[w]['excess_gross_vs_EW_pct_m'] = pg['excess_vs_EW_pct_m']
            o[w]['EW_net_CAGR'] = pstats(ewnr, b50, ewr, a, b)['CAGR']
            sub = info[(info.index >= pd.Timestamp(a)) & (info.index <= pd.Timestamp(b))]
            o[w]['avg_pool'] = round(float(sub['pool'].mean()), 1); o[w]['avg_held'] = round(float(sub['held'].mean()), 1)
            o[w]['holding_v60_median_NTD_m'] = round(float(sub['v60_med'].median()) / 1e6, 1)
            o[w]['holding_signal_median'] = round(float(sub['sig_med'].median()), 5)
            o[w]['holding_price_median'] = round(float(sub['price_med'].median()), 1)
        res[key] = o
        ps, pv, p2 = (o[w] for w in MAINW)
        print(f"{key:18s} tov {o['turnover_value_per_rebalance']} blk {o['blocked_share']} held {o['avg_held']}",
              '| S', ps['CAGR'], 'g', ps['CAGR_gross'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 't50', ps['t_vs_0050_m'], 'DD', ps['maxDD'],
              '| V', pv['CAGR'], 'g', pv['CAGR_gross'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'], 't50', pv['t_vs_0050_m'], 'DD', pv['maxDD'],
              '| 24-25', p2['CAGR'], p2['0050_CAGR'], p2['EW_CAGR'], p2['t_vs_EW_m'], p2['t_vs_0050_m'], flush=True)
        if key == 'MKT21_B_high': info.to_csv(HERE / 'monthly_counts_MKT21_B.csv')
        if key == 'MKT21_A_low': info.to_csv(HERE / 'monthly_counts_MKT21_A.csv')
    out['portfolios'] = res
    # (C) 最高減最低：兩腿各自扣成本／毛
    spread = {}
    for fam in ['MKT21', 'IND21', 'MKT63', 'exMAX10', 'inTVOL', 'TVOL60', 'MAX21']:
        if fam in ('TVOL60', 'MAX21'):
            hi, lo = (f'{fam}_Q5_highvol', f'{fam}_Q1_lowvol') if fam == 'TVOL60' else (f'{fam}_Q5_high', f'{fam}_Q1_low')
        else:
            hi, lo = f'{fam}_B_high', f'{fam}_A_low'
        spread[fam] = {}
        for w, (a, b) in WIN.items():
            o = {}
            for tag, suf in (('net', ''), ('gross', '_gross'), ('stress', '_stress')):
                h = navs[hi + suf].pct_change(); l_ = navs[lo + suf].pct_change()
                m = (h.index >= pd.Timestamp(a)) & (h.index <= pd.Timestamp(b))
                x = mo(h[m].fillna(0)) - mo(l_[m].fillna(0))
                o[f'B_minus_A_{tag}_pct_m'] = round(float(x.mean() * 100), 3); o[f't_{tag}'] = tstat(x)
            o['n_months'] = int(len(x))
            spread[fam][w] = o
    out['C_high_minus_low'] = spread
    for fam in spread: print('C', fam, {w: (spread[fam][w]['B_minus_A_gross_pct_m'], spread[fam][w]['t_gross'], spread[fam][w]['t_net']) for w in MAINW}, flush=True)
    fm = fama_macbeth()
    out['fama_macbeth'] = {w: fm_stats(fm, a, b) for w, (a, b) in WIN.items()}
    for w in MAINW: print('FM', w, json.dumps(out['fama_macbeth'][w], ensure_ascii=False), flush=True)
    out['signal_rank_corr_with_IVOL21_MKT'] = sig_corr(); print('CORR', out['signal_rank_corr_with_IVOL21_MKT'], flush=True)

    def passes(o):
        t = o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9
        ok_ew = (o['excess_vs_EW_pct_m'] > 0) and (t >= 2)
        ok_50 = o['CAGR'] >= o['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    SW, VW = MAINW[0], MAINW[1]
    tA = res['MKT21_A_low'][SW]['t_vs_EW_m']; tB = res['MKT21_B_high'][SW]['t_vs_EW_m']
    dkey = 'MKT21_A_low' if (np.nan_to_num(tA, nan=-9) > np.nan_to_num(tB, nan=-9)) else 'MKT21_B_high'
    vs = {'direction_chosen_in_select': dkey, 'select_t_A': tA, 'select_t_B': tB,
          'select': passes(res[dkey][SW]), 'valid': passes(res[dkey][VW])}
    k = vs['select']['pass'] + vs['valid']['pass']
    vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: {w: passes(o[w])['pass'] for w in MAINW[:2]} for kk, o in res.items()}
    out['verdict'] = vs
    pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()
