#!/usr/bin/env python3
"""高成交量報酬溢酬（GKM 2001）與異常量變體回測。規則見 results.json 的 PREREG（prereg.py 事前寫入，不改）。
價格陣列與成本從 exec_rebuild/exlib.py 匯入；子帳戶（cohort sleeves）引擎在本檔，換股成本邏輯與 exlib.sim_portfolio 相同。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402
tstat = L.tstat

P = L.P
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src']}
cols = list(L.COLS); keep = np.array([not c.startswith('91') for c in cols])
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
G = {k: v.loc[good] for k, v in D.items()}
IG = G['adj'].index; TG = len(IG); N = L.N
F = G['flag']; bad5 = F == 5
ADJ = G['adj'].mask(bad5); CR = G['close_raw'].mask(bad5); OP = G['open'].mask(bad5); V = G['value']
mkt4 = (F == 4).sum(axis=1) >= 50
FLd = (F.isin([3, 4, 5]) & ~((F == 4) & mkt4.to_numpy()[:, None])).to_numpy()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60n = v0.rolling(60, min_periods=60).mean().to_numpy()
VOL = (V / CR).where((V > 0) & (CR > 0)).to_numpy()            # 日成交量（股）
VOLc = np.where(FLd, np.nan, VOL)                                # flag 3/4/5 日剔除
ADJn, CRn, OPn = ADJ.to_numpy(), CR.to_numpy(), OP.to_numpy()
BASE = (V60n >= 1e7) & (CRn >= 10) & np.isfinite(ADJn) & keep[None, :]
OPEN_OK = np.isfinite(OPn) & np.isfinite(ADJn)
LPOS = {d: k for k, d in enumerate(L.IDX)}
LI = np.array([LPOS[d] for d in IG])                             # IG 位置 → L 位置
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - 0.006
GAP_UP = (L.AO_raw / L.AC_prev - 1) >= thrL
AO, AC = L.AO, L.AC
SRCn = G['src'].to_numpy()

# ---------------- 週曆與週量 ----------------
wk = IG.to_period('W-SUN')
grp = pd.Series(np.arange(TG), index=IG).groupby(wk)
last_of_week = grp.max().to_numpy(); first_of_week = grp.min().to_numpy()
NW = len(last_of_week)
VW = np.full((NW, N), np.nan); RW = np.full((NW, N), np.nan)
for k in range(NW):
    a, b = first_of_week[k], last_of_week[k] + 1
    blk = VOL[a:b]; cnt = np.isfinite(blk).sum(0)
    with np.errstate(invalid='ignore'):
        m = np.nanmean(np.where(np.isfinite(blk), blk, np.nan), axis=0) if True else None
    m[cnt < 3] = np.nan
    m[FLd[a:b].any(0)] = np.nan
    VW[k] = m
SA = pd.DataFrame(ADJn).ffill(limit=5).to_numpy()[last_of_week]
RW[1:] = SA[1:] / SA[:-1] - 1; RW[~np.isfinite(ADJn[last_of_week])] = np.nan

def rank_flags(X, look, minn):
    """X (T,N)。回傳 hi, lo, n：與前 look 期比較。"""
    T = X.shape[0]; hi = np.zeros((T, N), bool); lo = np.zeros((T, N), bool); nn = np.zeros((T, N), int)
    for t in range(look, T):
        W = X[t - look:t]; x = X[t]; ok = np.isfinite(W); n = ok.sum(0)
        below = (ok & (W < x)).sum(0); above = (ok & (W > x)).sum(0)
        need = np.ceil(0.9 * n)
        valid = np.isfinite(x) & (n >= minn)
        hi[t] = valid & (below >= need); lo[t] = valid & (above >= need); nn[t] = np.where(valid, n, 0)
    return hi, lo, nn

HIw, LOw, NNw = rank_flags(VW, 49, 40)
# Lee-Kim-Kim AV
LVW = np.log(VW); AV = np.full((NW, N), np.nan)
for k in range(52, NW):
    W = LVW[k - 52:k]; n = np.isfinite(W).sum(0)
    with np.errstate(invalid='ignore'):
        mu = np.nanmean(np.where(np.isfinite(W), W, np.nan), axis=0)
    av = LVW[k] - mu; av[n < 40] = np.nan; AV[k] = av
# 日版 GKM：當日 + 前 49 日
HId, LOd, NNd = rank_flags(VOLc, 49, 40)
# RVOL
VOLs = pd.DataFrame(VOLc)
M20 = VOLs.shift(1).rolling(20, min_periods=15).mean().to_numpy()
with np.errstate(invalid='ignore', divide='ignore'):
    RVOL = VOLc / M20
RVOL_SIG = (RVOL >= 2) & (CRn > OPn)

T_START = pd.Timestamp('2005-01-01')
SIGW = [k for k in range(NW) if IG[last_of_week[k]] >= T_START and last_of_week[k] + 1 < TG]

# ---------------- cohort 建構 ----------------
def make_cohorts(kind):
    """回傳 list of dict(e=IG位置, x=IG位置, pick=idx, pool=idx, sig=IG位置, wk=週序) 與資訊。"""
    info = {'cohorts': 0, 'blocked': 0, 'n_names': [], 'pool': [], 'empty': 0}
    raw = []
    if kind in ('main', 'd_low', 'b_av_1w', 'b_av_5w'):
        for k in SIGW:
            i = last_of_week[k]; ex = i + 1
            pool = BASE[i] & OPEN_OK[ex]
            if kind in ('main', 'd_low'):
                pool &= np.isfinite(VW[k]) & (NNw[k] >= 40)
            else:
                pool &= np.isfinite(AV[k])
            if pool.sum() < 100: continue
            if kind == 'main': cand = pool & HIw[k]
            elif kind == 'd_low': cand = pool & LOw[k]
            else:
                n = int(round(0.1 * pool.sum())); s = np.where(pool, AV[k], -np.inf)
                top = np.argsort(-s, kind='stable')[:n]; cand = np.zeros(N, bool); cand[top] = True
            raw.append(dict(sig=i, e=ex, cand=cand, pool=pool, wk=k))
    else:
        for i in range(TG - 1):
            if IG[i] < T_START: continue
            ex = i + 1; pool = BASE[i] & OPEN_OK[ex]
            if kind == 'a_daily':
                pool &= np.isfinite(VOLc[i]) & (NNd[i] >= 40); cand = pool & HId[i]
            else:
                pool &= np.isfinite(RVOL[i]); cand = pool & RVOL_SIG[i]
            if pool.sum() < 100: continue
            raw.append(dict(sig=i, e=ex, cand=cand, pool=pool, wk=None))
    out = []
    for c in raw:
        j = np.where(c['cand'])[0]; exL = LI[c['e']]
        blk = GAP_UP[exL, j]; info['blocked'] += int(blk.sum())
        c['pick'] = j[~blk]; c['pool'] = np.where(c['pool'])[0]; del c['cand']
        info['n_names'].append(len(c['pick'])); info['pool'].append(len(c['pool'])); info['empty'] += int(len(c['pick']) == 0)
        out.append(c)
    info['cohorts'] = len(out)
    info['n_names_mean'] = round(float(np.mean(info['n_names'])), 1); info['pool_mean'] = round(float(np.mean(info['pool'])), 1)
    info['frac_of_pool'] = round(float(np.sum(info['n_names']) / np.sum(info['pool'])), 4)
    del info['n_names'], info['pool']
    return out, info

DESIGN = {  # kind, K 子帳戶, H 最長持有交易日（None = 到同子帳戶下一批）
    'main_GKM_weekly_20d': ('main', 4, 20), 'd_low_volume_20d': ('d_low', 4, 20),
    'a_GKM_daily_20d': ('a_daily', 20, 20),
    'b_AV_top10_1w': ('b_av_1w', 1, None), 'b_AV_top10_5w': ('b_av_5w', 5, None),
    'c_RVOL2_up_5d': ('c_rvol', 5, 5), 'c_RVOL2_up_20d': ('c_rvol', 20, 20),
}

def assign(cohorts, K, H):
    for q, c in enumerate(cohorts):
        c['s'] = q % K
    for q, c in enumerate(cohorts):
        nxt = cohorts[q + K]['e'] if q + K < len(cohorts) else None
        x = c['e'] + H if H is not None else (nxt if nxt is not None else TG - 1)
        if nxt is not None: x = min(x, nxt)
        c['x'] = min(x, TG - 1); c['roll'] = (nxt is not None and c['x'] == nxt)
    return cohorts

def sim_sleeves(cohorts, K, buy, sell, which='pick'):
    acts = {}
    for c in cohorts:
        acts.setdefault(LI[c['e']], []).append((c['s'], c[which]))
        if not c['roll']: acts.setdefault(LI[c['x']], []).append((c['s'], np.array([], int)))
    # 同日同子帳戶：先出場再進場 → 進場會覆蓋（只保留最後一個動作，進場排在出場之後）
    for t in acts:
        d = {}
        for s, pk in sorted(acts[t], key=lambda z: len(z[1]) > 0): d[s] = pk
        acts[t] = list(d.items())
    t0 = LI[cohorts[0]['e']] - 1; t1 = len(L.IDX) - 1
    u = np.zeros((K, N)); cash = np.full(K, 1.0 / K); U = np.zeros(N)
    nav = np.empty(t1 - t0 + 1)
    for t in range(t0, t1 + 1):
        for s, pk in acts.get(t, []):
            Pz = np.nan_to_num(AO[t]); h = u[s] * Pz; Vt = h.sum() + cash[s]
            tgt = np.zeros(N); Vn = Vt
            if len(pk):
                w = 1.0 / len(pk)
                for _ in range(6):
                    tgt[:] = 0; tgt[pk] = w * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum(); Vn = Vt - cost
                U -= u[s]; u[s] = 0; u[s][pk] = tgt[pk] / Pz[pk]; U += u[s]; cash[s] = 0.0
            else:
                cost = sell * h.sum(); U -= u[s]; u[s] = 0; cash[s] = Vt - cost
        nav[t - t0] = np.nansum(U * AC[t]) + cash.sum()
    return pd.Series(nav, index=L.IDX[t0:t1 + 1])

def events_table(cohorts, buy, sell):
    rows = []
    for c in cohorts:
        e, x = LI[c['e']], LI[c['x']]
        pr = AO[x, c['pool']] / AO[e, c['pool']] - 1; ew = float(np.nanmean(pr))
        if len(c['pick']) == 0: continue
        r = AO[x, c['pick']] / AO[e, c['pick']] - 1
        net = (1 + r) * (1 - sell) / (1 + buy) - 1
        for j, rr, nn in zip(c['pick'], r, net):
            rows.append((IG[c['e']], cols[j], rr, nn, ew))
    ev = pd.DataFrame(rows, columns=['entry', 'code', 'gross', 'net', 'ew'])
    ev['ab_net'] = ev['net'] - ev['ew']; ev['ab_gross'] = ev['gross'] - ev['ew']
    return ev

WIN = {'select_2005-01_2019-04': [('2005-01-01', '2019-04-30')], 'valid_2019-05_2026-10': [('2019-05-01', '2026-10-31')],
       'y2024_2025': [('2024-01-01', '2025-12-31')], 'src3_2024-02-15_2025-09-23': [('2024-02-15', '2025-09-23')],
       'valid_ex_src3': [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')],
       'full_2005-01_2026-10': [('2005-01-01', '2026-10-31')]}

def inwin(idx, segs):
    m = np.zeros(len(idx), bool)
    for a, b in segs: m |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    return m

def ev_stats(ev, segs):
    e = ev[inwin(pd.DatetimeIndex(ev['entry']), segs)]
    if len(e) < 30: return None
    mon = e.groupby(e['entry'].dt.to_period('M'))
    o = {'n_events': int(len(e)), 'n_months': int(mon.ngroups),
         'mean_gross_pct': round(float(e['gross'].mean() * 100), 3), 'mean_net_pct': round(float(e['net'].mean() * 100), 3),
         'mean_ab_gross_pct': round(float(e['ab_gross'].mean() * 100), 3), 'mean_ab_net_pct': round(float(e['ab_net'].mean() * 100), 3),
         'hit_ab_net': round(float((e['ab_net'] > 0).mean()), 3),
         't_ab_net_month': tstat(mon['ab_net'].mean()), 't_ab_gross_month': tstat(mon['ab_gross'].mean()),
         't_net_month': tstat(mon['net'].mean())}
    return o

def stats(nav, segs, b50, ew):
    r = nav.pct_change(); m = inwin(r.index, segs) & r.notna().to_numpy()
    rr = r[m]
    if len(rr) < 20: return None
    span = 0.0
    for a, b in segs:
        x = rr[(rr.index >= pd.Timestamp(a)) & (rr.index <= pd.Timestamp(b))]
        if len(x): span += (x.index[-1] - r.index[max(r.index.get_loc(x.index[0]) - 1, 0)]).days / 365.25
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / span) - 1), 4), 'vol': round(float(rr.std() * math.sqrt(252)), 4),
         'maxDD': round(float((eq / eq.cummax() - 1).min()), 4), 'years': round(span, 2)}
    mon = lambda s: (1 + s).groupby(s.index.to_period('M')).prod() - 1
    sm = mon(rr)
    for nm, bn in (('0050', b50), ('EW', ew)):
        bd = bn.pct_change().reindex(rr.index).fillna(0); bm = mon(bd); ex = sm - bm
        eb = (1 + bd).cumprod()
        o[f'{nm}_CAGR'] = round(float(eb.iloc[-1] ** (1 / span) - 1), 4); o[f'{nm}_maxDD'] = round(float((eb / eb.cummax() - 1).min()), 4)
        o[f'excess_vs_{nm}_pct_m'] = round(float(ex.mean() * 100), 3); o[f't_vs_{nm}_m'] = tstat(ex); o[f'hit_vs_{nm}_m'] = round(float((ex > 0).mean()), 3)
    o['n_months'] = int(len(sm))
    return o

def yearly(nav):
    r = nav.pct_change().dropna(); return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

res = json.load(open(HERE / 'results.json'))
out = {'meta': {'bad_days_dropped': BAD, 'n_weeks': NW, 'first_signal_week': str(IG[last_of_week[SIGW[0]]].date()),
                'market_flag4_days_ignored': [str(x.date()) for x in IG[mkt4.to_numpy()]]}}
navs = {}; COH = {}
CR_N = L.cost_rates(0.001); CR_S = L.cost_rates('stress')
for name, (kind, K, H) in DESIGN.items():
    coh, info = make_cohorts(kind); coh = assign(coh, K, H); COH[name] = coh
    ew = sim_sleeves(coh, K, 0.0, 0.0, which='pool'); navs[f'{name}_EW'] = ew
    E50 = L.E50.reindex(ew.index)
    hold = [c['x'] - c['e'] for c in coh]
    info['hold_days_mean'] = round(float(np.mean(hold)), 2); info['hold_days_min'] = int(np.min(hold)); info['hold_days_max'] = int(np.max(hold))
    block = {'design': {'kind': kind, 'K_sleeves': K, 'H_days': H}, 'info': info, 'portfolio': {}, 'events': {}}
    for cname, cr in (('gross', (0.0, 0.0)), ('normal', CR_N), ('stress', CR_S)):
        nav = sim_sleeves(coh, K, cr[0], cr[1]); navs[f'{name}_{cname}'] = nav
        block['portfolio'][cname] = {w: stats(nav, s, E50, ew) for w, s in WIN.items()}
        if cname == 'normal': block['yearly_normal_pct'] = yearly(nav); block['yearly_EW_pct'] = yearly(ew)
        if cname != 'gross':
            ev = events_table(coh, cr[0], cr[1])
            block['events'][cname] = {w: ev_stats(ev, s) for w, s in WIN.items()}
            if cname == 'normal':
                ev.to_parquet(HERE / f'events_{name}.parquet')
                big = ev.reindex(ev['gross'].abs().sort_values(ascending=False).index).head(8)
                block['extreme_events'] = [[str(r.entry.date()), r.code, round(r.gross, 3)] for r in big.itertuples()]
    out[name] = block
    print(name, info, flush=True)
    for cn in ('gross', 'normal', 'stress'):
        for wn in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'valid_ex_src3', 'y2024_2025', 'src3_2024-02-15_2025-09-23'):
            x = block['portfolio'][cn][wn]
            ee = block['events'].get(cn, {}).get(wn) or {}
            print(f'  {cn:6s} {wn:26s} CAGR {x["CAGR"]:.4f} DD {x["maxDD"]:.3f} | EW {x["EW_CAGR"]:.4f} t_EW {x["t_vs_EW_m"]} | 0050 {x["0050_CAGR"]:.4f} t_50 {x["t_vs_0050_m"]}'
                  + (f' | ev ab_net {ee.get("mean_ab_net_pct")}% t {ee.get("t_ab_net_month")} n {ee.get("n_events")}' if ee else ''), flush=True)

# ---------------- 價量混雜診斷：本週報酬三分位內，高量 vs 低量 ----------------
coh = COH['main_GKM_weekly_20d']
rows = []
for c in coh:
    k = c['wk']; e, x = LI[c['e']], LI[c['x']]; pool = c['pool']
    rw = RW[k, pool]; okr = np.isfinite(rw)
    if okr.sum() < 60: continue
    q = pd.Series(rw[okr]).rank(pct=True).to_numpy(); terc = np.full(len(pool), -1); terc[okr] = np.minimum((q * 3).astype(int), 2)
    fr = AO[x, pool] / AO[e, pool] - 1; ew = np.nanmean(fr)
    # 診斷不排除開盤漲停（只看訊號本身），價格報酬開盤進出
    for g, lab in ((HIw[k, pool], 'high'), (LOw[k, pool], 'low'), (~HIw[k, pool] & ~LOw[k, pool], 'mid')):
        for tt in range(3):
            m = g & (terc == tt) & np.isfinite(fr)
            if m.sum(): rows.append((IG[c['e']], lab, tt, float(np.mean(fr[m]) - ew), int(m.sum())))
td = pd.DataFrame(rows, columns=['entry', 'grp', 'terc', 'ab', 'n'])
tdiag = {}
for wn in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024_2025'):
    s = td[inwin(pd.DatetimeIndex(td['entry']), WIN[wn])]; o = {}
    for tt, tl in ((0, 'loser_T1'), (1, 'mid_T2'), (2, 'winner_T3')):
        a = s[s.terc == tt]
        pv = a.pivot_table(index='entry', columns='grp', values='ab'); nv = a.pivot_table(index='entry', columns='grp', values='n')
        dd = (pv['high'] - pv['low']).dropna()
        mm = dd.groupby(dd.index.to_period('M')).mean()
        o[tl] = {'high_ab_gross_pct': round(float(pv['high'].mean() * 100), 3), 'low_ab_gross_pct': round(float(pv['low'].mean() * 100), 3),
                 'mid_ab_gross_pct': round(float(pv['mid'].mean() * 100), 3),
                 'high_minus_low_pct': round(float(dd.mean() * 100), 3), 't_month': tstat(mm),
                 'avg_n_high': round(float(nv['high'].mean()), 1), 'avg_n_low': round(float(nv['low'].mean()), 1)}
    tdiag[wn] = o
out['price_volume_tercile_diag_20d_open_to_open_gross_vs_poolEW'] = tdiag
print(json.dumps(tdiag, ensure_ascii=False, indent=1))

# 高量 − 低量（主版本 vs d 變體）月報酬價差
hl = {}
for cn in ('gross', 'normal'):
    a = navs[f'main_GKM_weekly_20d_{cn}'].pct_change(); b = navs[f'd_low_volume_20d_{cn}'].pct_change()
    o = {}
    for wn in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'y2024_2025'):
        m = inwin(a.index, WIN[wn]); x = a[m]; y = b[m]
        ma = (1 + x).groupby(x.index.to_period('M')).prod() - 1; mb = (1 + y).groupby(y.index.to_period('M')).prod() - 1
        o[wn] = {'mean_spread_pct_m': round(float((ma - mb).mean() * 100), 3), 't_month': tstat(ma - mb)}
    hl[cn] = o
out['high_minus_low_portfolio_spread'] = hl
print(hl)

# ---------------- src=3 口徑檢查 ----------------
def frac_series(H_, k_list, NN):
    fr = []
    for k in k_list:
        i = last_of_week[k]; pool = BASE[i] & np.isfinite(VW[k]) & (NN[k] >= 40)
        fr.append((IG[i], H_[k][pool].mean() if pool.sum() else np.nan))
    return pd.Series(dict(fr))
fh = frac_series(HIw, SIGW, NNw); fl = frac_series(LOw, SIGW, NNw)
src_chk = {}
for lab, d0 in (('to_src3_2024-02-15', '2024-02-15'), ('to_official_2025-09-24', '2025-09-24')):
    d0 = pd.Timestamp(d0); bi = fh.index[fh.index < d0][-26:]; ai = fh.index[fh.index >= d0][:26]
    src_chk[lab] = {'hi_frac_before': round(float(fh[bi].mean()), 4), 'hi_frac_after': round(float(fh[ai].mean()), 4),
                    'lo_frac_before': round(float(fl[bi].mean()), 4), 'lo_frac_after': round(float(fl[ai].mean()), 4)}
med = pd.Series(np.nanmedian(np.where(BASE[last_of_week] , VW, np.nan), axis=1), index=IG[last_of_week])
for lab, d0 in (('to_src3_2024-02-15', '2024-02-15'), ('to_official_2025-09-24', '2025-09-24')):
    d0 = pd.Timestamp(d0); src_chk[lab]['median_VW_ratio_after26_over_before26'] = round(float(med[med.index >= d0][:26].mean() / med[med.index < d0][-26:].mean()), 3)
src_chk['hi_frac_by_year'] = {int(y): round(float(g.mean()), 4) for y, g in fh.groupby(fh.index.year)}
src_chk['lo_frac_by_year'] = {int(y): round(float(g.mean()), 4) for y, g in fl.groupby(fl.index.year)}
out['src3_volume_check'] = src_chk
print(src_chk)
out['yearly_0050_pct'] = yearly(L.E50.dropna())
pd.DataFrame(navs).to_parquet(HERE / 'navs.parquet')
res['RESULTS'] = out
json.dump(res, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('done')
