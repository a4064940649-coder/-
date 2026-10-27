#!/usr/bin/env python3
"""SURGE × 主策略交叉：(a) 第二層篩選、(b) 排名合成、(c) 略過。先跑 cross_prereg.py。執行：python3 -I cross_run.py"""
import json, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'revenue_longhistory'))
import lhlib as L  # noqa: E402
T0 = time.time()
F = HERE / 'results_cross.json'
res = json.loads(F.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
res = {'PREREG': res['PREREG'], 'PREREG_sha256': res['PREREG_sha256']}
COLS, N, pos, IDX = L.COLS, L.N, L.pos, L.IDX

# ---------- SURGE（照抄 w3 run.py 主版本） ----------
d = pd.read_csv(L.LONG, dtype={'code': str, 'market': str, 'name': str})
d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
d = d.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first'); d = d[d['code'].isin(COLS)]
REV = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
REV = REV.reindex(pd.period_range(REV.index.min(), REV.index.max(), freq='M')).reindex(columns=COLS)
REV = REV.where(REV > 0)
IFRS_T = (REV.index >= pd.Period('2013-01', 'M')) & (REV.index <= pd.Period('2013-12', 'M'))
D_ = REV - REV.shift(12); D_.loc[IFRS_T] = np.nan


def surge(X, skip=False):
    if not skip:
        Xl = X.shift(1); mu = Xl.rolling(24, min_periods=24).mean(); sd = Xl.rolling(24, min_periods=24).std(ddof=1)
    else:
        mu = pd.DataFrame(np.nan, index=X.index, columns=X.columns); sd = mu.copy()
        pnum = pd.Series(np.arange(len(X.index)), index=X.index)
        for c in X.columns:
            s = X[c].dropna()
            if len(s) < 25: continue
            sl = s.shift(1); m_ = sl.rolling(24, min_periods=24).mean(); s_ = sl.rolling(24, min_periods=24).std(ddof=1)
            first = pd.Series(pnum.reindex(s.index).to_numpy(), index=s.index).shift(24)
            ok = (pnum.reindex(s.index) - first) <= 36
            mu.loc[s.index, c] = m_.where(ok); sd.loc[s.index, c] = s_.where(ok)
    return ((X - mu) / sd).where((sd > 0) & np.isfinite(sd) & X.notna())


SUR = {'main': surge(D_), 'ifrs_skip': surge(D_, skip=True)}
print('surge', round(time.time() - T0), flush=True)

S = L.Setup(L.yoy_long(), start='2005-01-01')
DATES = S.DATES
mm = [pd.Period(x, 'M') - 1 for x in S.dates]
def sur_at(X): return pd.DataFrame(X.reindex(mm).to_numpy(), index=DATES, columns=X.columns).reindex(columns=COLS)
SURd = {k: sur_at(v) for k, v in SUR.items()}

# 一致性檢查：w3 S_main 入選股在我方 SURGE 下是否有值且偏高
inf3 = pd.read_csv(HERE / 'info_S_main.csv.gz'); inf3 = inf3[inf3.picks.notna()]
chk = []
X = SUR['main']
for _, r in inf3.iterrows():
    p = pd.Period(r.rev_month, 'M'); row = X.loc[p]
    pk = r.picks.split(); v = row.reindex(pk)
    pct = row.rank(pct=True).reindex(pk)
    chk.append((v.notna().mean(), pct.min()))
chk = np.array(chk)
res['check_vs_w3_picks'] = {'months': int(len(chk)), 'share_picks_with_surge': round(float(chk[:, 0].mean()), 4), 'min_pick_pct_in_all_listed_min': round(float(np.nanmin(chk[:, 1])), 3),
                            'min_pick_pct_in_all_listed_median': round(float(np.nanmedian(chk[:, 1])), 3)}
print(res['check_vs_w3_picks'], flush=True)

el = S.t300 & S.s52.notna() & ~S.inD & S.yoy3.notna()
r52 = S.s52.where(el).rank(axis=1, pct=True); rrv = S.yoy3.where(el).rank(axis=1, pct=True)
MAIN = (0.5 * r52 + 0.5 * rrv)
GU = L.GAP_UP_OPEN_ERA


def select(n, mode, sk='main', top=60):
    SU = SURd[sk].where(el); RS = SU.rank(axis=1, pct=True)
    B = (r52 + rrv + RS.fillna(0.5)) / 3
    events, pools, fb = {}, {}, {}; held = set(); nblk = 0
    for dd in S.dates[:-1]:
        i = pos(dd); ex = i + 1
        okb = el.loc[dd].to_numpy(); trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        msc = MAIN.loc[dd].to_numpy()
        ok = okb & trad & np.isfinite(msc)
        if ok.sum() < 50: continue
        su = SU.loc[dd].to_numpy()
        nsu = int((ok & np.isfinite(su)).sum())
        fallback = (mode == 'main') or nsu < 100
        order = msc.copy(); order[~ok] = np.nan
        cand = np.argsort(-np.nan_to_num(order, nan=-9))[:int(ok.sum())]   # 與 lhlib 相同
        if not fallback:
            if mode == 'a':
                t6 = cand[:top]; s = su[t6]
                key = np.where(np.isfinite(s), -s, np.inf)
                cand = np.concatenate([t6[np.argsort(key, kind='stable')], cand[top:]])
            elif mode == 'b':
                o2 = B.loc[dd].to_numpy().copy(); o2[~ok] = np.nan
                cand = np.argsort(-np.nan_to_num(o2, nan=-9))[:int(ok.sum())]
        pick = []
        for j in cand:
            if GU[ex, j] and j not in held:
                nblk += 1; continue
            pick.append(j)
            if len(pick) == n: break
        events[ex] = ('open', np.array(pick)); pools[ex] = ('open', np.where(ok)[0]); fb[dd] = fallback
        held = set(pick)
    return events, pools, fb, nblk


t0 = pos(S.dates[0]); t1 = pos(S.dates[-1])
SEG = {'select_2006-04_2013-12': ('2006-04-01', '2013-12-31'), 'valid_2014-01_2020-12': ('2014-01-01', '2020-12-31'),
       'hold_2021-03_2026-10': ('2021-03-01', '2026-10-31'), 'A_2008-02_2013-01sig': ('2008-02-01', '2013-01-31')}


def paired(nav, ref, a, b, active=None):
    nd = nav.reindex(DATES); rd = ref.reindex(DATES)
    pr = (nd.shift(-1) / nd - 1).iloc[:-1]; rr = (rd.shift(-1) / rd - 1).iloc[:-1]
    m = (pr.index >= pd.Timestamp(a)) & (pr.index <= pd.Timestamp(b))
    if active is not None: m &= np.array([active.get(x, False) for x in pr.index])
    ex = (pr - rr)[m].dropna()
    if len(ex) < 6: return {'n_months': int(len(ex))}
    out = {'n_months': int(len(ex)), 'excess_vs_main_pct_m': round(float(ex.mean() * 100), 2), 't_vs_main_m': L.tstat(ex), 'hit_vs_main': round(float((ex > 0).mean()), 3)}
    if active is not None:   # 只在有效月的累積報酬
        out['cum_cross_pct'] = round(float((1 + pr[m]).prod() - 1) * 100, 1); out['cum_main_pct'] = round(float((1 + rr[m]).prod() - 1) * 100, 1)
    return out


b1, s1 = L.cost_rates(0.001); bS, sS = L.cost_rates('stress')
ev0, pools0, _, _ = select(30, 'main')
ew, _ = L.sim_portfolio(pools0, 0.0, 0.0, t0, t1)
# 主策略重現檢查
evL, _ = S.selection(30)
res['check_main_equals_lhlib_n30'] = all(k in evL and np.array_equal(evL[k][1], ev0[k][1]) for k in ev0) and len(evL) == len(ev0)
navs = {'EW_pool': ew}; out = {}
REF = {}
for n in (30, 10):
    evm, _, _, nbm = select(n, 'main')
    REF[n] = {c: L.sim_portfolio(evm, *(L.cost_rates(c)), t0, t1)[0] for c in (0.001, 'stress')}
    navs[f'main_n{n}'] = REF[n][0.001]
    out[f'main_n{n}'] = {'skipped_limit_up': nbm, 'windows': {k: L.daily_stats(REF[n][0.001], [s], DATES, L.E50, ew) for k, s in SEG.items()},
                         'windows_stress': {k: {kk: v[kk] for kk in ('CAGR', 'maxDD')} for k, s in SEG.items() if (v := L.daily_stats(REF[n]['stress'], [s], DATES))}}
    print('main', n, {k: v['CAGR'] for k, v in out[f'main_n{n}']['windows'].items()}, flush=True)
    for mode in ('a', 'b'):
        for sk in ('main', 'ifrs_skip'):
            key = f'{mode}{n}' + ('' if sk == 'main' else '_DIAG_ifrs_skip')
            ev, _, fb, nb = select(n, mode, sk)
            o = {'skipped_limit_up': nb, 'fallback_months': int(sum(fb.values())), 'active_months': int(sum(not v for v in fb.values())),
                 'fallback_signal_dates_range': [str(min(k for k, v in fb.items() if v).date()), str(max(k for k, v in fb.items() if v).date())] if any(fb.values()) else None}
            ov = [len(set(ev[k][1]) & set(evm[k][1])) / n for k in ev if not fb[DATES[np.searchsorted(DATES, IDX[k - 1])]]]
            o['overlap_with_main_active'] = round(float(np.mean(ov)), 3) if ov else None
            act = {k: (not v) for k, v in fb.items()}
            for cname, c in (('slip0.001', 0.001), ('stress', 'stress')):
                nav, tov = L.sim_portfolio(ev, *L.cost_rates(c), t0, t1)
                if cname == 'slip0.001': navs[key] = nav
                w = {}
                for k, s in SEG.items():
                    st = L.daily_stats(nav, [s], DATES, L.E50, ew) if cname == 'slip0.001' else {kk: v for kk, v in L.daily_stats(nav, [s], DATES).items() if kk in ('CAGR', 'maxDD')}
                    st['main_CAGR'] = L.daily_stats(REF[n][c], [s], DATES)['CAGR']
                    st['paired_vs_main'] = paired(nav, REF[n][c], *s)
                    st['paired_vs_main_activeOnly'] = paired(nav, REF[n][c], *s, active=act)
                    w[k] = st
                o[cname] = {'turnover': round(tov, 3), 'windows': w}
            out[key] = o
            w = o['slip0.001']['windows']
            print(key, {k: (v['CAGR'], v['main_CAGR'], v['paired_vs_main'].get('t_vs_main_m'), v['paired_vs_main_activeOnly'].get('t_vs_main_m')) for k, v in w.items()}, o['fallback_months'], round(time.time() - T0), flush=True)
res['results'] = out
res['0050'] = {k: L.daily_stats(L.E50.iloc[t0:t1 + 1], [s], DATES)['CAGR'] for k, s in SEG.items()}
# 判定
V = {}
for key in ('a30', 'a10', 'b30', 'b10'):
    w = out[key]['slip0.001']['windows']
    ok = {p: bool(w[p]['CAGR'] > w[p]['main_CAGR'] and (w[p]['paired_vs_main'].get('t_vs_main_m') or -9) >= 2) for p in ('valid_2014-01_2020-12', 'hold_2021-03_2026-10')}
    beat = {p: bool(w[p]['CAGR'] > w[p]['main_CAGR']) for p in ('valid_2014-01_2020-12', 'hold_2021-03_2026-10')}
    V[key] = {'pass': ok, 'beat_CAGR': beat}
res['verdict_by_combo'] = V
F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'cross_nav.parquet')
print(json.dumps(V, ensure_ascii=False), res['check_main_equals_lhlib_n30'], round(time.time() - T0))
