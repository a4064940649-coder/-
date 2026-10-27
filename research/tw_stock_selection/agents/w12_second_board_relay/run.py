#!/usr/bin/env python3
"""一進二／二板接力回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
漲停價 import agents/w3_limitup_pullback/lim.py；成本 import exec_rebuild/exlib.cost_rates；架構複製 w12_first_board_gap_down/run.py（不 import）。"""
import json, math, sys, warnings
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'w3_limitup_pullback'))
import lim as Q  # noqa: E402
import exlib as L  # noqa: E402
from btlib import tstat  # noqa: E402

X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
IDX, T, N, cols = Q.IDX, Q.T, Q.N, Q.cols
ADJ, CR, OP, HI, V, F = Q.ADJ, Q.CR, Q.OP, Q.HI, Q.V, Q.F
fac = ADJ / CR
AO = (OP * fac).to_numpy(); AC = ADJ.to_numpy(); ACf = ADJ.ffill().to_numpy()
CRn = CR.to_numpy(); OPn = OP.to_numpy()
VOL = (V / CR).to_numpy()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
FLAGOK = Q.FLAGOK.to_numpy()
UNIV = ((V60 >= 1e7) & (CR >= 10) & Q.FLAGOK & ADJ.notna()).to_numpy()
LU = Q.CLOSE_LU.copy()
OPEN_LU = Q.OPEN_LU
LUf = pd.DataFrame(LU.astype(float), index=IDX)
def lu_any(lag_from, lag_to):
    """t−lag_to .. t−lag_from 任一日收盤漲停"""
    w = lag_to - lag_from + 1
    return (LUf.shift(lag_from).rolling(w, min_periods=1).max() >= 1).to_numpy()
LUm1 = np.vstack([np.zeros((1, N), bool), LU[:-1]]); LUm2 = np.vstack([np.zeros((2, N), bool), LU[:-2]])
VOLm1 = np.vstack([np.full((1, N), np.nan), VOL[:-1]])

with np.errstate(invalid='ignore'):
    xdn = Q.REF * (1 - Q.lim)
    tkd = Q.tick_of(np.nan_to_num(xdn, nan=0.0), (IDX < pd.Timestamp('2005-03-01'))[:, None])
    LIMDN = np.ceil(xdn / tkd - 1e-6) * tkd
    OPEN_LD = OPn <= LIMDN + 0.5 * tkd
    CLOSE_LD = CRn <= LIMDN + 0.5 * tkd

dsp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
DISP = np.zeros((T, N), bool); CI = {c: j for j, c in enumerate(cols)}
for r in dsp.itertuples():
    if r.code in CI and pd.notna(r.start) and pd.notna(r.end):
        s = IDX.searchsorted(r.start); e = IDX.searchsorted(r.end, 'right') - 1
        if e >= s: DISP[s:e + 1, CI[r.code]] = True
DISP_START = dsp['start'].min()

UNIVlag = np.vstack([np.zeros((1, N), bool), UNIV[:-1]])
ACprev = np.vstack([np.full((1, N), np.nan), AC[:-1]])
with np.errstate(invalid='ignore', divide='ignore'):
    OCr = np.clip(AC / AO - 1, -0.5, 1.0); COr = np.clip(AO / ACprev - 1, -0.5, 1.0)
def segmean(x, m):
    y = np.where(m & np.isfinite(x), x, np.nan)
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nan_to_num(np.nanmean(y, axis=1))
POOL_OC, POOL_CO = segmean(OCr, UNIVlag), segmean(COr, UNIVlag)
LST_OC, LST_CO = segmean(OCr, np.ones((T, N), bool)), segmean(COr, np.ones((T, N), bool))
ew_cc = Q.RET.where(pd.DataFrame(UNIVlag, index=IDX, columns=cols)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0); E50n = e50.to_numpy()
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))
BUY, SELL = L.cost_rates(0.001); BUY_S, SELL_S = L.cost_rates('stress')
DT_START = pd.Timestamp('2014-01-06'); DT_TAX_CUT = pd.Timestamp('2017-04-28')
LASTV = np.array([np.max(np.where(np.isfinite(AC[:, j]))[0]) if np.isfinite(AC[:, j]).any() else -1 for j in range(N)])


def bench(seg_oc, seg_co, t1, tx, kind):
    g = 1 + seg_oc[t1] if (tx > t1 or kind == 'c') else 1.0
    for t in range(t1 + 1, tx + 1):
        g *= 1 + seg_co[t]
        if t < tx or kind == 'c': g *= 1 + seg_oc[t]
    return g - 1


def sell_point(j, t, kind, maxd=10):
    t0 = t; delayed = False
    while t < T and t <= t0 + maxd:
        if kind == 'o':
            if np.isfinite(AO[t, j]) and not OPEN_LD[t, j]: return t, 'o', AO[t, j], delayed
            kind = 'c'
        else:
            if np.isfinite(AC[t, j]) and not CLOSE_LD[t, j]: return t, 'c', AC[t, j], delayed
            kind = 'o'; t += 1
        delayed = True
    if LASTV[j] >= 0 and LASTV[j] < t0 + maxd and t >= LASTV[j]:
        return LASTV[j], 'c', AC[LASTV[j], j], True
    return None


def day0_mask(key):
    base = UNIV & ~DISP & LU
    if key == 'CONTROL_FIRST':
        return base & ~lu_any(1, 10)
    if key == 'THREE_BOARD':
        return base & LUm1 & LUm2 & ~lu_any(3, 12)
    m = base & LUm1 & ~lu_any(2, 11)
    with np.errstate(invalid='ignore'):
        if key == 'VOL_SHRINK': m &= (VOL <= VOLm1)
        if key == 'VOL_EXPAND': m &= (VOL > VOLm1)
    return m


def build(key):
    D0 = day0_mask(key)
    info = {'day0': int(D0.sum()), 'no_day1_open': 0, 'day1_flag_bad': 0, 'day1_disposal': 0, 'day1_open_limitup': 0, 'gap_below_0.97': 0,
            'overlap_skipped': 0, 'entered': 0}
    rows = []; busy = {}
    for t0, j in sorted(map(tuple, np.argwhere(D0))):
        t1 = t0 + 1
        if t1 >= T or not np.isfinite(AO[t1, j]) or not np.isfinite(AC[t0, j]): info['no_day1_open'] += 1; continue
        if not FLAGOK[t1, j]: info['day1_flag_bad'] += 1; continue
        if DISP[t1, j]: info['day1_disposal'] += 1; continue
        if OPEN_LU[t1, j]: info['day1_open_limitup'] += 1; continue
        g = AO[t1, j] / AC[t0, j]
        if not (g >= 0.97 - 1e-9): info['gap_below_0.97'] += 1; continue
        if busy.get(j, -1) >= t1: info['overlap_skipped'] += 1; continue
        info['entered'] += 1
        p0 = AO[t1, j]; nb = 0
        # C：連板持有
        k = t1
        while k < t1 + 10 and k < T and np.isfinite(AC[k, j]) and LU[k, j]:
            k += 1
        nb = k - t1  # Day1 起連續收盤漲停天數
        spC = sell_point(j, k + 1, 'o') if k < t1 + 10 else sell_point(j, t1 + 10, 'o')
        mx = t1
        for ver in 'ABC':
            if ver == 'A':
                if IDX[t1] < DT_START: continue
                sp = sell_point(j, t1, 'c', 0)
                forced = sp is None or sp[0] != t1 or sp[1] != 'c'
                if forced: sp = sell_point(j, t1 + 1, 'o')
            elif ver == 'B':
                sp = sell_point(j, t1 + 1, 'o'); forced = False
            else:
                sp = spC; forced = False
            if sp is None: continue
            tx, kind, p1, dl = sp; mx = max(mx, tx)
            raw = p1 / p0 - 1
            if ver == 'A' and not forced:
                tax = 0.0015 if IDX[t1] >= DT_TAX_CUT else 0.003
                net = raw - (0.0023 + 0.002 + (tax - 0.0015)); net_s = raw - 0.01
            else:
                net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1; net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
            ew = bench(POOL_OC, POOL_CO, t1, tx, kind); ewl = bench(LST_OC, LST_CO, t1, tx, kind)
            te = tx if kind == 'c' else tx - 1
            b50 = E50n[te] / E50n[t0] - 1 if te > t0 else E50n[t1] / E50n[t0] - 1
            rows.append((ver, cols[j], j, t0, t1, tx, kind, IDX[t0], IDX[t1], g, nb, VOL[t0, j] / VOLm1[t0, j] if VOLm1[t0, j] > 0 else np.nan,
                         raw, net, net_s, ew, ewl, b50, bool(SRC3[t0 - 2:tx + 1].any()), bool(dl or forced)))
        busy[j] = mx
    df = pd.DataFrame(rows, columns=['ver', 'code', 'j', 't0', 't1', 'tx', 'kind', 'd0', 'entry', 'gap', 'n_lu_after', 'vol_ratio',
                                     'raw', 'net', 'net_s', 'ew', 'ewl', 'b50', 'src3', 'delayed'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']
    df['abl'] = df['net'] - df['ewl']; df['ab50'] = df['net'] - df['b50']; df['ab_gross'] = df['raw'] - df['ew']
    df['hold_days'] = df['tx'] - df['t1']
    return df, info, D0


def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    r = lambda x: round(float(x) * 100, 3)
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'mean_gap_pct': r(df['gap'].mean() - 1), 'mean_hold_days': round(float(df['hold_days'].mean()), 2),
            'mean_raw_pct': r(df['raw'].mean()), 'median_raw_pct': r(df['raw'].median()), 'mean_net_pct': r(df['net'].mean()),
            'mean_EWpool_pct': r(df['ew'].mean()),
            'mean_ab_gross_pct': r(df['ab_gross'].mean()), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': r(df['ab'].mean()), 'median_ab_net_pct': r(df['ab'].median()),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': r(df['ab_s'].mean()), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_ab_listedEW_net_pct': r(df['abl'].mean()), 't_month_ab_listedEW_net': tstat(gm['abl'].mean()),
            'mean_net_vs_0050_pct': r(df['ab50'].mean()), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3),
            'share_day1_close_LU': round(float((df['n_lu_after'] >= 1).mean()), 3),
            'share_delayed_or_forced': round(float(df['delayed'].mean()), 4),
            'mean_ab_net_ex_top5pct': r(df['ab'][df['ab'] <= df['ab'].quantile(0.95)].mean())}


def portfolio(df, cost):
    R = np.zeros((T, N)); W = np.zeros((T, N), np.float32); cst = np.zeros(T)
    for e in df.itertuples():
        j, t1, tx, kind = e.j, e.t1, e.tx, e.kind
        W[t1:tx + 1, j] = 1
        R[t1, j] = (AC[t1, j] if np.isfinite(AC[t1, j]) else ACf[t1, j]) / AO[t1, j] - 1 if (tx > t1 or kind == 'c') else 0.0
        for t in range(t1 + 1, tx + 1):
            end = AO[t, j] if (t == tx and kind == 'o') else ACf[t, j]
            R[t, j] = end / ACf[t - 1, j] - 1
        if cost == 'net': cst[t1] += e.raw - e.net
        elif cost == 'stress': cst[t1] += e.raw - e.net_s
    n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(n > 0, (W * np.nan_to_num(R)).sum(1) / np.where(n > 0, n, 1), 0.0)
        c = np.where(n > 0, cst / np.where(n > 0, n, 1), 0.0)
    return pd.Series(g - c, index=IDX), pd.Series(n, index=IDX)


def pstats(r, npos, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    rr = r[m]; bb = r50[m]; ee = ew_cc[m]
    yrs = (rr.index[-1] - rr.index[0]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod()
        return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}
    o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
    o['vol'] = round(float(rr.std() * math.sqrt(252)), 4)
    o['share_days_invested'] = round(float((npos[m] > 0).mean()), 3)
    o['avg_positions_when_invested'] = round(float(npos[m][npos[m] > 0].mean()), 2) if (npos[m] > 0).any() else 0
    mo = lambda x: (1 + x).groupby(x.index.to_period('M')).prod() - 1
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    return o


WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'), 'd2021_select_2021-01_2023-12': ('2021-01-01', '2023-12-31'),
       'd2021_valid_2024-01_2026-10': ('2024-01-01', '2026-10-31')}
SUB = {'select_sub_7pct_2005-01_2015-05': ('2005-01-01', '2015-05-31'), 'select_sub_10pct_2015-06_2019-04': ('2015-06-01', '2019-04-30')}
SEL, VAL = 'select_2005-01_2019-04', 'valid_2019-05_2026-10'
KEYS = ['BASE', 'VOL_SHRINK', 'VOL_EXPAND', 'THREE_BOARD', 'CONTROL_FIRST']


def passes(o, w):
    e = o[f'event_{w}']; p = o[f'port_{w}']
    if e.get('n', 0) < 30 or e.get('n_months', 0) < 24: return {'thin': True, 'pass': False, 'beat_EW_t2': None, 'not_lose_0050': None}
    ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
    ok_50 = p['CAGR'] >= p['0050_CAGR']
    return {'thin': False, 'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}


def main():
    out = {'meta': {'bad_days_dropped': Q.BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'disposal_list_start': str(DISP_START.date())}}
    res = {}; yearly_ev = {}; counts = {}
    for key in KEYS:
        df, info, D0 = build(key)
        res[key] = dict(info)
        dc = df[df['ver'] == 'C']
        counts[key] = {'day0_by_year': {int(y): int(v) for y, v in pd.Series(D0.sum(1), index=IDX).groupby(IDX.year).sum().items()},
                       'entered_by_year': {int(y): int(v) for y, v in dc.groupby(dc['d0'].dt.year).size().items()}}
        for ver in 'ABC':
            dv = df[df['ver'] == ver]
            o = {'n_events': int(len(dv))}
            for wn, (a, b) in {**WIN, **SUB}.items():
                o[f'event_{wn}'] = ev_stats(dv[(dv['d0'] >= pd.Timestamp(a)) & (dv['d0'] <= pd.Timestamp(b))])
            vv = dv[dv['d0'] >= pd.Timestamp('2019-05-01')]
            o['event_valid_ex_src3'] = ev_stats(vv[~vv['src3']]); o['event_touch_src3'] = ev_stats(dv[dv['src3']])
            rn, npos = portfolio(dv, 'net'); rs, _ = portfolio(dv, 'stress'); rg, _ = portfolio(dv, 'gross')
            for wn, (a, b) in WIN.items():
                if ver == 'A' and wn == SEL: a = '2014-01-06'
                o[f'port_{wn}'] = pstats(rn, npos, a, b)
                o[f'port_{wn}']['CAGR_stress'] = pstats(rs, npos, a, b)['CAGR']
                o[f'port_{wn}']['CAGR_gross'] = pstats(rg, npos, a, b)['CAGR']
            o['pass'] = {SEL: passes(o, SEL), VAL: passes(o, VAL)}
            res[key][ver] = o
            yearly_ev[f'{key}_{ver}'] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2)}
                                         for y, g in dv.groupby(dv['d0'].dt.year)}
            es, ev_ = o[f'event_{SEL}'], o[f'event_{VAL}']; ps, pv = o[f'port_{SEL}'], o[f'port_{VAL}']
            print(key, ver, 'n', len(dv), '| S n', es.get('n'), 'ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'gross', es.get('mean_ab_gross_pct'),
                  'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], '| V n', ev_.get('n'), 'ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'),
                  'gross', ev_.get('mean_ab_gross_pct'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], flush=True)
            if key == 'BASE':
                pd.DataFrame({'net': rn, 'stress': rs, 'gross': rg, 'npos': npos}).to_parquet(HERE / f'nav_base_{ver}.parquet')
        print(key, info, flush=True)
        df.drop(columns=['j']).to_csv(HERE / f'events_{key}.csv.gz', index=False)
    # src 3 漲停偵測率
    lu = pd.Series(LU.sum(1), index=IDX)
    big = pd.Series(((Q.CC >= np.where(IDX < pd.Timestamp('2015-06-01'), 0.065, 0.095)[:, None]) & np.isfinite(Q.CC)).sum(1), index=IDX)
    seg = pd.Series(np.select([SRC3, IDX >= pd.Timestamp('2025-09-24')], ['src3', 'src4'], 'src12'), index=IDX)
    det = {}
    for y in range(2019, 2027):
        for s in ('src12', 'src3', 'src4'):
            m = (IDX.year == y) & (seg == s).to_numpy()
            if m.sum() == 0: continue
            det[f'{y}_{s}'] = {'days': int(m.sum()), 'close_LU_per_day': round(float(lu[m].mean()), 2), 'cc_ge_9.5pct_per_day': round(float(big[m].mean()), 2),
                               'LU_over_big': round(float(lu[m].sum() / max(big[m].sum(), 1)), 3)}
    out['limit_up_detection_by_segment'] = det
    out['event_counts'] = counts
    out['results'] = res; out['yearly_events'] = yearly_ev
    pr = res['BASE']['C']; vs = {w: pr['pass'][w] for w in (SEL, VAL)}
    thin = any(pr['pass'][w]['thin'] for w in (SEL, VAL))
    k = sum(pr['pass'][w]['pass'] for w in (SEL, VAL))
    vs['verdict'] = '無法測試' if thin else ('可用' if k == 2 else ('待觀察' if k == 1 else '不可用'))
    ab_both = [v for v in 'AB' if all(res['BASE'][v]['pass'][w]['pass'] for w in (SEL, VAL))]
    if vs['verdict'] == '不可用' and ab_both: vs['verdict'] = '待觀察'
    if vs['verdict'] == '可用' and False: pass
    vs['variants_pass_count'] = {f'{kk}_{v}': sum(res[kk][v]['pass'][w]['pass'] for w in (SEL, VAL)) for kk in res for v in 'ABC'}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))


if __name__ == '__main__':
    main()
