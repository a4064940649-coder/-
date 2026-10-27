#!/usr/bin/env python3
"""首板低開回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
漲停價 import agents/w3_limitup_pullback/lim.py；成本 import exec_rebuild/exlib.cost_rates；統計架構仿 w3_limitup_pullback/run.py（複製，不 import）。"""
import json, math, sys
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
AOd = OP * fac; AHd = HI * fac
AO = AOd.to_numpy(); AC = ADJ.to_numpy(); ACf = ADJ.ffill().to_numpy()
CRn = CR.to_numpy(); OPn = OP.to_numpy()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
FLAGOK = Q.FLAGOK.to_numpy()
UNIV = ((V60 >= 1e7) & (CR >= 10) & Q.FLAGOK & ADJ.notna()).to_numpy()
LU = Q.CLOSE_LU.copy()
LUf = pd.DataFrame(LU.astype(float), index=IDX)
LU_prev10 = (LUf.shift(1).rolling(10, min_periods=1).max() >= 1).to_numpy()
lo60 = ADJ.rolling(60, min_periods=40).min(); hi60 = AHd.rolling(60, min_periods=40).max()
with np.errstate(invalid='ignore', divide='ignore'):
    RP = ((ADJ - lo60) / (hi60 - lo60)).where(hi60 > lo60).to_numpy()

# 跌停價（升降單位表同 lim.py，無條件進位）
with np.errstate(invalid='ignore'):
    xdn = Q.REF * (1 - Q.lim)
    tkd = Q.tick_of(np.nan_to_num(xdn, nan=0.0), (IDX < pd.Timestamp('2005-03-01'))[:, None])
    LIMDN = np.ceil(xdn / tkd - 1e-6) * tkd
    OPEN_LD = OPn <= LIMDN + 0.5 * tkd
    CLOSE_LD = CRn <= LIMDN + 0.5 * tkd

# 處置（disposal_reconcile rlib.records 同定義）
dsp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
DISP = np.zeros((T, N), bool); CI = {c: j for j, c in enumerate(cols)}
for r in dsp.itertuples():
    if r.code in CI and pd.notna(r.start) and pd.notna(r.end):
        s = IDX.searchsorted(r.start); e = IDX.searchsorted(r.end, 'right') - 1
        if e >= s: DISP[s:e + 1, CI[r.code]] = True
DISP_START = dsp['start'].min()

# 基準：分段（開→收、收→開）等權
UNIVlag = np.vstack([np.zeros((1, N), bool), UNIV[:-1]])
ACprev = np.vstack([np.full((1, N), np.nan), AC[:-1]])
with np.errstate(invalid='ignore', divide='ignore'):
    OCr = np.clip(AC / AO - 1, -0.5, 1.0); COr = np.clip(AO / ACprev - 1, -0.5, 1.0)
def segmean(x, m):
    y = np.where(m & np.isfinite(x), x, np.nan)
    with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore')
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
    """從 (t, kind) 起找第一個有價且非跌停的時點；回傳 (t, kind, price, delayed)。下市 → 最後收盤。"""
    t0 = t; delayed = False
    while t < T and t <= t0 + maxd:
        if kind == 'o':
            if np.isfinite(AO[t, j]) and not OPEN_LD[t, j]: return t, 'o', AO[t, j], delayed
            kind = 'c'
        else:
            if np.isfinite(AC[t, j]) and not CLOSE_LD[t, j]: return t, 'c', AC[t, j], delayed
            kind = 'o'; t += 1
        delayed = True
    if LASTV[j] < t0 + maxd and LASTV[j] >= 0 and t >= LASTV[j]:
        return LASTV[j], 'c', AC[LASTV[j], j], True
    return None


def day0_mask(rp_on):
    m = LU & UNIV & ~LU_prev10 & ~DISP
    if rp_on:
        with np.errstate(invalid='ignore'): m &= (RP <= 0.5)
    return m


def build(rp_on, glo, ghi, hi_incl):
    D0 = day0_mask(rp_on)
    info = {'day0': int(D0.sum()), 'no_day1_open': 0, 'day1_flag_bad': 0, 'day1_disposal': 0, 'gap_out': 0, 'open_limit_down': 0}
    rows = []
    for t0, j in np.argwhere(D0):
        t1 = t0 + 1
        if t1 >= T or not np.isfinite(AO[t1, j]) or not np.isfinite(AC[t0, j]): info['no_day1_open'] += 1; continue
        if not FLAGOK[t1, j]: info['day1_flag_bad'] += 1; continue
        if DISP[t1, j]: info['day1_disposal'] += 1; continue
        g = AO[t1, j] / AC[t0, j]
        if not (g >= glo - 1e-9 and (g <= ghi + 1e-9 if hi_incl else g < ghi - 1e-9)): info['gap_out'] += 1; continue
        if OPEN_LD[t1, j]: info['open_limit_down'] += 1; continue
        p0 = AO[t1, j]
        for ver in 'ABC':
            if ver == 'A':
                if IDX[t1] < DT_START: continue
                sp = sell_point(j, t1, 'c', 0)
                forced = sp is None or sp[0] != t1 or sp[1] != 'c'
                if forced: sp = sell_point(j, t1 + 1, 'o')
            elif ver == 'B':
                sp = sell_point(j, t1 + 1, 'o'); forced = False
            else:
                sp = sell_point(j, t1 + 1, 'c'); forced = False
            if sp is None: continue
            tx, kind, p1, dl = sp
            raw = p1 / p0 - 1
            if ver == 'A' and not forced:
                tax = 0.0015 if IDX[t1] >= DT_TAX_CUT else 0.003
                net = raw - (0.0023 + 0.002 + (tax - 0.0015)); net_s = raw - 0.01
                net_fee = raw - (2 * L.FEE + tax + 0.002)
            else:
                net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1; net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1; net_fee = net
            ew = bench(POOL_OC, POOL_CO, t1, tx, kind); ewl = bench(LST_OC, LST_CO, t1, tx, kind)
            te = tx if kind == 'c' else tx - 1
            b50 = E50n[te] / E50n[t0] - 1 if te > t0 else E50n[t1] / E50n[t0] - 1
            rows.append((ver, cols[j], j, t0, t1, tx, kind, IDX[t0], IDX[t1], g, RP[t0, j], raw, net, net_s, net_fee, ew, ewl, b50,
                         bool(SRC3[t0:tx + 1].any()), bool(dl or forced)))
    df = pd.DataFrame(rows, columns=['ver', 'code', 'j', 't0', 't1', 'tx', 'kind', 'd0', 'entry', 'gap', 'rp', 'raw', 'net', 'net_s', 'net_fee',
                                     'ew', 'ewl', 'b50', 'src3', 'delayed'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab_fee'] = df['net_fee'] - df['ew']
    df['abl'] = df['net'] - df['ewl']; df['ab50'] = df['net'] - df['b50']; df['ab_gross'] = df['raw'] - df['ew']
    return df, info


def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    r = lambda x: round(float(x) * 100, 3)
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'mean_gap_pct': r(df['gap'].mean() - 1),
            'mean_raw_pct': r(df['raw'].mean()), 'median_raw_pct': r(df['raw'].median()), 'mean_net_pct': r(df['net'].mean()),
            'mean_EWpool_pct': r(df['ew'].mean()),
            'mean_ab_gross_pct': r(df['ab_gross'].mean()), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': r(df['ab'].mean()), 'median_ab_net_pct': r(df['ab'].median()),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': r(df['ab_s'].mean()), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_ab_net_feecfg_pct': r(df['ab_fee'].mean()), 't_month_ab_net_feecfg': tstat(gm['ab_fee'].mean()),
            'mean_ab_listedEW_net_pct': r(df['abl'].mean()), 't_month_ab_listedEW_net': tstat(gm['abl'].mean()),
            'mean_net_vs_0050_pct': r(df['ab50'].mean()), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3),
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
        if cost == 'net':
            c_in, c_out = e.raw - e.net, 0.0  # 以事件總成本（報酬差）近似，分到進場日
            cst[t1] += (1 + e.raw) - (1 + e.net)
        elif cost == 'stress':
            cst[t1] += e.raw - e.net_s
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
SPECS = [('PRIMARY', True, 0.96, 0.99, True), ('GAP_SHALLOW', True, 0.98, 1.00, False), ('GAP_DEEP', True, 0.94, 0.97, True), ('NO_RP', False, 0.96, 0.99, True)]


def passes(o, w):
    e = o[f'event_{w}']; p = o[f'port_{w}']
    if e.get('n', 0) < 30 or e.get('n_months', 0) < 24: return {'thin': True, 'pass': False, 'beat_EW_t2': None, 'not_lose_0050': None}
    ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
    ok_50 = p['CAGR'] >= p['0050_CAGR']
    return {'thin': False, 'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}


def main():
    d0p = day0_mask(True); d0n = day0_mask(False)
    out = {'meta': {'bad_days_dropped': Q.BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'disposal_list_start': str(DISP_START.date()),
                    'first_board_day0_noRP': int(d0n.sum()), 'first_board_day0_rp_le_0.5': int(d0p.sum()),
                    'day0_by_year_rp': {int(y): int(v) for y, v in pd.Series(d0p.sum(1), index=IDX).groupby(IDX.year).sum().items()},
                    'day0_by_year_noRP': {int(y): int(v) for y, v in pd.Series(d0n.sum(1), index=IDX).groupby(IDX.year).sum().items()}}}
    res = {}; yearly_ev = {}
    for key, rp_on, glo, ghi, incl in SPECS:
        df, info = build(rp_on, glo, ghi, incl)
        res[key] = {'params': {'rp_filter': rp_on, 'gap_lo': glo, 'gap_hi': ghi, 'gap_hi_inclusive': incl}, **info}
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
            if key == 'PRIMARY':
                pd.DataFrame({'net': rn, 'stress': rs, 'gross': rg, 'npos': npos}).to_parquet(HERE / f'nav_primary_{ver}.parquet')
        df.drop(columns=['j']).to_csv(HERE / f'events_{key}.csv.gz', index=False)
    out['results'] = res; out['yearly_events'] = yearly_ev
    pr = res['PRIMARY']['C']; vs = {w: pr['pass'][w] for w in (SEL, VAL)}
    thin = any(pr['pass'][w]['thin'] for w in (SEL, VAL))
    k = sum(pr['pass'][w]['pass'] for w in (SEL, VAL))
    vs['verdict'] = '無法測試' if thin else ('可用' if k == 2 else ('待觀察' if k == 1 else '不可用'))
    ab_both = [v for v in 'AB' if all(res['PRIMARY'][v]['pass'][w]['pass'] for w in (SEL, VAL))]
    if vs['verdict'] == '不可用' and ab_both: vs['verdict'] = '待觀察'
    vs['variants_pass_count'] = {f'{kk}_{v}': sum(res[kk][v]['pass'][w]['pass'] for w in (SEL, VAL)) for kk in res for v in 'ABC'}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))


if __name__ == '__main__':
    main()
