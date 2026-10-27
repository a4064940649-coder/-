#!/usr/bin/env python3
"""炸板次日反包回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
漲停價 import agents/w3_limitup_pullback/lim.py；成本 import exec_rebuild/exlib.cost_rates；
事件統計／組合架構複製自 w12_first_board_gap_down/run.py（不 import、不改原檔）。"""
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
ADJ, CR, OP, HI, V = Q.ADJ, Q.CR, Q.OP, Q.HI, Q.V
fac = ADJ / CR
AO = (OP * fac).to_numpy(); AH = (HI * fac).to_numpy(); AC = ADJ.to_numpy(); ACf = ADJ.ffill().to_numpy()
CRn = CR.to_numpy(); OPn = OP.to_numpy(); HIn = HI.to_numpy()
FLAGOK = Q.FLAGOK.to_numpy()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
UNIV = ((V60 >= 1e7) & (CR >= 10) & Q.FLAGOK & ADJ.notna()).to_numpy()

LIMUP, tk, REF = Q.LIMUP, Q.tk, Q.REF
with np.errstate(invalid='ignore'):
    TOUCH = (HIn >= LIMUP - 0.5 * tk) & FLAGOK
    CLOSE_LU = Q.CLOSE_LU
    NOTSEAL = CRn < LIMUP - 1.5 * tk                      # 收盤 < 漲停價 − 1 檔（嚴格）
    RED3 = CRn >= REF * 1.03 - 1e-9
    OPEN_LU = OPn >= LIMUP - 0.5 * tk
HITANY = pd.DataFrame((TOUCH | CLOSE_LU).astype(float), index=IDX)
PREV10 = (HITANY.shift(1).rolling(10, min_periods=1).max() >= 1).to_numpy()

with np.errstate(invalid='ignore'):
    xdn = REF * (1 - Q.lim)
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
D0 = TOUCH & NOTSEAL & RED3 & UNIV & ~PREV10 & ~DISP


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
    if LASTV[j] < t0 + maxd and LASTV[j] >= 0 and t >= LASTV[j]:
        return LASTV[j], 'c', AC[LASTV[j], j], True
    return None


def costs(raw, ti, daytrade):
    if daytrade:
        tax = 0.0015 if IDX[ti] >= DT_TAX_CUT else 0.003
        return raw - (0.0023 + 0.002 + (tax - 0.0015)), raw - 0.01
    return (1 + raw) * (1 - SELL) / (1 + BUY) - 1, (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1


def row(name, j, t0, ti, sp, extra, daytrade=False):
    tx, kind, p1, dl = sp
    raw = p1 / AO[ti, j] - 1
    net, net_s = costs(raw, ti, daytrade)
    ew = bench(POOL_OC, POOL_CO, ti, tx, kind); ewl = bench(LST_OC, LST_CO, ti, tx, kind)
    te = tx if kind == 'c' else tx - 1
    b50 = E50n[te] / E50n[ti - 1] - 1 if te > ti - 1 else 0.0
    return (name, cols[j], j, t0, ti, tx, kind, IDX[t0], IDX[ti], raw, net, net_s, ew, ewl, b50,
            bool(SRC3[t0:tx + 1].any()), bool(dl), extra.get('stop', False), tx - ti)


def build_main(name, H, thr):
    info = {'day0': int(D0.sum()), 'no_day1_close': 0, 'not_fanbao': 0, 'day2_unavailable': 0, 'disposal': 0, 'day2_open_LU': 0,
            'overlap_skip': 0, 'tail_or_no_exit': 0}
    rows = []; last_exit = np.full(N, -1)
    for t0, j in sorted(map(tuple, np.argwhere(D0))):
        t1, te = t0 + 1, t0 + 2
        if te >= T or not np.isfinite(AC[t1, j]) or not FLAGOK[t1, j]: info['no_day1_close'] += 1; continue
        ok = AC[t1, j] > AH[t0, j] if thr == 'high' else AC[t1, j] > AC[t0, j] * 1.05
        if not ok: info['not_fanbao'] += 1; continue
        if not np.isfinite(AO[te, j]) or not FLAGOK[te, j]: info['day2_unavailable'] += 1; continue
        if DISP[t1, j] or DISP[te, j]: info['disposal'] += 1; continue
        if OPEN_LU[te, j]: info['day2_open_LU'] += 1; continue
        if te <= last_exit[j]: info['overlap_skip'] += 1; continue
        if te + H >= T: info['tail_or_no_exit'] += 1; continue
        stop_t = None
        for t in range(te, te + H):
            if np.isfinite(AC[t, j]) and AC[t, j] < AC[t0, j]: stop_t = t; break
        sp = sell_point(j, (stop_t + 1) if stop_t is not None else te + H, 'o')
        if sp is None: info['tail_or_no_exit'] += 1; continue
        rows.append(row(name, j, t0, te, sp, {'stop': stop_t is not None})); last_exit[j] = sp[0]
    return rows, info


def build_aggr():
    info = {'day0': int(D0.sum()), 'day1_unavailable': 0, 'disposal': 0, 'day1_open_LU': 0, 'no_exit': 0}
    rows = []
    for t0, j in sorted(map(tuple, np.argwhere(D0))):
        t1 = t0 + 1
        if t1 + 1 >= T or not np.isfinite(AO[t1, j]) or not FLAGOK[t1, j]: info['day1_unavailable'] += 1; continue
        if DISP[t1, j]: info['disposal'] += 1; continue
        if OPEN_LU[t1, j]: info['day1_open_LU'] += 1; continue
        if IDX[t1] >= DT_START:
            sp = sell_point(j, t1, 'c', 0)
            if sp is not None and sp[0] == t1 and sp[1] == 'c':
                rows.append(row('AGGR_A', j, t0, t1, sp, {}, daytrade=True))
            else:
                sp = sell_point(j, t1 + 1, 'o')
                if sp is not None: rows.append(row('AGGR_A', j, t0, t1, (*sp[:3], True), {}))
        sp = sell_point(j, t1 + 1, 'o')
        if sp is None: info['no_exit'] += 1; continue
        rows.append(row('AGGR_B', j, t0, t1, sp, {}))
    return rows, info


COLN = ['ver', 'code', 'j', 't0', 'ti', 'tx', 'kind', 'd0', 'entry', 'raw', 'net', 'net_s', 'ew', 'ewl', 'b50', 'src3', 'delayed', 'stop', 'hold']


def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    r = lambda x: round(float(x) * 100, 3)
    return {'n': int(len(df)), 'n_months': int(gm.ngroups),
            'mean_raw_pct': r(df['raw'].mean()), 'median_raw_pct': r(df['raw'].median()), 'mean_net_pct': r(df['net'].mean()),
            'mean_EWpool_pct': r(df['ew'].mean()),
            'mean_ab_gross_pct': r(df['ab_gross'].mean()), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': r(df['ab'].mean()), 'median_ab_net_pct': r(df['ab'].median()),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': r(df['ab_s'].mean()), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_ab_listedEW_net_pct': r(df['abl'].mean()), 't_month_ab_listedEW_net': tstat(gm['abl'].mean()),
            'mean_net_vs_0050_pct': r(df['ab50'].mean()), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3),
            'share_stop': round(float(df['stop'].mean()), 3), 'mean_hold_days': round(float(df['hold'].mean()), 2),
            'share_delayed_or_forced': round(float(df['delayed'].mean()), 4),
            'mean_ab_net_ex_top5pct': r(df['ab'][df['ab'] <= df['ab'].quantile(0.95)].mean())}


def portfolio(df, cost):
    R = np.zeros((T, N)); W = np.zeros((T, N), np.float32); cst = np.zeros(T)
    for e in df.itertuples():
        j, ti, tx, kind = e.j, e.ti, e.tx, e.kind
        W[ti:tx + 1, j] = 1
        R[ti, j] = (AC[ti, j] if np.isfinite(AC[ti, j]) else ACf[ti, j]) / AO[ti, j] - 1 if (tx > ti or kind == 'c') else 0.0
        for t in range(ti + 1, tx + 1):
            end = AO[t, j] if (t == tx and kind == 'o') else ACf[t, j]
            R[t, j] = end / ACf[t - 1, j] - 1
        if cost == 'net': cst[ti] += e.raw - e.net
        elif cost == 'stress': cst[ti] += e.raw - e.net_s
    n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(n > 0, (W * np.nan_to_num(R)).sum(1) / np.where(n > 0, n, 1), 0.0)
        c = np.where(n > 0, cst / np.where(n > 0, n, 1), 0.0)
    return pd.Series(g - c, index=IDX), pd.Series(n, index=IDX)


def pstats(r, npos, segs):
    m = np.zeros(len(r), bool); yrs = 0.0
    for a, b in segs:
        mm = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
        if mm.sum() < 2: continue
        m |= mm; ii = r.index[mm]; yrs += (ii[-1] - ii[0]).days / 365.25
    rr = r[m]; bb = r50[m]; ee = ew_cc[m]
    def st(x):
        eq = (1 + x).cumprod()
        return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {'years': round(yrs, 2)}
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


SEL, VAL = 'select_2005-01_2019-04', 'valid_judge_2019-05_2026-10_ex_src3'
# (事件 Day0 期間, 是否剔除 src3 事件, 組合日報酬分段)
WIN = {
 SEL: ('2005-01-01', '2019-04-30', False, [('2005-01-01', '2019-04-30')]),
 VAL: ('2019-05-01', '2026-10-31', True, [('2019-05-01', '2024-02-14'), ('2025-09-24', '2026-10-31')]),
 'valid_all_2019-05_2026-10_incl_src3': ('2019-05-01', '2026-10-31', False, [('2019-05-01', '2026-10-31')]),
 'y2024_2025_incl_src3': ('2024-01-01', '2025-12-31', False, [('2024-01-01', '2025-12-31')]),
 'y2024_2025_ex_src3': ('2024-01-01', '2025-12-31', True, [('2025-09-24', '2025-12-31')]),
 'd2021_select_2021-01_2023-12': ('2021-01-01', '2023-12-31', False, [('2021-01-01', '2023-12-31')]),
 'd2021_valid_2024-01_2026-10_ex_src3': ('2024-01-01', '2026-10-31', True, [('2025-09-24', '2026-10-31')]),
 'd2021_valid_2024-01_2026-10_incl_src3': ('2024-01-01', '2026-10-31', False, [('2024-01-01', '2026-10-31')]),
 'select_sub_7pct_2005-01_2015-05': ('2005-01-01', '2015-05-31', False, [('2005-01-01', '2015-05-31')]),
 'select_sub_10pct_2015-06_2019-04': ('2015-06-01', '2019-04-30', False, [('2015-06-01', '2019-04-30')]),
}


def passes(o, w):
    e = o[f'event_{w}']; p = o[f'port_{w}']
    if e.get('n', 0) < 30 or e.get('n_months', 0) < 24: return {'thin': True, 'pass': False, 'beat_EW_t2': None, 'not_lose_0050': None}
    ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
    ok_50 = p['CAGR'] >= p['0050_CAGR']
    return {'thin': False, 'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}


def main():
    seg = np.select([SRC3, IDX >= pd.Timestamp('2025-09-24')], ['src3', 'src4'], 'src12')
    d0s = pd.Series(D0.sum(1), index=IDX); tch = pd.Series((TOUCH & UNIV).sum(1), index=IDX); clu = pd.Series((CLOSE_LU & UNIV).sum(1), index=IDX)
    diag = {}
    for y in range(2019, 2027):
        for s in ('src12', 'src3', 'src4'):
            mk = (IDX.year == y) & (seg == s)
            if mk.sum() == 0: continue
            diag[f'{y}_{s}'] = {'days': int(mk.sum()), 'broken_day0_per_day': round(float(d0s[mk].mean()), 2),
                                'touch_per_day': round(float(tch[mk].mean()), 2), 'closeLU_per_day': round(float(clu[mk].mean()), 2),
                                'broken_over_touch': round(float(d0s[mk].sum() / max(tch[mk].sum(), 1)), 3)}
    out = {'meta': {'bad_days_dropped': Q.BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'disposal_list_start': str(DISP_START.date()), 'day0_broken_total': int(D0.sum()),
                    'day0_by_year': {int(y): int(v) for y, v in d0s.groupby(IDX.year).sum().items()},
                    'src_segment_diag_2019on': diag}}
    allrows = []; infos = {}
    for name, H, thr in [('MAIN_H5', 5, 'high'), ('MAIN_H3', 3, 'high'), ('MAIN_H10', 10, 'high'), ('THR105_H5', 5, 'c105')]:
        rws, info = build_main(name, H, thr); allrows += rws; infos[name] = info
    rws, info = build_aggr(); allrows += rws; infos['AGGR'] = info
    df = pd.DataFrame(allrows, columns=COLN)
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['abl'] = df['net'] - df['ewl']
    df['ab50'] = df['net'] - df['b50']; df['ab_gross'] = df['raw'] - df['ew']
    res = {}; yearly_ev = {}
    for ver in ['MAIN_H5', 'MAIN_H3', 'MAIN_H10', 'THR105_H5', 'AGGR_A', 'AGGR_B']:
        dv = df[df['ver'] == ver]
        o = {'n_events': int(len(dv)), 'build_info': infos['AGGR' if ver.startswith('AGGR') else ver]}
        navs = {}
        for src_ex in (False, True):
            dd = dv[~dv['src3']] if src_ex else dv
            navs[src_ex] = (portfolio(dd, 'net'), portfolio(dd, 'stress')[0], portfolio(dd, 'gross')[0])
        for wn, (a, b, ex, segs) in WIN.items():
            e = dv[(dv['d0'] >= pd.Timestamp(a)) & (dv['d0'] <= pd.Timestamp(b))]
            if ex: e = e[~e['src3']]
            o[f'event_{wn}'] = ev_stats(e)
            if ver == 'AGGR_A' and wn in (SEL, 'select_sub_7pct_2005-01_2015-05'):
                segs = [(max(pd.Timestamp(s0), DT_START), s1) for s0, s1 in segs if pd.Timestamp(s1) >= DT_START]
            (rn, npos), rs, rg = navs[ex]
            if not segs: o[f'port_{wn}'] = None; continue
            o[f'port_{wn}'] = pstats(rn, npos, segs)
            o[f'port_{wn}']['CAGR_stress'] = pstats(rs, npos, segs)['CAGR']
            o[f'port_{wn}']['CAGR_gross'] = pstats(rg, npos, segs)['CAGR']
        o['event_touch_src3'] = ev_stats(dv[dv['src3']])
        o['pass'] = {SEL: passes(o, SEL), VAL: passes(o, VAL)}
        res[ver] = o
        yearly_ev[ver] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2),
                                   'src3_events': int(g['src3'].sum())} for y, g in dv.groupby(dv['d0'].dt.year)}
        es, ev_ = o[f'event_{SEL}'], o[f'event_{VAL}']; ps, pv = o[f'port_{SEL}'], o[f'port_{VAL}']
        print(ver, 'n', len(dv), '| S n', es.get('n'), 'ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'gross', es.get('mean_ab_gross_pct'),
              'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], '| V n', ev_.get('n'), 'ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'),
              'gross', ev_.get('mean_ab_gross_pct'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], flush=True)
        if ver == 'MAIN_H5':
            (rn, npos), rs, rg = navs[False]
            pd.DataFrame({'net': rn, 'stress': rs, 'gross': rg, 'npos': npos}).to_parquet(HERE / 'nav_main_h5.parquet')
    df.drop(columns=['j']).to_csv(HERE / 'events.csv.gz', index=False)
    out['results'] = res; out['yearly_events'] = yearly_ev
    pr = res['MAIN_H5']; vs = {w: pr['pass'][w] for w in (SEL, VAL)}
    thin = any(pr['pass'][w]['thin'] for w in (SEL, VAL))
    k = sum(pr['pass'][w]['pass'] for w in (SEL, VAL))
    vs['verdict'] = '無法測試' if thin else ('可用' if k == 2 else ('待觀察' if k == 1 else '不可用'))
    others = [v for v in res if v != 'MAIN_H5' and all(res[v]['pass'][w]['pass'] for w in (SEL, VAL))]
    if vs['verdict'] == '不可用' and others: vs['verdict'] = '待觀察'
    vs['variants_pass_count'] = {v: sum(res[v]['pass'][w]['pass'] for w in (SEL, VAL)) for v in res}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))


if __name__ == '__main__':
    main()
