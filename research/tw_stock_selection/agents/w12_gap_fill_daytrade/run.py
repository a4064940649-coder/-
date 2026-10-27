#!/usr/bin/env python3
"""窓埋め現股當沖回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
資料與漲跌停價 import agents/w3_limitup_pullback/lim.py；成本 import exec_rebuild/exlib.cost_rates；架構仿 w12_first_board_gap_down/run.py（複製改寫，不 import）。"""
import json, math, sys, warnings
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np, pandas as pd
warnings.simplefilter('ignore')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'w3_limitup_pullback'))
import lim as Q  # noqa: E402
import exlib as L  # noqa: E402
from btlib import tstat  # noqa: E402

X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
IDX, T, N, cols = Q.IDX, Q.T, Q.N, Q.cols
ADJ, CR, OP, V = Q.ADJ, Q.CR, Q.OP, Q.V
fac = ADJ / CR
AO = (OP * fac).to_numpy(); AC = ADJ.to_numpy(); ACf = ADJ.ffill().to_numpy()
CRn = CR.to_numpy(); OPn = OP.to_numpy(); REF = Q.REF
FLAGOK = Q.FLAGOK.to_numpy()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
UNIV = ((V60 >= 1e7) & (CR >= 10) & Q.FLAGOK & ADJ.notna()).to_numpy()
UNIVlag = np.vstack([np.zeros((1, N), bool), UNIV[:-1]])
MA60 = ADJ.rolling(60, min_periods=60).mean().to_numpy()
ABOVE_lag = np.vstack([np.zeros((1, N), bool), (AC > MA60)[:-1]])
BELOW_lag = np.vstack([np.zeros((1, N), bool), (AC < MA60)[:-1]])

with np.errstate(invalid='ignore'):
    tku = Q.tk
    xdn = REF * (1 - Q.lim)
    tkd = Q.tick_of(np.nan_to_num(xdn, nan=0.0), (IDX < pd.Timestamp('2005-03-01'))[:, None])
    LIMDN = np.ceil(xdn / tkd - 1e-6) * tkd
    OPEN_LD = OPn <= LIMDN + 0.5 * tkd
    CLOSE_LD = CRn <= LIMDN + 0.5 * tkd
    OPEN_LU = OPn >= Q.LIMUP - 0.5 * tku
    CLOSE_LU = Q.CLOSE_LU
    G = OPn / REF

# 處置
dsp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
DISP = np.zeros((T, N), bool); CI = {c: j for j, c in enumerate(cols)}
for r in dsp.itertuples():
    if r.code in CI and pd.notna(r.start) and pd.notna(r.end):
        s = IDX.searchsorted(r.start); e = IDX.searchsorted(r.end, 'right') - 1
        if e >= s: DISP[s:e + 1, CI[r.code]] = True

# 基準（分段等權）
ACprev = np.vstack([np.full((1, N), np.nan), AC[:-1]])
with np.errstate(invalid='ignore', divide='ignore'):
    OCr = np.clip(AC / AO - 1, -0.5, 1.0); COr = np.clip(AO / ACprev - 1, -0.5, 1.0)
def segmean(x, m):
    y = np.where(m & np.isfinite(x), x, np.nan)
    return np.nan_to_num(np.nanmean(y, axis=1))
POOL_OC, POOL_CO = segmean(OCr, UNIVlag), segmean(COr, UNIVlag)
LST_OC, LST_CO = segmean(OCr, np.ones((T, N), bool)), segmean(COr, np.ones((T, N), bool))
ew_cc = Q.RET.where(pd.DataFrame(UNIVlag, index=IDX, columns=cols)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0); E50n = e50.to_numpy()
S3a, S3b = pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23')
SRC3 = (IDX >= S3a) & (IDX <= S3b)
BUY, SELL = L.cost_rates(0.001); BUY_S, SELL_S = L.cost_rates('stress')
LONG_START = pd.Timestamp('2014-01-06'); SHORT_START = pd.Timestamp('2017-04-28'); TAX_CUT = pd.Timestamp('2017-04-28')
BORROW = 0.0008
LASTV = np.array([np.max(np.where(np.isfinite(AC[:, j]))[0]) if np.isfinite(AC[:, j]).any() else -1 for j in range(N)])


def bench(seg_oc, seg_co, t1, tx, kind):
    g = 1 + seg_oc[t1] if (tx > t1 or kind == 'c') else 1.0
    for t in range(t1 + 1, tx + 1):
        g *= 1 + seg_co[t]
        if t < tx or kind == 'c': g *= 1 + seg_oc[t]
    return g - 1


def exit_point(j, t, kind, side, maxd=10):
    """從 (t, kind) 起找第一個有價且可成交的時點（多：非跌停；空：非漲停）。下市 → 最後收盤。"""
    t0 = t; delayed = False
    while t < T and t <= t0 + maxd:
        if kind == 'o':
            blk = OPEN_LD[t, j] if side > 0 else OPEN_LU[t, j]
            if np.isfinite(AO[t, j]) and not blk: return t, 'o', AO[t, j], delayed
            kind = 'c'
        else:
            blk = CLOSE_LD[t, j] if side > 0 else CLOSE_LU[t, j]
            if np.isfinite(AC[t, j]) and not blk: return t, 'c', AC[t, j], delayed
            kind = 'o'; t += 1
        delayed = True
    if 0 <= LASTV[j] < t0 + maxd and t >= LASTV[j]:
        return LASTV[j], 'c', AC[LASTV[j], j], True
    return None


def base_mask():
    with np.errstate(invalid='ignore'):
        return UNIVlag & FLAGOK & np.isfinite(OPn) & np.isfinite(REF) & (OPn > 0) & ~DISP


def build(gap, trend, mode):
    """mode: 'DT'（當沖收盤出）或 'ON'（隔日開盤出）。"""
    B = base_mask()
    with np.errstate(invalid='ignore'):
        Lm = B & (G <= 1 - gap + 1e-9) & ~OPEN_LD
        Sm = B & (G >= 1 + gap - 1e-9) & ~OPEN_LU
    info = {'long_candidates_gap_only': int((B & (G <= 1 - gap + 1e-9)).sum()), 'long_open_limitdown_excluded': int((B & (G <= 1 - gap + 1e-9) & OPEN_LD).sum()),
            'short_candidates_gap_only': int((B & (G >= 1 + gap - 1e-9)).sum()), 'short_open_limitup_excluded': int((B & (G >= 1 + gap - 1e-9) & OPEN_LU).sum())}
    if trend:
        Lm &= ABOVE_lag; Sm &= BELOW_lag
    dts = IDX.to_numpy()
    if mode == 'DT':
        Lm &= (dts >= np.datetime64(LONG_START))[:, None]; Sm &= (dts >= np.datetime64(SHORT_START))[:, None]
    rows = []
    for side, M in ((1, Lm), (-1, Sm)):
        for t1, j in np.argwhere(M):
            p0 = AO[t1, j]
            if mode == 'DT':
                sp = exit_point(j, t1, 'c', side, 0)
                forced = sp is None or sp[0] != t1 or sp[1] != 'c'
                if forced: sp = exit_point(j, t1 + 1, 'o', side)
            else:
                sp = exit_point(j, t1 + 1, 'o', side); forced = False
            if sp is None: continue
            tx, kind, p1, dl = sp
            ratio = p1 / p0
            raw = ratio - 1 if side > 0 else 1 - ratio
            raw_rule = ratio - 1 if side > 0 else 1 / ratio - 1
            if mode == 'DT' and not forced:
                tax = 0.0015 if IDX[t1] >= TAX_CUT else 0.003
                rt = 0.0023 + (tax - 0.0015)
                net = raw - (rt + 0.002); net_s = raw - 0.01; net_noslip = raw - rt
            else:
                bor = BORROW if side < 0 else 0.0
                if side > 0:
                    net = ratio * (1 - SELL) / (1 + BUY) - 1; net_s = ratio * (1 - SELL_S) / (1 + BUY_S) - 1
                    net_noslip = ratio * (1 - SELL + 0.001) / (1 + BUY - 0.001) - 1
                else:
                    net = (1 - SELL - bor) - ratio * (1 + BUY); net_s = (1 - SELL_S - bor) - ratio * (1 + BUY_S)
                    net_noslip = (1 - SELL + 0.001 - bor) - ratio * (1 + BUY - 0.001)
            ew = bench(POOL_OC, POOL_CO, t1, tx, kind); ewl = bench(LST_OC, LST_CO, t1, tx, kind)
            te = tx if kind == 'c' else tx - 1
            b50 = E50n[te] / E50n[t1 - 1] - 1
            rows.append((side, cols[j], j, t1, tx, kind, IDX[t1], G[t1, j], raw, raw_rule, net, net_s, net_noslip, ew, ewl, b50,
                         bool(SRC3[t1:tx + 1].any()), bool(dl or forced)))
    df = pd.DataFrame(rows, columns=['side', 'code', 'j', 't1', 'tx', 'kind', 'entry', 'gap', 'raw', 'raw_rule', 'net', 'net_s', 'net_noslip',
                                     'ew', 'ewl', 'b50', 'src3', 'forced'])
    s = df['side']
    df['ab'] = df['net'] - s * df['ew']; df['ab_s'] = df['net_s'] - s * df['ew']; df['ab_noslip'] = df['net_noslip'] - s * df['ew']
    df['ab_gross'] = df['raw'] - s * df['ew']; df['ab_rule_gross'] = df['raw_rule'] - s * df['ew']
    df['abl'] = df['net'] - s * df['ewl']; df['ab_opp'] = df['net'] - df['ew']; df['ab50'] = df['net'] - df['b50']
    return df.sort_values(['entry', 'code']).reset_index(drop=True), info


def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    r = lambda x: round(float(x) * 100, 3)
    return {'n': int(len(df)), 'n_long': int((df['side'] > 0).sum()), 'n_short': int((df['side'] < 0).sum()), 'n_months': int(gm.ngroups),
            'mean_gap_pct': r(df['gap'].mean() - 1), 'mean_raw_pct': r(df['raw'].mean()), 'median_raw_pct': r(df['raw'].median()),
            'mean_net_pct': r(df['net'].mean()), 'mean_EWpool_dir_pct': r((df['side'] * df['ew']).mean()),
            'mean_ab_gross_pct': r(df['ab_gross'].mean()), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_gross_ruleformula_pct': r(df['ab_rule_gross'].mean()),
            'mean_ab_net_pct': r(df['ab'].mean()), 'median_ab_net_pct': r(df['ab'].median()),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': r(df['ab_s'].mean()), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_ab_net_noslip_pct': r(df['ab_noslip'].mean()), 't_month_ab_net_noslip': tstat(gm['ab_noslip'].mean()),
            'mean_ab_listedEW_net_pct': r(df['abl'].mean()), 't_month_ab_listedEW_net': tstat(gm['abl'].mean()),
            'mean_ab_oppcost_net_pct': r(df['ab_opp'].mean()), 't_month_ab_oppcost_net': tstat(gm['ab_opp'].mean()),
            'mean_net_vs_0050_pct': r(df['ab50'].mean()), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_raw': round(float((df['raw'] > 0).mean()), 3), 'hit_ab_net': round(float((df['ab'] > 0).mean()), 3),
            'share_forced_or_delayed': round(float(df['forced'].mean()), 4),
            'mean_ab_net_ex_top5pct': r(df['ab'][df['ab'] <= df['ab'].quantile(0.95)].mean()),
            'events_per_day_mean': round(float(df.groupby('entry').size().mean()), 2)}


def portfolio(df, cost):
    """日曆組合（prereg_amendments 1）：每個事件的整段報酬（扣成本）歸屬到進場日，當日所有事件等權平均；無事件日 = 0。
    當沖：資金開盤進、收盤出；隔夜：開盤進、隔日開盤出、資金在隔日開盤轉給新一批事件（一次只有一批）。"""
    col = {'net': 'net', 'stress': 'net_s', 'gross': 'raw'}[cost]
    g = df.groupby('t1')[col].mean(); n = df.groupby('t1').size()
    r = np.zeros(T); r[g.index.to_numpy()] = g.to_numpy(); c = np.zeros(T); c[n.index.to_numpy()] = n.to_numpy()
    return pd.Series(r, index=IDX), pd.Series(c, index=IDX)


def pstats(r, npos, segs):
    m = np.zeros(len(r), bool); yrs = 0.0
    for a, b in segs:
        mm = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
        if mm.sum() == 0: continue
        f = np.argmax(mm); l = len(mm) - 1 - np.argmax(mm[::-1])
        yrs += (r.index[l] - r.index[max(f - 1, 0)]).days / 365.25; m |= mm
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
    o['n_months'] = int(len(ms))
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    return o


VA, VB = '2019-05-01', '2026-10-31'
WIN = {'select_2005-01_2019-04': [('2005-01-01', '2019-04-30')],
       'valid_ex_src3': [(VA, '2024-02-14'), ('2025-09-24', VB)],
       'valid_incl_src3_2019-05_2026-10': [(VA, VB)],
       'valid_2024-01_2025-12': [('2024-01-01', '2025-12-31')],
       'src3_only': [('2024-02-15', '2025-09-23')],
       'd2021_select_2021-01_2023-12': [('2021-01-01', '2023-12-31')],
       'd2021_valid_2024-01_2026-10': [('2024-01-01', VB)],
       'd2021_valid_ex_src3': [('2025-09-24', VB)],
       'select_sub_2014-01_2017-04': [('2014-01-06', '2017-04-27')], 'select_sub_2017-04_2019-04': [('2017-04-28', '2019-04-30')]}
SEL, VAL = 'select_2005-01_2019-04', 'valid_ex_src3'
SPECS = [('PRIMARY', 0.03, False, 'DT', 0), ('G2', 0.02, False, 'DT', 0), ('G5', 0.05, False, 'DT', 0), ('TREND3', 0.03, True, 'DT', 0),
         ('LONG3', 0.03, False, 'DT', 1), ('SHORT3', 0.03, False, 'DT', -1), ('TREND3_LONG', 0.03, True, 'DT', 1), ('TREND3_SHORT', 0.03, True, 'DT', -1),
         ('ON3', 0.03, False, 'ON', 0), ('ON3_LONG', 0.03, False, 'ON', 1), ('ON3_SHORT', 0.03, False, 'ON', -1)]


def ev_window(df, segs, ex_src3):
    m = np.zeros(len(df), bool)
    for a, b in segs: m |= ((df['entry'] >= pd.Timestamp(a)) & (df['entry'] <= pd.Timestamp(b))).to_numpy()
    if ex_src3: m &= ~df['src3'].to_numpy()
    return df[m]


def passes(o):
    out = {}
    for w in (SEL, VAL):
        e = o[f'event_{w}']; p = o[f'port_{w}']
        if e.get('n', 0) < 30 or e.get('n_months', 0) < 24:
            out[w] = {'thin': True, 'pass': False}; continue
        ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
        ok_50 = p['CAGR'] >= p['0050_CAGR']
        out[w] = {'thin': False, 'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50),
                  'portfolio_t_vs_EW_m': p['t_vs_EW_m']}
    return out


def main():
    out = {'meta': {'bad_days_dropped': Q.BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'disposal_list_start': str(dsp['start'].min().date())}}
    res = {}; yearly_ev = {}
    cache = {}
    for key, gap, trend, mode, sidef in SPECS:
        ck = (gap, trend, mode)
        if ck not in cache: cache[ck] = build(gap, trend, mode)
        df, info = cache[ck]
        if sidef: df = df[df['side'] == sidef].reset_index(drop=True)
        o = {'params': {'gap': gap, 'trend_filter': trend, 'mode': mode, 'side': sidef or 'both'}, 'build_info': info, 'n_events': int(len(df))}
        dts = IDX.to_numpy()
        for wn, segs in WIN.items():
            ex = (wn == VAL) or (wn == 'd2021_valid_ex_src3')
            o[f'event_{wn}'] = ev_stats(ev_window(df, segs, ex))
            o[f'event_{wn}_long'] = ev_stats(ev_window(df[df['side'] > 0], segs, ex))
            o[f'event_{wn}_short'] = ev_stats(ev_window(df[df['side'] < 0], segs, ex))
        rn, npos = portfolio(df, 'net'); rs, _ = portfolio(df, 'stress'); rg, _ = portfolio(df, 'gross')
        for wn, segs in WIN.items():
            if mode == 'DT' and wn == SEL: segs = [('2014-01-06' if sidef >= 0 else '2017-04-28', '2019-04-30')]
            if mode == 'DT' and sidef < 0 and wn == 'select_sub_2014-01_2017-04': continue
            o[f'port_{wn}'] = pstats(rn, npos, segs)
            o[f'port_{wn}']['CAGR_stress'] = pstats(rs, npos, segs)['CAGR']; o[f'port_{wn}']['CAGR_gross'] = pstats(rg, npos, segs)['CAGR']
        o['pass'] = passes(o)
        res[key] = o
        yearly_ev[key] = {int(y): {'n': int(len(g)), 'n_long': int((g['side'] > 0).sum()), 'n_short': int((g['side'] < 0).sum()),
                                   'ab_net_pct': round(g['ab'].mean() * 100, 3), 'ab_gross_pct': round(g['ab_gross'].mean() * 100, 3),
                                   'ab_net_long_pct': round(g.loc[g['side'] > 0, 'ab'].mean() * 100, 3) if (g['side'] > 0).any() else None,
                                   'ab_net_short_pct': round(g.loc[g['side'] < 0, 'ab'].mean() * 100, 3) if (g['side'] < 0).any() else None}
                          for y, g in df.groupby(df['entry'].dt.year)}
        es, ev_ = o[f'event_{SEL}'], o[f'event_{VAL}']; ps, pv = o[f'port_{SEL}'], o[f'port_{VAL}']
        print(key, 'n', len(df), '| S n', es.get('n'), 'gross', es.get('mean_ab_gross_pct'), 'ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'),
              'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'tEW', ps['t_vs_EW_m'], '| V n', ev_.get('n'), 'gross', ev_.get('mean_ab_gross_pct'), 'ab', ev_.get('mean_ab_net_pct'),
              't', ev_.get('t_month_ab_net'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'tEW', pv['t_vs_EW_m'], flush=True)
        if key == 'PRIMARY':
            pd.DataFrame({'net': rn, 'stress': rs, 'gross': rg, 'npos': npos}).to_parquet(HERE / 'nav_primary.parquet')
        if key in ('PRIMARY', 'G2', 'G5', 'TREND3', 'ON3'):
            df.drop(columns=['j']).to_csv(HERE / f'events_{key}.csv.gz', index=False)
    out['results'] = res; out['yearly_events'] = yearly_ev
    pr = res['PRIMARY']['pass']
    thin = any(pr[w]['thin'] for w in (SEL, VAL)); k = sum(pr[w]['pass'] for w in (SEL, VAL))
    vs = {'PRIMARY': pr, 'verdict': '無法測試' if thin else ('可用' if k == 2 else ('待觀察' if k == 1 else '不可用'))}
    vs['variants_pass_count'] = {kk: sum(res[kk]['pass'][w]['pass'] for w in (SEL, VAL)) for kk in res}
    if vs['verdict'] == '不可用' and any(v == 2 for kk, v in vs['variants_pass_count'].items() if kk != 'PRIMARY'): vs['verdict'] = '待觀察'
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))


if __name__ == '__main__':
    main()
