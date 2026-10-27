#!/usr/bin/env python3
"""漲停回馬槍回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）與 prereg_amendments。
事件／組合統計架構沿用 w4_choke_volume_first_red/run.py（複製，不 import），成本用 exec_rebuild/exlib.cost_rates。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lim as Q  # noqa: E402
import exlib as L  # noqa: E402  成本模型（只讀取）
from btlib import tstat  # noqa: E402

X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
IDX, T, N, cols = Q.IDX, Q.T, Q.N, Q.cols
ADJ, CR, OP, V, F = Q.ADJ, Q.CR, Q.OP, Q.V, Q.F
fac = ADJ / CR
AO_raw = OP * fac
ACf = ADJ.ffill()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
UNIV = ((V60 >= 1e7) & (CR >= 10) & Q.FLAGOK & ADJ.notna()).to_numpy()
VOL = (V / CR).where((V > 0) & (CR > 0))
VMA20P = VOL.shift(1).rolling(20, min_periods=15).mean().to_numpy()
VOLn = VOL.to_numpy()
LU = Q.CLOSE_LU.copy()
LUf = pd.DataFrame(LU.astype(float), index=IDX)
LU_prev10 = (LUf.shift(1).rolling(10, min_periods=1).max() >= 1).to_numpy()          # t-10..t-1
LU_prev2_11 = (LUf.shift(2).rolling(10, min_periods=1).max() >= 1).to_numpy()        # t-11..t-2
LU_prev1 = np.vstack([np.zeros((1, N), bool), LU[:-1]])
VOLCOND = VOLn >= 2 * VMA20P
FLAGOK = Q.FLAGOK.to_numpy()

AOn = AO_raw.to_numpy(); ACn = ACf.to_numpy(); ADJn = ADJ.to_numpy()
PX = AO_raw.bfill().to_numpy(); PX = np.where(np.isfinite(PX), PX, ACn)
OPEN_LU = Q.OPEN_LU

# 處置（2021-09 起）、注意股（2021-10 起）
dsp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
DISP = np.zeros((T, N), bool); CI = {c: j for j, c in enumerate(cols)}
for r in dsp.itertuples():
    if r.code in CI and pd.notna(r.start) and pd.notna(r.end):
        DISP[(IDX >= r.start) & (IDX <= r.end), CI[r.code]] = True
att = pd.read_csv(X / 'attention_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
ATT = set((d, c) for d, c in zip(att['date'], att['code']))
ATT_START = att['date'].min()

# 基準
UNIVs = pd.DataFrame(UNIV, index=IDX).shift(1, fill_value=False).to_numpy()
oo = (AO_raw.shift(-1) / AO_raw - 1).to_numpy()
oo = np.clip(oo, -0.5, 1.0)
def cumlog(mask):
    x = np.where(mask & np.isfinite(oo), oo, np.nan)
    with np.errstate(invalid='ignore'):
        m = np.nanmean(x, axis=1)
    m = np.nan_to_num(m)
    return np.concatenate([[0.0], np.cumsum(np.log1p(m))])
LEW = cumlog(UNIVs)                                   # 同池等權（開盤對開盤）
LEWL = cumlog(np.isfinite(AOn))                       # 上市等權
ew_cc = Q.RET.where(pd.DataFrame(UNIVs, index=IDX, columns=cols)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0); E50n = e50.to_numpy()
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))
BUY, SELL = L.cost_rates(0.001)
BUY_S, SELL_S = L.cost_rates('stress')


def day0_mask(board2):
    base = LU & VOLCOND & UNIV
    if board2:
        return base & LU_prev1 & ~LU_prev2_11
    return base & ~LU_prev10


def build_events(D0, r, W, H):
    sigs = []; n_inval = 0; n_none = 0
    for t0, j in np.argwhere(D0):
        a0 = ADJn[t0, j]; o0 = AOn[t0, j]; v_0 = VOLn[t0, j]
        if not (np.isfinite(a0) and np.isfinite(o0) and np.isfinite(v_0)): continue
        found = None
        for k in range(1, W + 1):
            s = t0 + k
            if s >= T: break
            a = ADJn[s, j]
            if not np.isfinite(a): continue
            if a < o0:
                n_inval += 1; break
            v = VOLn[s, j]
            if a < a0 and a >= a0 * 0.92 and np.isfinite(v) and v <= r * v_0 and UNIV[s, j] and not DISP[s, j]:
                found = s; break
        else:
            n_none += 1
        if found is not None: sigs.append((j, t0, found))
    sigs.sort(key=lambda x: (x[0], x[2]))
    ev = []; free = {}; nblock = 0; nno = 0; ndisp = 0
    for j, t0, s in sigs:
        if s < free.get(j, -1): continue
        e = s + 1
        if e >= T: continue
        if not np.isfinite(AOn[e, j]): nno += 1; continue
        if OPEN_LU[e, j]: nblock += 1; continue
        if DISP[e, j]: ndisp += 1; continue
        stop = AOn[t0, j]; tp = AOn[e, j] * 1.10
        hi = min(e + H, T)
        a = ADJn[e:hi, j]
        with np.errstate(invalid='ignore'):
            h1 = a < stop; h2 = a >= tp
        hit = np.where(h1 | h2)[0]
        if len(hit):
            x = e + hit[0] + 1; why = 'stop' if h1[hit[0]] else 'tp'
        else:
            x = e + H; why = 'time'
        ev.append((j, t0, s, e, x, x < T, why))
        free[j] = x
    return ev, {'n_signals': len(sigs), 'invalidated_break': n_inval, 'no_pullback': n_none, 'blocked_open_limit_up': nblock, 'no_open': nno, 'disposal_entry': ndisp}


def event_table(ev):
    rows = []
    for j, t0, s, e, x, done, why in ev:
        if not done: continue
        p0 = AOn[e, j]; p1 = PX[x, j]; raw = p1 / p0 - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1
        net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        ew = math.expm1(LEW[x] - LEW[e]); ewl = math.expm1(LEWL[x] - LEWL[e])
        b50 = E50n[x - 1] / E50n[s] - 1
        rows.append((cols[j], IDX[t0], IDX[s], IDX[e], x - e, s - t0, why, raw, net, net_s, ew, ewl, b50, bool(SRC3[t0:x + 1].any()),
                     AOn[e, j] / ACn[s, j] - 1, (IDX[t0], cols[j]) in ATT, bool(IDX[t0] >= ATT_START)))
    df = pd.DataFrame(rows, columns=['code', 'd0', 'sig', 'entry', 'hold', 'lag', 'why', 'raw', 'net', 'net_s', 'ew', 'ewl', 'b50', 'src3', 'gap', 'att_d0', 'att_known'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['abl'] = df['net'] - df['ewl']
    df['ab50'] = df['net'] - df['b50']; df['ab_gross'] = df['raw'] - df['ew']
    return df


def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 2),
            'exit_mix': {k: round(float((df['why'] == k).mean()), 3) for k in ('stop', 'tp', 'time')},
            'mean_overnight_gap_pct': round(df['gap'].mean() * 100, 3),
            'mean_raw_pct': round(df['raw'].mean() * 100, 3), 'mean_net_pct': round(df['net'].mean() * 100, 3),
            'mean_EWpool_pct': round(df['ew'].mean() * 100, 3),
            'mean_ab_gross_pct': round(df['ab_gross'].mean() * 100, 3), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': round(df['ab'].mean() * 100, 3), 'median_ab_net_pct': round(df['ab'].median() * 100, 3),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': round(df['ab_s'].mean() * 100, 3), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_ab_listedEW_net_pct': round(df['abl'].mean() * 100, 3), 't_month_ab_listedEW_net': tstat(gm['abl'].mean()),
            'mean_net_vs_0050_pct': round(df['ab50'].mean() * 100, 3), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3)}


def portfolio(ev, buy, sell):
    R = np.zeros((T, N), np.float64); W = np.zeros((T, N), np.float32); ent = np.zeros(T); exi = np.zeros(T)
    for j, t0, s, e, x, done, why in ev:
        xe = x if done else T - 1
        W[e:xe + 1, j] += 1
        R[e, j] = ACn[e, j] / AOn[e, j] - 1
        if xe > e:
            R[e + 1:xe + 1, j] = np.nan_to_num(ACn[e + 1:xe + 1, j] / ACn[e:xe, j] - 1)
        if done:
            R[x, j] = PX[x, j] / ACn[x - 1, j] - 1
            exi[x] += 1
        ent[e] += 1
    W = np.minimum(W, 1); n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(n > 0, (W * R).sum(1) / np.where(n > 0, n, 1), 0.0)
        c = np.where(n > 0, (ent * buy + exi * sell) / np.where(n > 0, n, 1), 0.0)
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
    o['avg_positions_when_invested'] = round(float(npos[m][npos[m] > 0].mean()), 1) if (npos[m] > 0).any() else 0
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
SEL, VAL = 'select_2005-01_2019-04', 'valid_2019-05_2026-10'
SPECS = [('PRIMARY', 0.5, 3, 5, False)] + [(f'r{r}_W{w}_H{h}', r, w, h, False) for r in (0.5, 0.6) for w in (3, 5) for h in (3, 5, 10)
                                           if not (r == 0.5 and w == 3 and h == 5)] + [('BOARD2', 0.5, 3, 5, True)]


def passes(o, w):
    e = o[f'event_{w}']; p = o[f'port_{w}']
    ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
    ok_50 = p['CAGR'] >= p['0050_CAGR']
    return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}


def main():
    out = {'meta': {'bad_days_dropped': Q.BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'close_limit_up_stock_days': int(LU.sum()), 'close_limit_up_in_univ': int((LU & UNIV).sum()),
                    'day0_first_board': int(day0_mask(False).sum()), 'day0_board2': int(day0_mask(True).sum()),
                    'day0_first_board_by_year': {int(y): int(v) for y, v in pd.Series(day0_mask(False).sum(1), index=IDX).groupby(IDX.year).sum().items()}}}
    res = {}; yearly_ev = {}
    for key, r, w, h, b2 in SPECS:
        ev, info = build_events(day0_mask(b2), r, w, h)
        df = event_table(ev)
        o = {'params': {'r': r, 'W': w, 'H': h, 'board2': b2}, **info, 'events_taken': len(ev)}
        for wn, (a, b) in WIN.items():
            o[f'event_{wn}'] = ev_stats(df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))])
        dv = df[df['sig'] >= pd.Timestamp('2019-05-01')]
        o['event_valid_ex_src3'] = ev_stats(dv[~dv['src3']])
        o['event_touch_src3'] = ev_stats(df[df['src3']])
        o['event_select_sub_2005-2015-05_7pct'] = ev_stats(df[(df['sig'] >= '2005-01-01') & (df['sig'] < '2015-06-01')])
        o['event_select_sub_2015-06_2019-04_10pct'] = ev_stats(df[(df['sig'] >= '2015-06-01') & (df['sig'] <= '2019-04-30')])
        k = df[df['att_known']]
        o['event_attention_known_period'] = {'att_d0': ev_stats(k[k['att_d0']]), 'not_att_d0': ev_stats(k[~k['att_d0']])}
        rn, npos = portfolio(ev, BUY, SELL); rs, _ = portfolio(ev, BUY_S, SELL_S); rg, _ = portfolio(ev, 0.0, 0.0)
        for wn, (a, b) in WIN.items():
            o[f'port_{wn}'] = pstats(rn, npos, a, b)
            o[f'port_{wn}']['CAGR_stress'] = pstats(rs, npos, a, b)['CAGR']
            o[f'port_{wn}']['CAGR_gross'] = pstats(rg, npos, a, b)['CAGR']
        yearly_ev[key] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2)}
                          for y, g in df.groupby(df['sig'].dt.year)}
        o['pass'] = {SEL: passes(o, SEL), VAL: passes(o, VAL)}
        res[key] = o
        es, ev_ = o[f'event_{SEL}'], o[f'event_{VAL}']; ps, pv = o[f'port_{SEL}'], o[f'port_{VAL}']
        print(key, 'n', len(df), '| S n', es.get('n'), 'ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'CAGR', ps['CAGR'], '50', ps['0050_CAGR'],
              '| V n', ev_.get('n'), 'ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], flush=True)
        if key in ('PRIMARY', 'BOARD2'):
            df.to_csv(HERE / f'events_{key}.csv.gz', index=False)
            if key == 'PRIMARY':
                pd.DataFrame({'net': rn, 'stress': rs, 'gross': rg, 'npos': npos, 'r0050': r50, 'ew_pool': ew_cc}).to_parquet(HERE / 'nav_primary.parquet')
    out['results'] = res; out['yearly_events'] = yearly_ev
    pr = res['PRIMARY']; vs = dict(pr['pass'])
    k = vs[SEL]['pass'] + vs[VAL]['pass']
    thin = any(pr[f'event_{w}'].get('n', 0) < 30 or pr[f'event_{w}'].get('n_months', 0) < 24 for w in [SEL, VAL])
    vs['verdict'] = '無法測試' if thin else ('可用' if k == 2 else ('待觀察' if k == 1 else '不可用'))
    vs['variants_pass_count'] = {kk: sum(o['pass'][w]['pass'] for w in (SEL, VAL)) for kk, o in res.items()}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))


if __name__ == '__main__':
    main()
