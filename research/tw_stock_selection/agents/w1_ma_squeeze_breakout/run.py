#!/usr/bin/env python3
"""均線糾結（5/10/20）帶量長紅突破 回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
架構沿用 w1_granville_pullback_ma60/run.py（事件表、組合、統計同口徑），成本用 exec_rebuild/exlib.cost_rates。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成本模型（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
names = ['adj', 'close_raw', 'open', 'value', 'flag', 'src', 'ret']
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in names}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if len(c) == 4 and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
F = D['flag']; bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); V = D['value']
fac = ADJ / CR
AO_raw = OP * fac
ACf = ADJ.ffill()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
UNIV = (V60 >= 1e7) & (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()
VOL = V / CR
VOL5p = VOL.shift(1).rolling(5, min_periods=5).mean()      # t−5..t−1
MA = {n: ADJ.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60)}
MX = np.maximum(np.maximum(MA[5], MA[10]), MA[20]); MN = np.minimum(np.minimum(MA[5], MA[10]), MA[20])
SPREAD = (MX - MN) / ADJ
DRET = ADJ / ADJ.shift(1) - 1
BASE = UNIV & ((ADJ / MA[60] - 1).abs() <= 0.10) & (ADJ > MX) & (DRET >= 0.03) & (CR > OP)

lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
AC_prev = ACf.shift(1).to_numpy()
GAP_UP = (AO_raw.to_numpy() / AC_prev - 1) >= lim - 0.006
AOn = AO_raw.to_numpy(); ACn = ACf.to_numpy(); ADJn = ADJ.to_numpy(); MA20n = MA[20].to_numpy()
PX = AO_raw.bfill().to_numpy(); PX = np.where(np.isfinite(PX), PX, ACn)

oo = (AO_raw.shift(-1) / AO_raw - 1).where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0)
ew_oo = oo.mean(axis=1).fillna(0).to_numpy()
LEW = np.concatenate([[0.0], np.cumsum(np.log1p(ew_oo))])
ew_cc = D['ret'].where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0); E50n = e50.to_numpy()
SRC3 = (IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23'))

BUY, SELL = L.cost_rates(0.001)
BUY_S, SELL_S = L.cost_rates('stress')

def signals(X, Nw, K, incl_today=False):
    ok = (SPREAD <= X).astype(float).where(SPREAD.notna())
    sq = ok.rolling(Nw, min_periods=Nw).min() == 1
    if not incl_today: sq = sq.shift(1, fill_value=False)
    S = BASE & sq & (VOL >= K * VOL5p)
    return S.to_numpy()

def build_events(S, exit_rule):
    ev = []; nblock = 0; nnoopen = 0
    for j in range(N):
        free_from = -1
        for t in np.where(S[:, j])[0]:
            if t < free_from or IDX[t] < pd.Timestamp('2005-01-01'): continue
            e = t + 1
            if e >= T: continue
            if not np.isfinite(AOn[e, j]): nnoopen += 1; continue
            if GAP_UP[e, j]: nblock += 1; continue
            if exit_rule.startswith('A'):
                x = e + int(exit_rule[1:])
            else:
                cap = 120
                seg = ADJn[e:min(e + cap, T), j] < MA20n[e:min(e + cap, T), j]
                hit = np.where(seg)[0]
                x = e + hit[0] + 1 if len(hit) else e + cap
            ev.append((j, t, e, x, x < T)); free_from = x
    return ev, nblock, nnoopen

def event_table(ev):
    rows = []
    for j, t, e, x, done in ev:
        if not done: continue
        p0 = AOn[e, j]; p1 = PX[x, j]; raw = p1 / p0 - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1
        net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        ew = math.expm1(LEW[x] - LEW[e]); b50 = E50n[x - 1] / E50n[t] - 1
        gap = AOn[e, j] / ACn[t, j] - 1
        rows.append((cols[j], IDX[t], IDX[e], x - e, raw, net, net_s, ew, b50, bool(SRC3[t:x + 1].any()), gap))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'hold', 'raw', 'net', 'net_s', 'ew', 'b50', 'src3', 'gap'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']
    df['ab_gross'] = df['raw'] - df['ew']
    return df

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31')}

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M')); gq = df.groupby(df['entry'].dt.to_period('Q'))
    r = lambda x: round(float(x) * 100, 3)
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 1),
            'mean_raw_pct': r(df['raw'].mean()), 'mean_net_pct': r(df['net'].mean()), 'mean_EW_pct': r(df['ew'].mean()),
            'mean_gap_pct': r(df['gap'].mean()),
            'mean_ab_gross_pct': r(df['ab_gross'].mean()), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': r(df['ab'].mean()), 'median_ab_net_pct': r(df['ab'].median()),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_quarter_ab_net': tstat(gq['ab'].mean()),
            't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': r(df['ab_s'].mean()), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_net_vs_0050_pct': r(df['ab50'].mean()), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3)}

def portfolio(ev, buy, sell, idle_ew=False):
    R = np.zeros((T, N), np.float32); W = np.zeros((T, N), np.float32); ent = np.zeros(T); exi = np.zeros(T)
    for j, t, e, x, done in ev:
        xe = x if done else T - 1
        W[e:xe + 1, j] += 1
        R[e, j] = ACn[e, j] / AOn[e, j] - 1
        if xe > e: R[e + 1:xe + 1, j] = np.nan_to_num(ACn[e + 1:xe + 1, j] / ACn[e:xe, j] - 1)
        if done:
            R[x, j] = PX[x, j] / ACn[x - 1, j] - 1 if x > e else PX[x, j] / AOn[e, j] - 1
            exi[x] += 1
        ent[e] += 1
    W = np.minimum(W, 1); n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(n > 0, (W * R).sum(1) / np.where(n > 0, n, 1), 0.0)
        c = np.where(n > 0, (ent * buy + exi * sell) / np.where(n > 0, n, 1), 0.0)
    out = g - c
    if idle_ew: out = np.where(n > 0, out, ew_cc.to_numpy())
    return pd.Series(out, index=IDX), pd.Series(n, index=IDX)

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

def run_one(S, ex):
    ev, nb, nno = build_events(S, ex); df = event_table(ev)
    o = {'events_taken': len(ev), 'blocked_gap_up': nb, 'no_open': nno}
    for w, (a, b) in WIN.items():
        o[f'event_{w}'] = ev_stats(df[(df['sig'] >= a) & (df['sig'] <= b)])
    o['event_valid_ex_src3'] = ev_stats(df[(df['sig'] >= '2019-05-01') & ~df['src3']])
    o['event_select_sub_2005-2011'] = ev_stats(df[(df['sig'] >= '2005-01-01') & (df['sig'] <= '2011-12-31')])
    o['event_select_sub_2012-2019-04'] = ev_stats(df[(df['sig'] >= '2012-01-01') & (df['sig'] <= '2019-04-30')])
    r, npos = portfolio(ev, BUY, SELL); rs, _ = portfolio(ev, BUY_S, SELL_S); rg, _ = portfolio(ev, 0.0, 0.0)
    ri, _ = portfolio(ev, BUY, SELL, idle_ew=True)
    for w, (a, b) in WIN.items():
        o[f'port_{w}'] = pstats(r, npos, a, b)
        o[f'port_{w}']['CAGR_stress'] = pstats(rs, npos, a, b)['CAGR']
        o[f'port_{w}']['CAGR_gross'] = pstats(rg, npos, a, b)['CAGR']
        pi = pstats(ri, npos, a, b)
        o[f'port_{w}']['idleEW'] = {k: pi[k] for k in ('CAGR', 'maxDD', 't_vs_EW_m', 't_vs_0050_m')}
    ye = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2)}
          for y, g in df.groupby(df['sig'].dt.year)}
    return o, ye, df

def main():
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'base_days_before_squeeze': int(BASE.to_numpy().sum())}}
    res = {}; yearly_ev = {}
    for sqn, X, Nw in (('S5_20', 0.05, 20), ('S3_10', 0.03, 10)):
        for K in (2, 3):
            S = signals(X, Nw, K)
            for ex in ('A5', 'A20', 'A60', 'B'):
                key = f'{sqn}_V{K}_{ex}'
                o, ye, df = run_one(S, ex); o['raw_signal_days'] = int(S[IDX >= pd.Timestamp('2005-01-01')].sum())
                res[key] = o; yearly_ev[key] = ye
                es, ev_ = o['event_select_2005-01_2019-04'], o['event_valid_2019-05_2026-10']
                ps, pv = o['port_select_2005-01_2019-04'], o['port_valid_2019-05_2026-10']
                print(key, 'n', len(df), '| S ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'g', es.get('mean_ab_gross_pct'), es.get('t_month_ab_gross'),
                      'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'],
                      '| V ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'), 'g', ev_.get('mean_ab_gross_pct'), ev_.get('t_month_ab_gross'),
                      'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], flush=True)
                if key == 'S5_20_V2_A20':
                    df.to_csv(Path(L.P).parent / 'w1_ma_squeeze_primary_events.csv.gz', index=False)
    S = signals(0.05, 20, 2, incl_today=True)
    o, ye, df = run_one(S, 'A20'); res['SENS_S5_20_inclToday_V2_A20'] = o; yearly_ev['SENS_S5_20_inclToday_V2_A20'] = ye
    print('SENS incl today n', len(df), o['event_select_2005-01_2019-04'].get('mean_ab_net_pct'), o['event_select_2005-01_2019-04'].get('t_month_ab_net'),
          o['event_valid_2019-05_2026-10'].get('mean_ab_net_pct'), o['event_valid_2019-05_2026-10'].get('t_month_ab_net'))
    out['results'] = res; out['yearly_events'] = yearly_ev
    def passes(o, w):
        e = o[f'event_{w}']; p = o[f'port_{w}']
        ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
        ok_50 = p['CAGR'] >= p['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['S5_20_V2_A20']
    vs = {'select': passes(pr, 'select_2005-01_2019-04'), 'valid': passes(pr, 'valid_2019-05_2026-10')}
    k = vs['select']['pass'] + vs['valid']['pass']
    vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: {w: passes(o, w) for w in ['select_2005-01_2019-04', 'valid_2019-05_2026-10']} for kk, o in res.items()}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps({k: vs[k] for k in ('select', 'valid', 'verdict')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
