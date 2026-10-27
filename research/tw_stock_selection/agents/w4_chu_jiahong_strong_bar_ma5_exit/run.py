#!/usr/bin/env python3
"""朱家泓 多頭排列＋漲幅 3.5% 長紅過前高，跌破 5 日線出場。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
事件／組合統計架構沿用 w4_choke_volume_first_red/run.py（複製改寫，不 import），成本用 exec_rebuild/exlib.cost_rates。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild')); sys.path.insert(0, str(HERE.parents[1]))
import exlib as L  # noqa: E402  成本模型（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
names = ['adj', 'close_raw', 'open', 'high', 'value', 'flag', 'src']
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in names}
D['ret'] = pd.read_parquet(P / 'ret.parquet')
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if c.isdigit() and len(c) == 4 and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
F = D['flag']; bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); HI = D['high'].mask(bad5); V = D['value']
fac = ADJ / CR
AO_raw = OP * fac
AH_raw = HI * fac
ACf = ADJ.ffill()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
FOK = F.isin([0, 1, 2]) & ADJ.notna()
UNIV = (V60 >= 1e7) & (CR >= 10) & FOK
VOL = (V / CR).where((V > 0) & (CR > 0))
MA = {n: ADJ.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60)}
C1 = (MA[5] > MA[10]) & (MA[10] > MA[20]) & (MA[20] > MA[60])
C2 = MA[20] > MA[20].shift(5)
C3 = ADJ / ACf.shift(1) - 1 >= 0.035
C4 = ADJ > AH_raw.shift(1)
C5 = VOL >= 1.5 * VOL.shift(1).rolling(5, min_periods=5).mean()

lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
AC_prev = ACf.shift(1).to_numpy()
GAP_UP = (AO_raw.to_numpy() / AC_prev - 1) >= lim - 0.006
AOn = AO_raw.to_numpy(); AHn = AH_raw.to_numpy(); ACn = ACf.to_numpy(); ADJn = ADJ.to_numpy(); MA5n = MA[5].to_numpy()
PX = AO_raw.bfill().to_numpy()
PX = np.where(np.isfinite(PX), PX, ACn)

def ew_series(mask):
    m = mask.shift(1, fill_value=False)
    oo = (AO_raw.shift(-1) / AO_raw - 1).where(m).clip(-0.5, 1.0).mean(axis=1).fillna(0).to_numpy()
    oc = (ADJ / AO_raw - 1).where(m).clip(-0.5, 1.0).mean(axis=1).fillna(0).to_numpy()
    cc = D['ret'].where(m).clip(-0.5, 1.0).mean(axis=1).fillna(0)
    return np.concatenate([[0.0], np.cumsum(np.log1p(oo))]), oc, cc
LEW, EWOC, ew_cc = ew_series(UNIV)
LALL, ALLOC, all_cc = ew_series(FOK)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0)
E50n = e50.to_numpy()
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))

BUY, SELL = L.cost_rates(0.001)
BUY_S, SELL_S = L.cost_rates('stress')
H = 60

def signals(vol):
    S = UNIV & C1 & C2 & C3 & C4
    if vol: S = S & C5
    return S.fillna(False).to_numpy()

def build_events(S, tp):
    """回傳 (j, t, e, x, done, why, exit_px, intraday)。intraday=True 表示 x 日盤中觸價出場。"""
    ev = []; nblock = 0; nnoopen = 0
    for j in range(N):
        ts = np.where(S[:, j])[0]
        free_from = -1
        for t in ts:
            if t < free_from: continue
            e = t + 1
            if e >= T: continue
            if not np.isfinite(AOn[e, j]): nnoopen += 1; continue
            if GAP_UP[e, j]: nblock += 1; continue
            p0 = AOn[e, j]; stop = AOn[t, j]
            hi = min(e + H, T)
            x = None; why = 'time'; xp = np.nan; intra = False
            for s in range(e, hi):
                if tp is not None:
                    tg = p0 * (1 + tp)
                    h = AHn[s, j]
                    if np.isfinite(h) and h >= tg:
                        o = AOn[s, j] if s > e else p0
                        x = s; why = 'target'; xp = max(o, tg) if np.isfinite(o) else tg; intra = True; break
                a = ADJn[s, j]
                if not np.isfinite(a): continue
                if a < stop: x = s + 1; why = 'stop'; break
                if np.isfinite(MA5n[s, j]) and a < MA5n[s, j]: x = s + 1; why = 'ma5'; break
            if x is None: x = e + H
            done = x < T
            if done and not intra: xp = PX[x, j]
            ev.append((j, t, e, x, done, why, xp, intra))
            free_from = x + (1 if intra else 0)
    return ev, nblock, nnoopen

def event_table(ev):
    rows = []
    for j, t, e, x, done, why, xp, intra in ev:
        if not done: continue
        p0 = AOn[e, j]
        raw = xp / p0 - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1
        net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        if intra:
            ew = (1 + math.expm1(LEW[x] - LEW[e])) * (1 + EWOC[x]) - 1
            al = (1 + math.expm1(LALL[x] - LALL[e])) * (1 + ALLOC[x]) - 1
            b50 = E50n[x] / E50n[t] - 1
        else:
            ew = math.expm1(LEW[x] - LEW[e]); al = math.expm1(LALL[x] - LALL[e]); b50 = E50n[x - 1] / E50n[t] - 1
        touch3 = bool(SRC3[t:x + 1].any())
        gap = AOn[e, j] / ACn[t, j] - 1
        rows.append((cols[j], IDX[t], IDX[e], x - e + (1 if intra else 0), why, raw, net, net_s, ew, al, b50, touch3, gap))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'hold', 'why', 'raw', 'net', 'net_s', 'ew', 'allew', 'b50', 'src3', 'gap'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']
    df['ab_gross'] = df['raw'] - df['ew']; df['ab_all'] = df['net'] - df['allew']
    return df

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31')}

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 1),
         'exit_mix': {k: round(float((df['why'] == k).mean()), 3) for k in ('ma5', 'stop', 'target', 'time')},
         'mean_overnight_gap_pct': round(df['gap'].mean() * 100, 3),
         'mean_raw_pct': round(df['raw'].mean() * 100, 3), 'mean_net_pct': round(df['net'].mean() * 100, 3),
         'mean_EW_pct': round(df['ew'].mean() * 100, 3),
         'mean_ab_gross_pct': round(df['ab_gross'].mean() * 100, 3), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
         'mean_ab_net_pct': round(df['ab'].mean() * 100, 3), 'median_ab_net_pct': round(df['ab'].median() * 100, 3),
         't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
         'mean_ab_net_stress_pct': round(df['ab_s'].mean() * 100, 3), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
         'mean_ab_net_vs_allEW_pct': round(df['ab_all'].mean() * 100, 3), 't_month_vs_allEW': tstat(gm['ab_all'].mean()),
         'mean_net_vs_0050_pct': round(df['ab50'].mean() * 100, 3), 't_month_vs_0050': tstat(gm['ab50'].mean()),
         'hit_ab_net': round(float((df['ab'] > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3)}

def portfolio(ev, buy, sell):
    R = np.zeros((T, N), np.float32); W = np.zeros((T, N), np.float32)
    ent = np.zeros(T); exi = np.zeros(T)
    for j, t, e, x, done, why, xp, intra in ev:
        xe = x if done else T - 1
        W[e:xe + 1, j] += 1
        R[e, j] = ACn[e, j] / AOn[e, j] - 1
        if xe > e:
            seg = ACn[e + 1:xe + 1, j] / ACn[e:xe, j] - 1
            R[e + 1:xe + 1, j] = np.nan_to_num(seg)
        if done:
            R[x, j] = xp / ACn[x - 1, j] - 1 if x > e else xp / AOn[e, j] - 1
            exi[x] += 1
        ent[e] += 1
    W = np.minimum(W, 1)
    n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(n > 0, (W * R).sum(1) / n, 0.0)
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
    o['share_days_invested'] = round(float((npos[m] > 0).mean()), 3); o['avg_positions_when_invested'] = round(float(npos[m][npos[m] > 0].mean()), 1)
    mo = lambda x: (1 + x).groupby(x.index.to_period('M')).prod() - 1
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    return o

SPECS = [('PRIMARY', True, None), ('TP8', True, 0.08), ('TP15', True, 0.15), ('NOVOL', False, None)]

def main():
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'cond_counts_in_univ': {k: int((UNIV & c).sum().sum()) for k, c in [('UNIV', UNIV), ('C1', C1), ('C2', C2), ('C3', C3), ('C4', C4), ('C5', C5),
                                                                                         ('C1-4', C1 & C2 & C3 & C4), ('C1-5', C1 & C2 & C3 & C4 & C5)]}}}
    res = {}; yearly_ev = {}
    for key, vol, tp in SPECS:
        S = signals(vol)
        ev, nb, nno = build_events(S, tp)
        df = event_table(ev)
        o = {'raw_signal_days': int(S.sum()), 'events_taken': len(ev), 'blocked_gap_up': nb, 'no_open': nno,
             'exit_reasons_all': df['why'].value_counts().to_dict()}
        for w, (a, b) in WIN.items():
            o[f'event_{w}'] = ev_stats(df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))])
        o['event_valid_ex_src3'] = ev_stats(df[(df['sig'] >= pd.Timestamp('2019-05-01')) & ~df['src3']])
        o['event_select_sub_2005-2011'] = ev_stats(df[(df['sig'] >= '2005-01-01') & (df['sig'] <= '2011-12-31')])
        o['event_select_sub_2012-2019-04'] = ev_stats(df[(df['sig'] >= '2012-01-01') & (df['sig'] <= '2019-04-30')])
        r, npos = portfolio(ev, BUY, SELL)
        rs, _ = portfolio(ev, BUY_S, SELL_S)
        rg, _ = portfolio(ev, 0.0, 0.0)
        for w, (a, b) in WIN.items():
            o[f'port_{w}'] = pstats(r, npos, a, b)
            o[f'port_{w}']['CAGR_stress'] = pstats(rs, npos, a, b)['CAGR']
            o[f'port_{w}']['CAGR_gross'] = pstats(rg, npos, a, b)['CAGR']
            o[f'port_{w}']['t_vs_EW_m_gross'] = pstats(rg, npos, a, b)['t_vs_EW_m']
        yearly_ev[key] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2)}
                          for y, g in df.groupby(df['sig'].dt.year)}
        res[key] = o
        es, ev_ = o['event_select_2005-01_2019-04'], o['event_valid_2019-05_2026-10']
        ps, pv = o['port_select_2005-01_2019-04'], o['port_valid_2019-05_2026-10']
        print(key, 'n', len(df), '| S ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'gross', es.get('mean_ab_gross_pct'), 'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'],
              '| V ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'), 'gross', ev_.get('mean_ab_gross_pct'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], flush=True)
        df.to_csv(Path(L.P).parent / f'w4_chu_{key}_events.csv.gz', index=False)
    out['results'] = res; out['yearly_events'] = yearly_ev
    def passes(o, w):
        e = o[f'event_{w}']; p = o[f'port_{w}']
        ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
        ok_50 = p['CAGR'] >= p['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['PRIMARY']
    vs = {'select': passes(pr, 'select_2005-01_2019-04'), 'valid': passes(pr, 'valid_2019-05_2026-10')}
    k = vs['select']['pass'] + vs['valid']['pass']
    thin = any(pr[f'event_{w}'].get('n', 0) < 30 or pr[f'event_{w}'].get('n_months', 0) < 24 for w in ['select_2005-01_2019-04', 'valid_2019-05_2026-10'])
    vs['verdict'] = '無法測試' if thin else ('可用' if k == 2 else ('待觀察' if k == 1 else '不可用'))
    vs['variants_pass'] = {kk: {w: passes(o, w) for w in WIN} for kk, o in res.items()}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8'))
    d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()
