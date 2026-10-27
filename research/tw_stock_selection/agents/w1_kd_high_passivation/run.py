#!/usr/bin/env python3
"""KD 高檔鈍化（羅威派）回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
事件引擎與統計口徑沿用 w1_granville_pullback_ma60/run.py 的寫法（複製，不 import、不修改原檔），成本用 exec_rebuild/exlib.py。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成本模型（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
names = ['adj', 'close_raw', 'open', 'high', 'value', 'flag', 'src']
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in names}
D['ret'] = pd.read_parquet(P / 'ret.parquet')
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
F = D['flag']; bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); HI = D['high'].mask(bad5); V = D['value']
fac = ADJ / CR
AO_raw = OP * fac
AH = np.fmax(np.fmax(HI * fac, ADJ), AO_raw).where(ADJ.notna())
LOWP = np.fmin(AO_raw, ADJ).where(ADJ.notna())
ACf = ADJ.ffill()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
FLAGWIN = F.isin([3, 4, 5]).astype(float).rolling(10, min_periods=1).max().fillna(0).astype(bool)   # [t-9, t]
UNIV = (V60 >= 1e7) & (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()
ELIG = (UNIV & ~FLAGWIN).to_numpy()

def kd(Hm, Lm, Cm, warm=20):
    """Hm/Lm/Cm: ndarray (T,N)。回傳 K, D（不可用處 NaN）。"""
    H9 = pd.DataFrame(Hm).rolling(9, min_periods=9).max().to_numpy()
    L9 = pd.DataFrame(Lm).rolling(9, min_periods=9).min().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        rsv = np.where(H9 > L9, (Cm - L9) / (H9 - L9) * 100, 50.0)
    rsv = np.where(np.isfinite(H9) & np.isfinite(L9) & np.isfinite(Cm), np.clip(rsv, 0, 100), np.nan)
    Tn, Nn = Cm.shape
    K = np.full((Tn, Nn), np.nan); Dd = np.full((Tn, Nn), np.nan)
    kp = np.full(Nn, 50.0); dp = np.full(Nn, 50.0); cnt = np.zeros(Nn, int)
    for t in range(Tn):
        r = rsv[t]; ok = np.isfinite(r)
        kp = np.where(ok, kp * 2 / 3 + np.nan_to_num(r) / 3, kp)
        dp = np.where(ok, dp * 2 / 3 + kp / 3, dp)
        cnt += ok
        use = ok & (cnt >= warm)
        K[t] = np.where(use, kp, np.nan); Dd[t] = np.where(use, dp, np.nan)
    return K, Dd

def streak(Kn, thr=80):
    s = np.zeros(Kn.shape, int); prev = np.zeros(Kn.shape[1], int)
    for t in range(Kn.shape[0]):
        prev = np.where(Kn[t] > thr, prev + 1, 0); s[t] = prev
    return s

Kn, Dn = kd(AH.to_numpy(), LOWP.to_numpy(), ADJ.to_numpy())
STK = streak(Kn)

# 漲停（依年代）
lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
AC_prev = ACf.shift(1).to_numpy()
GAP_UP = (AO_raw.to_numpy() / AC_prev - 1) >= lim - 0.006
AOn = AO_raw.to_numpy(); ACn = ACf.to_numpy(); ADJn = ADJ.to_numpy()
PX = AO_raw.bfill().to_numpy(); PX = np.where(np.isfinite(PX), PX, ACn)
MA5 = ADJ.rolling(5, min_periods=5).mean().to_numpy(); MA20 = ADJ.rolling(20, min_periods=20).mean().to_numpy()

oo = (AO_raw.shift(-1) / AO_raw - 1).where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0)
ew_oo = oo.mean(axis=1).fillna(0).to_numpy()
LEW = np.concatenate([[0.0], np.cumsum(np.log1p(ew_oo))])
ew_cc = D['ret'].where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0); E50n = e50.to_numpy()
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))
BUY, SELL = L.cost_rates(0.001); BUY_S, SELL_S = L.cost_rates('stress')

def sig_v1(n): return (STK == n) & ELIG

def sig_v2(look=20, lb=20):
    S = np.zeros((T, N), bool); stats = {'episodes': 0, 'no_cross': 0, 'broke_low': 0, 'new_episode': 0, 'not_elig': 0, 'ok': 0}
    for j in range(N):
        k, d, st, a = Kn[:, j], Dn[:, j], STK[:, j], ADJn[:, j]
        for t3 in np.where(st == 3)[0]:
            ts = t3 - 2
            if ts - lb < 0: continue
            stats['episodes'] += 1
            te = t3 + 1
            while te < T and not (np.isfinite(k[te]) and k[te] <= 80): te += 1
            if te >= T: continue
            lref = np.nanmin(a[ts - lb:ts]) if np.isfinite(a[ts - lb:ts]).any() else np.nan
            hit = None; reason = 'no_cross'
            for s in range(te + 1, min(te + look + 1, T)):
                if st[s] >= 3: reason = 'new_episode'; break
                if np.isfinite(k[s]) and np.isfinite(k[s - 1]) and k[s - 1] <= d[s - 1] and k[s] > d[s]:
                    hit = s; break
            if hit is None: stats[reason] += 1; continue
            seg = a[ts:hit + 1]
            if not np.isfinite(lref) or np.nanmin(seg) < lref: stats['broke_low'] += 1; continue
            if not ELIG[hit, j]: stats['not_elig'] += 1; continue
            S[hit, j] = True; stats['ok'] += 1
    return S, stats

def exit_fn(rule):
    def f(j, e):
        if rule in ('C5', 'C20'): return e + int(rule[1:])
        if rule == 'V2':
            seg = ADJn[e:min(e + 20, T), j] < MA20[e:min(e + 20, T), j]
            h = np.where(seg)[0]; return e + h[0] + 1 if len(h) else e + 20
        cap = 250
        if rule == 'A': seg = ADJn[e:min(e + cap, T), j] < MA5[e:min(e + cap, T), j]
        else: seg = Kn[e:min(e + cap, T), j] < 80
        h = np.where(seg)[0]; return e + h[0] + 1 if len(h) else e + cap
    return f

def build_events(S, rule):
    ev = []; nblock = 0; nnoopen = 0; xf = exit_fn(rule)
    for j in range(N):
        free_from = -1
        for t in np.where(S[:, j])[0]:
            if t < free_from: continue
            e = t + 1
            if e >= T: continue
            if not np.isfinite(AOn[e, j]): nnoopen += 1; continue
            if GAP_UP[e, j]: nblock += 1; continue
            x = xf(j, e); done = x < T
            ev.append((j, t, e, x, done)); free_from = x
    return ev, nblock, nnoopen

def event_table(ev):
    rows = []
    for j, t, e, x, done in ev:
        if not done: continue
        p0 = AOn[e, j]; p1 = PX[x, j]; raw = p1 / p0 - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1; net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        ew = math.expm1(LEW[x] - LEW[e]); b50 = E50n[x - 1] / E50n[t] - 1
        rows.append((cols[j], IDX[t], IDX[e], x - e, raw, net, net_s, ew, b50, bool(SRC3[t:x + 1].any())))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'hold', 'raw', 'net', 'net_s', 'ew', 'b50', 'src3'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']; df['ab_gross'] = df['raw'] - df['ew']
    return df

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31')}

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 1),
            'mean_raw_pct': round(df['raw'].mean() * 100, 3), 'mean_net_pct': round(df['net'].mean() * 100, 3),
            'mean_EW_pct': round(df['ew'].mean() * 100, 3), 'mean_ab_gross_pct': round(df['ab_gross'].mean() * 100, 3),
            'mean_ab_net_pct': round(df['ab'].mean() * 100, 3), 'median_ab_net_pct': round(df['ab'].median() * 100, 3),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': round(df['ab_s'].mean() * 100, 3), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_net_vs_0050_pct': round(df['ab50'].mean() * 100, 3), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3)}

def portfolio(ev, buy, sell):
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
    return pd.Series(g - c, index=IDX), pd.Series(n, index=IDX)

def pstats(r, npos, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    rr = r[m]; bb = r50[m]; ee = ew_cc[m]
    yrs = (rr.index[-1] - rr.index[0]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod(); return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
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

def approx_check():
    """2025-09-24 起官方 OHLC（真實 low）vs 近似 low 的 KD 比較（同一份原始價資料）。"""
    h = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    h = h[h['code'].isin(cols)]
    pv = {k: h.pivot_table(index='date', columns='code', values=k, aggfunc='last').sort_index() for k in ['open', 'high', 'low', 'close']}
    c = pv['close'].where(pv['close'] > 0); o = pv['open'].where(pv['open'] > 0).reindex_like(c); hh = pv['high'].where(pv['high'] > 0).reindex_like(c); lo = pv['low'].where(pv['low'] > 0).reindex_like(c)
    Kt, _ = kd(hh.to_numpy(), lo.to_numpy(), c.to_numpy())
    Ka, _ = kd(hh.to_numpy(), np.fmin(o, c).to_numpy(), c.to_numpy())
    ok = np.isfinite(Kt) & np.isfinite(Ka)
    st_t, st_a = streak(np.where(ok, Kt, np.nan)), streak(np.where(ok, Ka, np.nan))
    e_t, e_a = st_t == 3, st_a == 3
    inter = (e_t & e_a).sum(); uni = (e_t | e_a).sum()
    # 放寬：真 low 的事件日 ±2 日內近似版也有事件
    ea_w = pd.DataFrame(e_a).rolling(5, center=True, min_periods=1).max().to_numpy().astype(bool)
    return {'dates': f'{c.index[0].date()}~{c.index[-1].date()}', 'n_stock_days': int(ok.sum()), 'n_stocks': int(c.shape[1]),
            'K_mean_abs_err': round(float(np.abs(Kt - Ka)[ok].mean()), 2), 'K_mean_err_approx_minus_true': round(float((Ka - Kt)[ok].mean()), 2),
            'K_p90_abs_err': round(float(np.percentile(np.abs(Kt - Ka)[ok], 90)), 2),
            'K>80_share_true': round(float((Kt[ok] > 80).mean()), 4), 'K>80_share_approx': round(float((Ka[ok] > 80).mean()), 4),
            'K>80_agreement': round(float(((Kt > 80) == (Ka > 80))[ok].mean()), 4),
            'N3_events_true': int(e_t.sum()), 'N3_events_approx': int(e_a.sum()), 'N3_same_day_jaccard': round(float(inter / uni), 3),
            'N3_true_events_matched_within_pm2d': round(float((e_t & ea_w).sum() / e_t.sum()), 3)}

def main():
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'share_K_gt80_univ': round(float((Kn[ELIG] > 80).mean()), 4)}}
    out['approx_check'] = approx_check(); print('approx', out['approx_check'], flush=True)
    specs = [(f'N{n}_{ex}', ('v1', n), ex) for n in (3, 5, 10) for ex in ('A', 'B', 'C5', 'C20')] + [('V2_pullback_cross', ('v2', None), 'V2')]
    res = {}; yearly_ev = {}; cache = {}
    for key, (kind, n), ex in specs:
        if (kind, n) not in cache:
            if kind == 'v1': cache[(kind, n)] = (sig_v1(n), None)
            else: cache[(kind, n)] = sig_v2()
        S, v2st = cache[(kind, n)]
        ev, nb, nno = build_events(S, ex); df = event_table(ev)
        o = {'raw_signal_days': int(S.sum()), 'events_taken': len(ev), 'blocked_gap_up': nb, 'no_open': nno}
        if v2st: o['v2_funnel'] = v2st
        for w, (a, b) in WIN.items():
            o[f'event_{w}'] = ev_stats(df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))])
        o['event_valid_ex_src3'] = ev_stats(df[(df['sig'] >= pd.Timestamp('2019-05-01')) & ~df['src3']])
        o['event_select_sub_2005-2011'] = ev_stats(df[(df['sig'] >= '2005-01-01') & (df['sig'] <= '2011-12-31')])
        o['event_select_sub_2012-2019-04'] = ev_stats(df[(df['sig'] >= '2012-01-01') & (df['sig'] <= '2019-04-30')])
        r, npos = portfolio(ev, BUY, SELL); rs, _ = portfolio(ev, BUY_S, SELL_S); rg, _ = portfolio(ev, 0.0, 0.0)
        for w, (a, b) in WIN.items():
            o[f'port_{w}'] = pstats(r, npos, a, b)
            o[f'port_{w}']['CAGR_stress'] = pstats(rs, npos, a, b)['CAGR']; o[f'port_{w}']['CAGR_gross'] = pstats(rg, npos, a, b)['CAGR']
        yearly_ev[key] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2)}
                          for y, g in df.groupby(df['sig'].dt.year)}
        res[key] = o
        es, ev_ = o['event_select_2005-01_2019-04'], o['event_valid_2019-05_2026-10']
        ps, pv = o['port_select_2005-01_2019-04'], o['port_valid_2019-05_2026-10']
        print(key, 'n', len(df), 'hold', es.get('avg_hold_days'), '| S ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'g', es.get('mean_ab_gross_pct'),
              'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'],
              '| V ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'), 'g', ev_.get('mean_ab_gross_pct'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], flush=True)
        if key == 'N3_A': df.to_csv(HERE / 'events_N3_A.csv.gz', index=False)
        if key == 'V2_pullback_cross': df.to_csv(HERE / 'events_V2.csv.gz', index=False)
    out['results'] = res; out['yearly_events'] = yearly_ev
    def passes(o, w):
        e = o[f'event_{w}']; p = o[f'port_{w}']
        ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
        ok_50 = p['CAGR'] >= p['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    vs = {}
    for nm in ('N3_A', 'V2_pullback_cross'):
        pr = res[nm]
        v = {'select': passes(pr, 'select_2005-01_2019-04'), 'valid': passes(pr, 'valid_2019-05_2026-10')}
        k = v['select']['pass'] + v['valid']['pass']; v['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
        vs[nm] = v
    vs['PRIMARY_verdict'] = vs['N3_A']['verdict']
    vs['variants_pass'] = {kk: {w: passes(o, w)['pass'] for w in ['select_2005-01_2019-04', 'valid_2019-05_2026-10']} for kk, o in res.items()}
    out['verdict'] = vs
    f = HERE / 'results.json'; d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()
