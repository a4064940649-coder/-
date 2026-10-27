#!/usr/bin/env python3
"""地量見地價回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
架構複製自 w4_dryup_volume_rebound/run.py（不 import）；成本用 exec_rebuild/exlib.cost_rates。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild')); sys.path.insert(0, str(HERE.parents[1]))
import exlib as L  # noqa: E402  成本模型、0050（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
names = ['adj', 'close_raw', 'open', 'value', 'flag', 'src', 'ret']
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in names}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if c.isdigit() and len(c) == 4 and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
F = D['flag']; bad5 = (F == 5); FOK = F.isin([0, 1, 2])
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); VAL = D['value']
fac = ADJ / CR
AO_raw = OP * fac
ACf = ADJ.ffill()
v0 = VAL.fillna(0).where(VAL.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
POOL = (V60 >= 1e7) & (CR >= 10) & FOK & ADJ.notna()          # 當日池；訊號用前一日
POOL_PREV = POOL.shift(1, fill_value=False)
VOL = (VAL / CR).where((VAL > 0) & (CR > 0) & FOK & ADJ.notna())
TRUNC = [pd.Timestamp('2025-01-21'), pd.Timestamp('2025-04-10')]
VOL.loc[VOL.index.isin(TRUNC)] = np.nan
MA60 = ADJ.rolling(60, min_periods=60).mean()
LOWPOS = ADJ < MA60
VMA5 = VOL.rolling(5, min_periods=5).mean()
CONF = (VOL >= 2 * VMA5) & (ADJ > ADJ.shift(1)) & FOK
CONFn = CONF.fillna(False).to_numpy()
V60P = V60.shift(1).to_numpy()

def trough(Lw):
    llv = VOL.rolling(Lw, min_periods=int(0.8 * Lw)).min()
    S = VOL.notna() & (VOL <= llv) & LOWPOS & POOL_PREV & FOK
    return S.fillna(False).to_numpy()

lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
AC_prev = ACf.shift(1).to_numpy()
AOn = AO_raw.to_numpy(); ACn = ACf.to_numpy(); ADJn = ADJ.to_numpy()
with np.errstate(invalid='ignore', divide='ignore'):
    GAP_UP = (AOn / AC_prev - 1) >= lim - 0.006
PX = AO_raw.bfill().to_numpy(); PX = np.where(np.isfinite(PX), PX, ACn)

oo = (AO_raw.shift(-1) / AO_raw - 1).where(POOL_PREV).clip(-0.5, 1.0)
ew_oo = oo.mean(axis=1).fillna(0).to_numpy()
LEW = np.concatenate([[0.0], np.cumsum(np.log1p(ew_oo))])
ew_cc = D['ret'].where(POOL_PREV).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0); E50n = e50.to_numpy()
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))
BUY, SELL = L.cost_rates(0.001); BUY_S, SELL_S = L.cost_rates('stress')

def candidates(S, ver, H):
    """所有候選（未去重、未設上限）：(prio, e, j, t, x, why)。"""
    out = {}; nblock = 0; nnoopen = 0; nnoconf = 0
    for j in range(N):
        for t in np.where(S[:, j])[0]:
            if ver == 'A':
                c = t
            else:
                w = np.where(CONFn[t + 1:min(t + 11, T), j])[0]
                if len(w) == 0: nnoconf += 1; continue
                c = t + 1 + w[0]
            e = c + 1
            if e >= T: continue
            if (j, e) in out: continue            # 同一次確認對應多個地量日 → 取最早地量日
            if not np.isfinite(AOn[e, j]): nnoopen += 1; out[(j, e)] = None; continue
            if GAP_UP[e, j]: nblock += 1; out[(j, e)] = None; continue
            why = 'time'; x = e + H
            if ver == 'B':
                hi = min(e + H, T); a = ADJn[e:hi, j]
                with np.errstate(invalid='ignore'):
                    hit = np.where(a < 0.93 * ADJn[t, j])[0]
                if len(hit): x = e + hit[0] + 1; why = 'stop'
            pr = V60P[t, j]
            out[(j, e)] = (-(pr if np.isfinite(pr) else 0), e, j, t, x, why)
    cand = sorted(v for v in out.values() if v is not None)
    cand.sort(key=lambda r: (r[1], r[0]))
    return cand, {'blocked_gap_up': nblock, 'no_open': nnoopen, 'no_confirm_in_10d': nnoconf}

def accept(cand, cap):
    held_until = np.full(N, -1); active = []; ev = []; nskip_cap = 0
    for pr, e, j, t, x, why in cand:
        if held_until[j] > e: continue
        if cap is not None:
            active = [a for a in active if a > e]
            if len(active) >= cap: nskip_cap += 1; continue
            active.append(x)
        held_until[j] = x
        ev.append((j, t, e, x, x < T, why))
    return ev, nskip_cap

def event_table(ev):
    rows = []
    for j, t, e, x, done, why in ev:
        if not done: continue
        p0 = AOn[e, j]; p1 = PX[x, j]; raw = p1 / p0 - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1; net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        ew = math.expm1(LEW[x] - LEW[e]); b50 = E50n[x - 1] / E50n[e - 1] - 1
        rows.append((cols[j], IDX[t], IDX[e], x - e, why, raw, net, net_s, ew, b50, bool(SRC3[t:x + 1].any())))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'hold', 'why', 'raw', 'net', 'net_s', 'ew', 'b50', 'src3'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']; df['ab_gross'] = df['raw'] - df['ew']
    return df

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31')}

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    r3 = lambda x: round(float(x) * 100, 3)
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 1),
            'stop_share': round(float((df['why'] == 'stop').mean()), 3),
            'mean_raw_pct': r3(df['raw'].mean()), 'mean_net_pct': r3(df['net'].mean()), 'mean_EW_pct': r3(df['ew'].mean()),
            'mean_ab_gross_pct': r3(df['ab_gross'].mean()), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': r3(df['ab'].mean()), 'median_ab_net_pct': r3(df['ab'].median()), 't_month_ab_net': tstat(gm['ab'].mean()),
            't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': r3(df['ab_s'].mean()), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_net_vs_0050_pct': r3(df['ab50'].mean()), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3)}

def portfolio(ev, buy, sell, slots):
    R = np.zeros((T, N), np.float32); W = np.zeros((T, N), np.float32); ent = np.zeros(T); exi = np.zeros(T)
    for j, t, e, x, done, why in ev:
        xe = x if done else T - 1
        W[e:xe + 1, j] = 1
        R[e, j] = ACn[e, j] / AOn[e, j] - 1
        if xe > e: R[e + 1:xe + 1, j] = np.nan_to_num(ACn[e + 1:xe + 1, j] / ACn[e:xe, j] - 1)
        if done:
            R[x, j] = PX[x, j] / ACn[x - 1, j] - 1 if x > e else PX[x, j] / AOn[e, j] - 1
            exi[x] += 1
        ent[e] += 1
    n = W.sum(1)
    den = np.full(T, float(slots)) if slots else np.where(n > 0, n, 1)
    g = np.where(n > 0, (W * R).sum(1) / den, 0.0); c = np.where(n > 0, (ent * buy + exi * sell) / den, 0.0)
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
    o['share_days_invested'] = round(float((npos[m] > 0).mean()), 3)
    o['avg_positions'] = round(float(npos[m].mean()), 1)
    mo = lambda x: (1 + x).groupby(x.index.to_period('M')).prod() - 1
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    return o

SPECS = [(v, Lw, H) for v in ('B', 'A') for Lw in (120, 60, 250) for H in (20, 10, 60)]

def src3_check(S, Lw):
    P1 = POOL_PREV.to_numpy() & np.isfinite(VOL.to_numpy())
    rate = pd.Series(S.sum(1) / np.maximum(P1.sum(1), 1), index=IDX)
    seg = {'official_2005-01_2024-01': rate['2005-01-01':'2023-12-31'], 'src3_2024-02-15_2025-09-23': rate['2024-02-15':'2025-09-23'],
           'src4_2025-09-24_2026-10': rate['2025-09-24':]}
    o = {k: {'mean_daily_trough_rate_pct': round(float(v.mean()) * 100, 3), 'p99_pct': round(float(v.quantile(0.99)) * 100, 2)} for k, v in seg.items()}
    o['first10d_after_2024-02-15_pct'] = round(float(rate['2024-02-15':].iloc[:10].mean()) * 100, 3)
    o['first20d_after_2025-09-24_pct'] = round(float(rate['2025-09-24':].iloc[:20].mean()) * 100, 3)
    o['top10_days'] = {str(d.date()): round(float(v) * 100, 1) for d, v in rate.sort_values(ascending=False).head(10).items()}
    return o

def main():
    out = {'meta': {'bad_days_dropped': BAD, 'truncated_days_V_nan': [str(d.date()) for d in TRUNC], 'n_stocks': N,
                    'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S]}}
    res = {}; yearly_ev = {}; chk = {}
    for ver, Lw, H in SPECS:
        key = f'{ver}_L{Lw}_H{H}'
        S = trough(Lw)
        if Lw not in chk: chk[Lw] = src3_check(S, Lw)
        cand, cstat = candidates(S, ver, H)
        evU, _ = accept(cand, None); evC, nskip = accept(cand, 20)
        dfU = event_table(evU); dfC = event_table(evC)
        o = {'trough_days': int(S.sum()), 'candidates': len(cand), **cstat, 'events_uncapped': len(evU), 'events_cap20': len(evC), 'skipped_cap20': nskip}
        for w, (a, b) in WIN.items():
            sel = lambda df: df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))]
            o[f'event_{w}'] = ev_stats(sel(dfU)); o[f'event_cap20_{w}'] = ev_stats(sel(dfC))
        o['event_valid_ex_src3'] = ev_stats(dfU[(dfU['sig'] >= pd.Timestamp('2019-05-01')) & ~dfU['src3']])
        r, npos = portfolio(evC, BUY, SELL, 20); rs, _ = portfolio(evC, BUY_S, SELL_S, 20); rg, _ = portfolio(evC, 0.0, 0.0, 20)
        rf, nf = portfolio(evU, BUY, SELL, None)
        for w, (a, b) in WIN.items():
            p = pstats(r, npos, a, b); p['CAGR_stress'] = pstats(rs, npos, a, b)['CAGR']; p['CAGR_gross'] = pstats(rg, npos, a, b)['CAGR']
            o[f'port_cap20_{w}'] = p
            pf = pstats(rf, nf, a, b)
            o[f'port_fullEW_{w}'] = {k: pf[k] for k in ('CAGR', 'maxDD', 't_vs_0050_m', 't_vs_EW_m', 'avg_positions', 'share_days_invested')}
        yearly_ev[key] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2)}
                          for y, g in dfU.groupby(dfU['sig'].dt.year)}
        res[key] = o
        es, ev_ = o['event_select_2005-01_2019-04'], o['event_valid_2019-05_2026-10']
        ps, pv = o['port_cap20_select_2005-01_2019-04'], o['port_cap20_valid_2019-05_2026-10']
        print(key, 'nU', len(dfU), '| S ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'],
              '| V ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], flush=True)
        if (Lw, H) == (120, 20): dfU.to_csv(HERE / f'events_{key}.csv.gz', index=False)
    out['results'] = res; out['yearly_events'] = yearly_ev; out['src3_check'] = {f'L{k}': v for k, v in chk.items()}
    def passes(o, w):
        e = o[f'event_{w}']; p = o[f'port_cap20_{w}']
        ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
        ok_50 = p['CAGR'] >= p['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['B_L120_H20']
    vs = {'select': passes(pr, 'select_2005-01_2019-04'), 'valid': passes(pr, 'valid_2019-05_2026-10')}
    k = vs['select']['pass'] + vs['valid']['pass']
    thin = any(pr[f'event_{w}'].get('n', 0) < 30 or pr[f'event_{w}'].get('n_months', 0) < 24 for w in ['select_2005-01_2019-04', 'valid_2019-05_2026-10'])
    vs['verdict'] = '無法測試' if thin else ('可用' if k == 2 else ('待觀察' if k == 1 else '不可用'))
    vs['variants_pass'] = {kk: {w: passes(o, w)['pass'] for w in ['select_2005-01_2019-04', 'valid_2019-05_2026-10']} for kk, o in res.items()}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()
