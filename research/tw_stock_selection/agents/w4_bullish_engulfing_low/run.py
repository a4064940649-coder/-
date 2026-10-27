#!/usr/bin/env python3
"""低檔多頭吞噬＋放量＋確認 K 回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
事件／組合統計架構複製自 w4_choke_volume_first_red/run.py（不 import），成本用 exec_rebuild/exlib.cost_rates。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild')); sys.path.insert(0, str(HERE.parents[1]))
import exlib as L  # noqa: E402
from btlib import tstat  # noqa: E402

P = L.P
names = ['adj', 'close_raw', 'open', 'value', 'flag', 'src']
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in names}
D['ret'] = pd.read_parquet(P / 'ret.parquet')
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if c.isdigit() and len(c) == 4 and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
F = D['flag'].fillna(0); bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); V = D['value']
fac = ADJ / CR
AO = OP * fac                                   # 還原開盤
ACf = ADJ.ffill()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
FLOK = ~F.isin([3, 4, 5])
UNIV = (V60 >= 1e7) & (CR >= 10) & ADJ.notna()
VOL = (V / CR).where((V > 0) & (CR > 0))
VMA5p = VOL.shift(1).rolling(5, min_periods=5).mean()
MA20 = ADJ.rolling(20, min_periods=20).mean()
BODY = (CR - OP) / OP                           # 原始價實體（紅正黑負）
BODYp = BODY.shift(1)
C_flag = FLOK & FLOK.shift(1, fill_value=False)
C_vol = VOL >= 1.5 * VMA5p
ret10 = ADJ.shift(1) / ADJ.shift(11) - 1
# 多頭吞噬
B1 = BODYp <= -0.015
B2 = (BODY > 0) & (AO <= ADJ.shift(1)) & (ADJ >= AO.shift(1)) & (BODY >= 1.5 * (-BODYp))
B3 = (ret10 <= -0.05) & (ADJ < MA20)
# 空頭吞噬（對照）
S1 = BODYp >= 0.015
S2 = (BODY < 0) & (AO >= ADJ.shift(1)) & (ADJ <= AO.shift(1)) & ((-BODY) >= 1.5 * BODYp)
S3 = (ret10 >= 0.05) & (ADJ > MA20)
up1 = ADJ.shift(-1) > ADJ; dn1 = ADJ.shift(-1) < ADJ

lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
AC_prev = ACf.shift(1).to_numpy()
GAP_UP = (AO.to_numpy() / AC_prev - 1) >= lim - 0.006
AOn = AO.to_numpy(); ACn = ACf.to_numpy(); ADJn = ADJ.to_numpy()
PX = AO.bfill().to_numpy(); PX = np.where(np.isfinite(PX), PX, ACn)

oo = (AO.shift(-1) / AO - 1).where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0)
ew_oo = oo.mean(axis=1).fillna(0).to_numpy()
LEW = np.concatenate([[0.0], np.cumsum(np.log1p(ew_oo))])
ew_cc = D['ret'].where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0); E50n = e50.to_numpy()
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))
GAPWIN = (IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2024-04-15'))   # 回看跨 2024-01~02 缺口

BUY, SELL = L.cost_rates(0.001)
BUY_S, SELL_S = L.cost_rates('stress')

def signals(side, conf, vol=True):
    base = UNIV & C_flag
    S = base & ((B1 & B2 & B3) if side == 'bull' else (S1 & S2 & S3))
    if vol: S = S & C_vol
    if conf: S = S & (up1 if side == 'bull' else dn1)
    return S.fillna(False).to_numpy()

H = 10
def build_events(S, lag, side):
    ev = []; nblock = 0; nnoopen = 0; reasons = {'stop': 0, 'time': 0}
    for j in range(N):
        ts = np.where(S[:, j])[0]; free_from = -1
        for t in ts:
            if t < free_from: continue
            e = t + lag
            if e >= T: continue
            if not np.isfinite(AOn[e, j]): nnoopen += 1; continue
            if GAP_UP[e, j]: nblock += 1; continue
            stop = AOn[t, j]
            hi = min(e + H, T); a = ADJn[e:hi, j]
            with np.errstate(invalid='ignore'):
                h = (a < stop) if side == 'bull' else (a > stop)
            hit = np.where(h)[0]
            if len(hit): x = e + hit[0] + 1; why = 'stop'
            else: x = e + H; why = 'time'
            done = x < T
            if done: reasons[why] += 1
            ev.append((j, t, e, x, done, why)); free_from = x
    return ev, nblock, nnoopen, reasons

def event_table(ev):
    rows = []
    for j, t, e, x, done, why in ev:
        if not done: continue
        p0 = AOn[e, j]; p1 = PX[x, j]; raw = p1 / p0 - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1
        net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        ew = math.expm1(LEW[x] - LEW[e]); b50 = E50n[x - 1] / E50n[e - 1] - 1
        rows.append((cols[j], IDX[t], IDX[e], x - e, why, raw, net, net_s, ew, b50, bool(SRC3[t:x + 1].any()), bool(GAPWIN[t]),
                     AOn[e, j] / ACn[e - 1, j] - 1))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'hold', 'why', 'raw', 'net', 'net_s', 'ew', 'b50', 'src3', 'gapwin', 'gap'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']
    df['ab_gross'] = df['raw'] - df['ew']
    return df

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'), 'src3_2024-02-15_2025-09-23': ('2024-02-15', '2025-09-23')}

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 1),
            'stop_share': round(float((df['why'] == 'stop').mean()), 3), 'mean_entry_gap_pct': round(df['gap'].mean() * 100, 3),
            'mean_raw_pct': round(df['raw'].mean() * 100, 3), 'mean_net_pct': round(df['net'].mean() * 100, 3),
            'mean_EW_pct': round(df['ew'].mean() * 100, 3), 'mean_ab_gross_pct': round(df['ab_gross'].mean() * 100, 3),
            't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': round(df['ab'].mean() * 100, 3), 'median_ab_net_pct': round(df['ab'].median() * 100, 3),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': round(df['ab_s'].mean() * 100, 3), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_net_vs_0050_pct': round(df['ab50'].mean() * 100, 3), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3)}

def portfolio(ev, buy, sell):
    R = np.zeros((T, N), np.float32); W = np.zeros((T, N), np.float32); ent = np.zeros(T); exi = np.zeros(T)
    for j, t, e, x, done, why in ev:
        xe = x if done else T - 1
        W[e:xe + 1, j] += 1
        R[e, j] = ACn[e, j] / AOn[e, j] - 1
        if xe > e: R[e + 1:xe + 1, j] = np.nan_to_num(ACn[e + 1:xe + 1, j] / ACn[e:xe, j] - 1)
        if done:
            R[x, j] = PX[x, j] / ACn[x - 1, j] - 1; exi[x] += 1
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

SPECS = [('CONF', 'bull', True, True, 2), ('NOCONF', 'bull', False, True, 1), ('CONF_NOVOL', 'bull', True, False, 2),
         ('BEAR_CONF', 'bear', True, True, 2), ('BEAR_NOCONF', 'bear', False, True, 1)]

def main():
    cc = {k: int((UNIV & C_flag & c).sum().sum()) for k, c in [('B1', B1), ('B1B2', B1 & B2), ('B1B2B3', B1 & B2 & B3), ('B1B2B3V', B1 & B2 & B3 & C_vol),
                                                                ('B1B2B3V_conf', B1 & B2 & B3 & C_vol & up1)]}
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S], 'cond_counts_in_univ': cc}}
    res = {}; yearly_ev = {}
    for key, side, conf, vol, lag in SPECS:
        S = signals(side, conf, vol)
        ev, nb, nno, reasons = build_events(S, lag, side)
        df = event_table(ev)
        o = {'raw_signal_days': int(S.sum()), 'events_taken': len(ev), 'blocked_gap_up': nb, 'no_open': nno, 'exit_reasons': reasons}
        for w, (a, b) in WIN.items():
            o[f'event_{w}'] = ev_stats(df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))])
        vv = df[df['sig'] >= pd.Timestamp('2019-05-01')]
        o['event_valid_ex_src3'] = ev_stats(vv[~vv['src3']])
        o['event_valid_ex_src3_ex_gapwin'] = ev_stats(vv[~vv['src3'] & ~vv['gapwin']])
        o['event_select_sub_2005-2011'] = ev_stats(df[(df['sig'] >= '2005-01-01') & (df['sig'] <= '2011-12-31')])
        o['event_select_sub_2012-2019-04'] = ev_stats(df[(df['sig'] >= '2012-01-01') & (df['sig'] <= '2019-04-30')])
        r, npos = portfolio(ev, BUY, SELL); rs, _ = portfolio(ev, BUY_S, SELL_S); rg, _ = portfolio(ev, 0.0, 0.0)
        for w, (a, b) in WIN.items():
            o[f'port_{w}'] = pstats(r, npos, a, b)
            o[f'port_{w}']['CAGR_stress'] = pstats(rs, npos, a, b)['CAGR']; o[f'port_{w}']['CAGR_gross'] = pstats(rg, npos, a, b)['CAGR']
        yearly_ev[key] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'ab_gross_pct': round(g['ab_gross'].mean() * 100, 2)}
                          for y, g in df.groupby(df['sig'].dt.year)}
        res[key] = o
        es, ev_ = o['event_select_2005-01_2019-04'], o['event_valid_2019-05_2026-10']
        ps, pv = o['port_select_2005-01_2019-04'], o['port_valid_2019-05_2026-10']
        print(key, 'n', len(df), '| S n', es.get('n'), 'abG', es.get('mean_ab_gross_pct'), 'ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'),
              'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'],
              '| V n', ev_.get('n'), 'abG', ev_.get('mean_ab_gross_pct'), 'ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'),
              'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], flush=True)
        df.to_csv(Path(L.P).parent / f'w4_engulf_{key}_events.csv.gz', index=False)
    out['results'] = res; out['yearly_events'] = yearly_ev
    def passes(o, w):
        e = o[f'event_{w}']; p = o[f'port_{w}']
        ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
        ok_50 = p['CAGR'] >= p['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['CONF']; ws = ['select_2005-01_2019-04', 'valid_2019-05_2026-10']
    vs = {'select': passes(pr, ws[0]), 'valid': passes(pr, ws[1])}
    k = vs['select']['pass'] + vs['valid']['pass']
    thin = any(pr[f'event_{w}'].get('n', 0) < 30 or pr[f'event_{w}'].get('n_months', 0) < 24 for w in ws)
    vs['verdict'] = '無法測試' if thin else ('可用' if k == 2 else ('待觀察' if k == 1 else '不可用'))
    vs['variants_pass'] = {kk: {w: passes(o, w) for w in ws} for kk, o in res.items() if not kk.startswith('BEAR')}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(out['meta']['cond_counts_in_univ'])); print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()
