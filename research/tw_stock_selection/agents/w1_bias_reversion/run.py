#!/usr/bin/env python3
"""葛蘭碧買點4：負乖離過大買進。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。
骨架沿用 w1_granville_pullback_ma60/run.py（同股池、同事件／組合統計），成本只從 exec_rebuild/exlib 匯入。"""
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
AH = (HI * fac).to_numpy()
ACf = ADJ.ffill()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
UNIV = (V60 >= 1e7) & (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()

# flag 3/4/5（全市場同日 flag4 >= 50 檔的面板缺口日不算）
mkt4 = (F == 4).sum(axis=1) >= 50
FL = (F.isin([3, 4, 5]) & ~((F == 4) & mkt4.to_numpy()[:, None])).astype(float)
MKT4_DAYS = [str(x.date()) for x in IDX[mkt4.to_numpy()]]

# 處置（t 或 t+1 在處置期間）
dsp = pd.read_csv(L.X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['start', 'end'])
INDISP = np.zeros((T, N), bool); CJ = {c: j for j, c in enumerate(cols)}
for _, r in dsp.iterrows():
    if r['code'] in CJ and pd.notna(r['start']) and pd.notna(r['end']):
        INDISP[(IDX >= r['start']) & (IDX <= r['end']), CJ[r['code']]] = True
DISP_TT1 = INDISP | np.vstack([INDISP[1:], np.zeros((1, N), bool)])

lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
AC_prev = ACf.shift(1).to_numpy()
GAP_UP = (AO_raw.to_numpy() / AC_prev - 1) >= lim - 0.006
LOCK_DN = (AH / AC_prev - 1) <= -(lim - 0.006)                # 整天鎖跌停（最高價也在跌停）
AOn = AO_raw.to_numpy(); ACn = ACf.to_numpy(); ADJn = ADJ.to_numpy()
PXo = AO_raw.to_numpy()
# 出場價：x 日開盤；停牌或鎖跌停 → 之後第一個可賣開盤；下市 → 最後收盤
SELLABLE = np.isfinite(PXo) & ~LOCK_DN
last_valid = np.where(np.isfinite(ADJn), np.arange(T)[:, None], -1); last_valid = np.maximum.accumulate(last_valid, axis=0)
def exit_px(x, j):
    k = x
    while k < T and not SELLABLE[k, j]:
        if last_valid[T - 1, j] < k:      # 之後再也沒有價格（下市）→ 最後收盤
            return ACn[T - 1, j], k, True
        k += 1
    if k >= T: return np.nan, k, False
    return PXo[k, j], k, False

oo = (AO_raw.shift(-1) / AO_raw - 1)
oo = oo.where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0)
ew_oo = oo.mean(axis=1).fillna(0).to_numpy()
LEW = np.concatenate([[0.0], np.cumsum(np.log1p(ew_oo))])
ew_cc = D['ret'].where(UNIV.shift(1, fill_value=False)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50 = L.e50.reindex(IDX.union(L.e50.index)).ffill().reindex(IDX)
r50 = e50.pct_change().fillna(0)
E50n = e50.to_numpy()
SRC3 = ((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2025-09-23')))
DELIST_IDX = last_valid[T - 1]          # 每檔最後有價日

BUY, SELL = L.cost_rates(0.001)
BUY_S, SELL_S = L.cost_rates('stress')

MA240 = ADJ.rolling(240, min_periods=240).mean()
MA240UP = (MA240 >= MA240.shift(20)).to_numpy()
UPDAY = (ADJ > ADJ.shift(1)).to_numpy()
UNIVn = UNIV.to_numpy()
MAS = {}
def ma(n):
    if n not in MAS: MAS[n] = ADJ.rolling(n, min_periods=n).mean().to_numpy()
    return MAS[n]

def signals(n, thr, trig, filt):
    M = ma(n)
    with np.errstate(invalid='ignore', divide='ignore'):
        B = ADJn / M - 1
    flagged = (FL.rolling(n, min_periods=1).max().to_numpy() > 0)
    S = UNIVn & (B <= thr) & ~flagged & ~DISP_TT1
    if trig == 'red': S &= UPDAY
    if filt == 'ma240_up': S &= MA240UP
    return S, M

def build_events(S, M, ex):
    ev = []; nblock = 0; nnoopen = 0; nlock = 0
    for j in range(N):
        ts = np.where(S[:, j])[0]; free_from = -1
        for t in ts:
            if t < free_from: continue
            e = t + 1
            if e >= T: continue
            if not np.isfinite(AOn[e, j]): nnoopen += 1; continue
            if GAP_UP[e, j]: nblock += 1; continue
            if ex == 'A':
                seg = ADJn[e:min(e + 20, T), j] >= M[e:min(e + 20, T), j]
                hit = np.where(seg)[0]
                x = e + hit[0] + 1 if len(hit) else e + 20
            else:
                x = e + int(ex[1:])
            if x >= T:
                ev.append((j, t, e, x, x, np.nan, False, False)); free_from = T; continue
            px, xs, dl = exit_px(x, j)
            if xs > x and not dl: nlock += 1
            done = np.isfinite(px)
            ev.append((j, t, e, x, min(xs, T - 1), px, bool(done), dl))
            free_from = xs if not dl else T
    return ev, nblock, nnoopen, nlock

def event_table(ev):
    rows = []
    for j, t, e, x, xs, px, done, dl in ev:
        if not done: continue
        p0 = AOn[e, j]; raw = px / p0 - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1
        net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        xe = min(xs, T - 1)
        ew = math.expm1(LEW[xe] - LEW[e])
        b50 = E50n[xe - 1] / E50n[t] - 1
        touch3 = bool(SRC3[t:xe + 1].any())
        gap = AOn[e, j] / ACn[t, j] - 1                  # 隔夜跳空（t 收→t+1 開）
        delist_in_hold = bool(DELIST_IDX[j] <= xe + 5 and DELIST_IDX[j] < T - 1)
        rows.append((cols[j], IDX[t], IDX[e], xe - e, raw, net, net_s, ew, b50, touch3, gap, delist_in_hold or dl))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'hold', 'raw', 'net', 'net_s', 'ew', 'b50', 'src3', 'gap', 'delist'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']
    df['ab_gross'] = df['raw'] - df['ew']
    # t 收盤進場（不可成交）的毛異常：(1+raw)(1+gap)−1 − EW(含 t→t+1 開盤的近似：忽略)
    df['raw_close_entry'] = (1 + df['raw']) * (1 + df['gap']) - 1
    return df

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31')}

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 1),
            'mean_raw_pct': round(df['raw'].mean() * 100, 3), 'mean_net_pct': round(df['net'].mean() * 100, 3),
            'median_net_pct': round(df['net'].median() * 100, 3),
            'mean_EW_pct': round(df['ew'].mean() * 100, 3),
            'mean_ab_gross_pct': round(df['ab_gross'].mean() * 100, 3),
            'mean_ab_net_pct': round(df['ab'].mean() * 100, 3), 'median_ab_net_pct': round(df['ab'].median() * 100, 3),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': round(df['ab_s'].mean() * 100, 3), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_net_vs_0050_pct': round(df['ab50'].mean() * 100, 3), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': round(float((df['ab'] > 0).mean()), 3), 'hit_raw': round(float((df['raw'] > 0).mean()), 3),
            'mean_gap_pct': round(df['gap'].mean() * 100, 3), 'mean_raw_if_close_entry_pct': round(df['raw_close_entry'].mean() * 100, 3),
            'p01_raw_pct': round(df['raw'].quantile(0.01) * 100, 2), 'p99_raw_pct': round(df['raw'].quantile(0.99) * 100, 2),
            'n_delist': int(df['delist'].sum()), 'top_month_share_n': round(float(gm.size().max() / len(df)), 3)}

def portfolio(ev, buy, sell):
    R = np.zeros((T, N), np.float32); W = np.zeros((T, N), np.float32)
    ent = np.zeros(T); exi = np.zeros(T)
    for j, t, e, x, xs, px, done, dl in ev:
        xe = xs if done else T - 1
        W[e:xe + 1, j] += 1
        R[e, j] = ACn[e, j] / AOn[e, j] - 1
        if xe > e:
            R[e + 1:xe + 1, j] = np.nan_to_num(ACn[e + 1:xe + 1, j] / ACn[e:xe, j] - 1)
        if done:
            R[xe, j] = px / ACn[xe - 1, j] - 1 if xe > e else px / AOn[e, j] - 1
            exi[xe] += 1
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

SETS = {'a_B20': (20, -0.10), 'b_B13': (13, -0.06), 'c_B60': (60, -0.07)}
PRIMARY = 'a_B20_touch_none_A'

def passes(o, w):
    e = o[f'event_{w}']; p = o[f'port_{w}']
    ok_ew = (e.get('mean_ab_net_pct', -1) > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
    ok_50 = p['CAGR'] >= p['0050_CAGR']
    return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}

def main():
    out = {'meta': {'bad_days_dropped': BAD, 'market_flag4_days_ignored': MKT4_DAYS, 'n_stocks': N,
                    'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S]}}
    res = {}; yearly_ev = {}
    for sn, (n, thr) in SETS.items():
        for trig in ('touch', 'red'):
            for filt in ('none', 'ma240_up'):
                S, M = signals(n, thr, trig, filt)
                for ex in ('A', 'B5', 'B10', 'B20'):
                    key = f'{sn}_{trig}_{filt}_{ex}'
                    ev, nb, nno, nlk = build_events(S, M, ex)
                    df = event_table(ev)
                    o = {'raw_signal_days': int(S.sum()), 'events_taken': len(ev), 'blocked_gap_up': nb, 'no_open': nno, 'exit_delayed_lockdown': nlk}
                    for w, (a, b) in WIN.items():
                        o[f'event_{w}'] = ev_stats(df[(df['sig'] >= a) & (df['sig'] <= b)])
                    o['event_valid_ex_src3'] = ev_stats(df[(df['sig'] >= '2019-05-01') & ~df['src3']])
                    o['event_select_sub_2005-2011'] = ev_stats(df[(df['sig'] >= '2005-01-01') & (df['sig'] <= '2011-12-31')])
                    o['event_select_sub_2012-2019-04'] = ev_stats(df[(df['sig'] >= '2012-01-01') & (df['sig'] <= '2019-04-30')])
                    o['event_select_ex_delist_DIAG'] = ev_stats(df[(df['sig'] >= '2005-01-01') & (df['sig'] <= '2019-04-30') & ~df['delist']])
                    r, npos = portfolio(ev, BUY, SELL); rs, _ = portfolio(ev, BUY_S, SELL_S); rg, _ = portfolio(ev, 0.0, 0.0)
                    for w, (a, b) in WIN.items():
                        o[f'port_{w}'] = pstats(r, npos, a, b)
                        o[f'port_{w}']['CAGR_stress'] = pstats(rs, npos, a, b)['CAGR']
                        o[f'port_{w}']['CAGR_gross'] = pstats(rg, npos, a, b)['CAGR']
                    yearly_ev[key] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'net_pct': round(g['net'].mean() * 100, 2)}
                                      for y, g in df.groupby(df['sig'].dt.year)}
                    o['pass'] = {w: passes(o, w) for w in ['select_2005-01_2019-04', 'valid_2019-05_2026-10', 'valid_2024-01_2025-12']}
                    res[key] = o
                    es, ev_ = o['event_select_2005-01_2019-04'], o['event_valid_2019-05_2026-10']
                    ps, pv = o['port_select_2005-01_2019-04'], o['port_valid_2019-05_2026-10']
                    print(key, 'n', len(df), '| S ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'],
                          '| V ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], flush=True)
                    if key == PRIMARY:
                        df.to_csv(HERE / 'primary_events.csv.gz', index=False)
    out['results'] = res; out['yearly_events'] = yearly_ev
    pr = res[PRIMARY]
    vs = {'primary': PRIMARY, 'select': pr['pass']['select_2005-01_2019-04'], 'valid': pr['pass']['valid_2019-05_2026-10'],
          'valid_2024_2025_info': pr['pass']['valid_2024-01_2025-12']}
    k = vs['select']['pass'] + vs['valid']['pass']
    vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: [o['pass']['select_2005-01_2019-04']['pass'], o['pass']['valid_2019-05_2026-10']['pass']] for kk, o in res.items()}
    vs['n_variants_both'] = sum(all(v) for v in vs['variants_pass'].values())
    vs['n_variants_select_only'] = sum(v[0] and not v[1] for v in vs['variants_pass'].values())
    vs['n_variants_valid_only'] = sum(v[1] and not v[0] for v in vs['variants_pass'].values())
    vs['n_beatEW_t2_select'] = sum(o['pass']['select_2005-01_2019-04']['beat_EW_t2'] for o in res.values())
    vs['n_beatEW_t2_valid'] = sum(o['pass']['valid_2019-05_2026-10']['beat_EW_t2'] for o in res.values())
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8'))
    d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps({k: v for k, v in vs.items() if k != 'variants_pass'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
