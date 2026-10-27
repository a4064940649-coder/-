#!/usr/bin/env python3
"""投信首買／投信認養中小型股回測。規則與參數見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成本模型與 0050（只讀取）
from btlib import tstat  # noqa: E402

P = L.P; X = L.X
# ---------------- 價格（面板日曆） ----------------
names = ['adj', 'close_raw', 'open', 'value', 'flag', 'src', 'ret']
D = {k: pd.read_parquet(P / f'{k}.parquet').loc['2021-01-01':] for k in names}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if len(c) == 4 and c.isdigit() and not c.startswith('0') and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
PIDX = D['adj'].index; TP = len(PIDX); N = len(cols)
F = D['flag']; bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); V = D['value']
fac = ADJ / CR
AO_raw = OP * fac
ACf = ADJ.ffill()
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
LIQ = (V60 >= 1e7) & (CR >= 10) & (F.fillna(0).isin([0, 1, 2])) & ADJ.notna()

sh = pd.read_csv(X / 'shares_hist.csv', dtype={'code': str}).sort_values(['code', 'ym'])
SH = sh.groupby('code')['shares'].last().reindex(cols)                 # 股（2026-10 值，前視）
MCAP = CR * SH.to_numpy()[None, :]
SMALL_MCAP = LIQ & (MCAP < 150e8)
rk = V60.where(LIQ).rank(axis=1, pct=True)
SMALL_PROXY = LIQ & (rk <= 0.5)

lim = 0.10
AC_prev = ACf.shift(1).to_numpy()
GAP_UP = (AO_raw.to_numpy() / AC_prev - 1) >= lim - 0.006
AOn = AO_raw.to_numpy(); ACn = ACf.to_numpy()
PX = AO_raw.bfill().to_numpy(); PX = np.where(np.isfinite(PX), PX, ACn)

def ew_index(pool):
    oo = (AO_raw.shift(-1) / AO_raw - 1).where(pool.shift(1, fill_value=False)).clip(-0.5, 1.0)
    e = oo.mean(axis=1).fillna(0).to_numpy()
    return np.concatenate([[0.0], np.cumsum(np.log1p(e))])            # LEW[d] = log 指數於 d 日開盤
LEW = ew_index(LIQ); LEWS = ew_index(SMALL_MCAP)
ew_cc = D['ret'].where(LIQ.shift(1, fill_value=False)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
ews_cc = D['ret'].where(SMALL_MCAP.shift(1, fill_value=False)).clip(-0.5, 1.0).mean(axis=1).fillna(0)
e50 = L.e50.reindex(PIDX.union(L.e50.index)).ffill().reindex(PIDX)
r50 = e50.pct_change().fillna(0); E50n = e50.to_numpy()
SRC3 = ((PIDX >= pd.Timestamp('2024-02-15')) & (PIDX <= pd.Timestamp('2025-09-23')))

BUY, SELL = L.cost_rates(0.001)
BUY_S, SELL_S = L.cost_rates('stress')

# ---------------- 投信（inst 日曆） ----------------
ins = pd.read_csv(X / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'trust'], parse_dates=['date'])
ins = ins[ins['code'].isin(cols)]
TR = ins.pivot_table(index='date', columns='code', values='trust', aggfunc='sum').reindex(columns=cols).fillna(0.0)
IIDX = TR.index; TI = len(IIDX)
assert PIDX[(PIDX >= IIDX[0]) & (PIDX <= IIDX[-1])].difference(IIDX).empty
ppos = pd.Series(np.arange(TP), index=PIDX)
PP = ppos.reindex(IIDX).fillna(-1).astype(int).to_numpy()                      # inst 日 → 面板位置（-1＝面板缺）
NEXTP = np.searchsorted(PIDX.values, IIDX.values)                              # inst 日 → 第一個 >= 該日的面板位置
CENSOR_P = int(np.searchsorted(PIDX.values, IIDX[-1].to_datetime64(), side='right'))  # 2026-10-07 之後第一個面板日

def to_inst(df): return df.reindex(IIDX).fillna(False).astype(bool).to_numpy()
LIQ_I = to_inst(LIQ); SM_I = to_inst(SMALL_MCAP); SP_I = to_inst(SMALL_PROXY)
SHk = SH.to_numpy() / 1000.0                                                   # 張
buy = TR > 0; sel = TR < 0
nb60 = buy.astype(float).shift(3).rolling(60, min_periods=60).sum() == 0
streak3 = buy & buy.shift(1, fill_value=False) & buy.shift(2, fill_value=False)
cum3 = TR.rolling(3, min_periods=3).sum()
thr = np.where(np.isfinite(SHk), np.minimum(SHk * 0.001, 300.0), 300.0)
FB = (nb60 & streak3 & (cum3 >= thr[None, :])).to_numpy()
FB_PRI = (cum3 / SHk[None, :]).to_numpy()
SELL3 = (sel & sel.shift(1, fill_value=False) & sel.shift(2, fill_value=False)).to_numpy()
SELL5 = sel.astype(float).rolling(5, min_periods=5).min().fillna(0).to_numpy() == 1
RATIO = TR.cumsum() / SHk[None, :]
AD = ((RATIO > 0.03) & (RATIO.shift(1) <= 0.03) & (RATIO.shift(1).rolling(20, min_periods=20).min() < 0.01)).to_numpy()
RATIOn = RATIO.to_numpy()
SIG_START = int(np.searchsorted(IIDX.values, np.datetime64('2022-01-03')))

def build_events(S, POOL, mode, pri):
    ev = []; nblock = 0; nnoopen = 0; nsig = 0
    for j in range(N):
        ts = np.where(S[:, j] & POOL[:, j])[0]
        ts = ts[ts >= SIG_START]
        free_from = -1
        for t in ts:
            nsig += 1
            if t < free_from: continue
            e = t + 1
            if e >= TI: continue
            ep = PP[e]
            if ep < 0 or not np.isfinite(AOn[ep, j]): nnoopen += 1; continue
            if GAP_UP[ep, j]: nblock += 1; continue
            x = None; why = 'cap'
            if mode == 'FB':
                cap = 60
                for s in range(e + 2, min(e + cap, TI)):
                    if SELL3[s, j]: x = s + 1; why = 'sell3'; break
                if x is None: x = e + cap
            elif mode == 'HOLD60':
                x = e + 60
            else:  # AD
                cap = 250
                for s in range(e, min(e + cap, TI)):
                    if RATIOn[s, j] > 0.15: x = s + 1; why = 'r>15%'; break
                    if s >= e + 4 and SELL5[s, j]: x = s + 1; why = 'sell5'; break
                if x is None: x = e + cap
            if x >= TI: xp = CENSOR_P; why = 'censored'
            else: xp = int(NEXTP[x])
            if xp >= TP: xp = TP - 1; why = 'censored'
            tp = PP[t]
            ev.append(dict(j=j, t=t, tp=tp, e=e, ep=ep, x=x, xp=xp, why=why, pri=float(pri[t, j]) if np.isfinite(pri[t, j]) else 0.0))
            free_from = x
    return ev, dict(raw_signals=nsig, events=len(ev), blocked_gap_up=nblock, no_open=nnoopen)

def event_table(ev):
    rows = []
    for v in ev:
        j, tp, ep, xp = v['j'], v['tp'], v['ep'], v['xp']
        p0 = AOn[ep, j]; p1 = PX[xp, j]
        raw = p1 / p0 - 1
        net = (1 + raw) * (1 - SELL) / (1 + BUY) - 1
        net_s = (1 + raw) * (1 - SELL_S) / (1 + BUY_S) - 1
        ew = math.expm1(LEW[xp] - LEW[ep]); ews = math.expm1(LEWS[xp] - LEWS[ep])
        b50 = E50n[xp - 1] / E50n[tp] - 1
        rows.append((cols[j], IIDX[v['t']], PIDX[ep], PIDX[xp], xp - ep, v['why'], raw, net, net_s, ew, ews, b50, bool(SRC3[tp:xp + 1].any())))
    df = pd.DataFrame(rows, columns=['code', 'sig', 'entry', 'exit', 'hold', 'why', 'raw', 'net', 'net_s', 'ew', 'ews', 'b50', 'src3'])
    df['ab'] = df['net'] - df['ew']; df['ab_s'] = df['net_s'] - df['ew']; df['ab50'] = df['net'] - df['b50']
    df['ab_gross'] = df['raw'] - df['ew']; df['ab_small'] = df['net'] - df['ews']
    return df

WIN = {'select_2022_2023': ('2022-01-01', '2023-12-31'), 'valid_2024_2026': ('2024-01-01', '2026-10-31'),
       'valid_2024_2025': ('2024-01-01', '2025-12-31')}
r3 = lambda x: round(float(x), 3)

def ev_stats(df):
    if len(df) < 5: return {'n': int(len(df))}
    gm = df.groupby(df['entry'].dt.to_period('M'))
    return {'n': int(len(df)), 'n_months': int(gm.ngroups), 'avg_hold_days': round(float(df['hold'].mean()), 1),
            'exit_reason': df['why'].value_counts().to_dict(),
            'mean_raw_pct': r3(df['raw'].mean() * 100), 'mean_net_pct': r3(df['net'].mean() * 100), 'mean_EW_pct': r3(df['ew'].mean() * 100),
            'mean_ab_gross_pct': r3(df['ab_gross'].mean() * 100), 't_month_ab_gross': tstat(gm['ab_gross'].mean()),
            'mean_ab_net_pct': r3(df['ab'].mean() * 100), 'median_ab_net_pct': r3(df['ab'].median() * 100),
            't_month_ab_net': tstat(gm['ab'].mean()), 't_date_ab_net': tstat(df.groupby('entry')['ab'].mean()),
            'mean_ab_net_stress_pct': r3(df['ab_s'].mean() * 100), 't_month_ab_net_stress': tstat(gm['ab_s'].mean()),
            'mean_ab_vs_smallEW_pct': r3(df['ab_small'].mean() * 100), 't_month_ab_vs_smallEW': tstat(gm['ab_small'].mean()),
            'mean_net_vs_0050_pct': r3(df['ab50'].mean() * 100), 't_month_vs_0050': tstat(gm['ab50'].mean()),
            'hit_ab_net': r3((df['ab'] > 0).mean()), 'hit_raw': r3((df['raw'] > 0).mean()),
            'ab_net_trim1pct_mean_pct': r3(df['ab'].clip(df['ab'].quantile(0.01), df['ab'].quantile(0.99)).mean() * 100)}

def portfolio20(ev, buy, sell, slots=20):
    ent = {};
    for k, v in enumerate(ev): ent.setdefault(v['ep'], []).append(k)
    held = {}; cash = 1.0; nav = np.empty(TP); npos = np.zeros(TP); skipped = 0
    for d in range(TP):
        for k in [k for k, h in held.items() if ev[k]['xp'] == d]:
            u = held.pop(k); cash += u * PX[d, ev[k]['j']] * (1 - sell)
        if d in ent:
            pv = cash + sum(u * (AOn[d, ev[k]['j']] if np.isfinite(AOn[d, ev[k]['j']]) else ACn[d - 1, ev[k]['j']]) for k, u in held.items())
            for k in sorted(ent[d], key=lambda k: -ev[k]['pri']):
                if len(held) >= slots or cash <= 1e-9: skipped += 1; continue
                amt = min(pv / slots, cash); cash -= amt
                held[k] = amt / (1 + buy) / AOn[d, ev[k]['j']]
        nav[d] = cash + sum(u * ACn[d, ev[k]['j']] for k, u in held.items())
        npos[d] = len(held)
    return pd.Series(nav, index=PIDX), pd.Series(npos, index=PIDX), skipped

def pstats(nav, npos, a, b):
    r = nav.pct_change().fillna(0)
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))
    rr = r[m]; bb = r50[m]; ee = ew_cc[m]; es = ews_cc[m]
    yrs = (rr.index[-1] - rr.index[0]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod()
        return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}
    o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb)
    o['EW_CAGR'], o['EW_maxDD'] = st(ee); o['smallEW_CAGR'], o['smallEW_maxDD'] = st(es)
    o['vol'] = round(float(rr.std() * math.sqrt(252)), 4)
    o['avg_positions'] = round(float(npos[m].mean()), 1); o['avg_invested_share'] = round(float(npos[m].mean() / 20), 3)
    mo = lambda x: (1 + x).groupby(x.index.to_period('M')).prod() - 1
    ms, mb, me, mss = mo(rr), mo(bb), mo(ee), mo(es)
    o['n_months'] = int(len(ms))
    o['excess_vs_0050_pct_m'] = r3((ms - mb).mean() * 100); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = r3((ms - me).mean() * 100); o['t_vs_EW_m'] = tstat(ms - me)
    o['excess_vs_smallEW_pct_m'] = r3((ms - mss).mean() * 100); o['t_vs_smallEW_m'] = tstat(ms - mss)
    yr = lambda x: {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in x.groupby(x.index.year)}
    o['yearly_pct'] = yr(rr); o['yearly_0050_pct'] = yr(bb); o['yearly_EW_pct'] = yr(ee); o['yearly_smallEW_pct'] = yr(es)
    return o

PWIN = {'select_2022_2023': ('2022-01-03', '2023-12-29'), 'valid_2024_2026': ('2024-01-01', '2026-10-22'),
        'valid_2024_2025': ('2024-01-01', '2025-12-31')}

def main():
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'n_without_shares': int(SH.isna().sum()), 'cost_buy': BUY, 'cost_sell': SELL,
                    'stress': [BUY_S, SELL_S], 'inst_range': [str(IIDX[0].date()), str(IIDX[-1].date())],
                    'avg_pool_liq': round(float(LIQ.loc['2022':].sum(1).mean()), 1), 'avg_pool_small_mcap': round(float(SMALL_MCAP.loc['2022':].sum(1).mean()), 1),
                    'avg_pool_small_proxy': round(float(SMALL_PROXY.loc['2022':].sum(1).mean()), 1)}}
    specs = {'FB_MCAP': (FB, SM_I, 'FB', FB_PRI), 'FB_PROXY': (FB, SP_I, 'FB', FB_PRI), 'FB_HOLD60': (FB, SM_I, 'HOLD60', FB_PRI),
             'AD_MCAP': (AD, SM_I, 'AD', RATIOn), 'AD_PROXY': (AD, SP_I, 'AD', RATIOn)}
    res = {}; yearly_ev = {}
    for key, (S, POOL, mode, pri) in specs.items():
        ev, diag = build_events(S, POOL, mode, pri)
        df = event_table(ev)
        o = {'diag': diag}
        for w, (a, b) in WIN.items():
            o[f'event_{w}'] = ev_stats(df[(df['sig'] >= pd.Timestamp(a)) & (df['sig'] <= pd.Timestamp(b))])
        o['event_valid_ex_src3'] = ev_stats(df[(df['sig'] >= '2024-01-01') & ~df['src3']])
        o['event_valid_2026'] = ev_stats(df[df['sig'] >= '2026-01-01'])
        nav, npos, skp = portfolio20(ev, BUY, SELL); nav_s, _, _ = portfolio20(ev, BUY_S, SELL_S); nav_g, _, _ = portfolio20(ev, 0.0, 0.0)
        o['port_skipped_full'] = skp
        for w, (a, b) in PWIN.items():
            o[f'port_{w}'] = pstats(nav, npos, a, b)
            o[f'port_{w}']['CAGR_stress'] = pstats(nav_s, npos, a, b)['CAGR']; o[f'port_{w}']['CAGR_gross'] = pstats(nav_g, npos, a, b)['CAGR']
        yearly_ev[key] = {int(y): {'n': int(len(g)), 'ab_net_pct': round(g['ab'].mean() * 100, 2), 'raw_pct': round(g['raw'].mean() * 100, 2),
                                   't_month_ab_net': tstat(g.groupby(g['entry'].dt.to_period('M'))['ab'].mean())}
                          for y, g in df.groupby(df['sig'].dt.year)}
        res[key] = o
        df.to_csv(HERE / f'events_{key}.csv.gz', index=False)
        es, ev_ = o['event_select_2022_2023'], o['event_valid_2024_2026']
        ps, pv = o['port_select_2022_2023'], o['port_valid_2024_2026']
        print(key, diag, '| S n', es.get('n'), 'ab', es.get('mean_ab_net_pct'), 't', es.get('t_month_ab_net'), 'CAGR', ps['CAGR'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'],
              '| V n', ev_.get('n'), 'ab', ev_.get('mean_ab_net_pct'), 't', ev_.get('t_month_ab_net'), 'CAGR', pv['CAGR'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], flush=True)
    out['results'] = res; out['yearly_events'] = yearly_ev
    def passes(o, w):
        e = o[f'event_{w}']; p = o[f'port_{w}']
        if e.get('n', 0) < 30 or e.get('n_months', 0) < 12: return {'insufficient': True, 'pass': False}
        ok_ew = (e['mean_ab_net_pct'] > 0) and ((e.get('t_month_ab_net') or -9) >= 2)
        ok_50 = p['CAGR'] >= p['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['FB_MCAP']
    vs = {'select': passes(pr, 'select_2022_2023'), 'valid': passes(pr, 'valid_2024_2026')}
    k = vs['select']['pass'] + vs['valid']['pass']
    if vs['select'].get('insufficient') and vs['valid'].get('insufficient'): vs['verdict'] = '無法測試'
    else: vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: {w: passes(o, w) for w in ['select_2022_2023', 'valid_2024_2026']} for kk, o in res.items()}
    out['verdict'] = vs
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8'))
    d = {'PREREG': d['PREREG'], **out}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()
