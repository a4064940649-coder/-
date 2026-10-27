#!/usr/bin/env python3
"""外資買超強度月選股（FI20 前 20 檔，M2）。規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
import exlib as L  # noqa: E402  成交引擎價格陣列與成本（只讀取）
from btlib import tstat  # noqa: E402

PRE = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['PREREG']
WIN = {k: tuple(v) for k, v in PRE['windows'].items()}
P = L.P; X = L.X
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'open', 'value', 'flag']}
good = D['adj'].notna().sum(axis=1) >= 300
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if len(c) == 4 and c.isdigit() and not c.startswith('0') and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; N = len(cols)
F = D['flag']; ADJ = D['adj']; CR = D['close_raw']; V = D['value']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
BASEU = (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()

# ---------- 外資買超強度 ----------
INS = pd.read_csv(X / 'inst_5y.csv.gz', dtype={'code': str}, usecols=['date', 'code', 'foreign'], parse_dates=['date'])
INST0 = INS['date'].min(); INST1 = INS['date'].max()
FOR = INS[INS['code'].isin(cols)].pivot_table(index='date', columns='code', values='foreign', aggfunc='sum')
FOR = FOR.reindex(index=IDX, columns=cols)
inrange = (IDX >= INST0) & (IDX <= INST1)
FOR = FOR.fillna(0.0).where(pd.Series(inrange, index=IDX), np.nan, axis=0) if False else FOR.fillna(0.0)
FOR.loc[~inrange] = np.nan
VOL = V / CR
valid = V.notna() & CR.notna() & (CR > 0) & FOR.notna()
FSH = (FOR * 1000).where(valid, 0.0); VSH = VOL.where(valid, 0.0)
POS_I0 = int(np.searchsorted(IDX, INST0))

def fi(n):
    sF = FSH.rolling(n, min_periods=1).sum(); sV = VSH.rolling(n, min_periods=1).sum(); c = valid.astype(float).rolling(n, min_periods=1).sum()
    out = (sF / sV).where((c >= 0.75 * n) & (sV > 0))
    full = np.arange(len(IDX)) >= POS_I0 + n - 1       # 整個窗都在 inst 期間內
    out.loc[~full] = np.nan
    return out
FI20 = fi(20); FI60 = fi(60)

# ---------- 處置、市值 ----------
dsp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
dsp = dsp[dsp['code'].isin(cols) & dsp['end'].notna()]
sh = pd.read_csv(X / 'shares_hist.csv', dtype={'code': str}).sort_values('ym').drop_duplicates('code').set_index('code')['shares']
SH = np.array([sh.get(c, np.nan) for c in cols], float)

_s = pd.Series(IDX, index=IDX)
ME = [d for d in _s.groupby([IDX.year, IDX.month]).last() if pd.Timestamp('2021-10-01') <= d <= INST1]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in ME}
LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[x] for x in cols])
GAP_UP_L = L.GAP_UP_OPEN
CI = {c: j for j, c in enumerate(cols)}

def disposal_mask(d):
    m = np.zeros(N, bool)
    s = dsp[(dsp['date'] <= d) & (dsp['end'] >= d)]
    for c in s['code']: m[CI[c]] = True
    return m

POOL, EXI, TOP50, DISP = {}, {}, {}, {}
for d in ME:
    ex = LPOS[NEXT[d]]
    dm = disposal_mask(d); DISP[d] = int((dm & BASEU.loc[d].to_numpy() & (V60.loc[d] >= 1e7).to_numpy()).sum())
    u = (BASEU.loc[d] & (V60.loc[d] >= 1e7)).to_numpy() & ~dm
    trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
    POOL[d] = u & trad; EXI[d] = ex
    mc = SH * CR.loc[d].to_numpy(); mc = np.where(np.isfinite(mc), mc, -1)
    t50 = np.zeros(N, bool); t50[np.argsort(-mc)[:50]] = True; TOP50[d] = t50

def build(spec):
    kind = spec['kind']; sig = FI60 if spec.get('sig') == 'FI60' else FI20
    events, pools, info = {}, {}, []
    held = set(); nblk = 0; ncand = 0; nskip = 0
    for d in ME:
        ex = EXI[d]; s = sig.loc[d].to_numpy()
        ok = POOL[d] & np.isfinite(s)
        if spec.get('both'):
            f60 = FI60.loc[d].to_numpy(); ok = ok & np.isfinite(f60)
        if spec.get('ex50'): ok = ok & ~TOP50[d]
        if not np.isfinite(s[POOL[d]]).any():
            continue                                   # 訊號尚未有效（暖身）
        if ok.sum() < 50:
            nskip += 1; continue
        poolidx = np.where(ok)[0]
        elig = ok.copy()
        if spec.get('both'): elig = elig & (s > 0) & (f60 > 0)
        blocked = GAP_UP_L[ex, LCOL]
        idx = np.where(elig)[0]
        order = idx[np.argsort(-s[idx], kind='stable')]
        if kind == 'topN' or kind == 'topQ':
            n = spec['n'] if kind == 'topN' else int(round(len(poolidx) / 10))
            pick = []; ncand += n
            for j in order:
                if len(pick) >= n: break
                if blocked[j] and LCOL[j] not in held: nblk += 1; continue
                pick.append(j)
        elif kind == 'dec':
            r = pd.Series(s[poolidx]).rank(method='first')
            g = pd.qcut(r, 10, labels=False).to_numpy()
            mem = poolidx[g == spec['g']]; ncand += len(mem)
            pick = [j for j in mem if not (blocked[j] and LCOL[j] not in held)]; nblk += len(mem) - len(pick)
        pick = np.array(pick, dtype=int)
        if len(pick) == 0:
            nskip += 1; continue
        events[ex] = ('open', LCOL[pick], np.full(len(pick), 1.0 / len(pick)))
        pools[ex] = ('open', LCOL[poolidx], np.full(len(poolidx), 1.0 / len(poolidx)))
        held = set(LCOL[pick])
        info.append((d, int(ok.sum()), len(pick), float(np.mean(s[pick])), float(np.median(s[poolidx])),
                     int(TOP50[d][pick].sum()), float(np.nanmedian(V60.loc[d].to_numpy()[pick])) / 1e6,
                     float(np.nanmedian(CR.loc[d].to_numpy()[pick])), DISP[d], ' '.join(cols[j] for j in pick)))
    inf = pd.DataFrame(info, columns=['d', 'pool', 'held', 'sig_mean_pick', 'sig_median_pool', 'n_top50', 'v60_med_m', 'price_med', 'n_disposal_excl', 'members']).set_index('d')
    return events, pools, inf, nblk, ncand, nskip

def sim(events, buy, sell, t0, t1):
    """exlib.sim_portfolio 同邏輯（同 w12_betting_against_beta），支援目標權重。"""
    u = np.zeros(L.N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk, wt = events[t]
            Pz = L.AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0)
            Vt = h.sum() + cash
            tgt = np.zeros(L.N); Vn = Vt
            for _ in range(6):
                tgt[:] = 0; tgt[pk] = wt * Vn
                cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - cost
            tov.append(np.clip(h - tgt, 0, None).sum() / Vt if Vt > 0 else 0)
            u = np.zeros(L.N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
    return pd.Series(nav, index=L.IDX[t0:t1 + 1]), (float(np.mean(tov[1:])) if len(tov) > 1 else float('nan'))

def mo(x): return (1 + x).groupby(x.index.to_period('M')).prod() - 1

def pstats(r, bench, ew, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
    if m.sum() < 40: return None
    rr = r[m]; bb = bench.reindex(rr.index).fillna(0); ee = ew.reindex(rr.index).fillna(0)
    i0 = int(np.argmax(m)); yrs = (rr.index[-1] - r.index[max(i0 - 1, 0)]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod(); return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {'start': str(rr.index[0].date()), 'end': str(rr.index[-1].date()), 'years': round(yrs, 2)}
    o['CAGR'], o['maxDD'] = st(rr); o['0050_CAGR'], o['0050_maxDD'] = st(bb); o['EW_CAGR'], o['EW_maxDD'] = st(ee)
    o['vol'] = round(float(rr.std() * math.sqrt(252)), 4)
    ms, mb, me = mo(rr), mo(bb), mo(ee)
    o['n_months'] = int(len(ms))
    o['excess_vs_0050_pct_m'] = round(float((ms - mb).mean() * 100), 3); o['t_vs_0050_m'] = tstat(ms - mb)
    o['excess_vs_EW_pct_m'] = round(float((ms - me).mean() * 100), 3); o['t_vs_EW_m'] = tstat(ms - me)
    o['hit_vs_EW_m'] = round(float((ms > me).mean()), 3)
    o['yearly_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    o['yearly_0050_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in bb.groupby(bb.index.year)}
    o['yearly_EW_pct'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in ee.groupby(ee.index.year)}
    return o

def mstats(x, a, b):
    xx = x[(x.index >= pd.Period(a, 'M')) & (x.index <= pd.Period(b, 'M'))].dropna()
    return {'mean_pct_m': round(float(xx.mean() * 100), 3), 't': tstat(xx), 'n_months': int(len(xx)),
            'hit': round(float((xx > 0).mean()), 3),
            'yearly_sum_pct': {int(y): round(float(g.sum() * 100), 1) for y, g in xx.groupby(xx.index.year)}}

def main():
    COST = {'base': L.cost_rates(0.001), 'stress': L.cost_rates('stress')}
    t1 = len(L.IDX) - 1
    b50 = L.E50.pct_change()
    out = {'meta': {'n_stocks': N, 'inst_range': [str(INST0.date()), str(INST1.date())], 'costs_buy_sell': COST,
                    'month_ends': [str(d.date()) for d in ME], 'last_day': str(L.IDX[-1].date())}}
    specs = {
        'FI20_TOP20': {'kind': 'topN', 'n': 20},
        'FI60_TOP20': {'kind': 'topN', 'n': 20, 'sig': 'FI60'},
        'FI20_D10': {'kind': 'topQ'},
        'FI20_POS_FI60_POS_TOP20': {'kind': 'topN', 'n': 20, 'both': True},
        'FI20_EX50_TOP20': {'kind': 'topN', 'n': 20, 'ex50': True},
    }
    for g in range(10): specs[f'DEC{g + 1}'] = {'kind': 'dec', 'g': g}
    res = {}; navs = {}; ewc = {}; mrets = {}
    for key, spec in specs.items():
        ev, pools, info, nb, nc, nskip = build(spec)
        t0 = min(ev)
        pk = (spec.get('sig', 'FI20'), bool(spec.get('both')), bool(spec.get('ex50')))
        if pk not in ewc:
            g_, _ = sim(pools, 0.0, 0.0, t0, t1); ewc[pk] = g_; navs['EW_' + '_'.join(map(str, pk))] = g_
        ewr = ewc[pk].pct_change()
        nav, tov = sim(ev, *COST['base'], t0, t1); nav_s, _ = sim(ev, *COST['stress'], t0, t1); nav_g, _ = sim(ev, 0.0, 0.0, t0, t1)
        navs[key] = nav; navs[key + '_gross'] = nav_g; navs[key + '_stress'] = nav_s
        mrets[key] = mo(nav.pct_change().dropna()); mrets[key + '_gross'] = mo(nav_g.pct_change().dropna())
        o = {'first_exec': str(L.IDX[t0].date()), 'n_rebalances': len(ev), 'blocked_gap_up': nb, 'cand_total': nc,
             'blocked_share': round(nb / nc, 4) if nc else None, 'months_skipped': nskip, 'turnover_value_per_rebalance': round(tov, 3)}
        for w, (a, b) in WIN.items():
            o[w] = pstats(nav.pct_change(), b50, ewr, a, b)
            if o[w] is None: continue
            ps_ = pstats(nav_s.pct_change(), b50, ewr, a, b); pg = pstats(nav_g.pct_change(), b50, ewr, a, b)
            o[w]['CAGR_stress'] = ps_['CAGR']; o[w]['t_stress_vs_EW_m'] = ps_['t_vs_EW_m']; o[w]['t_stress_vs_0050_m'] = ps_['t_vs_0050_m']
            o[w]['CAGR_gross'] = pg['CAGR']; o[w]['t_gross_vs_EW_m'] = pg['t_vs_EW_m']; o[w]['excess_gross_vs_EW_pct_m'] = pg['excess_vs_EW_pct_m']
            o[w]['t_gross_vs_0050_m'] = pg['t_vs_0050_m']
            sub = info[(info.index >= pd.Timestamp(a) - pd.Timedelta(days=31)) & (info.index <= pd.Timestamp(b))]
            o[w]['avg_pool'] = round(float(sub['pool'].mean()), 1); o[w]['avg_held'] = round(float(sub['held'].mean()), 1)
            o[w]['avg_sig_pick'] = round(float(sub['sig_mean_pick'].mean()), 4); o[w]['avg_sig_pool_median'] = round(float(sub['sig_median_pool'].mean()), 4)
            o[w]['avg_n_top50_in_pick'] = round(float(sub['n_top50'].mean()), 2)
            o[w]['holding_v60_median_NTD_m'] = round(float(sub['v60_med_m'].median()), 1)
            o[w]['holding_price_median'] = round(float(sub['price_med'].median()), 1)
        res[key] = o
        if key in ('FI20_TOP20',): info.to_csv(HERE / 'monthly_FI20_TOP20.csv')
        ps, pv, p2 = (o[w] for w in ('select', 'valid', 'valid_2024_2025'))
        print(f"{key:24s} first {o['first_exec']} tov {o['turnover_value_per_rebalance']} blk {o['blocked_share']} skip {nskip}",
              '| S', ps['CAGR'], 'g', ps['CAGR_gross'], 'st', ps['CAGR_stress'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 't50', ps['t_vs_0050_m'], 'DD', ps['maxDD'],
              '| V', pv['CAGR'], 'g', pv['CAGR_gross'], 'st', pv['CAGR_stress'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'], 't50', pv['t_vs_0050_m'], 'DD', pv['maxDD'],
              '| 24-25', p2['CAGR'], p2['0050_CAGR'], p2['EW_CAGR'], p2['t_vs_EW_m'], p2['t_vs_0050_m'], flush=True)
    out['portfolios'] = res
    # 十分位價差
    spread = {}
    for tag in ('gross', 'net'):
        suf = '_gross' if tag == 'gross' else ''
        x = (mrets['DEC10' + suf] - mrets['DEC1' + suf]).dropna()
        spread[tag] = {w: mstats(x, a, b) for w, (a, b) in WIN.items()}
    out['decile_spread_D10_minus_D1'] = spread
    out['decile_gross_mean_pct_m'] = {w: {f'D{g + 1}': round(float(mrets[f'DEC{g + 1}_gross'][(mrets[f'DEC{g + 1}_gross'].index >= pd.Period(a, 'M')) & (mrets[f'DEC{g + 1}_gross'].index <= pd.Period(b, 'M'))].mean() * 100), 3) for g in range(10)} for w, (a, b) in WIN.items()}
    print('spread', json.dumps(spread, ensure_ascii=False))
    print('deciles', json.dumps(out['decile_gross_mean_pct_m']))
    # 月度 rank IC（FI20 vs 換股開盤→下次換股開盤 報酬）
    ics = {}
    exs = [EXI[d] for d in ME]
    for k, d in enumerate(ME[:-1]):
        s = FI20.loc[d].to_numpy(); ok = POOL[d] & np.isfinite(s)
        if ok.sum() < 50: continue
        a0 = L.AO[exs[k], LCOL]; a1 = L.AO[exs[k + 1], LCOL]
        r = a1 / a0 - 1; ok = ok & np.isfinite(r)
        ics[str(d.date())] = float(pd.Series(s[ok]).rank().corr(pd.Series(r[ok]).rank()))
    ic = pd.Series(ics); ic.index = pd.PeriodIndex(pd.to_datetime(ic.index), freq='M')
    out['rank_IC_FI20'] = {w: mstats(ic, a, b) for w, (a, b) in WIN.items()}
    print('IC', json.dumps(out['rank_IC_FI20']))
    # 判定
    segs = {}
    for w in ('select', 'valid'):
        o = res['FI20_TOP20'][w]
        segs[w] = {'t_vs_EW_m': o['t_vs_EW_m'], 'excess_vs_EW_pct_m': o['excess_vs_EW_pct_m'], 'CAGR': o['CAGR'], '0050_CAGR': o['0050_CAGR'],
                   'EW_CAGR': o['EW_CAGR'],
                   'pass': bool(o['t_vs_EW_m'] >= 2 and o['excess_vs_EW_pct_m'] > 0 and o['CAGR'] >= o['0050_CAGR'])}
    npass = sum(v['pass'] for v in segs.values())
    out['verdict'] = {'segments': segs, 'grade': {2: '可用', 1: '待觀察', 0: '不可用'}[npass]}
    print('VERDICT', json.dumps(out['verdict'], ensure_ascii=False))
    pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
    f = HERE / 'results.json'; d0 = json.loads(f.read_text(encoding='utf-8'))
    d0 = {'PREREG': d0['PREREG'], **out}
    f.write_text(json.dumps(d0, ensure_ascii=False, indent=1), encoding='utf-8')

if __name__ == '__main__':
    main()
