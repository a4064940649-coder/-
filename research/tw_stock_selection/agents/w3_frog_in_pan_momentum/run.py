#!/usr/bin/env python3
"""溫水煮青蛙動能（FIP）：PRET 前 20% × ID 低半，3 個月 3 批重疊（M2 成交）＋變體 (a)(b)(c)(d)＋診斷＋動能家族直接比較。
規則見 results.json 的 PREREG（prereg.py，事前寫入）。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild'))
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import exlib as L  # noqa: E402  成交引擎價格陣列與成本（只讀取）
from btlib import tstat  # noqa: E402

P = L.P
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'value', 'flag']}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
for k in D: D[k] = D[k].loc[good]
cols = [c for c in D['adj'].columns if len(c) == 4 and c.isdigit() and not c.startswith('91')]
for k in D: D[k] = D[k][cols]
IDX = D['adj'].index; N = len(cols)
F = D['flag']; ADJ = D['adj']; CR = D['close_raw']; V = D['value']
v0 = V.fillna(0).where(V.notna().cumsum() > 0)
V60 = v0.rolling(60, min_periods=60).mean()
BASEU = (CR >= 10) & F.isin([0, 1, 2]) & ADJ.notna()

# ---- 訊號：PRET、%pos、%neg、ID（窗 = r 位置 i−251 … i−21） ----
R = ADJ / ADJ.shift(1) - 1
R = R.where(~F.isin([3, 4, 5]))
WIN_N, SKIP = 231, 21
val = R.notna().astype(float)
nv = val.rolling(WIN_N, min_periods=1).sum().shift(SKIP)
npos = (R > 0).astype(float).rolling(WIN_N, min_periods=1).sum().shift(SKIP)
nneg = (R < 0).astype(float).rolling(WIN_N, min_periods=1).sum().shift(SKIP)
slog = np.log1p(R.fillna(0)).rolling(WIN_N, min_periods=1).sum().shift(SKIP)
okw = nv >= 200
PRET = (np.exp(slog) - 1).where(okw)
PPOS = (npos / nv).where(okw); PNEG = (nneg / nv).where(okw)
ID = (np.sign(PRET) * (PNEG - PPOS)).where(okw)

_s = pd.Series(IDX, index=IDX)
ME = list(_s.groupby([IDX.year, IDX.month]).last())
ME = [d for d in ME if pd.Timestamp('2004-12-01') <= d <= pd.Timestamp('2026-09-30')]
NEXT = {d: IDX[IDX.get_loc(d) + 1] for d in ME}
LPOS = {d: i for i, d in enumerate(L.IDX)}
LCOL = np.array([L.CI[x] for x in cols])
thrL = np.where(L.IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None] - L.LTOL
GAP_UP_L = (L.AO_raw / L.AC_prev - 1) >= thrL
INDISP = L.in_disposal(pd.DatetimeIndex(ME)).reindex(columns=cols).fillna(False).astype(bool)
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
MA200_ON = (tx > tx.rolling(200, min_periods=200).mean()).reindex(pd.DatetimeIndex(ME), method='ffill')

def pool_at(d):
    ex = LPOS[NEXT[d]]
    u = (BASEU.loc[d] & (V60.loc[d] >= 1e7) & ~INDISP.loc[d]).to_numpy()
    trad = np.isfinite(L.OPEN_RAW[ex, LCOL]) & np.isfinite(L.ADJ_RAW[ex, LCOL])
    sig = np.isfinite(PRET.loc[d].to_numpy()) & np.isfinite(ID.loc[d].to_numpy())
    return ex, u & trad & sig

def orders(d, kind):
    """回傳 (ex, 偏好排序 [cleaned col idx], 目標檔數, 股池 idx)。"""
    ex, ok = pool_at(d)
    n = int(ok.sum())
    if n < 50: return None
    idx = np.where(ok)[0]
    pr = PRET.loc[d].to_numpy()[idx]; idv = ID.loc[d].to_numpy()[idx]
    byP = idx[np.argsort(-pr, kind='stable')]
    frac = 0.1 if kind == 'FIP10_LOW' else 0.2
    m = int(math.floor(frac * n)); M = byP[:m]
    if kind == 'MOM20':
        return ex, M, len(M), idx
    pM = PRET.loc[d].to_numpy()[M]; iM = ID.loc[d].to_numpy()[M]
    if kind.endswith('HIGH'):
        o = np.lexsort((-pM, -iM))          # ID 高→低，同 ID 依 PRET 高
    else:
        o = np.lexsort((-pM, iM))           # ID 低→高，同 ID 依 PRET 高
    return ex, M[o], int(math.ceil(m / 2)), idx

def schedule(kind, H, ma=False):
    """每個 tranche 的 events（LCOL 索引）。"""
    tr = [dict(ev={}, held=set(), cash=True) for _ in range(H)]
    nblk = 0; ncand = 0; info = []; m = -1
    for d in ME:
        o = orders(d, kind)
        if o is None: continue
        m += 1
        ex, order, nt, poolidx = o
        blocked = GAP_UP_L[ex, LCOL]
        on = bool(MA200_ON.loc[d]) if ma else True
        for k, T in enumerate(tr):
            if not on:
                if not T['cash']:
                    T['ev'][ex] = ('open', np.array([], dtype=int)); T['held'] = set(); T['cash'] = True
                continue
            if not (T['cash'] or m % H == k): continue
            pick = []
            for j in order:
                if len(pick) >= nt: break
                if blocked[j] and LCOL[j] not in T['held']: nblk += 1; continue
                pick.append(j)
            ncand += nt
            pk = LCOL[np.array(pick, dtype=int)]
            T['ev'][ex] = ('open', pk); T['held'] = set(pk); T['cash'] = False
        info.append((d, len(poolidx), nt, on, float(np.nanmean(ID.loc[d].to_numpy()[order[:nt]])), float(np.nanmedian(PRET.loc[d].to_numpy()[order[:nt]])),
                     float(np.nanmean(PPOS.loc[d].to_numpy()[order[:nt]])), float(np.nanmedian(V60.loc[d].to_numpy()[order[:nt]]))))
    inf = pd.DataFrame(info, columns=['d', 'pool', 'target_n', 'filter_on', 'ID_mean', 'PRET_median', 'pctpos_mean', 'v60_median']).set_index('d')
    return [T['ev'] for T in tr], nblk, ncand, inf

def sim(events, buy, sell, t0, t1, cap=1.0):
    u = np.zeros(L.N); cash = cap; nav = np.empty(t1 - t0 + 1); sold = []
    for t in range(t0, t1 + 1):
        if t in events:
            mode, pk = events[t]; Pz = L.AO[t]
            h = np.where(u > 0, u * np.nan_to_num(Pz), 0.0); Vt = h.sum() + cash
            if len(pk) == 0:
                cash = Vt - sell * h.sum(); sold.append(h.sum()); u = np.zeros(L.N)
            else:
                tgt = np.zeros(L.N); w = 1.0 / len(pk); Vn = Vt
                for _ in range(6):
                    tgt[:] = 0; tgt[pk] = w * Vn
                    cost = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                    Vn = Vt - cost
                sold.append(np.clip(h - tgt, 0, None).sum())
                u = np.zeros(L.N); u[pk] = tgt[pk] / Pz[pk]; cash = 0.0
        nav[t - t0] = np.nansum(u * L.AC[t]) + cash
    return pd.Series(nav, index=L.IDX[t0:t1 + 1]), float(np.sum(sold))

def sim_tranches(evs, buy, sell, t0, t1):
    H = len(evs); tot = None; sold = 0.0
    for ev in evs:
        nav, s = sim(ev, buy, sell, t0, t1, 1.0 / H); sold += s
        tot = nav if tot is None else tot + nav
    yrs = (L.IDX[t1] - L.IDX[t0]).days / 365.25
    return tot, round(sold / float(tot.mean()) / yrs, 2)      # 年化單邊周轉（賣出額 / 平均 NAV）

WIN = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
       'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'), 'valid_survivor_2019-05_2023-12': ('2019-05-01', '2023-12-31'),
       'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31')}
MAINW = list(WIN)[:3]

def mo(x): return (1 + x).groupby(x.index.to_period('M')).prod() - 1

def pstats(r, bench, ew, a, b):
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
    rr = r[m]; bb = bench.reindex(rr.index).fillna(0); ee = ew.reindex(rr.index).fillna(0)
    yrs = (rr.index[-1] - r.index[max(np.argmax(m) - 1, 0)]).days / 365.25
    def st(x):
        eq = (1 + x).cumprod(); return round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), round(float((eq / eq.cummax() - 1).min()), 4)
    o = {}
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

def pick_return_check(evs):
    """月換股名單：個股 d+1 開盤→下一換股開盤報酬，列極端值與持有期 flag 3/4/5。"""
    ev = evs[0]; ks = sorted(ev); rows = []
    Fl = pd.read_parquet(P / 'flag.parquet').reindex(index=L.IDX, columns=L.COLS).to_numpy()
    for a, b in zip(ks[:-1], ks[1:]):
        for j in ev[a][1]:
            r = L.AO[b, j] / L.AO[a, j] - 1
            fl = np.isin(Fl[a:b + 1, j], [3, 4, 5]).any()
            rows.append((str(L.IDX[a].date()), L.COLS[j], round(float(r), 4), bool(fl)))
    df = pd.DataFrame(rows, columns=['entry', 'code', 'ret', 'flag345'])
    return {'n': int(len(df)), 'n_flag345_in_hold': int(df['flag345'].sum()), 'ret_p1_p50_p99': [round(float(x), 4) for x in df['ret'].quantile([0.01, 0.5, 0.99])],
            'top10': df.nlargest(10, 'ret').to_dict('records'), 'bottom5': df.nsmallest(5, 'ret').to_dict('records'),
            'mean_ret_ex_flag345_minus_all_pct': round(float((df.loc[~df['flag345'], 'ret'].mean() - df['ret'].mean()) * 100), 3)}

def main():
    BUY, SELL = L.cost_rates(0.001); BUY_S, SELL_S = L.cost_rates('stress')
    out = {'meta': {'bad_days_dropped': BAD, 'n_stocks': N, 'cost_buy': BUY, 'cost_sell': SELL, 'stress': [BUY_S, SELL_S],
                    'last_rebalance_signal': str(ME[-1].date()), 'last_day': str(L.IDX[-1].date())}}
    first = next(d for d in ME if orders(d, 'FIP20_LOW') is not None)
    out['meta']['first_signal'] = str(first.date())
    t0 = LPOS[NEXT[first]]; t1 = len(L.IDX) - 1
    out['meta']['first_trade'] = str(L.IDX[t0].date())
    print('first signal', first.date(), 'trade', L.IDX[t0].date(), flush=True)
    # 訊號分布
    samp = [d for d in ME if d >= first][::12]
    allid = np.concatenate([ID.loc[d].to_numpy()[pool_at(d)[1]] for d in samp]); allp = np.concatenate([PRET.loc[d].to_numpy()[pool_at(d)[1]] for d in samp])
    out['signal_distribution_in_pool'] = {'ID_pct': {q: round(float(np.percentile(allid, q)), 3) for q in (5, 25, 50, 75, 95)},
                                          'PRET_pct': {q: round(float(np.percentile(allp, q)), 3) for q in (5, 25, 50, 75, 95)},
                                          'spearman_PRET_ID_pool_mean': round(float(np.mean([pd.Series(PRET.loc[d].to_numpy()[pool_at(d)[1]]).corr(pd.Series(ID.loc[d].to_numpy()[pool_at(d)[1]]), method='spearman') for d in samp])), 3)}
    b50 = L.E50.reindex(L.IDX[t0:t1 + 1]).pct_change()
    # EW（同池、月換股）
    pools = {}
    for d in ME:
        if d < first: continue
        ex, ok = pool_at(d)
        if ok.sum() >= 50: pools[ex] = ('open', LCOL[np.where(ok)[0]])
    ewg, _ = sim(pools, 0.0, 0.0, t0, t1); ewn, _ = sim(pools, BUY, SELL, t0, t1)
    navs = {'EW_gross': ewg, 'EW_net': ewn, '0050': L.E50.reindex(ewg.index)}
    ewr = ewg.pct_change()
    specs = {'FIP20_LOW_H3': ('FIP20_LOW', 3, False), 'FIP20_LOW_H1': ('FIP20_LOW', 1, False), 'FIP10_LOW_H3': ('FIP10_LOW', 3, False),
             'FIP20_HIGH_H3': ('FIP20_HIGH', 3, False), 'FIP20_LOW_H3_MA200': ('FIP20_LOW', 3, True), 'MOM20_H3_diag': ('MOM20', 3, False)}
    res = {}
    for key, (kind, H, ma) in specs.items():
        evs, nb, nc, info = schedule(kind, H, ma)
        if key == 'FIP20_LOW_H1': out['data_check_FIP20_LOW_H1_picks'] = pick_return_check(evs)
        nav, tov = sim_tranches(evs, BUY, SELL, t0, t1); nav_s, _ = sim_tranches(evs, BUY_S, SELL_S, t0, t1); nav_g, _ = sim_tranches(evs, 0.0, 0.0, t0, t1)
        navs[key] = nav; navs[key + '_gross'] = nav_g; navs[key + '_stress'] = nav_s
        r = nav.pct_change()
        o = {'blocked_gap_up': nb, 'target_slots_total': nc, 'blocked_share': round(nb / nc, 4) if nc else None,
             'annual_oneway_turnover': tov, 'avg_pool': round(float(info['pool'].mean()), 1), 'avg_target_n': round(float(info['target_n'].mean()), 1),
             'filter_on_share': round(float(info['filter_on'].mean()), 3)}
        for w, (a, b) in WIN.items():
            o[w] = pstats(r, b50, ewr, a, b)
            ps_ = pstats(nav_s.pct_change(), b50, ewr, a, b); pg = pstats(nav_g.pct_change(), b50, ewr, a, b)
            o[w]['CAGR_stress'] = ps_['CAGR']; o[w]['t_stress_vs_EW_m'] = ps_['t_vs_EW_m']; o[w]['t_stress_vs_0050_m'] = ps_['t_vs_0050_m']
            o[w]['CAGR_gross'] = pg['CAGR']; o[w]['t_gross_vs_EW_m'] = pg['t_vs_EW_m']
            o[w]['EW_net_CAGR'] = pstats(ewn.pct_change(), b50, ewr, a, b)['CAGR']
            sub = info[(info.index >= pd.Timestamp(a)) & (info.index <= pd.Timestamp(b))]
            if len(sub):
                o[w]['avg_pool'] = round(float(sub['pool'].mean()), 1); o[w]['avg_target_n'] = round(float(sub['target_n'].mean()), 1)
                o[w]['ID_mean_pick'] = round(float(sub['ID_mean'].mean()), 3); o[w]['PRET_median_pick'] = round(float(sub['PRET_median'].median()), 3)
                o[w]['pctpos_mean_pick'] = round(float(sub['pctpos_mean'].mean()), 3); o[w]['v60_median_pick_NTD_m'] = round(float(sub['v60_median'].median()) / 1e6, 1)
                o[w]['filter_on_share'] = round(float(sub['filter_on'].mean()), 3)
        res[key] = o
        ps, pv, p2 = (o[w] for w in MAINW)
        print(key, 'tov', tov, 'blk', o['blocked_share'], 'n', o['avg_target_n'],
              '| S', ps['CAGR'], 'g', ps['CAGR_gross'], 'st', ps['CAGR_stress'], '50', ps['0050_CAGR'], 'EW', ps['EW_CAGR'], 'tEW', ps['t_vs_EW_m'], 't50', ps['t_vs_0050_m'], 'DD', ps['maxDD'],
              '| V', pv['CAGR'], 'g', pv['CAGR_gross'], 'st', pv['CAGR_stress'], '50', pv['0050_CAGR'], 'EW', pv['EW_CAGR'], 'tEW', pv['t_vs_EW_m'], 't50', pv['t_vs_0050_m'], 'DD', pv['maxDD'],
              '| 24-25', p2['CAGR'], p2['0050_CAGR'], p2['EW_CAGR'], p2['t_vs_EW_m'], p2['t_vs_0050_m'], flush=True)
        if key == 'FIP20_LOW_H3': info.to_csv(HERE / 'monthly_info_FIP20_LOW.csv')
    out['portfolios'] = res
    # 診斷：LOW − HIGH、LOW − MOM20 毛價差（日曆月）
    sp = {}
    for nm, (x1, x2) in {'LOW_minus_HIGH_gross': ('FIP20_LOW_H3_gross', 'FIP20_HIGH_H3_gross'), 'LOW_minus_MOM20_gross': ('FIP20_LOW_H3_gross', 'MOM20_H3_diag_gross')}.items():
        sp[nm] = {}
        q1 = navs[x1].pct_change(); q2 = navs[x2].pct_change()
        for w, (a, b) in WIN.items():
            m = (q1.index >= pd.Timestamp(a)) & (q1.index <= pd.Timestamp(b))
            x = mo(q1[m].fillna(0)) - mo(q2[m].fillna(0))
            sp[nm][w] = {'pct_m': round(float(x.mean() * 100), 3), 't': tstat(x), 'n_months': int(len(x))}
    out['spreads'] = sp; print('SPREADS', json.dumps(sp), flush=True)
    # ---- 動能家族直接比較（mlib 股池／日曆／成本） ----
    import mlib as M
    Dm = list(M.LDATES)
    el_u, _, _ = M.universe(Dm)
    idm = pd.DataFrame(ID.reindex(pd.DatetimeIndex(Dm)).reindex(columns=M.COLS).to_numpy(), index=pd.DatetimeIndex(Dm), columns=M.COLS)
    P2m = M.at(M.P2, Dm)
    el = el_u & M.at(M.P1, Dm).notna() & P2m.notna() & M.at(M.P3, Dm).notna() & M.at(M.P6, Dm).notna() & idm.notna()
    sc = pd.DataFrame(np.nan, index=el.index, columns=M.COLS)
    for d in Dm:
        e = el.loc[d].to_numpy(); n = int(e.sum())
        if n < 50: continue
        p2 = P2m.loc[d].to_numpy(); iv = idm.loc[d].to_numpy()
        s = np.full(M.N, np.nan); ix = np.where(e)[0]
        s[ix] = pd.Series(p2[ix]).rank(pct=True).to_numpy()
        top = ix[np.argsort(-p2[ix], kind='stable')][:int(math.floor(0.2 * n))]
        o = np.lexsort((-p2[top], iv[top]))
        s[top[o]] = 10 + np.arange(len(top))[::-1] / len(top)
        sc.loc[d] = s
    evF, pools_m = M.select(sc, el, Dm, 30); nbF = M.select.nblk
    evP, _ = M.select(P2m, el, Dm, 30); nbP = M.select.nblk
    EWm, _ = M.sim(pools_m, 'gross', Dm); E50 = L.E50_NAV
    Wm = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
          'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31')}
    mf = {'note': 'mlib.universe（前 300 流動、處置排除）＋ P1/P2/P3/P6/ID 都有值的共同股池；mlib 日曆；M2；統計 mlib.stats（換股日窗 t，對共同股池毛等權）。',
          'avg_pool': round(float(el.sum(axis=1)[el.sum(axis=1) > 0].mean()), 1), 'blocked': {'FIP': nbF, 'P2': nbP}}
    exs = {}
    for nm, ev in (('FIP_MLIB_n30', evF), ('P2_MLIB_n30', evP)):
        mf[nm] = {}
        for cost in ('gross', 0.001, 'stress'):
            nav, tovm = M.sim(ev, cost, Dm)
            mf[nm][str(cost)] = {'turnover_per_rebalance': round(tovm, 3), 'windows': {k: M.stats(nav, a, b, Dm, E50, EWm) for k, (a, b) in Wm.items()}}
            if cost == 0.001: exs[nm] = M.month_rets(nav, Dm)
        navs[nm] = M.sim(ev, 0.001, Dm)[0].reindex(ewg.index)
    dx = exs['FIP_MLIB_n30'] - exs['P2_MLIB_n30']
    mf['FIP_minus_P2_net_pct_m'] = {k: {'pct_m': round(float(dx[(dx.index >= pd.Timestamp(a)) & (dx.index <= pd.Timestamp(b))].mean() * 100), 3),
                                        't': tstat(dx[(dx.index >= pd.Timestamp(a)) & (dx.index <= pd.Timestamp(b))])} for k, (a, b) in Wm.items()}
    try:
        ml = json.loads((HERE.parent / 'momentum_family' / 'results_long.json').read_text(encoding='utf-8'))
        mf['momentum_family_reported_P2_n30_slip0.1'] = {k: ml['P2_mom12_1']['n30_0.001']['windows'].get(k) for k in ('select_2005-01_2019-04', 'valid_2019-05_2026-10', 'valid_2024-01_2025-12')}
    except Exception as e:
        mf['momentum_family_reported_P2_n30_slip0.1'] = str(e)
    out['momentum_family_compare'] = mf
    for nm in ('FIP_MLIB_n30', 'P2_MLIB_n30'):
        print(nm, {k: (v['CAGR'], v['t_vs_EW_m'], v['t_vs_0050_m'], v['EW_CAGR'] if 'EW_CAGR' in v else None) for k, v in mf[nm]['0.001']['windows'].items() if v}, flush=True)
    print('FIP-P2', mf['FIP_minus_P2_net_pct_m'], flush=True)
    # ---- 判定 ----
    def passes(o):
        t = o['t_vs_EW_m'] if np.isfinite(o['t_vs_EW_m']) else -9
        ok_ew = (o['excess_vs_EW_pct_m'] > 0) and (t >= 2); ok_50 = o['CAGR'] >= o['0050_CAGR']
        return {'beat_EW_t2': bool(ok_ew), 'not_lose_0050': bool(ok_50), 'pass': bool(ok_ew and ok_50)}
    pr = res['FIP20_LOW_H3']
    vs = {'select': passes(pr['select_2005-01_2019-04']), 'valid': passes(pr['valid_2019-05_2026-10'])}
    k = vs['select']['pass'] + vs['valid']['pass']
    vs['verdict'] = '可用' if k == 2 else ('待觀察' if k == 1 else '不可用')
    vs['variants_pass'] = {kk: {w: passes(o[w]) for w in MAINW[:2]} for kk, o in res.items()}
    out['verdict'] = vs
    pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')); d.update(out)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
    print(json.dumps(vs, ensure_ascii=False))

if __name__ == '__main__':
    main()
