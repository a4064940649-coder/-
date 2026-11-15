#!/usr/bin/env python3
"""S14 事後描述（看過 results.json 之後才跑；標「事後」，不影響判定）。
事件層級（扣成本前）逐日累積報酬 0～60 日、逐年、上市與上櫃、事件密度、新制（2026-08-10 起）、單筆張數（預收款券門檻）、C 對 C0 逐年。
python3 -I -B posthoc.py → results_posthoc.json"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s14lib
from s14lib import s4lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
P = s14lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U_tw = stlib.universe(P, 'twse') & ~INNOV
U_all = stlib.universe(P, 'all') & ~INNOV
DT = s14lib.disp_table()
CH = s14lib.disp_chains(P, DT, 'first')
CH = CH[CH.k >= P.t0 - 1].reset_index(drop=True)
HMAX = 60
out = {}


def tick_cost(o):
    """嚴成本來回（近似）：買賣各半個跳動點＋0.3% 滑價、手續費 0.1425%×0.28 兩邊、賣出稅 0.3%（不含最低 20 元）。"""
    return 2 * (0.5 * cslib.tick(o) / o + 0.003) + 2 * 0.001425 * 0.28 + 0.003


def event_rows(ch, U):
    rows = []
    for r in ch.itertuples():
        s, e, j, k = r.s_row, r.e_row, r.j, r.k
        if k < 0 or s >= P.T or not U[k, j]:
            continue
        o = P.O[s, j]
        if not (np.isfinite(o) and o > 0) or o >= P.LIMUP[s, j] - 1e-9:
            continue
        a0 = P.AO[s, j]; b0 = P.E50O[s]
        cur = np.full(HMAX + 1, np.nan); ex = np.full(HMAX + 1, np.nan)
        for d in range(HMAX + 1):
            t = s + d
            if t >= P.T:
                break
            g = P.G[t, j]
            if np.isfinite(g):
                cur[d] = g / a0 - 1; ex[d] = cur[d] - (P.E50[t] / b0 - 1)
        x = e + 1
        hold = P.AO[x, j] / a0 - 1 if x < P.T and np.isfinite(P.AO[x, j]) else np.nan
        hold_b = P.E50O[x] / b0 - 1 if x < P.T else np.nan
        post20 = P.AO[min(x + 20, P.T - 1), j] / P.AO[x, j] - 1 if x + 20 < P.T and np.isfinite(P.AO[x, j]) else np.nan
        post20_b = P.E50O[x + 20] / P.E50O[x] - 1 if x + 20 < P.T else np.nan
        rows.append(dict(code=r.code, ent=P.IDX[s], year=int(pd.Timestamp(P.IDX[s]).year), hold_days=int(e - s + 1), n_ext=int(getattr(r, 'n_ext', 0)),
                         hold_ret=hold, hold_ex=hold - hold_b, net_ex=hold - hold_b - tick_cost(o), post20_ex=post20 - post20_b, cur=cur, ex=ex,
                         new_regime=bool(P.IDX[s] >= np.datetime64('2026-08-10'))))
    return rows


def tstat(v):
    v = pd.Series(v).dropna()
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 else float('nan')


def summarize(rows, label):
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    ym = df['ent'].dt.to_period('M')
    mm = df.groupby(ym)['net_ex'].mean()
    s = dict(n=int(len(df)), hold_ret_mean=float(df.hold_ret.mean()), hold_ex_mean=float(df.hold_ex.mean()), hold_ex_median=float(df.hold_ex.median()),
             net_ex_mean=float(df.net_ex.mean()), net_ex_t_month=tstat(mm), win_rate_net_ex=float((df.net_ex > 0).mean()), post20_ex_mean=float(df.post20_ex.mean()),
             post20_ex_t=tstat(df.post20_ex))
    return s


rows = event_rows(CH, U_tw)
df = pd.DataFrame(rows)
C = np.vstack(df['cur'].to_numpy()); X = np.vstack(df['ex'].to_numpy())
days = [0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 15, 20, 30, 40, 50, 60]
out['event_curve_first_twse'] = dict(n=int(len(df)), note='扣成本前；第 d 日＝進場日開盤到第 d 個交易日收盤；超額＝減 0050 同期（0050 從進場日開盤起算）',
                                     mean={d: float(np.nanmean(C[:, d])) for d in days}, median={d: float(np.nanmedian(C[:, d])) for d in days},
                                     excess_mean={d: float(np.nanmean(X[:, d])) for d in days}, excess_median={d: float(np.nanmedian(X[:, d])) for d in days},
                                     n_obs={d: int(np.isfinite(C[:, d]).sum()) for d in days})
out['event_curve_first_twse_by_period'] = {}
for nm, m in (('2015_2018', df.year <= 2018), ('2019_2023', (df.year >= 2019) & (df.year <= 2023)), ('2024_', df.year >= 2024)):
    out['event_curve_first_twse_by_period'][nm] = dict(n=int(m.sum()), excess_mean={d: float(np.nanmean(X[m.to_numpy(), d])) for d in days})
out['event_hold_first_twse'] = summarize(rows, 'all')
out['event_hold_by_year'] = {int(y): summarize([r for r in rows if r['year'] == y], str(y)) for y in sorted(df.year.unique())}
out['event_hold_new_regime'] = summarize([r for r in rows if r['new_regime']], 'new')
out['event_hold_old_regime_2026'] = summarize([r for r in rows if (not r['new_regime']) and r['year'] == 2026], 'old2026')
out['event_hold_extended_vs_not'] = dict(extended=summarize([r for r in rows if r['n_ext'] > 0], 'ext'), not_extended=summarize([r for r in rows if r['n_ext'] == 0], 'noext'))
dens = df.groupby(df['ent'].dt.to_period('M')).size()
allm = pd.period_range(P.IDX[P.t0], P.IDX[-1], freq='M')
dens = dens.reindex(allm, fill_value=0)
out['density_per_month'] = {int(y): dict(mean=float(g.mean()), max=int(g.max()), zero_months=int((g == 0).sum())) for y, g in dens.groupby(dens.index.year)}

# 上市 vs 上櫃（screener：disposal_5y＋warnings_hist 的處置列；第一次＋第二次都算；2021-10 起）
d5 = pd.read_csv(s14lib.EXTRA / 'disposal_5y.csv.gz', dtype={'code': str})
wh = pd.read_csv(s14lib.EXTRA / 'warnings_hist.csv.gz', dtype={'code': str}); wh = wh[wh.flag == '處置']
sc = pd.concat([d5, wh], ignore_index=True).drop_duplicates(['code', 'start', 'end'])
sc = sc.rename(columns={'date': 'announce'})
for c in ('announce', 'start', 'end'):
    sc[c] = pd.to_datetime(sc[c])
sc['kind'] = 'regular'; sc['nth'] = 1; sc['ordr'] = np.arange(len(sc))
CHs = s14lib.disp_chains(P, sc, 'all')
CHs = CHs[CHs.s_row >= int(np.searchsorted(P.IDX.values, np.datetime64('2021-10-01')))]
mk = np.array([P.MK[r.k, r.j] if r.k >= 0 else 0 for r in CHs.itertuples()])
r_tw = event_rows(CHs[mk == 1], U_all); r_tp = event_rows(CHs[mk == 2], U_all)
out['twse_vs_tpex_screener_2021_10_'] = dict(note='screener 名單（第一次＋第二次）、股池＝上市＋上櫃流動性（非創新板）、同一持有規則；扣成本前與近似嚴成本',
                                            twse=summarize(r_tw, 'tw'), tpex=summarize(r_tp, 'tp'),
                                            twse_by_year={y: summarize([r for r in r_tw if r['year'] == y], '') for y in range(2021, 2027)},
                                            tpex_by_year={y: summarize([r for r in r_tp if r['year'] == y], '') for y in range(2021, 2027)})

# 單筆張數（第一次處置：單筆 ≥ 10 張或當日累積 ≥ 30 張要預收款券）
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str})
d = tr[tr.run == 'D']
out['lots_D_main'] = dict(n=int(len(d)), ge10=int((d.shares >= 10000).sum()), ge30=int((d.shares >= 30000).sum()),
                          ge10_from=str(d[d.shares >= 10000].ent_date.min()), pnl_share_ge10=float(d[d.shares >= 10000].pnl.sum() / d.pnl.sum()),
                          pnl_share_ge30=float(d[d.shares >= 30000].pnl.sum() / d.pnl.sum()),
                          by_year_ge30={int(y): int(v) for y, v in d[d.shares >= 30000].groupby(d.ent_date.str[:4].astype(int)).size().items()})
out['D_main_pnl_by_year'] = {int(y): float(v) for y, v in d.groupby(d.ent_date.str[:4].astype(int)).pnl.sum().items()}
out['D_main_trade_ret_by_year'] = {int(y): dict(n=int(len(g)), mean=float(g.ret.mean()), median=float(g.ret.median())) for y, g in d.groupby(d.ent_date.str[:4].astype(int))}

# C 對 C0 逐年（主跑）
nv = pd.read_csv(H / 'navs.csv.gz', parse_dates=['date']).set_index('date')
ye = nv.groupby(nv.index.year).last(); ye.loc[2014] = nv.iloc[0]; ye = ye.sort_index()
yr = ye.pct_change().dropna()
out['yearly_main'] = {int(y): {c: float(yr.loc[y, c]) for c in ('0050', 'E0', 'D', 'DA', 'C', 'CA', 'C0')} for y in yr.index}
out['yearly_C_minus_C0_pp'] = {int(y): float((yr.loc[y, 'C'] - yr.loc[y, 'C0']) * 100) for y in yr.index}

# 晚進場、準時出場（事後；PREREG 的位移是進出場一起晚 s 天，這裡拆開看損失來自晚進還是晚出；100 萬，不影響判定）
from s14lib import s2lib, s10lib, COSTS
RT = s2lib.rev_tables(P)
ADV20 = pd.DataFrame(np.nan_to_num(P.AMT)).rolling(20, min_periods=20).mean().to_numpy()
CHf = s14lib.disp_chains(P, DT, 'first')
bench = pd.Series(P.E50[P.t0:], index=P.IDX[P.t0:])
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]; D24 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]; DE = P.IDX[-1]
simu = lambda S, cap: s10lib.simulate_s10(P, S, U_tw, sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap)[0]
late = {}
for sft in (1, 2, 3):
    c2 = CHf.copy(); c2['k'] += sft; c2['s_row'] += sft; c2 = c2[c2.s_row <= c2.e_row]
    SDl = s14lib.make_signal_disp(P, U_tw, c2, shift=0, adv20=ADV20, keep_info=False)
    SE = s14lib.surge_signal(P, U_tw, RT, shift=sft)
    nD = simu(SDl, 1e6); nE = simu(SE, 5e5); nDh = simu(SDl, 5e5)
    nC = nE + nDh; nC0 = nE + 5e5 * bench / bench.iloc[0]
    late[sft] = dict(D_since2019=stlib.cagr(nD, D19, DE), D_since2024=stlib.cagr(nD, D24, DE),
                     C_minus_C0_since2019_pp=(stlib.cagr(nC, D19, DE) - stlib.cagr(nC0, D19, DE)) * 100,
                     C_minus_C0_2019_2023_pp=(stlib.cagr(nC, D19, D24) - stlib.cagr(nC0, D19, D24)) * 100,
                     C_minus_C0_since2024_pp=(stlib.cagr(nC, D24, DE) - stlib.cagr(nC0, D24, DE)) * 100)
out['late_entry_on_time_exit_100w'] = late
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k not in ('event_hold_by_year',)}, ensure_ascii=False, indent=0, default=str)[:9000])
