#!/usr/bin/env python3
"""S15 事後描述（看過 results.json 之後才跑；標「事後」，不影響判定）。
事件層級（扣成本前）進場後 1～60 日累積超額、正好持有 20 日的報酬、逐年、同日排序（RS 三分組）、訊號密度、每筆期望值與勝率、和 SURGE 交易重疊、C 對 C0 逐年。
python3 -I -B posthoc.py → results_posthoc.json"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s15lib
from s15lib import s4lib, stlib, cslib

H = Path(__file__).resolve().parent
t0 = time.time()
P = s15lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s15lib.pb_features(P, U)
sig0, rs = F['sig0'], F['rs']
del F
HMAX = 60
DAYS = [1, 2, 3, 5, 10, 15, 19, 20, 30, 40, 50, 60]
out = {'_note': '事後描述，不影響判定。事件層級為扣成本前（還原價、開盤進場），對 0050 用含息還原開盤／收盤。'}


def tick_cost(o):
    """嚴成本來回（近似）：買賣各半個跳動點＋0.3% 滑價、手續費 0.1425%×0.28 兩邊、賣出稅 0.3%（不含最低 20 元）。"""
    return 2 * (0.5 * cslib.tick(o) / o + 0.003) + 2 * 0.001425 * 0.28 + 0.003


def tstat(v):
    v = pd.Series(v).dropna()
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 else float('nan')


# ---------------------------------------------------------------- 事件層級
last = np.full(P.N, -10**6)
rows = []
for k in range(P.t0 - 1, P.T - 1):
    js = np.flatnonzero(sig0[k])
    if not len(js):
        continue
    rk = pd.Series(rs[k, js]).rank(ascending=False, pct=True).to_numpy()
    for j, q in zip(js, rk):
        indep = k - last[j] >= 20
        last[j] = k
        s = k + 1
        o = P.O[s, j]
        if not (np.isfinite(o) and o > 0) or o >= P.LIMUP[s, j] - 1e-9:
            continue
        a0 = P.AO[s, j]; b0 = P.E50O[s]
        ex = np.full(HMAX + 1, np.nan); cu = np.full(HMAX + 1, np.nan)
        for d in range(HMAX + 1):
            t = s + d
            if t >= P.T:
                break
            cu[d] = P.G[t, j] / a0 - 1; ex[d] = cu[d] - (P.E50[t] / b0 - 1)
        x = s + 20
        h20 = P.AO[x, j] / a0 - 1 if x < P.T else np.nan
        h20b = P.E50O[x] / b0 - 1 if x < P.T else np.nan
        x4 = s + 40
        h40 = P.AO[x4, j] / a0 - 1 if x4 < P.T else np.nan
        h40b = P.E50O[x4] / b0 - 1 if x4 < P.T else np.nan
        gap = o / P.C[k, j] - 1
        rows.append(dict(code=P.COLS[j], k=k, ent=P.IDX[s], year=int(P.IDX[s].year), indep=bool(indep), rs_pct=float(q), n_day=len(js), gap=gap,
                         h20=h20, h20_ex=h20 - h20b, h20_net_ex=h20 - h20b - tick_cost(o), h40_ex=h40 - h40b, cu=cu, ex=ex))
df = pd.DataFrame(rows)
EX = np.vstack(df['ex'].to_numpy()); CU = np.vstack(df['cu'].to_numpy())


def curve(mask):
    e = EX[mask]; c = CU[mask]
    return {int(d): dict(cum_mean=float(np.nanmean(c[:, d])), ex_mean=float(np.nanmean(e[:, d])), ex_median=float(np.nanmedian(e[:, d])),
                         ex_t=tstat(e[:, d]), n=int(np.isfinite(e[:, d]).sum())) for d in DAYS}


def summ(m):
    d = df[m]
    mm = d.groupby(d['ent'].dt.to_period('M'))['h20_net_ex'].mean()
    return dict(n=int(len(d)), h20_ex_mean=float(d.h20_ex.mean()), h20_ex_median=float(d.h20_ex.median()), h20_ex_t=tstat(d.h20_ex),
                h20_win_rate_ex=float((d.h20_ex > 0).mean()), h20_net_ex_mean=float(d.h20_net_ex.mean()), h20_net_ex_t_month=tstat(mm),
                h40_ex_mean=float(d.h40_ex.mean()), gap_mean=float(d.gap.mean()))


all_m = np.ones(len(df), bool); ind_m = df['indep'].to_numpy()
out['event_all'] = dict(summary=summ(all_m), curve=curve(all_m))
out['event_indep20'] = dict(summary=summ(ind_m), curve=curve(ind_m))
out['event_by_year_indep20'] = {int(y): summ(ind_m & (df['year'] == y).to_numpy()) for y in sorted(df['year'].unique())}
out['event_by_period_indep20'] = {'2015-07～2018': summ(ind_m & (df['year'] <= 2018).to_numpy()), '2019～2023': summ(ind_m & df['year'].between(2019, 2023).to_numpy()),
                                  '2024 起': summ(ind_m & (df['year'] >= 2024).to_numpy())}
terc = pd.cut(df['rs_pct'], [0, 1 / 3, 2 / 3, 1.0], labels=['RS 高（當日前 1/3）', 'RS 中', 'RS 低'])
out['event_by_rs_tercile_all（同日排序有沒有用）'] = {str(g): summ((terc == g).to_numpy()) for g in terc.cat.categories}
out['event_single_signal_days'] = summ((df['n_day'] == 1).to_numpy())

# ---------------------------------------------------------------- 主跑交易
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
bench = pd.Series(P.E50O, index=P.IDX)


def trade_block(x):
    x = x[x['why'] != 'open_end'].copy()
    b = bench.reindex(x['exit_date']).to_numpy() / bench.reindex(x['ent_date']).to_numpy() - 1
    x['ex'] = x['ret'] - b
    w = x['ret'] > 0
    yrs = x.groupby(x['ent_date'].dt.year)
    return dict(n=int(len(x)), win_rate=float(w.mean()), avg_win=float(x.loc[w, 'ret'].mean()), avg_loss=float(-x.loc[~w, 'ret'].mean()),
                expectancy=float(w.mean() * x.loc[w, 'ret'].mean() - (1 - w.mean()) * (-x.loc[~w, 'ret'].mean())),
                median_ret=float(x['ret'].median()), ex_mean=float(x['ex'].mean()), ex_t=tstat(x['ex']), ex_win_rate=float((x['ex'] > 0).mean()),
                big_loss_share_lt_minus15=float((x['ret'] < -0.15).mean()),
                by_year={int(y): dict(n=int(len(g)), mean_ret=float(g['ret'].mean()), ex_mean=float(g['ex'].mean()), win_rate=float((g['ret'] > 0).mean())) for y, g in yrs})


out['trades_after_cost'] = {c: trade_block(tr[tr['run'] == c]) for c in ('P', 'P40', 'E0')}
# 和 SURGE 交易重疊（主跑 P 對主跑 E0）
pe = tr[tr['run'] == 'P']; ee = tr[tr['run'] == 'E0']
ov = 0
for r in pe.itertuples():
    g = ee[ee['code'] == r.code]
    if len(g) and ((g['ent_date'] < r.exit_date) & (g['exit_date'] > r.ent_date)).any():
        ov += 1
top10 = ee[ee['why'] != 'open_end'].sort_values('pnl', ascending=False).head(10)
caught = []
for r in top10.itertuples():
    g = pe[pe['code'] == r.code]
    caught.append(dict(code=r.code, ent=str(r.ent_date.date()), ret=round(float(r.ret), 3),
                       P_trades_same_period=int(((g['ent_date'] < r.exit_date) & (g['exit_date'] > r.ent_date)).sum()),
                       P_trades_any_time=int(len(g))))
out['overlap'] = dict(P_trades=int(len(pe)), P_trades_overlap_E0_same_stock_period=int(ov), share=float(ov / len(pe)), surge_top10_vs_P=caught)
# 每檔最賺／最賠
pp = pe[pe['why'] != 'open_end']
out['P_pnl_by_year_sell'] = {int(y): float(v) for y, v in pp.groupby(pp['exit_date'].dt.year)['pnl'].sum().items()}

# ---------------------------------------------------------------- 訊號密度與名額
s = pd.Series(sig0[P.t0 - 1:].sum(axis=1), index=P.IDX[P.t0 - 1:])
mon = s.groupby(s.index.to_period('M')).sum()
out['density'] = dict(signals_per_month_median=float(mon.median()), signals_per_month_mean=float(mon.mean()),
                      per_year={int(y): int(v) for y, v in s.groupby(s.index.year).sum().items()})
npos = np.zeros(P.T)
for r in pe.itertuples():
    a = P.IDX.get_loc(r.ent_date); b = P.IDX.get_loc(r.exit_date)
    npos[a:b] += 1
npos = npos[P.t0:]
out['density']['P_main_days_full_10'] = float((npos >= 10).mean())
out['density']['P_main_avg_positions'] = float(npos.mean())

# ---------------------------------------------------------------- 逐年（主跑淨值）
nv = pd.read_csv(H / 'navs.csv.gz', index_col='date', parse_dates=True)
yr = nv.groupby(nv.index.year).last()
first = nv.iloc[0]
yr = pd.concat([first.to_frame().T, yr]).pct_change().iloc[1:]
yr.index = sorted(set(nv.index.year))
out['yearly_main'] = {int(y): {c: round(float(yr.loc[y, c]) * 100, 2) for c in yr.columns} for y in yr.index}
out['yearly_C_minus_C0_pp'] = {int(y): round(float(yr.loc[y, 'C'] - yr.loc[y, 'C0']) * 100, 2) for y in yr.index}
out['yearly_P_minus_0050_pp'] = {int(y): round(float(yr.loc[y, 'P'] - yr.loc[y, '0050']) * 100, 2) for y in yr.index}
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k not in ('event_all', 'event_indep20')}, ensure_ascii=False, indent=1, default=str)[:12000])
print(json.dumps(out['event_indep20'], ensure_ascii=False, indent=1)[:5000])
print(json.dumps(out['event_all']['summary'], ensure_ascii=False))
