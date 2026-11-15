#!/usr/bin/env python3
"""S10 事後分析（標「事後」，不影響分類與判定）。讀 results.json、trades.csv.gz、navs.csv.gz，另載面板算「被賣後到下一個換股日的報酬」。
python3 -I -B posthoc.py → results_posthoc.json"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s10lib
from s10lib import s2lib, s4lib, stlib, EXITS

H = Path(__file__).resolve().parent
res = json.loads((H / 'results.json').read_text())
tr = pd.read_csv(H / 'trades.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
nv = pd.read_csv(H / 'navs.csv.gz', index_col=0, parse_dates=True)
CELLS = ['W10_70', 'WN_70']
out = {}

# 1. 逐年（主跑）
out['yearly'] = {c: {e: {y: round(v['strat'] * 100, 2) for y, v in res['combos'][f'{e}|{c}']['yearly'].items()} for e in EXITS} for c in CELLS}
out['yearly_0050'] = {y: round(v['e0050'] * 100, 2) for y, v in res['combos']['E0|W10_70']['yearly'].items()}
# 逐年：各出場減 E0（個百分點）與 E0 每年在 11 個裡的名次
yr = {}
for c in CELLS:
    Y = pd.DataFrame(out['yearly'][c])
    yr[c] = dict(minus_E0={e: (Y[e] - Y['E0']).round(2).to_dict() for e in EXITS[1:]},
                 E0_rank_by_year={y: int((Y.loc[y] > Y.loc[y, 'E0']).sum() + 1) for y in Y.index})
out['yearly_vs_E0'] = yr

# 2. 大贏家
BIG = ['2408', '3653', '1325', '2603']
bw = {}
for c in CELLS:
    e0 = tr[(tr['run'] == f'E0|{c}') & (tr['why'] != 'open_end')].sort_values('pnl', ascending=False)
    top = list(dict.fromkeys(BIG + e0['code'].head(5).tolist()))
    d = {}
    for code in top:
        d[code] = {}
        for e in EXITS:
            x = tr[(tr['run'] == f'{e}|{c}') & (tr['code'] == code)]
            d[code][e] = dict(n=int(len(x)), pnl_sum=round(float(x['pnl'].sum()), 0),
                              trades=[f"{r.ent_date.date()}→{r.exit_date.date()} {r.ret*100:+.1f}% {r.why}" for r in x.itertuples()])
    bw[c] = d
out['big_winners'] = bw

# 3. 持有天數、周轉、成本、出場原因、期望值、平均個股比重
st = {}
for c in CELLS:
    st[c] = {}
    for e in EXITS:
        R = res['combos'][f'{e}|{c}']; t = R['trades_full']
        st[c][e] = dict(n=t['n'], avg_hold=round(t['avg_hold_days'], 1), median_hold=t['median_hold_days'], expectancy_pct=round(t['expectancy'] * 100, 2),
                        win_rate=round(t['win_rate'] * 100, 1), avg_win=round(t['avg_win'] * 100, 2), avg_loss=round(t['avg_loss'] * 100, 2), why=t['why'],
                        turnover=round(R['turnover_oneway_ann'], 2), cost_pct_nav=round(R['cost_ann_pct_of_nav'] * 100, 2),
                        stock_cost_pct_nav=round(R['stock_cost_ann_pct_of_nav'] * 100, 2), avg_stockfrac=round(R['avg_stockfrac'] * 100, 1), avg_npos=round(R['avg_npos'], 2))
out['trade_stats'] = st

# 4. 最大回撤高低點
out['maxdd'] = {c: {e: dict(main=res['combos'][f'{e}|{c}']['dd_dates'], paths_2024_07_2025_04=res['combos'][f'{e}|{c}']['paths20']['n_dd_2024_07_to_2025_04'],
                            other_paths=sorted(set(tuple(v) for v in res['combos'][f'{e}|{c}']['paths_dd_dates'].values()
                                                   if not (v[0][:7] in ('2024-06', '2024-07', '2024-08') and v[1][:7] == '2025-04'))))
                    for e in EXITS} for c in CELLS}
# 2024-07 → 2025-04 這一段在主跑跌多少（高點到 2025-04-09）
seg = {}
for c in CELLS:
    seg[c] = {}
    for e in EXITS:
        s = nv[f'{e}|{c}'].loc['2024-06-01':'2025-04-30']
        seg[c][e] = round(float((s / s.cummax() - 1).min()) * 100, 2)
s = nv['0050'].loc['2024-06-01':'2025-04-30']; seg['0050'] = round(float((s / s.cummax() - 1).min()) * 100, 2)
out['dd_2024_2025_segment_main'] = seg

# 5. 最小買進門檻邊緣
out['minfrac_edge'] = {c: {e: dict(bought_50_55=res['combos'][f'{e}|{c}']['minfrac_edge']['bought_50_55'],
                                   skipped_45_50=res['combos'][f'{e}|{c}']['minfrac_edge']['skipped_45_50']) for e in EXITS} for c in CELLS}
out['minfrac_verdicts'] = {k: dict(base=(v['verdict_ret'], v['npos'], round(v['median_pp'], 2)),
                                   **{m: (x['verdict_ret'], x['npos'], round(x['median_pp'], 2)) for m, x in v['minfrac'].items()}) for k, v in res['pairs'].items()}

# 6. 被賣後到下一個換股日（下一個訊號日的次日開盤）的報酬：還原開盤 → 還原開盤；同期 0050
t0 = time.time()
P = s10lib.load()
U = stlib.universe(P, 'twse') & ~s4lib.innov_panel(P)[0]
RT = s2lib.rev_tables(P)
S = s10lib.make_signal_s10(P, U, RT, shift=0)
exec_rows = np.array(sorted(x['k'] + 1 for x in S['info']))
di = {d: i for i, d in enumerate(P.IDX)}
AO = np.where(np.isfinite(P.O) & (P.O > 0), P.O * P.G / P.CF, np.nan)
after = {}
for c in CELLS:
    after[c] = {}
    for e in ('TS15', 'TS25', 'TS15M', 'DMA60', 'MA20', 'R10', 'R15', 'RQ6'):
        x = tr[(tr['run'] == f'{e}|{c}') & (tr['why'].isin(['trail', 'ma_daily', 'requal', 'rule']))]
        rows = []
        for r in x.itertuples():
            j = P.ci[r.code]; xt = di[r.exit_date]
            nx = exec_rows[exec_rows > xt]
            if not len(nx) or not np.isfinite(AO[xt, j]):
                continue
            n1 = int(nx[0])
            a = AO[n1, j] if np.isfinite(AO[n1, j]) else P.G[n1 - 1, j]
            ret = a / AO[xt, j] - 1
            r50 = P.E50O[n1] / P.E50O[xt] - 1
            rows.append((r.why, ret, ret - r50, n1 - xt))
        if not rows:
            continue
        d = pd.DataFrame(rows, columns=['why', 'ret', 'ex', 'days'])
        after[c][e] = {w: dict(n=int(len(g)), mean_ret_pct=round(g['ret'].mean() * 100, 2), median_ret_pct=round(g['ret'].median() * 100, 2),
                               share_up=round(float((g['ret'] > 0).mean()) * 100, 1), mean_excess_0050_pct=round(g['ex'].mean() * 100, 2), mean_days=round(g['days'].mean(), 1))
                       for w, g in d.groupby('why')}
out['after_exit_to_next_rebalance'] = after

# 7. E0 的出場拆成「只因跌破季線」（MANO 會續抱）與「名次掉出前 20」；被賣後 1、2 個換股日的報酬
sig_rows = np.array(sorted(x['k'] for x in S['info']))
XE, XM, X2 = S['XV']['E0'], S['XV']['MANO'], S['XV']['MA20']
mech = {}
for c in CELLS:
    x = tr[(tr['run'] == f'E0|{c}') & (tr['why'] == 'rule')]
    rows = []
    for r in x.itertuples():
        j = P.ci[r.code]; xt = di[r.exit_date]
        fr = sig_rows[sig_rows < xt]
        if not len(fr):
            continue
        k = int(fr[-1])
        kind = 'MA60_only' if (XE[k, j] and not XM[k, j]) else 'rank'
        nx = exec_rows[exec_rows > xt]
        if len(nx) < 2 or not np.isfinite(AO[xt, j]):
            continue
        r1 = []
        for n1 in nx[:2]:
            a = AO[n1, j] if np.isfinite(AO[n1, j]) else P.G[n1 - 1, j]
            r1.append(a / AO[xt, j] - 1 - (P.E50O[n1] / P.E50O[xt] - 1))
        rows.append((kind, r1[0], r1[1], bool(P.F_MA20[k, j])))
    d = pd.DataFrame(rows, columns=['kind', 'ex1', 'ex2', 'above_ma20'])
    mech[c] = {kd: dict(n=int(len(g)), excess_0050_next_rebal_mean_pct=round(g['ex1'].mean() * 100, 2), median_pct=round(g['ex1'].median() * 100, 2),
                        excess_0050_two_rebal_mean_pct=round(g['ex2'].mean() * 100, 2), share_above_ma20=round(g['above_ma20'].mean() * 100, 1))
               for kd, g in d.groupby('kind')}
out['E0_exit_mechanism'] = mech

# 8. MANO、MA20 對 E0：主跑各窗、成本加倍、禁買 5 筆的差（個百分點）；依股票加總損益差最大的幾檔
cmp = {}
for c in CELLS:
    cmp[c] = {}
    E = res['combos'][f'E0|{c}']
    for e in ('MANO', 'MA20', 'TS15M', 'RQ6', 'TS15', 'DMA60', 'R10'):
        V = res['combos'][f'{e}|{c}']
        dw = {k: round((V['windows'][k]['cagr'] - E['windows'][k]['cagr']) * 100, 2) for k in V['windows']}
        dx2 = {k: round((V['costx2'][k]['cagr'] - E['costx2'][k]['cagr']) * 100, 2) for k in V['costx2']}
        db5 = {k: round((V['block5'][k]['cagr'] - E['block5'][k]['cagr']) * 100, 2) for k in V['block5']}
        pe = tr[tr['run'] == f'E0|{c}'].groupby('code')['pnl'].sum(); pv = tr[tr['run'] == f'{e}|{c}'].groupby('code')['pnl'].sum()
        dd = pv.sub(pe, fill_value=0).sort_values()
        cmp[c][e] = dict(windows=dw, costx2=dx2, block5=db5, top_plus={k: round(v) for k, v in dd.tail(5)[::-1].items()}, top_minus={k: round(v) for k, v in dd.head(5).items()},
                         total_pnl_diff=round(float(dd.sum())), top3_share_of_diff=round(float(dd.tail(3).sum() / dd.sum()), 3) if dd.sum() != 0 else None)
out['vs_E0_detail'] = cmp

# 9. 2024-06 以前的最大回撤（排除 2024-07 → 2025-04 那一段）
pre = {}
for c in CELLS:
    pre[c] = {e: round(float((lambda s: (s / s.cummax() - 1).min())(nv[f'{e}|{c}'].loc[:'2024-05-31'])) * 100, 2) for e in EXITS}
s50 = nv['0050'].loc[:'2024-05-31']; pre['0050'] = round(float((s50 / s50.cummax() - 1).min()) * 100, 2)
out['maxdd_before_2024_06'] = pre
# 10. 敏感度（事後）：(a) 整段不准買 2408（兩邊一起）；(b) 只看 2019-01 → 2023-12（不含 2024 起）——20 條路徑逐條配對對 E0
from s10lib import COSTS
CELLD = {'W10_70': dict(sat_w=0.7, sizing='W10', lock=False), 'WN_70': dict(sat_w=0.7, sizing='WN', lock=False)}
VS = ['E0', 'MANO', 'MA20', 'TS15M', 'RQ6', 'TS15']
ban = {P.ci['2408']: [(0, P.T - 1)]}
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]; D23 = P.IDX[P.IDX < pd.Timestamp('2024-01-01')][-1]
acc = {c: {e: dict(ban2408={}, to2023={}) for e in VS} for c in CELLS}
for sh in range(4):
    Sx = S if sh == 0 else s10lib.make_signal_s10(P, U, RT, shift=sh)
    for c in CELLS:
        for e in VS:
            S2, kw = s10lib.variant_S(Sx, e)
            for cap in (6e5, 8e5, 1e6, 1.5e6, 2e6):
                key = f'{int(cap/1e4)}萬|+{sh}'
                n1 = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELLD[c], **kw)[0]
                acc[c][e]['to2023'][key] = stlib.cagr(n1, D19, D23)
                n2 = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, block=ban, **CELLD[c], **kw)[0]
                acc[c][e]['ban2408'][key] = stlib.cagr(n2, D19, P.IDX[-1])
    if sh:
        del Sx
sens = {}
for c in CELLS:
    sens[c] = {}
    for e in VS[1:]:
        sens[c][e] = {}
        for kk in ('ban2408', 'to2023'):
            d = np.array([acc[c][e][kk][k] - acc[c]['E0'][kk][k] for k in acc[c]['E0'][kk]])
            sens[c][e][kk] = dict(npos=int((d > 0).sum()), median_pp=round(float(np.median(d)) * 100, 2), min_pp=round(float(d.min()) * 100, 2), max_pp=round(float(d.max()) * 100, 2),
                                  E0_median=round(float(np.median(list(acc[c]['E0'][kk].values()))) * 100, 2), var_median=round(float(np.median(list(acc[c][e][kk].values()))) * 100, 2))
out['sensitivity_ban2408_and_to2023'] = sens
print(json.dumps(sens, ensure_ascii=False))
out['_note'] = '全部是事後描述，不影響事前判定。after_exit：出場日還原開盤 → 下一個訊號日次日還原開盤（可以再買回的最早時點）。'
out['_meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out['after_exit_to_next_rebalance'], ensure_ascii=False))
print(out['_meta'])
