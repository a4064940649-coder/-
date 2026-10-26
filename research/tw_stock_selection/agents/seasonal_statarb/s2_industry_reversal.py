#!/usr/bin/env python3
"""檢定 2：產業內均值回歸（只做多）。

事前固定規則：
  產業：stock_list.csv 的 industry（上市、現行分類；已下市股無產業 → 不在池內，見 FINDINGS 偏誤說明）。
  訊號（t−1 收盤資訊，t 收盤成交，持有一個月）：
    r1m  = 還原價 t−1 / t−22 − 1
    rel  = r1m − 同產業可投資股 r1m 等權平均（產業至少 5 檔可投資股）
    mom  = 還原價 t−22 / t−253 − 1（12−1 月動能）> 0 才可買
  版本：
    2a 流動性前 300 名，mom>0，rel 最低 30 檔
    2b 產業中立：流動性前 300 名，每個產業（池內 ≥3 檔且 mom>0）取 rel 最低 1 檔
    2c 全部可投資股，mom>0，rel 最低 30 檔
    2d 同 2a 但不加 mom>0 條件（檢查動能濾網的貢獻）
    2e 同 2a 但用 t 收盤訊號、t 收盤成交（不可交易的理論版，只作參考）
  五分位：前 300 名且 mom>0 內，依 −rel 分五組，Q5（相對跌最多）− Q1。
  成本 0.5%（壓力 1%）× 周轉；t 收盤鎖漲停不買。
  挑選期 2005-01～2019-04、驗證期 2019-05～2026-10。
"""
import json, warnings
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
warnings.filterwarnings('ignore')

adj, C, H, V = load('adj', 'close_raw', 'high', 'value')
ind = industries()
cols = [c for c in adj.columns if c in ind.index]
adj, C, H, V = adj[cols], C[cols], H[cols], V[cols]
me = month_ends(adj.index)
R = period_returns(adj, me)
B = bench_returns(e50_series(), me)
lock = locked_limit_up(C, H).reindex(me).fillna(False)
A = adj.ffill(limit=10)

def signals(lag):
    r1 = (A / A.shift(21) - 1).shift(lag).reindex(me)
    mom = (A.shift(21) / A.shift(252) - 1).shift(lag).reindex(me)
    el, v60 = eligibility(me, C, V, lag=lag)
    el &= r1.notna() & mom.notna()
    # 產業平均（全部可投資股）
    r1e = r1.where(el)
    g = ind.reindex(cols)
    indmean = r1e.T.groupby(g).transform('mean').T
    indcnt = r1e.notna().T.groupby(g).transform('sum').T
    rel = (r1 - indmean).where(el & (indcnt >= 5))
    return rel, mom, el & ~lock, v60

res = {}
periods = {'IS_2005-01_2019-04': (IS_START, IS_END), 'OOS_2019-05_2026-10': (OOS_START, OOS_END)}

def record(name, port, pool, extra=None):
    for pn, (a, b) in periods.items():
        o = {'turnover': round(port['turnover'], 3), 'n_names': round(port['n_names'], 1),
             'gross': ann(cut(port['gross'], a, b), me),
             'net_0.5pct': eval_monthly(port['net'], me, B, pool, a, b),
             'net_1pct': eval_monthly(port['net_1pct'], me, B, pool, a, b)}
        if extra: o.update(extra(a, b))
        res[f'{name}|{pn}'] = o
        print(f"{name:22s} {pn:20s} n={o['n_names']} to={o['turnover']:.2f} gross {o['gross'].get('CAGR')} | net.5 {fmt(o['net_0.5pct'])} | net1 {o['net_1pct']['strat'].get('CAGR')}")

for lag in (1, 0):
    rel, mom, el, v60 = signals(lag)
    U300 = top_liq(el, v60, 300)
    pool300 = pool_ew(R, U300); poolall = pool_ew(R, el)
    good = (mom > 0)
    def topk(mask, k=30):
        s = rel.where(mask & rel.notna())
        return {t: list(s.loc[t].dropna().nsmallest(k).index) for t in R.index if s.loc[t].count() >= k}
    if lag == 1:
        rr = run_sort(-rel.where(U300 & good), R, U300 & good, nq=5, top_n=30, min_names=50)
        def q_extra(a, b, rr=rr):
            sp = cut((rr['Q5'] - rr['Q1']).dropna(), a, b)
            return {'Q5_Q1_pct_m': round(float(sp.mean() * 100), 3), 't_Q5_Q1': tstat(sp),
                    'quintile_CAGR': {f'Q{k}': ann(cut(rr[f'Q{k}'], a, b), me).get('CAGR') for k in range(1, 6)}}
        record('2a_top300_mom+_worst30', pick_portfolio(topk(U300 & good), R), pool300, q_extra)
        # 2b 產業中立
        g = ind.reindex(cols); picks = {}
        s = rel.where(U300 & good)
        for t in R.index:
            row = s.loc[t].dropna()
            if len(row) < 30: continue
            cnt = U300.loc[t][U300.loc[t]].index.to_series().map(g).value_counts()
            okind = set(cnt[cnt >= 3].index)
            pk = row.groupby(g.reindex(row.index)).idxmin()
            picks[t] = [c for i, c in pk.items() if i in okind]
        record('2b_top300_ind_neutral', pick_portfolio(picks, R), pool300)
        record('2c_all_elig_mom+_worst30', pick_portfolio(topk(el & good), R), poolall)
        rr2 = run_sort(-rel.where(U300), R, U300, nq=5, top_n=30, min_names=50)
        def q2(a, b, rr=rr2):
            sp = cut((rr['Q5'] - rr['Q1']).dropna(), a, b); return {'Q5_Q1_pct_m': round(float(sp.mean() * 100), 3), 't_Q5_Q1': tstat(sp)}
        record('2d_top300_no_mom_filter', pick_portfolio(topk(U300), R), pool300, q2)
    else:
        record('2e_top300_mom+_lag0_ref', pick_portfolio(topk(U300 & good), R), pool300)

(OUTDIR / 'results_s2.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for k in ['2a_top300_mom+_worst30|OOS_2019-05_2026-10']:
    print(k, res[k]['net_0.5pct']['yearly'], res[k.replace('OOS_2019-05_2026-10', 'IS_2005-01_2019-04')].get('t_Q5_Q1'), res[k].get('t_Q5_Q1'))
