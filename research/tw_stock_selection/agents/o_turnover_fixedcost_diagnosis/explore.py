#!/usr/bin/env python3
"""探索（只用挑選期：換股訊號日 ≤ 2019-04-30）。只描述，不定任何規則；結果 → explore.json。
python3 -I -B explore.py"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import diag_lib as DL; sys.path.pop(0)
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tfc')
CELLS = [('normal', '100萬'), ('stress', '100萬'), ('normal', '30萬'), ('stress', '30萬'), ('normal', '300萬'), ('normal', '1000萬')]
OUT = {'scope': '挑選期 2006-01～2019-04-30（訊號日）；seed 0；REAL_opt 為主、p90／carry 並列', 'cells': {}}
def sel(df):
    d = DL.cut(df, 'SEL'); assert pd.to_datetime(d.signal).max() <= DL.SEL_END; return d
for cell in CELLS:
    T = pickle.load(open(SCR / f'T_{cell[0]}_{cell[1]}.pkl', 'rb'))
    F = T['fills']; F = F[F.date <= DL.SEL_END]
    X = sel(T['exits']); N = sel(T['entries']); SL = sel(T['slots']); RB = sel(T['rebs'])
    o = {}
    for v in ('REAL_opt', 'REAL_p90', 'REAL_carry'):
        f = F[F.variant == v]; x = X[X.variant == v]; rb = RB[RB.variant == v]
        f = f.assign(rate=f.cost / f.notional)
        # (1) 逐筆成本分布
        dist = {}
        for kind in ('W', 'O', 'D'):
            ff = f[f.kind == kind]
            if len(ff): dist[kind] = dict(n=int(len(ff)), med=DL.med(ff.rate), p90=DL.q(ff.rate, 90), p99=DL.q(ff.rate, 99),
                                         tick_med=DL.med(ff.tk), share_notional=float(ff.notional.sum() / f.notional.sum()))
        comp = {c: float(f[c].sum() / f.cost.sum()) for c in ('tax', 'fee', 'tick_amt', 'imp_amt', 'odd_amt', 'stress_amt', 'fee_min_excess')}
        # 年成本拖累（pp／年）：Σ cost/avail，按年
        rb2 = rb.assign(y=pd.to_datetime(rb.date).dt.year)
        drag = {c: float((rb2[c] / rb2.avail).groupby([rb2.j, rb2.y]).sum().groupby(level=1).median().mean() * 100) for c in ('cost', 'tax', 'fee', 'tick_amt', 'imp_amt', 'odd_amt')}
        turn = {c: float((rb2[f'not_{c}'] / rb2.avail).groupby([rb2.j, rb2.y]).sum().groupby(level=1).median().mean()) for c in ('exit', 'rw_sell', 'entry', 'rw_buy')}
        # (2) edge vs 成本（三種配對）
        ec = {}
        for tag in ('B', 'A2', 'A1'):
            xx = x[np.isfinite(x[f'edge_{tag}'])]
            lt = xx[f'edge_{tag}'] < xx[f'rtc_{tag}']
            ec[tag] = dict(n=int(len(xx)), frac_edge_lt_cost=float(lt.mean()), edge_med=DL.med(xx[f'edge_{tag}']), rtc_med=DL.med(xx[f'rtc_{tag}']),
                           by_rb={b: dict(n=int(len(g)), frac=float((g[f'edge_{tag}'] < g[f'rtc_{tag}']).mean()), edge_med=DL.med(g[f'edge_{tag}']), rtc_med=DL.med(g[f'rtc_{tag}']))
                                  for b, g in xx.groupby('rb')})
        # 價格分類候選的機制一致性：Spearman(價格, 來回成本)
        xx = x[np.isfinite(x.edge_B)]
        sp = dict(exit_price=float(pd.Series(xx.price).corr(pd.Series(xx.rtc_B), method='spearman')),
                  min_price_A2=float(pd.Series(np.fmin(xx.price, xx.partner_price_A2)).corr(pd.Series(xx.rtc_A2), method='spearman')),
                  entry_mean_price=float(pd.Series(xx.entry_price_mean).corr(pd.Series(xx.rtc_B), method='spearman')))
        # 每筆成交價格級距：筆數占比 vs 成本占比
        pbt = f.groupby('pb').agg(n=('cost', 'size'), cost=('cost', 'sum'))
        pbt = {b: dict(n_share=float(r.n / pbt.n.sum()), cost_share=float(r.cost / pbt.cost.sum())) for b, r in pbt.iterrows()}
        # 換股檔數
        nx = rb.n_exits.describe().to_dict()
        # 反轉月份定義候選
        rb3 = rb.assign(rev_obs=np.sign(rb.m21) != np.sign(rb.m21_126), rev_post=np.sign(rb.m63) != np.sign(rb.r50_win))
        rv = {f: {nb: float(g[f].mean()) for nb, g in rb3.groupby(pd.cut(rb3.n_exits, [-1, 2, 4, 99], labels=['<=2', '3~4', '>=5']), observed=True)} for f in ('rev_obs', 'rev_post')}
        # 槽貢獻
        sl = SL[SL.variant == v]; band = sl[(sl.contrib >= -0.02) & (sl.contrib <= 0.02)]
        slot = dict(n=int(len(sl)), band_n=int(len(band)), band_contrib_sum=float(band.contrib.sum()), band_cost_sum=float(band.cost_pct.sum()),
                    all_contrib_sum=float(sl.contrib.sum()), all_cost_sum=float(sl.cost_pct.sum()), contrib_q=[DL.q(sl.contrib, p) for p in (5, 25, 50, 75, 95)])
        # 新進來源
        n_ = N[N.variant == v]
        src = {s: dict(n=int(len(g)), r_med=DL.med(g.r), buy_rate_med=DL.med(g.buy_rate)) for s, g in n_.groupby('src')}
        o[v] = dict(cost_dist=dist, cost_comp_share=comp, drag_pp_per_yr=drag, turnover_per_yr=turn, edge_vs_cost=ec, price_def_spearman=sp,
                    price_bucket_fill=pbt, n_exits_desc=nx, reversal_by_nswap=rv, slots=slot, entry_src=src)
    OUT['cells']['|'.join(cell)] = o
    print(cell, json.dumps({k: o['REAL_opt'][k] for k in ('drag_pp_per_yr', 'turnover_per_yr', 'price_def_spearman')}, ensure_ascii=False))
(HERE / 'explore.json').write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
