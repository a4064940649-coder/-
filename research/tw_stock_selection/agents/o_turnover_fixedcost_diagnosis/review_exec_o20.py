#!/usr/bin/env python3
"""反方審查（成本與可成交，o20）：只讀研究員匯出的 data/*.csv.gz、results.json 與 V7X 資料層（x_data，唯讀）。
不改研究員檔案。python3 -I -B review_exec_o20.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sys.path.insert(0, str(AG / 'o_exec_engine_rewrite_v7x')); import x_data as D; sys.path.pop(0)
rng = np.random.default_rng(20)
X = pd.read_csv(HERE / 'data/swaps_exitlevel_100n_REAL_opt_seed0.csv.gz', dtype={'code': str, 'partner_A0': str})
X = X[np.isfinite(X.edge_B)].copy(); X['sig'] = pd.to_datetime(X.signal); X['gap'] = X.edge_B - X.rtc_B
F = pd.read_csv(HERE / 'data/tradelog_100n_REAL_opt_seed0.csv.gz', dtype={'code': str}); F['date'] = pd.to_datetime(F.date)
F3 = pd.read_csv(HERE / 'data/tradelog_30n_REAL_opt_seed0.csv.gz', dtype={'code': str}); F3['date'] = pd.to_datetime(F3.date)
RB = pd.read_csv(HERE / 'data/rebs_allcells_seed0.csv.gz')
TP = {d: i for i, d in enumerate(D.IDX)}
O = {}
def boot_mean_diff(d, mask_a, mask_b, col='gap', B=2000):
    d = d.reset_index(drop=True); a = mask_a.to_numpy(); b = mask_b.to_numpy(); v = d[col].to_numpy()
    m = d.sig.dt.to_period('M').astype(str).to_numpy(); ids = [np.flatnonzero(m == u) for u in np.unique(m)]
    obs = v[a].mean() - v[b].mean(); out = []
    for _ in range(B):
        ix = np.concatenate([ids[i] for i in rng.integers(0, len(ids), len(ids))])
        aa, bb = a[ix], b[ix]
        if aa.any() and bb.any(): out.append(v[ix][aa].mean() - v[ix][bb].mean())
    out = np.array(out); return dict(obs=float(obs), p_le0=float(np.mean(out <= 0)), ci90=[float(np.percentile(out, 5)), float(np.percentile(out, 95))])

for per, d in (('Y19on', X[X.sig >= '2019-01-01']), ('S0618', X[X.sig < '2019-01-01'])):
    o = {}
    # 1. 成本本身並不均勻：rtc 分布、平均 vs 中位數
    o['rtc'] = dict(mean=float(d.rtc_B.mean()), med=float(d.rtc_B.median()), p90=float(d.rtc_B.quantile(.9)), p99=float(d.rtc_B.quantile(.99)))
    o['edge'] = dict(mean=float(d.edge_B.mean()), med=float(d.edge_B.median()), sd=float(d.edge_B.std()))
    o['gap_mean'] = float(d.gap.mean())
    # 2. 依來回成本五分位：平均淨 edge（經濟上相關的量）
    d = d.assign(cq=pd.qcut(d.rtc_B, 5, labels=False))
    o['by_rtc_quintile'] = {int(k): dict(n=int(len(g)), rtc_mean=float(g.rtc_B.mean()), edge_mean=float(g.edge_B.mean()), edge_med=float(g.edge_B.median()),
                                         gap_mean=float(g.gap.mean()), frac_below=float((g.gap < 0).mean())) for k, g in d.groupby('cq')}
    o['boot_gap_q0_minus_q4'] = boot_mean_diff(d, d.cq == 0, d.cq == 4)
    # 3. 跳動點占比（tick/price）級距：價格級距不是跳動點成本的單調代理
    tkr = D.tick(d.price.to_numpy()) / d.price.to_numpy(); d = d.assign(tkr=tkr)
    d['tb'] = pd.cut(d.tkr, [0, 0.001, 0.002, 0.003, 0.005, 1], labels=['<0.1%', '0.1~0.2%', '0.2~0.3%', '0.3~0.5%', '>=0.5%'])
    o['by_tick_ratio'] = {str(k): dict(n=int(len(g)), rtc_mean=float(g.rtc_B.mean()), edge_mean=float(g.edge_B.mean()), gap_mean=float(g.gap.mean()), frac_below=float((g.gap < 0).mean()))
                          for k, g in d.groupby('tb', observed=True)}
    pb = np.select([d.price < 10, d.price < 20, d.price < 50, d.price < 100, d.price < 500], ['<10', '10~20', '20~50', '50~100', '100~500'], '>=500')
    o['by_price_fine'] = {k: dict(n=int(len(g)), tkr_med=float(g.tkr.median()), rtc_mean=float(g.rtc_B.mean()), gap_mean=float(g.gap.mean()), frac_below=float((g.gap < 0).mean()))
                          for k, g in d.assign(pb=pb).groupby('pb')}
    # 4. 排名級距的平均淨 edge（研究員只報中位數）
    o['by_rank_mean'] = {k: dict(n=int(len(g)), edge_mean=float(g.edge_B.mean()), gap_mean=float(g.gap.mean())) for k, g in d.groupby('rb')}
    # 5. 順延成交（漲跌停／上限）：edge 以 e 開盤計，與實際成交價不符
    o['deferred_exits'] = dict(n=int((d.deferred > 0).sum()), share=float((d.deferred > 0).mean()), gap_mean_def=float(d.gap[d.deferred > 0].mean()) if (d.deferred > 0).any() else None)
    O[per] = o

# 6. 成交層：處置中買進（需預收款、分盤撮合）、開盤即漲停被擋後順延、處置收盤賣出、零股占比
def fill_audit(Fx, tag):
    o = {}
    for per, f in (('Y19on', Fx[Fx.date >= '2019-01-01']), ('S0618', Fx[Fx.date < '2019-01-01'])):
        t = f.date.map(TP).to_numpy(); j = f.code.map(D.CI).to_numpy()
        disp = D.DISP[t, j]
        buy = f.side > 0
        nb = f.notional[buy].sum()
        lu = (np.isfinite(D.REF[t, j])) & (f.open.to_numpy() >= D.REF[t, j] * 1.085)   # 開盤漲幅 ≥ 8.5%（未鎖死但接近漲停）
        r = dict(n=int(len(f)), buy_under_disp_n=int((buy & disp).sum()), buy_under_disp_notional_share=float(f.notional[buy & disp].sum() / nb),
                 sell_under_disp_n=int(((~buy) & disp).sum()), kind_D_n=int((f.kind == 'D').sum()),
                 deferred_fill_share_n=float((f.ctx == 'P').mean()), deferred_fill_share_notional=float(f.notional[f.ctx == 'P'].sum() / f.notional.sum()),
                 odd_share_n=float((f.kind == 'O').mean()), odd_share_notional=float(f.notional[f.kind == 'O'].sum() / f.notional.sum()),
                 buy_open_ge_8p5pct_n=int((buy & lu).sum()), buy_open_ge_8p5pct_notional_share=float(f.notional[buy & lu].sum() / nb),
                 short_fill_share=float(((f.req > f.shares) & (f.ctx == 'R')).mean()),
                 fee_at_min20_share=float((f.fee <= 20.0001).mean()))
        # 處置中買進的實際衝擊率（引擎以一般開盤成交處理）
        if (buy & disp).any():
            g = f[buy & disp]; r['disp_buy_cost_rate_mean'] = float(((g.px - g.open) / g.open).mean())
            r['disp_buy_kinds'] = g.kind.value_counts().to_dict()
        o[per] = r
    return o
O['fills_100'] = fill_audit(F, '100'); O['fills_30'] = fill_audit(F3, '30')
# 盯盤需求：一般 R 換股日之外仍有成交的交易日（順延、處置收盤賣）占比
def days_needed(Fx):
    o = {}
    for per, f in (('Y19on', Fx[Fx.date >= '2019-01-01']),):
        g = f.groupby('j').apply(lambda z: pd.Series(dict(reb_days=z.date[z.ctx == 'R'].nunique(), extra_days=z.date[(z.ctx == 'P') & ~z.date.isin(z.date[z.ctx == 'R'])].nunique(),
                                                          close_sell_days=z.date[z.kind == 'D'].nunique())))
        o[per] = g.median().to_dict()
    return o
O['days_100'] = days_needed(F); O['days_30'] = days_needed(F3)

# 7. 兩平口徑：會計拖累（占期初資產）vs 年化報酬口徑；研究員報「會計口徑 8.8pp 打平」
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['RESULTS']
s0 = R['A_seed0_cells']['REAL_opt|normal|100萬']; dr = R['B_drag']['normal|100萬|Y19on']['REAL_opt']
O['breakeven_units'] = dict(acct_drag_pp=dr['cost'], turn_exit=dr['turn_exit'], turn_rw=dr['turn_rw_sell'], turn_entry=dr['turn_entry'], turn_rwbuy=dr['turn_rw_buy'],
                            seed0_net_cagr_y19=s0['CAGR_Y19on_med'], shapley_vals=R['I_shapley']['100萬|REAL_opt|CAGR_Y19on']['values'])
BE = pd.read_csv(HERE / 'data/breakeven_table.csv.gz')
q = BE[(BE.period.isin(['Y19on', 'POSTODD'])) & (BE.price == 'ALL') & (BE.variant == 'REAL_opt')]
O['breakeven_mean_vs_med'] = q[['cell', 'period', 'rtc_med', 'rtc_mean', 'edge_med', 'frac_below', 'gross_excess_needed_if_all_swaps_like_this_pp', 'annual_cost_drag_pp']].to_dict('records')
# 8. 換股日處置賣出（收盤）與其他成交日是否需要盤中操作：D 類筆數
print(json.dumps(O, ensure_ascii=False, indent=1, default=float))
