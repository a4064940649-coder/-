# -*- coding: utf-8 -*-
"""審查 A：空位放 0050 的引擎是否偷到盤中／隔夜、79/81 為正是不是機械結果。python3 -I -B review_a_idle.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
D = L.Data()
sig = L.signals(D)
out = {}
intr = pd.read_csv(L.S + 'rep_vcp/etf_intraday.csv', index_col=0, parse_dates=True)['intr'].reindex(D.dates).values
out['intraday_missing_days'] = int(np.isnan(intr[np.searchsorted(D.dates.values, np.datetime64('2005-02-01')):]).sum())
etf = pd.Series(D.ETF, index=D.dates)
# 1. 口徑修正
for nm, s in (('default', sig), ('SegLen20', L.signals(D, dict(SegLen=20))), ('VolDry0.8', L.signals(D, dict(VolDryRatio=0.8)))):
    for mode in ('spec', 'intraday', 'conservative'):
        r = L.run_portfolio(D, s, 'E3', 'etf', idle_mode=mode, etf_intraday=intr)
        p = L.perf(D, r)
        out['%s_%s' % (nm, mode)] = dict(full_ex=p['full']['ex'], full_t=p['full']['t'], y19_ex=p['2019']['ex'], y19_t=p['2019']['t'])
        print(nm, mode, {k: round(v, 4) for k, v in out['%s_%s' % (nm, mode)].items()})
# 買進日／開盤賣出日的 0050 盤中報酬
r = L.run_portfolio(D, sig, 'E3', 'etf')
tr = r['trades']
di = {d: i for i, d in enumerate(D.dates)}
b = np.array([intr[di[d]] for d in tr.entry]); so = np.array([intr[di[d]] for d in tr.exit[tr.reason.isin(['ma50', 'gap_stop'])]])
out['etf_intraday_on_buy_days'] = dict(n=int(len(b)), mean=float(np.nanmean(b)), t=float(np.nanmean(b) / (np.nanstd(b, ddof=1) / np.sqrt(np.isfinite(b).sum()))))
out['etf_intraday_on_open_sell_days'] = dict(n=int(len(so)), mean=float(np.nanmean(so)), t=float(np.nanmean(so) / (np.nanstd(so, ddof=1) / np.sqrt(np.isfinite(so).sum()))))
out['etf_intraday_all_days_mean'] = float(np.nanmean(intr[np.searchsorted(D.dates.values, np.datetime64('2005-02-01')):]))
print(out['etf_intraday_on_buy_days'], out['etf_intraday_on_open_sell_days'], out['etf_intraday_all_days_mean'])
# 2. 81 組：0050 版超額與個股部位比例、現金版的關係
g = pd.read_csv(HERE + 'my_grid81.csv')
g['cash_full_ann'] = g.cash_full_ex + 0.14550
g['sleeve_ret_on_inv'] = g.cash_full_ann / g.cash_inv
g['pred_naive'] = g.cash_full_ann - g.cash_inv * 0.1455
out['grid_corr'] = dict(corr_ex_inv=float(np.corrcoef(g.etf_full_ex, g.etf_inv)[0, 1]), corr_ex_nsig=float(np.corrcoef(g.etf_full_ex, g.n_sig)[0, 1]),
                        corr_ex_pred=float(np.corrcoef(g.etf_full_ex, g.pred_naive)[0, 1]),
                        ex_per_unit_inv_med=float((g.etf_full_ex / g.etf_inv).median()), inv_min=float(g.etf_inv.min()), inv_med=float(g.etf_inv.median()), inv_max=float(g.etf_inv.max()),
                        slope_ex_on_inv=float(np.polyfit(g.etf_inv, g.etf_full_ex, 1)[0]), intercept=float(np.polyfit(g.etf_inv, g.etf_full_ex, 1)[1]),
                        corr_2019ex_inv=float(np.corrcoef(g.etf_2019_ex, g.etf_inv)[0, 1]), slope_2019=float(np.polyfit(g.etf_inv, g.etf_2019_ex, 1)[0]))
print(out['grid_corr'])
# 3. 安慰劑：同日期、同筆數，改買隨機股票（池／趨勢樣板＋相對強度通過者），跑 E3＋空位 0050
pool = (D.V / 1000.0 >= 500) & (D.barno >= 260) & D.has
tsig = L.signals(D, ablate=('contract', 'voldry', 'cross', 'breakvol', 'minvol'))        # 只留趨勢＋相對強度（560,527 股日）
tmask = np.zeros_like(pool); tmask[tsig.di.values, tsig.j.values] = True
tmask &= pool
del tsig
def placebo(base_sig, mask, seed):
    rng = np.random.RandomState(seed)
    rows = []
    for di_, k in base_sig.groupby('di').size().items():
        cand = np.nonzero(mask[di_])[0]
        if len(cand) == 0: continue
        pick = rng.choice(cand, min(k, len(cand)), replace=False)
        for j in pick:
            rows.append((D.codes[j], j, di_, D.dates[di_], rng.rand(), np.nan))
    return pd.DataFrame(rows, columns=['code', 'j', 'di', 'date', 'vbreak', 'pivot_adj'])
def run_pl(base_sig, mask, n, tag):
    res = []
    for sd in range(n):
        ps = placebo(base_sig, mask, 1000 + sd)
        pc = L.perf(D, L.run_portfolio(D, ps, 'E3', 'cash'))
        pe = L.perf(D, L.run_portfolio(D, ps, 'E3', 'etf'))
        res.append((pe['full']['ex'], pe['2019']['ex'], pe['full']['t'], pe['invested'], pc['full']['ex']))
    a = np.array(res)
    o = dict(n=n, etf_full_ex_mean=float(a[:, 0].mean()), etf_full_ex_med=float(np.median(a[:, 0])), etf_full_ex_p05=float(np.quantile(a[:, 0], .05)), etf_full_ex_p95=float(np.quantile(a[:, 0], .95)),
             share_pos=float((a[:, 0] > 0).mean()), etf_2019_ex_med=float(np.median(a[:, 1])), share_pos_2019=float((a[:, 1] > 0).mean()), t_med=float(np.median(a[:, 2])), inv_med=float(np.median(a[:, 3])),
             cash_full_ex_med=float(np.median(a[:, 4])))
    out[tag] = o
    print(tag, {k: round(v, 4) for k, v in o.items()})
    return a
a1 = run_pl(sig, pool, 200, 'placebo_pool_default_dates')
a2 = run_pl(sig, tmask, 200, 'placebo_trendRS_default_dates')
out['default_pctile_in_placebo_pool'] = float((a1[:, 0] < -0.0032).mean()); out['default_pctile_in_placebo_trendRS'] = float((a2[:, 0] < -0.0032).mean())
out['grid_median_pctile_in_placebo_pool'] = float((a1[:, 0] < 0.0126).mean()); out['grid_median_pctile_in_placebo_trendRS'] = float((a2[:, 0] < 0.0126).mean())
s20 = L.signals(D, dict(SegLen=20))
a3 = run_pl(s20, pool, 100, 'placebo_pool_SegLen20_dates')
a4 = run_pl(s20, tmask, 100, 'placebo_trendRS_SegLen20_dates')
out['SegLen20_pctile_in_placebo_pool'] = float((a3[:, 0] < 0.0243).mean()); out['SegLen20_pctile_in_placebo_trendRS'] = float((a4[:, 0] < 0.0243).mean())
print({k: v for k, v in out.items() if 'pctile' in k})
json.dump(out, open(HERE + 'review_a_idle.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
