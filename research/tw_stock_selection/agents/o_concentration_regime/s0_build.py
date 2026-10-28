#!/usr/bin/env python3
"""步驟 0：清點與挑選期訊號基礎表（PREREG 前）。
* 跳日清點（只看日期，不看報酬）：主策略淨值、0050、面板 GOOD 日，列出連續 > 5 個加權指數交易日沒有資料的區段。
* 決策日：面板規則（每月第一個日期 ≥ 11 的 GOOD 日）重算，並與 plan_V4.csv 的 signal 欄逐一核對。
* 訊號基礎表只建到挑選期結束（2023-01 決策日），起點 2016-12（L=12＋跳過 1 個月所需回看）。
* 同池等權重建後與 AB_V4|EW（o_v4 引擎的同池等權，開盤成交、排除處置）逐視窗核對（2017～2022）。
輸出：data/W_sel.parquet、explore.json 的 S0 區塊。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import cr_lib as L

SEL_END_DEC = '2023-01-31'     # 只建到 2023-01 決策日（挑選期最後一個視窗 2022-12 的終點）

P = L.Panel(end=SEL_END_DEC)
dates_all = P.official_dates('2016-12-01')
dates = [d for d in dates_all if d <= pd.Timestamp(SEL_END_DEC)]
W = P.windows(dates)
W = L.add_e50(W, L.e50_series()[lambda s: s.index <= pd.Timestamp(SEL_END_DEC)])
L.DATA.mkdir(exist_ok=True)
W.to_parquet(L.DATA / 'W_sel.parquet')

out = {}
# ---- 決策日核對 ----
plan = pd.read_csv(L.RAW / 'plan_V4.csv', parse_dates=['signal', 'exec'])
ps = set(plan['signal'][(plan['signal'] >= pd.Timestamp('2017-01-01')) & (plan['signal'] <= pd.Timestamp(SEL_END_DEC))])
mine = set(d for d in dates if d >= pd.Timestamp('2017-01-01'))
out['decision_dates_vs_plan_V4'] = dict(n_mine=len(mine), n_plan=len(ps), only_mine=sorted(str(d.date()) for d in mine - ps), only_plan=sorted(str(d.date()) for d in ps - mine))

# ---- 跳日清點（全期，只看日期）----
tx = pd.DatetimeIndex(sorted(set(pd.read_csv(L.RAW / 'taiex_twii.csv', parse_dates=['date'])['date'])))
def gaps(ix, name, a='2006-01-01', b='2026-10-22'):
    ix = pd.DatetimeIndex(sorted(set(ix))); txw = tx[(tx >= pd.Timestamp(a)) & (tx <= pd.Timestamp(b))]
    miss = txw[~txw.isin(ix)]
    res = [];
    if len(miss):
        pos = np.array([txw.get_loc(d) for d in miss]); s = 0
        for i in range(1, len(pos) + 1):
            if i == len(pos) or pos[i] != pos[i - 1] + 1:
                if i - s > 5: res.append([str(miss[s].date()), str(miss[i - 1].date()), int(i - s)])
                s = i
    return dict(name=name, n_missing_taiex_days=int(len(miss)), runs_gt5=res, extra_days_not_taiex=int((~ix[(ix >= pd.Timestamp(a)) & (ix <= pd.Timestamp(b))].isin(tx)).sum()))
nv4 = pd.read_parquet(L.RAW / 'navs_v4.parquet')['AB_V4|normal'].dropna().index
nv6 = pd.read_parquet(L.RAW / 'navs_v6.parquet')['AB_V6|normal'].dropna().index
e50 = L.e50_series().index
Pfull_idx = pd.read_parquet(L.PANEL / 'adj.parquet', columns=['2330']).index
adjn = pd.read_parquet(L.PANEL / 'adj.parquet').notna().sum(axis=1)
good = Pfull_idx[(adjn >= 300).to_numpy()]
out['skip_day_inventory'] = [gaps(nv4, 'AB_V4 淨值', '2006-04-11', '2026-10-12'), gaps(nv6, 'AB_V6 淨值', '2006-04-11', '2026-10-12'),
                             gaps(e50, '0050 含息（etf_0050.csv，已剔除 2026-09-20 週日壞資料）'), gaps(good, '面板 GOOD 日（有值 ≥ 300 檔）')]
out['panel_bad_days_lt300'] = [str(d.date()) for d in Pfull_idx[(adjn < 300).to_numpy()]]

# ---- 同池等權核對（挑選期之前與之內，2017～2022）----
ew = pd.read_parquet(L.RAW / 'navs_v4.parquet')['AB_V4|EW'].dropna()
ew_w = L.win_ret(ew, [d for d in dates if d in ew.index])
m = W['rEW'].reindex(ew_w.index)
x = pd.concat([m, ew_w], axis=1, keys=['mine', 'AB_V4_EW']).dropna()
x = x[(x.index >= pd.Timestamp('2017-01-01')) & (x.index <= pd.Timestamp('2022-12-31'))]
out['ew_check_vs_AB_V4_EW_2017_2022'] = dict(n=int(len(x)), corr=float(x.corr().iloc[0, 1]), mean_abs_diff_pp=float((x.mine - x.AB_V4_EW).abs().mean() * 100),
                                            max_abs_diff_pp=float((x.mine - x.AB_V4_EW).abs().max() * 100), mean_diff_pp=float((x.mine - x.AB_V4_EW).mean() * 100),
                                            note='我的同池等權：訊號日收盤到收盤、不排除處置；AB_V4|EW：exec 日開盤成交、排除處置。差異是時點口徑，不是錯誤。')
out['W_sel_rows'] = int(len(W)); out['W_sel_span'] = [str(W.index[0].date()), str(W['end'].iloc[-1].date())]
out['pool_size_min'] = int(W['n_pool'].min())

ex = L.load_json(L.HERE / 'explore.json'); ex['S0'] = out; L.save_json(ex, L.HERE / 'explore.json')
print(pd.Series({k: v for k, v in out.items() if k != 'skip_day_inventory'}).to_string()); print(out['skip_day_inventory'])
