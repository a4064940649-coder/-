#!/usr/bin/env python3
"""探索（不看策略報酬，只看執行量）→ explore.json
 (1) 各級距×月份名單：整股／零股拆分、取整造成的現金殘差、零股最低手續費實際費率（FIX 版金額；全部 40 seeds＋V4）
 (2) 開盤集合競價量：vol_curve（2026-10）OA 占全日、OA ÷ 9:05 K（κ）；Yahoo 5 分 K 9:05–9:10 量 ÷ ADV60 依名次／價位／σ／跳空分桶
 (3) 零股價差：oddlot_hist 買成本（ask/收盤−1）、賣成本（1−bid/收盤）、成交價偏離，依價位×名次分桶
 (4) 9:10 價格相對開盤：Yahoo 09:05 K 收盤 ÷ 開盤 −1，依跳空分桶、成交日子樣本
 (5) 參與率預覽：粗估 OA（名次分桶中位數×κ）下，各級距新進股整股委託占 OA 的比例（FIX）
用法：python3 -I -B explore.py
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import xlib as X, ecore as E

out = {'note': '只用執行量與價格資料；沒有計算任何策略淨值或報酬。'}
D = E.Data()
q = lambda s, ps=(0.1, 0.25, 0.5, 0.75, 0.9): {str(p): (None if not np.isfinite(v) else round(float(v), 5)) for p, v in s.quantile(list(ps)).items()}

# ---------------- (1) 拆分 ----------------
rows = []
for seed, plan in D.plans.items():
    prev = set()
    for e, pk, sg in plan:
        for j in pk:
            rows.append((seed, e, j, sg, j not in prev))
        prev = set(pk)
R = pd.DataFrame(rows, columns=['seed', 'e', 'j', 'sig', 'new'])
R['date'] = D.IDX[R.e]; R['p'] = D.PC[R.e, R.j]; R['vrank'] = D.VRANK[R.e - 1, R.j]
R['pre_odd'] = R.date < E.ODD_INTRADAY
split = {}
for name, tier in E.TIERS.items():
    T = tier * np.array([E.cpi_scale(d) for d in R.date]) / 10
    lots = np.floor(T / (1000 * R.p)); rem = T - lots * 1000 * R.p
    odd = np.floor(rem / R.p + 1e-9); resid = rem - odd * R.p
    oddv = odd * R.p
    fee20 = np.where(odd > 0, 20 / np.where(oddv > 0, oddv, np.nan), np.nan)
    d = pd.DataFrame({'pure_odd': lots == 0, 'has_odd': odd > 0, 'odd_share': oddv / T, 'resid_share': resid / T, 'fee20': fee20,
                      'pre': R.pre_odd, 'new': R.new, 'hi': R.p > 300, 'yr': R.date.dt.year})
    split[name] = {
        'pure_odd_slot_share_all': float(d.pure_odd.mean()), 'pure_odd_new_entries': float(d[d.new].pure_odd.mean()),
        'pure_odd_pre2020_10': float(d[d.pre].pure_odd.mean()), 'pure_odd_post2020_10': float(d[~d.pre].pure_odd.mean()),
        'has_odd_order_share': float(d.has_odd.mean()), 'odd_value_share_mean': float(d.odd_share.mean()),
        'odd_value_share_new_post': float(d[d.new & ~d.pre].odd_share.mean()),
        'cash_residual_share_mean': float(d.resid_share.mean()), 'cash_residual_share_p90': float(d.resid_share.quantile(0.9)),
        'odd_minfee20_effective_rate_median': float(np.nanmedian(d.fee20)), 'odd_minfee20_rate_p90': float(np.nanquantile(d.fee20, 0.9)),
        'price_gt300_share': float(d.hi.mean()), 'pure_odd_by_year': d.groupby('yr').pure_odd.mean().round(3).to_dict()}
out['1_split_FIX'] = split
out['1_note'] = ('pure_odd＝每檔目標金額買不到一張（只能零股）；odd_value_share＝零股金額占每檔目標；cash_residual＝不足 1 股留現金；'
                 'fee20＝零股委託若最低 20 元時的實際費率。金額用 FIX（2026 幣值×CPI 換算）；所有 40 seeds＋V4 的全部名額。')
out['1_n_slots'] = int(len(R)); out['1_new_per_month'] = float(R[R.new].groupby(['seed', 'e']).size().mean())

# ---------------- (2) 開盤集合競價量 ----------------
V = X.load_vc()
Vt = V[V.V60_off.notna()]
kap_c = (V.OA_central / V.bar0905_eq).replace([np.inf, -np.inf], np.nan).dropna()
kap_u = (V.OA_upper / V.bar0905_eq).replace([np.inf, -np.inf], np.nan).dropna()
out['2a_volcurve'] = {
    'n_stock_days': int(len(V)), 'n_twse_with_official': int(len(Vt)), 'days': sorted(str(d.date()) for d in V.date.unique()),
    'OA_central_share_of_day': q(Vt.oa_share_day), 'OA_upper_share_of_day': q(Vt.OA_upper / Vt.volume),
    'day_volume_over_V60': q(Vt.volume * Vt.close / Vt.V60_off),
    'kappa_OA_central_over_bar0905': q(kap_c), 'kappa_OA_upper_over_bar0905': q(kap_u),
    'kappa_ratio_of_sums_central': float(V.OA_central.sum() / V.bar0905_eq.sum()),
    'OA_share_by_vrank': {str(k): round(float(v), 4) for k, v in Vt.groupby(pd.cut(Vt.vrank_off, [0, 100, 200, 300, 900]), observed=True).oa_share_day.median().items()},
    'note': 'vol_curve 是 screener 的觀察名單（多為爆量股：當日量中位數為 V60 的 3.3 倍），OA÷V60 不具代表性；只用來估 OA 占全日與 κ（OA÷09:05–09:10 量）。'}
Y = X.load_5m()
out['2b_yahoo_to_official_volume'] = X.YAHOO_TO_OFFICIAL
Yv = Y[np.isfinite(Y.r_bar) & (Y.r_bar > 0) & Y.vrank_l1.notna() & (Y.vrank_l1 <= 300)].copy()
Yv['lr'] = np.log(Yv.r_bar)
b = {}
b['all'] = q(Yv.r_bar)
b['by_vrank'] = {str(k): round(float(v), 4) for k, v in Yv.groupby(pd.cut(Yv.vrank_l1, [0, 50, 100, 200, 300]), observed=True).r_bar.median().items()}
b['by_price'] = {str(k): round(float(v), 4) for k, v in Yv.groupby(pd.cut(Yv.prevc, [0, 30, 100, 300, 1e5]), observed=True).r_bar.median().items()}
b['by_sig'] = {str(k): round(float(v), 4) for k, v in Yv.groupby(pd.qcut(Yv.sig_l1, 4), observed=True).r_bar.median().items()}
b['by_absgap'] = {str(k): round(float(v), 4) for k, v in Yv.groupby(pd.cut(Yv.gap.abs(), [-1e-9, 0.005, 0.01, 0.02, 0.04, 1]), observed=True).r_bar.median().items()}
b['exec_days_vs_other'] = {'exec': round(float(Yv[Yv.is_exec].r_bar.median()), 4), 'other': round(float(Yv[~Yv.is_exec].r_bar.median()), 4), 'n_exec': int(Yv.is_exec.sum())}
b['corr_spearman'] = {c: round(float(Yv[['lr', c]].corr('spearman').iloc[0, 1]), 3) for c in ('vrank_l1', 'prevc', 'sig_l1')}
b['note'] = 'r_bar＝Yahoo 09:05–09:10 K 量 ÷ Yahoo ADV60 × Yahoo/官方量比（%.3f）；Yahoo 09:00 K 量 95%% 為 0（不能直接當開盤量）。只取前 300 名池。' % X.YAHOO_TO_OFFICIAL
out['2b_bar0905_over_ADV60'] = b
# 粗估開盤量占 ADV：κ×r_bar
kc = float(kap_c.median())
out['2c_implied_OA_over_ADV60_median'] = {'central': float(kc * Yv.r_bar.median()), 'exec_days': float(kc * Yv[Yv.is_exec].r_bar.median()),
                                         'by_vrank': {k: round(kc * v, 4) for k, v in b['by_vrank'].items()}}

# ---------------- (3) 零股價差 ----------------
Od = X.load_odd()
Op = Od[(Od.close >= 10) & Od.vrank_off.notna() & (Od.vrank_off <= 300)].copy()
Op['pb'] = pd.cut(Op.close, [0, 50, 100, 300, 1e5]); Op['rb'] = pd.cut(Op.vrank_off, [0, 50, 100, 200, 300])
out['3_oddlot'] = {
    'n_rows_pool300': int(len(Op)), 'days': [str(Od.date.min().date()), str(Od.date.max().date())],
    'buy_cost_ask_over_close': q(Op.buy_cost.dropna()), 'sell_cost_1_minus_bid_over_close': q(Op.sell_cost.dropna()),
    'price_dev_close': q(Op.dev_close.dropna()), 'half_spread': q(Op.hs.dropna()),
    'share_days_no_odd_trade': float((Op.shares == 0).mean()),
    'buy_cost_by_price': {str(k): round(float(v), 5) for k, v in Op.groupby('pb', observed=True).buy_cost.median().items()},
    'sell_cost_by_price': {str(k): round(float(v), 5) for k, v in Op.groupby('pb', observed=True).sell_cost.median().items()},
    'buy_cost_by_rank': {str(k): round(float(v), 5) for k, v in Op.groupby('rb', observed=True).buy_cost.median().items()},
    'sell_cost_by_rank': {str(k): round(float(v), 5) for k, v in Op.groupby('rb', observed=True).sell_cost.median().items()},
    'buy_cost_mean_by_price': {str(k): round(float(v), 5) for k, v in Op.groupby('pb', observed=True).buy_cost.mean().items()},
    'note': '買成本＝收盤時零股委買賣簿的 ask ÷ 整股收盤 −1；賣成本＝1 − bid ÷ 整股收盤（同時點 13:30）。正值＝對散戶不利。'}

# ---------------- (4) 9:10 價格相對開盤 ----------------
Yd = Y[np.isfinite(Y.drift910) & Y.vrank_l1.notna() & (Y.vrank_l1 <= 300) & np.isfinite(Y.gap)].copy()
gb = pd.cut(Yd.gap, [-1, -0.03, -0.01, 0, 0.01, 0.03, 1])
out['4_drift910'] = {
    'n': int(len(Yd)), 'all': q(Yd.drift910), 'mean_all': float(Yd.drift910.mean()),
    'mean_by_gap': {str(k): round(float(v), 5) for k, v in Yd.groupby(gb, observed=True).drift910.mean().items()},
    'n_by_gap': {str(k): int(v) for k, v in Yd.groupby(gb, observed=True).size().items()},
    'exec_days': {'n': int(Yd.is_exec.sum()), 'mean': float(Yd[Yd.is_exec].drift910.mean()), 'median': float(Yd[Yd.is_exec].drift910.median()),
                  'sd': float(Yd[Yd.is_exec].drift910.std())},
    'yahoo_open_eq_panel_open': float((np.abs(Y.y_open / Y.open_p - 1) < 1e-9).mean()),
    'note': 'p910＝Yahoo 09:05 K 收盤（9:10 整股價）；盤中零股 9:10 第一次撮合價以此加零股價差代理。'}

# ---------------- (5) 參與率預覽（FIX、新進股、粗估 OA）----------------
med_by_rank = Yv.groupby(pd.cut(Yv.vrank_l1, [0, 50, 100, 200, 300]), observed=True).r_bar.median()
edges = [0, 50, 100, 200, 300]
def oa_crude(rank):
    k = np.clip(np.searchsorted(edges, rank, side='left') - 1, 0, 3)
    return kc * med_by_rank.to_numpy()[k]
Rn = R[R.new].copy()
Rn['V60'] = D.V60[Rn.e - 1, Rn.j]
Rn['oa'] = oa_crude(Rn.vrank.fillna(300).to_numpy()) * Rn.V60
part = {}
for name, tier in E.TIERS.items():
    T = tier * np.array([E.cpi_scale(d) for d in Rn.date]) / 10
    lots = np.floor(T / (1000 * Rn.p)); Q = lots * 1000 * Rn.p
    r = Q / Rn.oa
    dd = pd.DataFrame({'r': r, 'early': Rn.date.dt.year <= 2018, 'small': Rn.vrank > 200})
    part[name] = {'gt10_all': float((dd.r > 0.10).mean()), 'gt20_all': float((dd.r > 0.20).mean()),
                  'gt10_2006_18': float((dd[dd.early].r > 0.10).mean()), 'gt10_2019_26': float((dd[~dd.early].r > 0.10).mean()),
                  'gt10_rank201_300': float((dd[dd.small].r > 0.10).mean()), 'median_part': float(dd.r.median()),
                  'gt1pct_ADV': float((Q / Rn.V60 > 0.01).mean())}
out['5_participation_preview_FIX_new_entries'] = part
out['5_note'] = '粗估：OA＝κ（vol_curve 中位數 %.3f）× Yahoo 名次分桶 r_bar 中位數 × V60（訊號日）。正式模型見 iterations.json。' % kc
json.dump(out, open(HERE / 'explore.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float)[:6000])
