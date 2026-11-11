#!/usr/bin/env python3
"""反方審查（資料與統計，o20）：只讀研究員匯出的 data/swaps_exitlevel_100n_REAL_opt_seed0.csv.gz 與 results.json。
不改研究員檔案。python3 -I -B review_stat_o20.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
HERE = Path(__file__).resolve().parent
X = pd.read_csv(HERE / 'data/swaps_exitlevel_100n_REAL_opt_seed0.csv.gz', dtype={'code': str})
X = X[np.isfinite(X.edge_B)].copy()
X['sig'] = pd.to_datetime(X.signal); X['gap'] = X.edge_B - X.rtc_B; X['below'] = X.gap < 0
X['mon'] = X.sig.dt.to_period('M').astype(str); X['y'] = X.sig.dt.year
P = {'Y19on': X[X.sig >= '2019-01-01'], 'S0618': X[(X.sig < '2019-01-01')]}
rng = np.random.default_rng(7)
def boot(df, stat, B=2000):
    m = df.mon.to_numpy(); u = np.unique(m); ids = [np.flatnonzero(m == x) for x in u]; df = df.reset_index(drop=True)
    out = np.empty(B)
    for b in range(B):
        out[b] = stat(df.iloc[np.concatenate([ids[i] for i in rng.integers(0, len(ids), len(ids))])])
    return stat(df), out
O = {}
for per, d in P.items():
    o = {}
    o['n'] = len(d); o['n_unique_code_e'] = int(d[['code', 'e']].drop_duplicates().shape[0]); o['n_months'] = int(d.mon.nunique())
    o['edge_sd'] = float(d.edge_B.std()); o['edge_iqr'] = float(d.edge_B.quantile(.75) - d.edge_B.quantile(.25)); o['rtc_med'] = float(d.rtc_B.median())
    o['edge_mean'] = float(d.edge_B.mean()); o['edge_med'] = float(d.edge_B.median()); o['frac_below'] = float(d.below.mean())
    # 去除換股日共同成分後的離散度（B 配對下同日所有 exit 共用 mean r_entry）
    o['r_exit_sd_within_reb'] = float((d.r - d.groupby(['j', 'k']).r.transform('mean')).std())
    # 檢定力：若某一類 edge 整體平移 −δ，edge<成本 比率要上升 10pp 需要多大 δ（用該期實際 gap 分布經驗計算）
    g = d.gap.to_numpy(); base = (g < 0).mean()
    for dlt in (0.005, 0.01, 0.02, 0.03, 0.05):
        o[f'frac_up_pp_if_shift_{dlt}'] = float(((g - dlt) < 0).mean() - base) * 100
    lo, hi = 0.0, 0.2
    for _ in range(40):
        mid = (lo + hi) / 2
        if ((g - mid) < 0).mean() - base < 0.10: lo = mid
        else: hi = mid
    o['shift_needed_for_10pp'] = hi
    # 平均數口徑的類別差（比中位數／比率更有檢定力）：>50 減 11～15 的 mean gap，月區塊 bootstrap
    def st_mean(z):
        a = z.gap[z.rb == '>50']; b = z.gap[z.rb == '11~15']; return a.mean() - b.mean()
    ob, bb = boot(d, st_mean); o['P4_meangap_gt50_minus_1115'] = dict(obs=float(ob), ci90=[float(np.percentile(bb, 5)), float(np.percentile(bb, 95))], p_le0=float((bb <= 0).mean()))
    def st_med(z):
        a = z.gap[z.rb == '>50']; b = z.gap[z.rb == '11~15']; return np.median(a) - np.median(b)
    ob, bb = boot(d, st_med); o['P4_medgap_ci90'] = dict(obs=float(ob), ci90=[float(np.percentile(bb, 5)), float(np.percentile(bb, 95))], p_le0=float((bb <= 0).mean()))
    # 低價 <50 vs ≥100：mean gap 差
    def st_px(z):
        return z.gap[z.price >= 100].mean() - z.gap[z.price < 50].mean()
    ob, bb = boot(d, st_px); o['P3_meangap_hi_minus_lo'] = dict(obs=float(ob), ci90=[float(np.percentile(bb, 5)), float(np.percentile(bb, 95))], p_le0=float((bb <= 0).mean()))
    # <20 元級距
    def st_20(z):
        return z.gap[z.price >= 20].mean() - z.gap[z.price < 20].mean()
    ob, bb = boot(d, st_20); o['lt20_meangap_rest_minus_lt20'] = dict(obs=float(ob), n_lt20=int((d.price < 20).sum()), n_unique_lt20=int(d[d.price < 20][['code', 'e']].drop_duplicates().shape[0]), ci90=[float(np.percentile(bb, 5)), float(np.percentile(bb, 95))], p_le0=float((bb <= 0).mean()))
    o['by_rb'] = {b: dict(n=int(len(q)), n_uniq=int(q[['code', 'e']].drop_duplicates().shape[0]), frac=float(q.below.mean()), gap_mean=float(q.gap.mean()), gap_med=float(q.gap.median())) for b, q in d.groupby('rb')}
    o['by_year_P4_meandiff'] = {int(y): float(q.gap[q.rb == '>50'].mean() - q.gap[q.rb == '11~15'].mean()) for y, q in d.groupby('y') if (q.rb == '11~15').any() and (q.rb == '>50').any()}
    # 平均每筆可達到的最小可偵測差（mean gap，兩類，80% 檢定力，單尾 5%）用 bootstrap SE
    o['MDE_mean_gap_P4'] = float((norm.ppf(0.95) + norm.ppf(0.8)) * np.std(bb if False else boot(d, st_mean)[1]))
    O[per] = o
# 同一筆換出在 13 j 間重複程度
O['dup'] = {per: float(len(d) / d[['code', 'e']].drop_duplicates().shape[0]) for per, d in P.items()}
O['dup_code_month'] = {per: float(len(d) / d[['code', 'mon']].drop_duplicates().shape[0]) for per, d in P.items()}
# A1 與 A2 配對是否其實相同
O['A1_eq_A2_share'] = float(np.mean(np.isclose(X.edge_A1, X.edge_A2, equal_nan=True)))
O['A1_eq_A2_share_when_nX_eq_nN'] = float(np.mean(np.isclose(X.edge_A1, X.edge_A2, equal_nan=True)[X.n_exits.to_numpy() == X.n_entries.to_numpy()]))
# 會計拖累 vs Shapley 衝擊差距
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['RESULTS']
dr = R['B_drag']['normal|100萬|Y19on']['REAL_opt']; sh = R['I_shapley']['100萬|REAL_opt|CAGR_Y19on']['shapley']
O['acct_vs_shapley'] = dict(acct_total=dr['cost'], acct_tick_imp_odd=dr['tick_amt'] + dr['imp_amt'] + dr['odd_amt'], shap_K_I_C=sh['跳動點／價差']['mean'] + sh['衝擊']['mean'] + sh['參與上限']['mean'],
                           acct_tax=dr['tax'], shap_tax=sh['證交稅']['mean'], acct_fee=dr['fee'], shap_fee=sh['手續費']['mean'])
print(json.dumps(O, ensure_ascii=False, indent=1, default=float))
