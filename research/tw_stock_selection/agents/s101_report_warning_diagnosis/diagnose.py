#!/usr/bin/env python3
"""S101 報告警訊診斷（2026-11-27）——只做描述性診斷，不改規則、不跑新參數組合、不引入外部資料。

執行: python3 -I -B diagnose.py      （數秒、< 500MB）
輸入: ../s75_continuous_2008_2026/v3_2026-11-25_limit7/nav_2008_2026_v3.csv.gz（V_BRIDGE、0050）、trades_v3.csv.gz（V_BRIDGE 627 筆）
      ../eng_full_report_2026-11-27/results.json（第 9 項兩腿數字、核對用）、build_report.py（檢定函數，直接 import）
      repo 既有加權指數檔（引擎本來就讀的 taiex_twii.csv，價格指數不含息）
輸出: results.json（同資料夾）

三列並排口徑：
  strategy ＝ 帳戶日報酬 × 100 萬（規格固定本金 PnL）
  b0050    ＝ 0050 含息日報酬 × 100 萬
  excess   ＝（帳戶 − 0050）日報酬 × 100 萬
"""
import sys, math, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, optimize
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RPT = AG / 'eng_full_report_2026-11-27'
V3 = AG / 's75_continuous_2008_2026' / 'v3_2026-11-25_limit7'
TWII_CSV = Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')
sys.path.insert(0, str(RPT))
import build_report as BR          # 只用函數：hac_tstat、deflated_sharpe、dd_episodes（不跑引擎、不寫報告）

CAP = 1_000_000
N_OURS = 4040
Z95 = stats.norm.ppf(0.95)


def sr_ann(x):
    x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(252))


def runs(mask):
    """回傳 True 連續區段 (i0, i1) 含端點。"""
    m = np.asarray(mask, bool); out = []; i = 0; n = len(m)
    while i < n:
        if m[i]:
            k = i
            while k + 1 < n and m[k + 1]:
                k += 1
            out.append((i, k)); i = k + 1
        else:
            i += 1
    return out


# ------------------------------------------------------------------ 資料
nav = pd.read_csv(V3 / 'nav_2008_2026_v3.csv.gz', index_col=0, parse_dates=True)
ns, nb = nav['V_BRIDGE'], nav['0050']
r_s = ns.pct_change().iloc[1:]; r_b = nb.pct_change().iloc[1:]; r_x = r_s - r_b
dates = r_s.index
S = {'strategy': r_s * CAP, 'b0050': r_b * CAP, 'excess': r_x * CAP}
rep = json.loads((RPT / 'results.json').read_text(encoding='utf-8'))
tr = pd.read_csv(V3 / 'trades_v3.csv.gz', parse_dates=['sig_date', 'ent_date', 'exit_date'])
tr = tr[tr['version'] == 'V_BRIDGE'].reset_index(drop=True)
assert len(tr) == 627
twii = pd.read_csv(TWII_CSV, parse_dates=['date']).set_index('date')['close'].sort_index()
r_tw = twii.pct_change().reindex(dates)
tw_missing = int(r_tw.isna().sum()); r_tw = r_tw.fillna(0)

out = {'meta': dict(id='S101', built='2026-11-27', sample=[str(nav.index[0].date()), str(nav.index[-1].date())], n_days=int(len(dates)),
                    note='描述性診斷；不改規則、不跑新參數組合；三列＝策略／0050 含息／策略−0050 日超額（皆 ×100 萬固定本金）',
                    twii_missing_days_filled0=tw_missing)}
yrs = len(dates) / 252; days_per_year = len(dates) / ((dates[-1] - nav.index[0]).days / 365.25)

# 核對報告數字
chk = dict(sr=sr_ann(S['strategy']), sr_rep=rep['core9_fixed_capital']['sharpe'])
out['check'] = chk
assert abs(chk['sr'] - chk['sr_rep']) < 1e-6, chk   # nav 檔四捨五入到 0.01 元，差 ~5e-9

# 基本三列
base = {}
for k, x in S.items():
    base[k] = dict(sr=sr_ann(x), mu_daily_pct=float(x.mean() / CAP * 100), sd_daily_pct=float(x.std(ddof=1) / CAP * 100),
                   hac_t=BR.hac_tstat(x)[0], skew=float(stats.skew(x, bias=False)), exkurt=float(stats.kurtosis(x, bias=False)),
                   total_pnl=float(x.sum()))
out['base_three_rows'] = base

# ================================================================== 第 7 項 Rolling 252d SR
roll = {k: (x.rolling(252).mean() / x.rolling(252).std(ddof=1) * math.sqrt(252)) for k, x in S.items()}
t7 = {}
for k, r in roll.items():
    rc = r.dropna(); sr_full = base[k]['sr']
    t7[k] = dict(pct_below_0=float((rc < 0).mean()), mean=float(rc.mean()), min=float(rc.min()),
                 iid_expect_pct_below_0=float(stats.norm.cdf(-sr_full)),   # iid 常態近似：1 年 SR 估計值 ~ N(SR, 1)
                 n_windows=int(len(rc)))
# 「門檻 15% 等價於全期 SR 要多高」（iid 近似：Φ(−SR) < 15% ⇔ SR > 1.036）
t7['iid_sr_needed_for_15pct'] = float(-stats.norm.ppf(0.15))
rs_, rb_, rx_ = roll['strategy'], roll['b0050'], roll['excess']
valid = rs_.notna()
neg = (rs_ < 0) & valid
t7['overlap'] = dict(strategy_neg_days=int(neg.sum()), of_which_0050_also_neg=int(((rb_ < 0) & neg).sum()),
                     share_0050_also_neg=float(((rb_ < 0) & neg).sum() / neg.sum()),
                     b0050_neg_days=int(((rb_ < 0) & valid).sum()),
                     b0050_neg_and_strategy_pos=int(((rb_ < 0) & (rs_ >= 0) & valid).sum()))
iv = []
for i0, i1 in runs(neg.to_numpy()):
    seg = slice(i0, i1 + 1)
    iv.append(dict(win_end_start=str(dates[i0].date()), win_end_end=str(dates[i1].date()),
                   window_cover=[str(dates[max(i0 - 251, 0)].date()), str(dates[i1].date())], n_days=int(i1 - i0 + 1),
                   strat_sr_min=float(rs_.iloc[seg].min()), strat_sr_mean=float(rs_.iloc[seg].mean()),
                   b0050_sr_mean=float(rb_.iloc[seg].mean()), b0050_sr_min=float(rb_.iloc[seg].min()),
                   b0050_neg_share=float((rb_.iloc[seg] < 0).mean()), excess_sr_mean=float(rx_.iloc[seg].mean())))
t7['strategy_neg_intervals'] = iv
# 合併間隔 ≤ 21 個交易日的區段（較易讀）
cl = []
for (i0, i1) in runs(neg.to_numpy()):
    if cl and i0 - cl[-1][1] <= 21:
        cl[-1][1] = i1
    else:
        cl.append([i0, i1])
t7['strategy_neg_clusters_gap21'] = [dict(win_end_start=str(dates[a].date()), win_end_end=str(dates[b].date()),
                                          window_cover=[str(dates[max(a - 251, 0)].date()), str(dates[b].date())],
                                          neg_days=int(neg.iloc[a:b + 1].sum()), span_days=int(b - a + 1),
                                          strat_sr_min=float(rs_.iloc[a:b + 1].min()), b0050_sr_mean=float(rb_.iloc[a:b + 1].mean()),
                                          b0050_neg_share_on_strat_neg_days=float(((rb_.iloc[a:b + 1] < 0) & neg.iloc[a:b + 1]).sum() / max(neg.iloc[a:b + 1].sum(), 1)),
                                          excess_sr_mean=float(rx_.iloc[a:b + 1].mean())) for a, b in cl]
t7['strategy_specific_neg'] = dict(days=int((neg & (rb_ >= 0)).sum()), pct_of_windows=float((neg & (rb_ >= 0)).sum() / valid.sum()),
                                   pct_if_only_market_driven=float((neg & (rb_ < 0)).sum() / valid.sum()),
                                   by_year={int(k): int(v) for k, v in (neg & (rb_ >= 0)).groupby(dates.year).sum().items() if v > 0})
# 依年份統計 SR<0 天數
yneg = neg.groupby(dates.year).sum(); t7['strategy_neg_days_by_year'] = {int(k): int(v) for k, v in yneg.items() if v > 0}
out['t07_rolling252'] = t7

# ================================================================== 第 8 項 水下
def uw_stats(x_pnl, nav_series):
    cum = np.concatenate([[0.0], np.cumsum(x_pnl)])
    dd_fixed = cum - np.maximum.accumulate(cum)
    v = np.asarray(nav_series, float); dd_c = v / np.maximum.accumulate(v) - 1
    o = dict(uw_fixed=float((dd_fixed[1:] < 0).mean()), uw_compound=float((dd_c[1:] < 0).mean()))
    for thr in (0.01, 0.05, 0.10):
        o[f'uw_fixed_deeper_{int(thr * 100)}pct_of_capital'] = float((dd_fixed[1:] < -thr * CAP).mean())
        o[f'uw_compound_deeper_{int(thr * 100)}pct'] = float((dd_c[1:] < -thr).mean())
    return o, cum, v

rel = (ns / nb)   # 策略相對 0050 的財富比（複利口徑的超額）
t8 = {}
eps = {}
for k, navser in (('strategy', ns), ('b0050', nb), ('excess', rel)):
    o, cum, v = uw_stats(S[k].to_numpy(), navser.to_numpy())
    dl = list(nav.index)
    ef = BR.dd_episodes(cum, dl); ec = BR.dd_episodes(v, dl)
    for e in ef + ec:
        e['len'] = e['d_down'] + (e['d_up'] if e['d_up'] is not None else (len(dl) - 1 - dl.index(pd.Timestamp(e['trough']))))
    o['n_episodes_fixed'] = len(ef); o['n_episodes_compound'] = len(ec)
    # 最長 3 段占全部水下天數的比例（複利）
    lens = sorted([e['len'] for e in ec], reverse=True)
    o['compound_top3_longest_share_of_uw_days'] = float(sum(lens[:3]) / max(sum(lens), 1))
    o['compound_longest_days'] = int(lens[0])
    t8[k] = o
    eps[k] = dict(top10_depth_compound=sorted(ec, key=lambda e: e['depth_pct'])[:10],
                  top10_length_compound=sorted(ec, key=lambda e: -e['len'])[:10],
                  top10_depth_fixed=sorted(ef, key=lambda e: e['depth'])[:10])
t8['episodes'] = eps
t8['mc_iid_bootstrap_tuw_from_report'] = rep['monte_carlo']['tuw']
# iid 常態模擬：同 μ/σ、同天數的「水下時間比例」分布（描述機制，不是策略試驗）
rng = np.random.default_rng(101)
sim = {}
for k in ('strategy', 'b0050', 'excess'):
    mu, sd = S[k].mean(), S[k].std(ddof=1); u = []
    for _ in range(2000):
        c = np.cumsum(rng.normal(mu, sd, len(dates))); c = np.concatenate([[0.0], c])
        u.append(float((c - np.maximum.accumulate(c) < 0)[1:].mean()))
    sim[k] = dict(p5=float(np.percentile(u, 5)), p50=float(np.percentile(u, 50)), p95=float(np.percentile(u, 95)))
# 要讓 iid 中位數水下 < 85% 需要的年化 SR（格點描述）
grid_sr = {}
for s_ann in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
    u = []
    for _ in range(500):
        c = np.cumsum(rng.normal(s_ann / math.sqrt(252), 1.0, len(dates))); c = np.concatenate([[0.0], c])
        u.append(float((c - np.maximum.accumulate(c) < 0)[1:].mean()))
    grid_sr[str(s_ann)] = float(np.median(u))
t8['iid_normal_sim_uw_fixed'] = sim
t8['iid_normal_uw_median_by_annual_sr'] = grid_sr
out['t08_underwater'] = t8

# ================================================================== 第 13 項 CAPM
def ols_alpha(y, x, hac=False):
    X = sm.add_constant(np.asarray(x, float))
    m = sm.OLS(np.asarray(y, float), X).fit(cov_type='HAC', cov_kwds={'maxlags': int(len(y) ** 0.25)}) if hac else sm.OLS(np.asarray(y, float), X).fit()
    return m

def capm_rows(y_daily, y_spec, bm_daily, bm_name):
    o = {}
    m = ols_alpha(y_spec, bm_daily); o['spec_formula'] = dict(alpha_a=float((1 + m.params[0]) ** 252 - 1), t=float(m.tvalues[0]), beta=float(m.params[1]), r2=float(m.rsquared))
    m = ols_alpha(y_daily, bm_daily); mh = ols_alpha(y_daily, bm_daily, hac=True)
    o['actual_daily'] = dict(alpha_a=float((1 + m.params[0]) ** 252 - 1), t=float(m.tvalues[0]), t_hac=float(mh.tvalues[0]), beta=float(m.params[1]), r2=float(m.rsquared))
    yd = pd.Series(np.asarray(y_daily), index=dates); bd = pd.Series(np.asarray(bm_daily), index=dates)
    ym = (1 + yd).groupby(dates.to_period('M')).prod() - 1; bmm = (1 + bd).groupby(dates.to_period('M')).prod() - 1
    m = ols_alpha(ym, bmm); mh = ols_alpha(ym, bmm, hac=True)
    o['monthly'] = dict(alpha_a=float((1 + m.params[0]) ** 12 - 1), t=float(m.tvalues[0]), t_hac=float(mh.tvalues[0]), beta=float(m.params[1]), r2=float(m.rsquared), n_months=int(len(ym)))
    return o

def spec_ret(pnl):
    eq = CAP + pd.Series(pnl).cumsum()
    return eq.pct_change().fillna(0).to_numpy()   # 規格公式（第一天 0）

t13 = {}
for k, rd in (('strategy', r_s), ('b0050', r_b), ('excess', r_x)):
    t13[f'{k}_vs_TAIEX'] = capm_rows(rd.to_numpy(), spec_ret(S[k].to_numpy()), r_tw.to_numpy(), 'TAIEX')
t13['strategy_vs_0050'] = capm_rows(r_s.to_numpy(), spec_ret(S['strategy'].to_numpy()), r_b.to_numpy(), '0050')
# 稀釋機制：規格公式的日報酬 = r_t × 100 萬 ÷ equity_{t-1}
eq = CAP + S['strategy'].cumsum(); scale = (CAP / eq.shift(1).fillna(CAP))
t13['dilution'] = dict(equity_end=float(eq.iloc[-1]), scale_mean=float(scale.mean()), scale_end=float(scale.iloc[-1]),
                       scale_by_year={int(y): float(v) for y, v in scale.groupby(dates.year).mean().items()},
                       spec_mean_daily_pct=float(spec_ret(S['strategy'].to_numpy()).mean() * 100), actual_mean_daily_pct=float(r_s.mean() * 100),
                       note='規格把固定本金累加 PnL 當成淨值再取 pct_change：分母從 100 萬長到約 546 萬，後段每天的報酬被除以 3～5 倍；基準（TAIEX）報酬卻沒有被縮小，所以 β 被壓低、α 被壓低、後段（策略最強的 2020 以後）權重最小。')
# 0050 含息 vs TAIEX 價格指數：股息造成的機械 α
t13['dividend_note'] = '加權指數檔是價格指數（不含息），0050 為含息；0050 對 TAIEX 的 α 約等於股息殖利率，屬口徑差，不是選股能力。'
out['t13_capm'] = t13

# ================================================================== 第 14 項 DSR
def dsr_z_parts(x):
    x = pd.Series(x).dropna().astype(float); n = len(x); sr = x.mean() / x.std(ddof=1)
    g3 = stats.skew(x, bias=False); g4 = stats.kurtosis(x, bias=False)
    return n, sr, g3, g4

def emax(N):
    if N <= 1:
        return 0.0
    return math.sqrt(2 * math.log(N)) - (math.log(math.log(max(N, 3))) + math.log(4 * math.pi)) / (2 * math.sqrt(2 * math.log(N)))

def dsr_prob(sr, n, g3, g4, N):
    v = max(1 - g3 * sr + g4 / 4 * sr ** 2, 1e-12)
    z = (sr - emax(N) / math.sqrt(n)) * math.sqrt(n - 1) / math.sqrt(v)
    return float(stats.norm.cdf(z))

def sr_needed(n, g3, g4, N, target=0.95):
    f = lambda s: dsr_prob(s, n, g3, g4, N) - target
    return float(optimize.brentq(f, 1e-6, 1.0))

def n_needed(sr, g3, g4, N, target=0.95):
    f = lambda n: dsr_prob(sr, n, g3, g4, N) - target
    if f(40) > 0:
        return 40.0
    hi = 40
    while f(hi) < 0:
        hi *= 2
        if hi > 1e8:
            return float('inf')
    return float(optimize.brentq(f, 40, hi))

t14 = {'formula_note': ('規格 DSR：門檻 SR* = E[max of N 個標準常態]/√n（假設 N 個試驗互相獨立且真實 SR 皆為 0、試驗間 SR 的標準差＝1/√n）；'
                        '變異數項用 1 − γ3·SR + (超額峰度/4)·SR²（原論文為 (γ4−1)/4，γ4＝原始峰度；日 SR 很小，差異可忽略）。')}
Ns = [1, 10, 50, 100, 200, 403, 500, 1000, 2000, 4040, 4140, 4500, 4540, 4600, 5000, 10000, 100000]
for k, x in S.items():
    n, sr, g3, g4 = dsr_z_parts(x)
    row = dict(n=n, sr_daily=float(sr), sr_annual=float(sr * math.sqrt(252)),
               dsr_by_N={str(N): dsr_prob(sr, n, g3, g4, N) for N in Ns})
    # 與 build_report 函數核對
    assert abs(row['dsr_by_N']['4500'] - BR.deflated_sharpe(x, 4500)[0]) < 1e-9
    need = {}
    for N in (200, 4040, 4140, 4500, 4540, 4600, 5000):
        s_need = sr_needed(n, g3, g4, N)
        nd = n_needed(sr, g3, g4, N) if sr > 0 else float('inf')
        need[str(N)] = dict(sr_daily_needed=s_need, sr_annual_needed=s_need * math.sqrt(252),
                            days_needed_at_current_sr=nd, years_needed_at_current_sr=nd / days_per_year if math.isfinite(nd) else None,
                            extra_years_vs_now=(nd - n) / days_per_year if math.isfinite(nd) else None)
    row['needed'] = need
    row['emax'] = {str(N): emax(N) for N in (200, 4040, 4500, 5000)}
    t14[k] = row
s = t14['strategy']['dsr_by_N']
t14['more_trials_impact'] = dict(
    spec_N4500=s['4500'], spec_plus100=s['4600'], spec_plus500=s['5000'],
    ours_N4040=s['4040'], ours_plus100=s['4140'], ours_plus500=s['4540'],
    years_needed_4500=t14['strategy']['needed']['4500']['years_needed_at_current_sr'],
    years_needed_4600=t14['strategy']['needed']['4600']['years_needed_at_current_sr'],
    years_needed_5000=t14['strategy']['needed']['5000']['years_needed_at_current_sr'],
    sr_ann_needed_4500=t14['strategy']['needed']['4500']['sr_annual_needed'],
    sr_ann_needed_4600=t14['strategy']['needed']['4600']['sr_annual_needed'],
    sr_ann_needed_5000=t14['strategy']['needed']['5000']['sr_annual_needed'])
out['t14_dsr'] = t14

# ================================================================== 第 9 項 兩腿
lg = rep['legs']; sat, bas = lg['satellite']['sharpe'], lg['base0050']['sharpe']
out['t09_legs'] = dict(satellite_sr=sat, base_sr=bas, min_sr=min(sat, bas), rule='min(兩腿 SR) > 0.5 → △；否則 ⚠（規格書改編版第 119 行；此項沒有 ✓ 的可能）',
                       verdict_recomputed='△' if min(sat, bas) > 0.5 else '⚠', verdict_report=rep['summary_rows'][8]['verdict'],
                       b0050_full_sr=base['b0050']['sr'],
                       note='底倉腿就是 0050 部位，其 SR（0.78）≈ 0050 本身（0.77）；決定 min 的是 0050 不是選股。0050 單一資產無兩腿 → N/A；超額列也無兩腿 → N/A。')

# ================================================================== 第 11 項 集中度
def top_share(x, q):
    a = np.sort(np.asarray(x, float))[::-1]; m = max(1, int(len(a) * q)); return float(a[:m].sum() / a.sum())

t11 = {}
for k, x in S.items():
    n, sr, g3, g4 = dsr_z_parts(x)
    t11[k] = dict(top5_daily=top_share(x, 0.05), top1_daily=top_share(x, 0.01),
                  normal_iid_approx_top5=float(0.05 + stats.norm.pdf(stats.norm.ppf(0.95)) / sr),   # 0.05 + φ(1.645)/日SR
                  share_positive_days=float((np.asarray(x) > 0).mean()), gross_pos_over_net=float(x[x > 0].sum() / x.sum()))
# 規格門檻 50% 等價的日 SR（常態近似）
t11['daily_sr_needed_for_top5_below_50pct'] = float(stats.norm.pdf(stats.norm.ppf(0.95)) / 0.45)
t11['annual_sr_needed_for_top5_below_50pct'] = t11['daily_sr_needed_for_top5_below_50pct'] * math.sqrt(252)
r_ = tr['ret'].to_numpy(float); pn = tr['pnl'].to_numpy(float); m5 = int(math.ceil(len(r_) * 0.05)); od = np.argsort(-r_, kind='stable')[:m5]
t11['trade_level'] = dict(n=int(len(r_)), n_top=m5, share_top5_of_trade_pnl=float(pn[od].sum() / pn.sum()),
                          threshold_ret=float(r_[od[-1]]), top10pct_share=float(pn[np.argsort(-r_, kind='stable')[:int(math.ceil(len(r_) * 0.10))]].sum() / pn.sum()))
# 交易超額（對同期 0050，收盤到收盤近似）：前 5% 占超額損益
e50 = nb
def _px(d):
    i = e50.index.searchsorted(d); i = min(i, len(e50) - 1); return float(e50.iloc[i])
b_ret = np.array([_px(b) / _px(a) - 1 for a, b in zip(tr['ent_date'], tr['exit_date'])])
tr['ret_0050'] = b_ret; tr['xret'] = tr['ret'] - b_ret; tr['xpnl'] = tr['xret'] * tr['cost']
t11['trade_level']['excess_pnl_sum'] = float(tr['xpnl'].sum())
t11['trade_level']['share_top5_of_excess_pnl'] = float(tr['xpnl'].to_numpy()[np.argsort(-tr['xret'].to_numpy(), kind='stable')[:m5]].sum() / tr['xpnl'].sum())
t11['trade_level']['rest95_mean_xret'] = float(np.delete(tr['xret'].to_numpy(), np.argsort(-tr['xret'].to_numpy(), kind='stable')[:m5]).mean())
out['t11_concentration'] = t11

# ================================================================== 第 5 項 IC
def sp(a, b):
    if len(a) < 5:
        return dict(rho=None, p=None, n=int(len(a)))
    r = stats.spearmanr(a, b); return dict(rho=float(r[0]), p=float(r[1]), n=int(len(a)))

t5 = {'all': dict(surge_ret=sp(tr['score'], tr['ret']), surge_xret=sp(tr['score'], tr['xret'])),
      'by_year': {}, 'subperiods': {}}
tr['y'] = tr['ent_date'].dt.year
for y, g in tr.groupby('y'):
    t5['by_year'][int(y)] = dict(surge_ret=sp(g['score'], g['ret']), surge_xret=sp(g['score'], g['xret']))
for lab, lo, hi in (('2008-2013', 2008, 2013), ('2014-2019', 2014, 2019), ('2020-2026', 2020, 2026),
                    ('2008-2015H1', None, None), ('2015H2-2026', None, None)):
    if lo is not None:
        g = tr[(tr['y'] >= lo) & (tr['y'] <= hi)]
    elif lab == '2008-2015H1':
        g = tr[tr['ent_date'] < '2015-06-01']
    else:
        g = tr[tr['ent_date'] >= '2015-06-01']
    t5['subperiods'][lab] = dict(surge_ret=sp(g['score'], g['ret']), surge_xret=sp(g['score'], g['xret']))
# 同一訊號日內（截面）IC：每個訊號日的名單內 Spearman，再取平均（只限 ≥4 筆的訊號日）
cs = []
for d, g in tr.groupby('sig_date'):
    if len(g) >= 4:
        r = stats.spearmanr(g['score'], g['ret'])[0]
        if np.isfinite(r):
            cs.append(r)
cs = np.array(cs)
t5['within_signal_date_ic'] = dict(n_dates=int(len(cs)), mean=float(cs.mean()), t=float(cs.mean() / cs.std(ddof=1) * math.sqrt(len(cs))), share_pos=float((cs > 0).mean()))
yr = np.array([v['surge_ret']['rho'] for v in t5['by_year'].values() if v['surge_ret']['rho'] is not None])
t5['yearly_summary'] = dict(n_years=int(len(yr)), n_pos=int((yr > 0).sum()), mean=float(yr.mean()), sd=float(yr.std(ddof=1)),
                            t_mean=float(yr.mean() / yr.std(ddof=1) * math.sqrt(len(yr))), n_sig_pos_p05=int(sum(1 for v in t5['by_year'].values() if v['surge_ret']['p'] is not None and v['surge_ret']['p'] < 0.05 and v['surge_ret']['rho'] > 0)))
t5['note'] = '逐筆口徑：樣本只限「季線上 SURGE 前 10 名且近 20 日漲幅 > 10%」的入選股（範圍截斷），截斷後 IC 本來就會變小；0050 與超額列無訊號 → N/A。xret＝該筆報酬 − 同期 0050（進出場日收盤近似）。'
out['t05_ic'] = t5

# ================================================================== 分類總表
out['classification'] = {
    '7': dict(cls='B 為底＋C 推過門檻', why='0050 同指標 {:.1%} 也不過；指標近似 Φ(−全期SR)，門檻 15% ⇔ SR>1.04；只計 0050 也為負的天數＝{:.1%}（會過），策略特有的 {:.1%}（2011～2013、2019 選股超額為負）把它推過 15%；超額列 {:.1%}'.format(t7['b0050']['pct_below_0'], t7['strategy_specific_neg']['pct_if_only_market_driven'], t7['strategy_specific_neg']['pct_of_windows'], t7['excess']['pct_below_0'])),
    '8': dict(cls='A（主）＋B', why='固定本金 {:.1%}／複利 {:.1%}；0050 {:.1%}／{:.1%}；iid 常態同 SR 模擬中位數 {:.1%}：只要有任何一點點低於前高就算水下，SR≈1 的序列天生 ~90%'.format(t8['strategy']['uw_fixed'], t8['strategy']['uw_compound'], t8['b0050']['uw_fixed'], t8['b0050']['uw_compound'], sim['strategy']['p50'])),
    '13': dict(cls='A', why='規格公式 t 1.26；實際日報酬 t {:.2f}、月報酬 t {:.2f}；分母膨脹到 {:.0f} 萬'.format(t13['strategy_vs_TAIEX']['actual_daily']['t'], t13['strategy_vs_TAIEX']['monthly']['t'], t13['dilution']['equity_end'] / 1e4)),
    '14': dict(cls='帳戶層 A＋C 各半；超額層 C', why='N=4500 需年化 SR {:.2f}（現 1.06）或約 {:.1f} 年樣本；有效獨立 N≈403（S98）時 {:.1%} 會過；超額 N=403 只 {:.1%}'.format(t14['strategy']['needed']['4500']['sr_annual_needed'], t14['strategy']['needed']['4500']['years_needed_at_current_sr'], t14['strategy']['dsr_by_N']['403'], t14['excess']['dsr_by_N']['403'])),
    '9': dict(cls='A', why='判定正確（min 0.78>0.5 → △）；此項規格上限就是 △'),
    '11': dict(cls='日版 A；逐筆版 C', why='日版 ≈ 0.05+φ(1.645)/日SR，0050 {:.0%}；逐筆 69.6% 是大贏家依賴（S82）'.format(t11['b0050']['top5_daily'])),
    '5': dict(cls='C（＋部分 A：範圍截斷）', why='逐年正 {}/{} 年，年 ρ 平均 {:.3f}'.format(t5['yearly_summary']['n_pos'], t5['yearly_summary']['n_years'], t5['yearly_summary']['mean'])),
}

# C 類問題對照已測方向（出處：s99 inventory.json tested_directions、BACKLOG 2026-11-27 S100、各題 FINDINGS）；候選方向只寫假說，不執行
out['c_issues_vs_tested'] = {
    'C1 盤整期／2011～2013、2019 選股超額為負（推高第 7 項；超額水下 2010-10～2018-04）': [
        'S23／S54／S80／S83 大盤／寬度／因子動能開關、環境溫度計：空位買 0050 版無效；留現金版＝少跌換少賺；領先指標 0/45',
        'S5～S8 底倉＋回撤上限：回撤 ≈ −23.1 − 1.58×每檔權重，單檔上限不是有效率的壓回撤方法',
        'S53 波動目標／S32 波動加權：6 格 Calmar 全輸，SURGE 高波動伴隨大漲',
        'S76／S77／S78 前段規則 147＋288 格；S81 名單放寬 192 格；S99 月中第二次檢查 6 格：全部不通過（前段弱是年代與整張限制，多賺來自多曝險）',
        'S30／S38／S47／S66／S87 閒置資金去處：沒有 ETF 贏 0050'],
    'C2 逐筆集中度（前 5% 占 69.6%、其餘 95% 平均輸 0050）': [
        'S13／S82 大贏家進場當下認不出（AUC 0.52～0.67），是抱出來的',
        'exit_rules 停損停利 48 版、S10／S48／S50 出場家族、S57／S74／S89 提早出場：輸家看得出來時已跌完大半',
        'S99 月中補買、S100 確認後加碼：都失敗且集中度上升（S100 69.6%→79.6%）'],
    'C3 名單內排序 IC 弱且不穩（逐年正 10/19）': [
        'S6／RW 名次加權、S33／S79 名單內價量重排 110 格：名單內排序全失敗',
        'S17／S34／S58／S20 SURGE 定義家族、S21／S31 產業內排名、S12／S44／S46 門檻：不改或較差'],
    'C4 超額 SR 0.65 不足以抵銷試驗數（超額 DSR N=4500 22%）': [
        'S52／S98 多重檢定：N=4,026 校正後 0/3 通過；十年實盤只有在 IR≈0.75 以上才看得出來'],
}
out['untested_candidates'] = [
    dict(id='U1＝S99 W1-B 上月買進名單沿用遞補', hypothesis='本月名單 < 5 檔時，上月買進名單中「本月仍在季線上前 20 名、近 20 日漲幅 > 10%、但目前未持有」者遞補。',
         logic='S76：SURGE 資訊可延續超過一個月；遞補來源是上月前 10 名（已驗證過的那一類），不是放寬門檻或加深名次。',
         affects='第 7 項（2011～2013 策略特有的 SR<0 天數）、第 14 項超額 DSR、超額水下；對第 11 項逐筆集中度可能略降',
         trials='先只數新增檔數（0 次）；若值得再做 2～3 格', data='現成（正式面板）', risk='符合者多半已持有，新增檔數可能很少；與 S81 遞補同樣前段不穩'),
    dict(id='U2＝S99 W2-B 破線賣出後站回買回', hypothesis='因跌破季線被賣、但 SURGE 仍在當月前 20 名者，同月站回季線且近 20 日漲幅 > 10% 時買回。',
         logic='S10：只因破季線被賣的 41 筆到下個換股日比 0050 多 +3.56%（t 1.8）；只針對「被洗掉的潛在大贏家」。',
         affects='第 11 項逐筆版、第 5 項（間接）', trials='先描述（0 次）；樣本約 41 筆，不足就結案；做的話 1～2 格',
         data='現成（訊號層即可）', risk='樣本太小、來回成本 1.2%、S10 已知 2019～2023 不賣反而較差'),
    dict(id='U3 訊號新鮮度分層（首次進入季線上前 10 名 vs 連續在榜）', hypothesis='同樣入選，首次進榜（上月不在前 10 名）與連續在榜的後續超額不同；若首次進榜明顯較好，可在名額不足時優先或只在排序平手時使用。',
         logic='營收公告後漂移在第一次驚奇最強（反應不足）；連續在榜者部分已被定價。與 S82「大贏家是抱出來的」不衝突：這裡只看進場，不動續抱。',
         affects='第 5 項 IC（直接）、第 11 項逐筆集中度', trials='先描述（0 次，用現有 627 筆＋名單歷史）；若描述成立最多 2 格',
         data='現成（SURGE 月排名歷史在正式面板）', risk='部分重疊：舊主策略的 o_rev_freshness_attention、o_hold_buffer_fresh（新鮮度救不了價格腿）；名單內排序類過去全失敗（S33／S79）'),
]


def jclean(o):
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o); return None if not math.isfinite(v) else round(v, 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o

(HERE / 'results.json').write_text(json.dumps(jclean(out), ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(jclean({k: out[k] for k in ('base_three_rows', 'classification')}), ensure_ascii=False, indent=1))
