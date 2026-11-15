#!/usr/bin/env python3
"""S19 前置：建面板與創新板旗標快取（scratchpad/s19tmp/），檢查面板和 S6～S18 相同，做「不碰報酬」的資料檢查 → data_check.json。
內容：成交股數單位、A10 與 B11 訊號日的關係、每月 L_MA 第 10／20 名 SURGE 的年度漂移、動態門檻表（只用前一年）、腳本股池 vs 研究股池的差異、
門檻版每月候選檔數（只數名單，不算報酬）。python3 -I -B prep.py"""
import sys, time, json, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s19lib
from s19lib import s4lib, stlib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s19lib.load()
INNOV, names = s4lib.innov_panel(P)
print('面板', P.T, P.N, str(P.IDX[-1].date()), '創新板', len(names), f'{time.time()-t:.0f}s', rss(), 'MB', flush=True)
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗：面板和 S6～S18 不同'
out = {'P0': dict(T=P.T, N=P.N, last=str(P.IDX[-1].date()), n_innov=len(names))}

# 1) 成交股數單位：成交金額 ÷（收盤 × 成交量）
m = P.has & np.isfinite(P.VOL) & (P.VOL > 0) & np.isfinite(P.AMT) & (P.AMT > 0)
ratio = P.AMT[m] / (P.C[m] * P.VOL[m])
out['vol_unit'] = dict(median_amt_over_c_x_vol=float(np.median(ratio)), p05=float(np.quantile(ratio, 0.05)), p95=float(np.quantile(ratio, 0.95)),
                       note='≈1 → VOL 是股數、AMT 是元；腳本的 Close × Volume(張) ÷ 10（萬元）≥ 1000 等於 Close × 股數 ≥ 1,000 萬元')
del ratio, m

S, n0 = s19lib.rev_std(P)
months = list(S.index)
avA = s19lib.a10_rows(P, months)
avB, dl = s19lib.b11_rows(P, months, 11)
U = stlib.universe(P, 'twse') & ~INNOV
UX, Cadj, liqX = s19lib.script_universe(P, INNOV)
out['n_sigma0'] = int(n0)

# 2) 訊號日
nA = s19lib.lma_nth(P, U, S, avA); nB = s19lib.lma_nth(P, U, S, avB)
mm = nA.merge(nB, on='month', suffixes=('_A', '_B'))
d = mm['k_B'] - mm['k_A']
out['signal_dates'] = dict(n_months=int(len(mm)), first=mm['month'].iloc[0], last=mm['month'].iloc[-1], B_minus_A_days=d.value_counts().sort_index().to_dict(),
                           early_months=[r.month for r in mm.itertuples() if r.k_B < r.k_A])

# 3) 年度漂移（研究股池、XQ 訊號日 B11 與研究訊號日 A10）
def yr_tab(df):
    df = df.copy(); df['year'] = df['date'].str[:4].astype(int)
    g = df.groupby('year')
    return {int(y): dict(n=int(len(x)), s10_med=float(x['s10'].median()), s10_q1=float(x['s10'].quantile(.25)), s10_q3=float(x['s10'].quantile(.75)),
                         s20_med=float(x['s20'].median()), s20_q1=float(x['s20'].quantile(.25)), s20_q3=float(x['s20'].quantile(.75)),
                         n_ma_med=float(x['n_ma'].median())) for y, x in g}
out['drift_B11'] = yr_tab(nB); out['drift_A10'] = yr_tab(nA)
out['overall_B11'] = dict(s10_med=float(nB['s10'].median()), s10_q1=float(nB['s10'].quantile(.25)), s10_q3=float(nB['s10'].quantile(.75)),
                          s20_med=float(nB['s20'].median()), s20_q1=float(nB['s20'].quantile(.25)), s20_q3=float(nB['s20'].quantile(.75)),
                          share_s10_ge_2_5=float((nB['s10'] >= 2.5).mean()), share_s20_ge_2_0=float((nB['s20'] >= 2.0).mean()))
# 4) 動態門檻表：Y 年用 Y−1 年（B11、研究股池、位移 0）的每月第 10／20 名中位數；第一年（2016）沒有前一年 → 2.5／2.0
dyn = {}
for y in range(2016, 2027):
    prev = out['drift_B11'].get(y - 1)
    dyn[y] = dict(min_surge=round(prev['s10_med'], 2), exit_surge=round(prev['s20_med'], 2), source=f'{y-1} 年 {prev["n"]} 個月中位數') if prev else dict(min_surge=2.5, exit_surge=2.0, source='沒有前一年，用腳本說明的 2.5／2.0')
out['dynamic_table'] = dyn

# 5) 股池差異（訊號列 B11）
rows = []
for r in nB.itertuples():
    k = r.k; s = S.loc[pd.Period(r.month, 'M')].to_numpy(); fin = np.isfinite(s)
    a = U[k] & fin; b = UX[k] & fin
    rows.append(dict(month=r.month, research=int(a.sum()), script=int(b.sum()), only_research=int((a & ~b).sum()), only_script=int((b & ~a).sum()),
                     ma_only_research=int((a & ~b & P.F_MA60[k]).sum()), ma_only_script=int((b & ~a & P.F_MA60[k]).sum())))
ud = pd.DataFrame(rows); ud['year'] = ud['month'].str[:4]
out['universe_diff'] = dict(mean=ud[['research', 'script', 'only_research', 'only_script', 'ma_only_research', 'ma_only_script']].mean().round(2).to_dict(),
                            by_year=ud.groupby('year')[['research', 'script', 'only_research', 'only_script']].mean().round(1).to_dict('index'))
# 還原價比例（腳本 MinPrice 看的是還原價）：訊號列上 Cadj ÷ 原始收盤 的中位數（研究股池內）
rr = []
for r in nB.itertuples():
    k = r.k; a = U[k]
    rr.append(dict(year=r.month[:4], ratio=float(np.nanmedian(Cadj[k, a] / P.C[k, a]))))
out['adj_ratio_median_by_year'] = pd.DataFrame(rr).groupby('year')['ratio'].median().round(3).to_dict()

# 6) 門檻版候選檔數（只數名單：腳本股池、MinSurge 2.5、RET20 > 10%、季線上），對研究版 B11 名單（L_MA 前 10 名且 RET20 > 10%）
cnt = []
for r in nB.itertuples():
    k = r.k; s = S.loc[pd.Period(r.month, 'M')].to_numpy(); fin = np.isfinite(s); sv = np.where(fin, s, -np.inf)
    with np.errstate(invalid='ignore'):
        ok_ma = U[k] & fin & P.F_MA60[k]
        o = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        res = set(int(j) for j in o[:10] if P.RET20[k, j] > 0.10)
        xt = set(np.flatnonzero(UX[k] & fin & (sv >= 2.5) & (P.RET20[k] > 0.10) & P.F_MA60[k]).tolist())
    cnt.append(dict(month=r.month, n_res=len(res), n_xt=len(xt), inter=len(res & xt)))
cd = pd.DataFrame(cnt)
out['list_counts_XT25_vs_XR'] = dict(n_res_mean=float(cd['n_res'].mean()), n_xt_mean=float(cd['n_xt'].mean()), n_xt_median=float(cd['n_xt'].median()),
                                    n_xt_max=int(cd['n_xt'].max()), share_res_in_xt=float(cd['inter'].sum() / max(cd['n_res'].sum(), 1)),
                                    months_xt_gt10=int((cd['n_xt'] > 10).sum()))
out['runtime_s'] = round(time.time() - t, 1); out['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: out[k] for k in ('vol_unit', 'signal_dates', 'overall_B11', 'dynamic_table', 'universe_diff', 'adj_ratio_median_by_year', 'list_counts_XT25_vs_XR')}, ensure_ascii=False, default=str))
print({y: (round(v['s10_med'], 2), round(v['s20_med'], 2), v['n']) for y, v in out['drift_B11'].items()})
print(out['runtime_s'], 's', out['maxrss_mb'], 'MB')
