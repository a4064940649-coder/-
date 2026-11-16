#!/usr/bin/env python3
"""S35 事後補充（不在 PREREG 內；主要檢定判定之後才寫）：上櫃除權息日處理的全流程敏感度。python3 -I -B posthoc_exright.py → results_posthoc.json
發現（run.py 之後）：上櫃除權息日的官方日 K 漲跌價差是「缺值」不是 0（7,079 格，前一日有成交；上市則是 0、已由 EXRIGHT_FIX 修正）。
引擎遇缺值改用「收盤 ÷ 前收 − 1」→ 上櫃除權息日把股利當成下跌（悲觀；平均 −4.32%）。PREREG 的悲觀界敏感度只動到漲跌價差＝0 的 1,772 格（前一日無成交），
所以幾乎沒變。這裡補做相反方向：上櫃這 7,079 格的含息日報酬改記 0（近似修正：上市稽核除權息日對參考價平均只 +0.36%），
重算還原價 G、季線、RET20、PTH → 重新產生上櫃 F1（名單會變）→ 同一個主要檢定。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s35lib
from s35lib import s20lib, s17lib, s10lib, s33lib, stlib, cslib, cluster_diff, within_month_diff, grp_stats

H = Path(__file__).resolve().parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s35lib.load()
C, has = P.C, P.has
CFt = pd.DataFrame(np.where(has, C, np.nan)).ffill().to_numpy()
prev = np.vstack([np.full((1, P.N), np.nan), CFt[:-1]])
pu = np.vstack([np.zeros((1, P.N), bool), ~has[:-1]])
base = has & np.isfinite(prev) & (np.abs(C - prev) > 1e-9) & (P.MK == 2)
zn = np.isnan(P.CHG) & base & ~pu                          # 上櫃除權息日（漲跌價差缺值、前一日有成交）
z0 = (P.CHG == 0) & base
out = dict(note='事後補充，不在 PREREG 內；不改主要判定',
           cells=dict(otc_chg_nan_prev_traded=int(zn.sum()), otc_chg_nan_mean_close_over_prev_pct=float(np.mean(C[zn] / prev[zn] - 1) * 100),
                      otc_chg_nan_close_below_prev=float((C[zn] < prev[zn]).mean()), otc_chg_zero_price_changed=int(z0.sum()),
                      otc_chg_zero_prev_untraded=int((z0 & pu).sum()), twse_chg_nan=int((np.isnan(P.CHG) & has & (P.MK == 1)).sum())))
del CFt, prev, pu, base, z0


def build(P):
    UU = s35lib.universes(P)
    RT = s17lib.rev_family(P, defs=['STD']); RT = dict(S=RT['S']['STD'], av=RT['av'])
    S0 = s10lib.make_signal_s10(P, UU['OTC'], RT, shift=0)
    G = s20lib.growth_features(P); RO = s35lib.ROE(P)
    df = s35lib.f1_table(P, UU['OTC'], RT, S0, G, RO)
    df = df[df['grp'] == 'F1'].reset_index(drop=True)
    IND, _, _ = s35lib.industry_matrix(P, sorted(set(df['k'])))
    o = s20lib.outcomes(P, S0['X'], [x['k'] for x in S0['info']], df, P.T - 1)
    df = df.join(o)
    peer, npeer = s35lib.peer_returns(P, df, UU['ALL'], IND, P.V60)
    df['peer'] = peer; df['ex50'] = df['excess']; df['exP'] = df['ret'] - df['peer']
    return df[np.isfinite(df['ret'])].reset_index(drop=True)


def summarize(df):
    d = df[np.isfinite(df['exP'])]
    r = dict(n_F1=int(len(df)), share_TA=float(df['TA'].mean()),
             primary_exP=cluster_diff(d['exP'], d['TA'].astype(float), d['month']),
             ex50=cluster_diff(df['ex50'], df['TA'].astype(float), df['month']),
             within_month_exP=within_month_diff(d['exP'].to_numpy(), d['TA'].to_numpy(), d['month'].to_numpy()),
             halves_exP={h: cluster_diff(g['exP'], g['TA'].astype(float), g['month'])['diff'] for h, g in d.groupby('half')},
             groups_exP=dict(TA=grp_stats(d.loc[d['TA'], 'exP']), other=grp_stats(d.loc[~d['TA'], 'exP'])),
             top10_exP_nTA=int(d.sort_values('exP', ascending=False).head(10)['TA'].sum()))
    return r


df0 = build(P)
out['as_registered'] = summarize(df0)
keys0 = set(zip(df0['month'], df0['code']))
print('base', out['as_registered']['primary_exP'], f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
# 近似修正：上櫃除權息日報酬記 0
P.R = np.where(zn, 0.0, P.R)
P.REF = np.where(zn, C, P.REF)
P.G = np.cumprod(1 + P.R, axis=0)
P.MA60 = stlib.rmean(P.G, 60)
P.F_MA60 = P.G > P.MA60
with np.errstate(invalid='ignore', divide='ignore'):
    P.RET20 = P.G / cslib.shift(P.G, 20) - 1
P.PTH = s33lib.pth_panel(P)
gc.collect()
df1 = build(P)
keys1 = set(zip(df1['month'], df1['code']))
out['otc_exday_return_zero'] = summarize(df1)
out['otc_exday_return_zero']['F1_overlap'] = dict(same=len(keys0 & keys1), only_registered=len(keys0 - keys1), only_fixed=len(keys1 - keys0))
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, default=str)[:5000])
