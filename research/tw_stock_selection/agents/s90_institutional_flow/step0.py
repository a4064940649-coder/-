#!/usr/bin/env python3
"""S90 第 0 步（登記前）：①重現正式標準路徑（48.86%／34.46%／381 筆／對網站 navs_official 逐日差 0）；②法人資料驗收（data_check.json）；
③只看訓練期（訊號日 2016-02～2021-12）現行已平倉交易的法人分組描述（train_desc，供 PREREG 的事前預測參考）。任何加了法人條件的模擬都不在這裡跑。
用法：python3 -I -B step0.py"""
import sys, json, time, resource, gzip
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s90lib as L
from s90lib import stlib
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
S0 = L.signal(P, U, RT, 0)
nav, lg, st = L.sim(P, S0, U)
site = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
A0 = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()), n_all=int(len(lg)),
          site_same_index=bool(site.index.equals(nav.index)), max_abs_diff_vs_site_nav=float((nav.round(2) - site).abs().max()), panel=[int(P.T), int(P.N), str(DE.date())])
A0['pass'] = bool(abs(A0['since2019'] - L.TARGET['since2019']) < 1e-12 and abs(A0['full'] - L.TARGET['full']) < 1e-12 and A0['n_closed'] == 381 and A0['n_all'] == 389
                  and A0['site_same_index'] and A0['max_abs_diff_vs_site_nav'] == 0)
print('A0', A0, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
assert A0['pass']

# ---- 法人資料驗收
FL, chk = L.build_flow(P)
np.savez_compressed(L.FLOW_NPZ, **FL)
FW = L.Flow(P, U, FL)
A, V, SH = FL['A'].astype(float), FL['VOL'].astype(float), FL['SH'].astype(float)
sigk = [x['k'] for x in S0['info']]
u = U & np.isfinite(V) & (V > 0)
q = lambda x: {f'p{int(p*100):02d}': float(np.quantile(x, p)) for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
dc = dict(build=chk, unit_check={}, ratio_to_volume={}, signal_days={}, coverage={})
for key in ('F', 'T', 'D', 'A'):
    x = FL[key].astype(float)[u] / V[u]
    dc['ratio_to_volume'][key] = dict(q(x), frac_zero=float((x == 0).mean()), frac_abs_gt_1=float((np.abs(x) > 1).mean()), n=int(u.sum()))
dc['unit_check'] = dict(note='單位＝股：|合計買賣超| ÷ 當日成交股數的分佈合理（中位數接近 0、極少超過 1）；若單位是張會小 1000 倍。',
                        median_abs_total_over_volume=float(np.median(np.abs(A[u]) / V[u])), frac_rows_abs_total_gt_volume=float((np.abs(A[u]) > V[u]).mean()),
                        frac_rows_multiple_of_1000=float(np.mean(A[u & (A != 0)] % 1000 == 0)))
yrs = pd.Series(P.IDX.year)
dc['coverage'] = {int(y): dict(days=int((yrs == y).sum()), days_with_inst=int(FL['HAVE'][(yrs == y).to_numpy()].sum()),
                               frac_univ_stockdays_any_inst=float(((FL['F'] != 0) | (FL['T'] != 0) | (FL['D'] != 0))[(yrs == y).to_numpy()][U[(yrs == y).to_numpy()]].mean()),
                               frac_univ_stockdays_trust_nonzero=float((FL['T'] != 0)[(yrs == y).to_numpy()][U[(yrs == y).to_numpy()]].mean()),
                               frac_univ_with_volume=float(np.isfinite(V)[(yrs == y).to_numpy()][U[(yrs == y).to_numpy()]].mean()))
                  for y in sorted(yrs.unique())}
dc['signal_days'] = dict(n=len(sigk), first=str(P.IDX[sigk[0]].date()), last=str(P.IDX[sigk[-1]].date()),
                         all_have_inst_file_on_signal_day=bool(FL['HAVE'][sigk].all()), all_have_25_days_lookback=bool(all(FL['HAVE'][k - 24:k + 1].all() for k in sigk)),
                         shift3_also_ok=bool(all(FL['HAVE'][min(k + 3, P.T - 1)] for k in sigk)),
                         targets_total=int(sum(len(x['targets']) for x in S0['info'])),
                         targets_with_volume_20d=int(sum(np.isfinite(FW.feat(x['k'])['av20'][np.flatnonzero(S0['sig'][x['k']])]).sum() for x in S0['info'])),
                         targets_with_shares=int(sum(np.isfinite(FW.feat(x['k'])['as20'][np.flatnonzero(S0['sig'][x['k']])]).sum() for x in S0['info'])))
# 與舊來源（tw-stock-screener inst_5y，2021-10-08 起、單位張含小數）逐筆比對：直接用原始檔（不經 float32 矩陣）
sys.path.insert(0, str(L.RAW))
from tsd_extra_load import load as tload
od = pd.read_csv(HERE.parent / 'o_concentration_regime' / 'raw' / 'inst_5y.csv.gz', dtype={'code': str})
nw = tload('inst_twse', od['date'].min(), od['date'].max(), columns=['date', 'code', 'foreign', 'trust', 'dealer', 'total']); nw['date'] = nw['date'].astype(str)
mg = od.merge(nw, on=['date', 'code'], how='inner', suffixes=('_o', ''))
dc['old_source_compare'] = dict(old_file='o_concentration_regime/raw/inst_5y.csv.gz', old_range=[od['date'].min(), od['date'].max()], n_overlap=int(len(mg)), n_new_twse_rows_in_range=int(len(nw)),
                                **{c + '_same': float(((mg[c + '_o'] * 1000 - mg[c]).abs() <= 1).mean()) for c in ('foreign', 'trust', 'dealer')},
                                total_same=float(((mg['inst_total'] * 1000 - mg['total']).abs() <= 1).mean()), note='舊檔 ×1000 後與新檔差 ≤ 1 股算相同')
del od, nw, mg
print(json.dumps({k: dc[k] for k in ('unit_check', 'signal_days', 'old_source_compare')}, ensure_ascii=False, default=str)[:3000])
print({k: {a: round(b, 4) for a, b in v.items()} for k, v in dc['ratio_to_volume'].items()})
(HERE / 'data_check.json').write_text(json.dumps(dc, ensure_ascii=False, indent=1, default=str))

# ---- 訓練期描述（只有訓練期）
TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
x = L.trade_features(P, FW, lg[(lg['why'] != 'open_end') & (lg['sig_date'] > TA) & (lg['sig_date'] <= TB)])
n5 = int(round(len(x) * 0.05))
x['bigwin'] = x['ret'].rank(ascending=False, method='first') <= n5; x['biglose'] = x['ret'].rank(ascending=True, method='first') <= n5
x['top10pnl'] = x['pnl'].rank(ascending=False, method='first') <= 10


def grp(lab):
    o = {}
    for g, y in x.groupby(lab, observed=True):
        o[str(g)] = dict(n=int(len(y)), win=float((y['ret'] > 0).mean()), mean=float(y['ret'].mean()), median=float(y['ret'].median()), pnl_sum=float(y['pnl'].sum()),
                         bigwin=int(y['bigwin'].sum()), biglose=int(y['biglose'].sum()), top10pnl=int(y['top10pnl'].sum()))
    return o


sgn = lambda v: np.where(v > 0, '買超', np.where(v < 0, '賣超', '零'))
D = {}
for c in ('fs20', 'ts20', 'as20', 'as5', 'fs5', 'ts5'):
    x['g'] = sgn(x[c]); D[c + '_sign'] = grp('g')
for c in ('p_av20', 'p_fv20', 'p_av5', 'p_fv5'):
    x['g'] = pd.cut(x[c], [-0.01, 0.2, 0.5, 0.8, 1.01], labels=['最低20%', '20~50%', '50~80%', '最高20%']); D[c] = grp('g')
x['g'] = np.where(x['TFB'], '投信初買', '否'); D['TFB'] = grp('g')
x['g'] = np.where(x['streak_a'] >= 3, '連買≥3日', np.where(x['streak_a'] >= 1, '連買1~2日', '當日未買超')); D['streak_a'] = grp('g')
x['g'] = np.where((x['fs20'] > 0) & (x['ts20'] > 0), '同買', np.where((x['fs20'] < 0) & (x['ts20'] < 0), '同賣', np.where(x['ts20'] == 0, '投信沒動', '一買一賣'))); D['f_t_20'] = grp('g')
res = dict(A0=A0, train_desc=dict(window=[str(TA.date()), str(TB.date())], n=int(len(x)), n5=n5, groups=D), meta=dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss()))
(HERE / 'step0.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
for k, v in D.items():
    print(k, {g: (o['n'], round(o['win'], 2), round(o['mean'], 3), round(o['median'], 3), o['bigwin'], o['biglose'], o['top10pnl']) for g, o in v.items()})
print(res['meta'])
