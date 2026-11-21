#!/usr/bin/env python3
"""S89 步驟 0（登記前）：驗收＋只看訓練期（訊號日 2016-02～2021-12）的現行交易描述。驗證期的逐筆報酬在這裡不計算、不輸出。
A0 原引擎標準路徑＝正式數字（48.86%／34.46%／−31.61%／381 筆）且與 xq_scripts/site/navs_official.csv.gz 的 E0 欄逐日差 0；
A1 副本引擎掛鉤全關，20 條路徑淨值與原引擎逐日差 0、交易紀錄逐筆相同；A2 20 條路徑中位數 46.42%。
用法：python3 -I -B step0.py"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s89lib as L
from s89lib import stlib
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], lastb('2019-01-01')
TA, TB = lastb('2016-02-01'), lastb('2022-01-01')
print('載入', f'{time.time()-t0:.0f}s', rss(), 'MB', (P.T, P.N, str(DE.date())), flush=True)
acc = dict(src_sha=L.SRC_SHA, panel=[int(P.T), int(P.N), str(DE.date())], A1_max_abs_nav_diff=0.0, A1_logs_same=True, paths={})
main = None
for s in L.SHIFTS:
    S = L.signal(P, U, RT, s)
    for cap in L.CAPS:
        n0, l0, _ = L.sim(P, S, U, cap=cap, orig=True)
        n1, l1, st1 = L.sim(P, S, U, cap=cap)
        d = float((n0 - n1).abs().max())
        acc['A1_max_abs_nav_diff'] = max(acc['A1_max_abs_nav_diff'], d)
        acc['A1_logs_same'] = bool(acc['A1_logs_same'] and l0.equals(l1) and 'ts_events' not in st1)
        acc['paths'][f'{int(cap/1e4)}萬|+{s}'] = stlib.cagr(n0, D19, DE)
        if s == 0 and cap == 1e6:
            main = (n0, l0)
    del S; gc.collect()
nav, lg = main
acc['A0'] = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int((lg['why'] != 'open_end').sum()), n_all=int(len(lg)))
site = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
acc['A0_site_same_index'] = bool(site.index.equals(nav.round(2).index))
acc['A0_max_abs_diff_vs_site_nav'] = float((site - nav.round(2)).abs().max())
acc['A2_paths_median'] = float(np.median(list(acc['paths'].values())))
T = L.TARGET
acc['pass'] = bool(abs(acc['A0']['since2019'] - T['since2019']) < 1e-12 and abs(acc['A0']['full'] - T['full']) < 1e-12 and abs(acc['A0']['maxdd'] - T['maxdd']) < 1e-12
                   and acc['A0']['n_closed'] == 381 and acc['A0_max_abs_diff_vs_site_nav'] == 0 and acc['A1_max_abs_nav_diff'] == 0 and acc['A1_logs_same']
                   and abs(acc['A2_paths_median'] - T['paths_median']) < 1e-12)
print('驗收', {k: v for k, v in acc.items() if k != 'paths'}, flush=True)
assert acc['pass'], '驗收失敗'

# ---- 訓練期描述（只用訊號日在 TA < d ≤ TB 的已平倉交易）
x = lg[(lg['sig_date'] > TA) & (lg['sig_date'] <= TB) & (lg['why'] != 'open_end')].copy()
x = x.join(L.early_path(P, x))
n5 = int(round(len(x) * 0.05))
x['bigwin'] = x['ret'].rank(ascending=False, method='first') <= n5
x['biglose'] = x['ret'].rank(ascending=True, method='first') <= n5
BINS = [-np.inf, -0.08, -0.03, 0.03, 0.08, np.inf]; LAB = ['≤−8%', '−8～−3%', '−3～+3%', '+3～+8%', '>+8%']
desc = dict(window=[str(TA.date()), str(TB.date())], n=int(len(x)), n5=n5, bigwin_ret_ge=float(x.loc[x['bigwin'], 'ret'].min()), biglose_ret_le=float(x.loc[x['biglose'], 'ret'].max()), groups={})
for n in (3, 5, 10):
    g = pd.cut(x[f'r{n}'], BINS, labels=LAB, right=True)
    tb = {}
    for lab in LAB:
        y = x[g == lab]
        tb[lab] = dict(n=int(len(y)), mean=float(y['ret'].mean()) if len(y) else None, median=float(y['ret'].median()) if len(y) else None, win=float((y['ret'] > 0).mean()) if len(y) else None,
                       pnl_sum=float(y['pnl'].sum()), bigwin=int(y['bigwin'].sum()), biglose=int(y['biglose'].sum()))
    desc['groups'][f'day{n}'] = tb
    W = x[x['bigwin']]; Z = x[x['biglose']]
    desc[f'bigwin_day{n}'] = dict(neg=int((W[f'r{n}'] < 0).sum()), le_m3=int((W[f'r{n}'] <= -.03).sum()), le_m5=int((W[f'r{n}'] <= -.05).sum()), le_m8=int((W[f'r{n}'] <= -.08).sum()), le_m10=int((W[f'r{n}'] <= -.10).sum()),
                                  min_le_m5=int((W[f'min{n}'] <= -.05).sum()), min_le_m8=int((W[f'min{n}'] <= -.08).sum()), n=int(len(W)))
    desc[f'biglose_day{n}'] = dict(neg=int((Z[f'r{n}'] < 0).sum()), le_m3=int((Z[f'r{n}'] <= -.03).sum()), le_m5=int((Z[f'r{n}'] <= -.05).sum()), le_m8=int((Z[f'r{n}'] <= -.08).sum()), le_m10=int((Z[f'r{n}'] <= -.10).sum()), n=int(len(Z)))
    desc[f'trigger_counts_day{n}'] = {f'le_m{k}': int((x[f'r{n}'] <= -k / 100).sum()) for k in (3, 5, 8, 10)}
desc['trigger_counts_day10']['lt_0'] = int((x['r10'] < 0).sum())
json.dump(dict(acceptance=acc, train_desc=desc, secs=round(time.time() - t0), peak_mb=rss()), open(HERE / 'out' / 'step0.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(desc, ensure_ascii=False, indent=0))
print('done', round(time.time() - t0), 's', rss(), 'MB')
