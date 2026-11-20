#!/usr/bin/env python3
"""S75 驗收 A1：正式引擎單獨跑官方面板（2015-01-05～2026-10-22，起算 2015-07-01；r55lib.run_current 原函式）→ 逐位重現 2019 起 48.76%／381 筆。
另存每月入選名單（給 accept_sig.py 比對）與淨值（給 run.py 做重新基準化比較）到 scratchpad/s75/。
python3 -I -B accept_is.py → accept_is.json"""
import sys, json, time, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s75lib as K
from s75lib import R, L, stlib
import numpy as np, pandas as pd

t0 = time.time()
R.point_cache(K.TMP / 'cache_is')
P, U, F, sg, nav, lg, st, fix = R.run_current()
assert (P.T, P.N, str(P.IDX[-1].date()), str(P.IDX[P.t0].date())) == (2876, 2142, '2026-10-22', '2015-07-01')
D0, DE, D19 = P.IDX[P.t0], P.IDX[-1], P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
x = lg[lg['why'] != 'open_end']
a = dict(since2019=stlib.cagr(nav, D19, DE), full=stlib.cagr(nav, D0, DE), maxdd=stlib.maxdd(nav, D0, DE), n_closed=int(len(x)), final_nav=float(nav.iloc[-1]))
a['bitwise_vs_ACC'] = bool(all(a[k] == L.ACC[k] for k in ('since2019', 'full', 'maxdd', 'n_closed')))
site = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
a['max_abs_diff_vs_navs_official_csv'] = float(np.abs(nav.round(2).reindex(site.index) - site).max())
a['n_signal_months'] = len(sg['info']); a['first_signal'] = sg['info'][0]['sig_date']
a['trade_stats'] = dict(win_rate=float((x['ret'] > 0).mean()), mean_ret=float(x['ret'].mean()), avg_hold_trading_days=float(x['hold_days'].mean()),
                        avg_npos=float(st['avg_npos']), avg_idle_frac=float(st['avg_idle_frac']))
a['pass'] = bool(a['bitwise_vs_ACC'])
a['runtime_s'] = round(time.time() - t0, 1); a['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
json.dump({i['month']: dict(sig_date=i['sig_date'], targets=i['targets']) for i in sg['info']}, open(K.TMP / 'is_targets.json', 'w'))
pd.DataFrame({'nav': nav, 'e50': pd.Series(P.E50, index=P.IDX).reindex(nav.index)}).to_csv(K.TMP / 'is_nav.csv')
(K.HERE / 'accept_is.json').write_text(json.dumps(a, ensure_ascii=False, indent=1))
print(json.dumps(a, ensure_ascii=False, indent=1))
assert a['pass'], 'A1 驗收失敗'
