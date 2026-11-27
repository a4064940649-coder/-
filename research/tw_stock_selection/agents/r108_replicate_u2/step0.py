#!/usr/bin/env python3
"""R108 第 0 步：重現正式基準（2015-07 起 34.46%、2019 起 48.86%、381 筆、20 條路徑中位數 46.42%、與 navs_official.csv.gz 逐日差 0）；
驗收自寫引擎 simulate_u2(rebuy=None) 與 s10lib.simulate_s10 逐日逐筆相同；並數 U2|any 標準路徑的筆數（與 S102 PREREG 登記前公開的筆數對照，不涉報酬）。
用法：python3 -I -B step0.py → step0.json"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r108lib as L
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P, U, RT = L.load()
from s19lib import s10lib, stlib, COSTS
D0, DE = P.IDX[P.t0], P.IDX[-1]
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
D19, TA, TB = lastb('2019-01-01'), lastb('2016-02-01'), lastb('2022-01-01')
KW = lambda cap: dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=cap)
out = {}
paths = {}; ident = []
for s in L.SHIFTS:
    S = s10lib.make_signal_s10(P, U, RT, shift=s)
    ROWS = L.row_scores(P, U, RT, S, s)
    for cap in L.CAPS:
        n1, l1, _ = s10lib.simulate_s10(P, S, U, **KW(cap))
        n2, l2, s2 = L.simulate_u2(P, S, U, ROWS, rebuy=None, capital=cap)
        cols = ['j', 'sig_t', 'ent_t', 'exit_t', 'cost', 'proceeds', 'why', 'shares']
        ident.append(bool(np.array_equal(n1.to_numpy(), n2.to_numpy(), equal_nan=True) and l1[cols].equals(l2[cols])))
        paths[f'{int(cap/1e4)}萬|+{s}'] = stlib.cagr(n1, D19, DE)
        if s == 0 and cap == 1e6:
            ref = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)['E0|W10_70']
            mine = n1.round(2)
            cl = l1[l1['why'] != 'open_end']
            out['std'] = dict(since2019=stlib.cagr(n1, D19, DE), full=stlib.cagr(n1, D0, DE), maxdd=stlib.maxdd(n1, D0, DE), n_closed=int(len(cl)), n_all=int(len(l1)),
                              nav_same_index=bool(mine.index.equals(ref.index)), nav_max_abs_diff=float((mine - ref).abs().max()),
                              avg_stockfrac=float(np.nanmean(s2['stockfrac'].to_numpy())))
            n3, l3, s3 = L.simulate_u2(P, S, U, ROWS, rebuy=dict(r5=False), capital=cap)
            rb = l3[l3['tag'] == 'rebuy']
            out['U2any_std_counts（對照 S102 PREREG seen_before_prereg，不含報酬）'] = dict(
                n_positions=int(len(l3)), rb_flag=s3['rb_flag'], rb_win=s3['rb_win'], rb_trig=s3['rb_trig'], rb_buy=s3['rb_buy'], rb_fail=s3['rb_fail'],
                rb_buy_train=int(((rb['ent_date'] > TA) & (rb['ent_date'] <= TB)).sum()), rb_buy_valid=int((rb['ent_date'] > TB).sum()),
                avg_stockfrac=float(np.nanmean(s3['stockfrac'].to_numpy())), n_rule_exits_current=int((l1['why'] == 'rule').sum()))
            n4, l4, s4 = L.simulate_u2(P, S, U, ROWS, rebuy=dict(r5=True), capital=cap)
            out['U2r5_std_counts'] = dict(n_positions=int(len(l4)), rb_flag=s4['rb_flag'], rb_trig=s4['rb_trig'], rb_buy=s4['rb_buy'], rb_fail=s4['rb_fail'])
            del n3, l3, s3, n4, l4, s4
        del n1, l1, n2, l2, s2
    print('shift', s, f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
out['paths_since2019'] = paths; out['paths_median'] = float(np.median(list(paths.values())))
out['A1_engine_identical_20paths'] = bool(all(ident))
a = out['std']
out['acceptance'] = dict(since2019=abs(a['since2019'] - L.TARGET['since2019']) < 1e-9, full=abs(a['full'] - L.TARGET['full']) < 1e-9, n_closed=a['n_closed'] == 381,
                         nav_diff0=a['nav_same_index'] and a['nav_max_abs_diff'] == 0, paths_median=abs(out['paths_median'] - L.TARGET['paths_median']) < 1e-9, A1=out['A1_engine_identical_20paths'])
out['acceptance']['pass'] = bool(all(out['acceptance'].values()))
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'step0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != 'paths_since2019'}, ensure_ascii=False, default=str))
