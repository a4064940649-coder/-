#!/usr/bin/env python3
"""R55 第 3 步：凍結現行規則在自建 2007～2015H1 面板上跑一次（stlib.START＝2008-01-02、s17lib.FIRST_MONTH＝2008-01），算主窗／次窗／卡住期／合併。
python3 -I -B step2_oos.py [cache 目錄名，預設 cache_pre] → scratchpad/r55/oos_<cache>.json（＋ nav、交易紀錄 csv）"""
import sys, json, time, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r55lib as R
from r55lib import stlib, s17lib, cslib
import numpy as np, pandas as pd

t0 = time.time()
CN = sys.argv[1] if len(sys.argv) > 1 else 'cache_pre'
R.point_cache(R.TMP / CN)
P, U, F, sg, nav, lg, st, fix = R.run_current(start='2008-01-02', first_month='2008-01')
assert str(P.IDX[P.t0].date()) == '2008-01-02' and str(P.IDX[-1].date()) == '2015-06-30'
info = sg['info']
bench = pd.Series(P.E50, index=P.IDX).reindex(nav.index)
rs, rb = R.monthly_returns(nav, bench)
isn = pd.read_csv(R.TMP / 'is_nav.csv', parse_dates=[0], index_col=0)
irs, irb = R.monthly_returns(isn['nav'], isn['e50'])
win = lambda a, b: (rs[(rs.index >= pd.Period(a, 'M')) & (rs.index <= pd.Period(b, 'M'))], rb[(rb.index >= pd.Period(a, 'M')) & (rb.index <= pd.Period(b, 'M'))])
out = dict(cache=CN, P=dict(T=P.T, N=P.N, t0=str(P.IDX[P.t0].date()), last=str(P.IDX[-1].date()), e50_open_missing=P.e50_open_missing,
                            exright_info=P.exright_info, bad_ret=P.bad_ret),
           signal=dict(n_signal_months=len(info), first=info[0]['sig_date'], last=info[-1]['sig_date'], first_month=info[0]['month'], last_month=info[-1]['month'],
                       n_targets=int(sum(x['n_tgt'] for x in info)), mean_targets=float(np.mean([x['n_tgt'] for x in info])),
                       months_zero_target=[x['month'] for x in info if x['n_tgt'] == 0]))
x = lg[lg['why'] != 'open_end']
pe = pd.Timestamp('2013-01-31')
out['trades'] = dict(n_buy=int(st['n_buy']), n_closed=int(len(x)), n_open_end=int((lg['why'] == 'open_end').sum()), n_all_logged=int(len(lg)),
                     n_entry_le_primary=int((lg['ent_date'] <= pe).sum()), n_closed_exit_le_primary=int((x['exit_date'] <= pe).sum()),
                     win_rate_closed=float((x['ret'] > 0).mean()), why=lg['why'].value_counts().to_dict(), n_delist=int(st['n_delist']),
                     avg_npos=float(st['avg_npos']), avg_idle_frac=float(st['avg_idle_frac']),
                     blk=dict((k, int(st[k])) for k in ('blk_limup', 'blk_noopen', 'blk_lot', 'blk_cap', 'blk_nocash', 'blk_min', 'noslot', 'sell_defer_limdn', 'sell_defer_noopen')))
W = {}
for k, (a, b) in dict(primary=('2008-01', '2013-01'), secondary=('2008-01', '2015-06'), stuck=('2013-02', '2015-06')).items():
    s1, s2 = win(a, b); W[k] = R.stats_block(s1, s2)
W['primary']['maxdd'] = R.maxdd(nav.loc[:'2013-01-31']); W['primary']['maxdd_0050'] = R.maxdd(bench.loc[:'2013-01-31'])
W['secondary']['maxdd'] = R.maxdd(nav); W['secondary']['maxdd_0050'] = R.maxdd(bench)
W['in_sample'] = R.stats_block(irs, irb)
p1, p2 = win('2008-01', '2013-01')
W['merged_primary_is'] = R.stats_block(pd.concat([p1, irs]), pd.concat([p2, irb]))
q1, q2 = win('2008-01', '2015-06')
W['merged_secondary_is'] = R.stats_block(pd.concat([q1, irs]), pd.concat([q2, irb]))
out['windows'] = W
out['yearly'] = R.yearly_excess(rs, rb)
pr = W['primary']
out['decision_primary'] = '支持' if pr['ex_ann_pp'] > 0 and pr['t_nw6'] > 1.65 else ('方向一致但不顯著' if pr['ex_ann_pp'] > 0 else '不支持')
sc = W['secondary']
out['decision_secondary'] = '支持' if sc['ex_ann_pp'] > 0 and sc['t_nw6'] > 1.65 else ('方向一致但不顯著' if sc['ex_ann_pp'] > 0 else '不支持')
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
pd.DataFrame({'nav': nav, 'e50': bench}).to_csv(R.TMP / f'oos_nav_{CN}.csv')
lg.drop(columns=['j']).to_csv(R.TMP / f'oos_trades_{CN}.csv', index=False)
pd.DataFrame(info).drop(columns=['targets']).assign(targets=[' '.join(x['targets']) for x in info]).to_csv(R.TMP / f'oos_signals_{CN}.csv', index=False)
(R.TMP / f'oos_{CN}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
