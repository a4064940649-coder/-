#!/usr/bin/env python3
"""挑選期 2006-07～2013-12：第二腿 AVG|B1|n30、主策略排除金融 n10、同池等權、0050。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L

days = L.reb_days('2006-07-01', '2013-12-31')
END = pd.Timestamp('2013-12-31')
t0 = L.IDX.get_loc(days[0]); t1 = L.IDX.get_loc(END)
nanmode = sys.argv[1] if len(sys.argv) > 1 else 'both'

pl2 = L.make_plan(days, 30, lambda cs, d: L.leg2_score(cs, d, nanmode=nanmode)[0])
pl2_10 = L.make_plan(days, 10, lambda cs, d: L.leg2_score(cs, d, nanmode=nanmode)[0])
plm = L.make_plan(days, 10, lambda cs, d: L.main_score(cs))
plew = [(L.pool(d)['e'], np.where(L.pool(d)['el'])[0], d) for d in days]

res = {}
navs = {}
for k, pl in [('leg2', pl2), ('leg2_n10', pl2_10), ('main', plm)]:
    for c in ['normal', 'stress', 'zero']:
        b, s = L.rates(c)
        nav, tov = L.simulate_w(L.to_w(pl), b, s, t0, t1)
        navs[(k, c)] = nav
        if c == 'normal': res[k + '_turnover'] = float(np.mean(tov[1:]))
ew, _ = L.simulate_w(L.to_w(plew), 0, 0, t0, t1)
e50 = L.E50.iloc[t0:t1 + 1] / L.E50.iloc[t0]
wd = days[1:] + [END]  # 研究員口徑：淨值從成交日收盤起算 → 第一個窗從 2006-08 訊號日開始，最後一窗到 2013-12-31，共 89 窗
X0 = L.IDX[t0 + 1]
out = {'n_days': len(days), 'first': str(days[0].date()), 'pool_mean': float(np.mean([len(p[1]) for p in plew]))}

def st(nav, tag):
    cg, dd = L.cagr_dd(nav.loc[X0:])
    wr = L.win_ret(nav, wd); we = L.win_ret(ew, wd); w5 = L.win_ret(e50, wd)
    o = {'CAGR': cg * 100, 'maxDD': dd * 100, 't_EW': L.tstat(wr - we), 'ex_EW_m': np.mean(wr - we) * 100,
         't_0050': L.tstat(wr - w5), 'ex_0050_m': np.mean(wr - w5) * 100, 'n_win': len(wr)}
    wd2 = days  # 對照：訊號日→訊號日（含 2006-07 窗、不含 12 月殘窗）
    wr2 = L.win_ret(nav, wd2); we2 = L.win_ret(ew, wd2); w52 = L.win_ret(e50, wd2)
    o.update({'t_EW_sig89': L.tstat(wr2 - we2), 't_0050_sig89': L.tstat(wr2 - w52)})
    return o

for (k, c), nav in navs.items():
    out[f'{k}|{c}'] = st(nav, k)
out['EW'] = dict(zip(['CAGR', 'maxDD'], [x * 100 for x in L.cagr_dd(ew.loc[X0:])]))
out['0050'] = dict(zip(['CAGR', 'maxDD'], [x * 100 for x in L.cagr_dd(e50.loc[X0:])]))
out['yearly'] = {'leg2': L.yearly(navs[('leg2', 'normal')].loc[X0:]), 'main': L.yearly(navs[('main', 'normal')].loc[X0:]), 'EW': L.yearly(ew.loc[X0:]), '0050': L.yearly(e50.loc[X0:])}
out.update(res)
# 相關、配對差
wr2 = L.win_ret(navs[('leg2', 'normal')], wd); wm = L.win_ret(navs[('main', 'normal')], wd); we = L.win_ret(ew, wd)
w10 = L.win_ret(navs[('leg2_n10', 'normal')], wd)
out['corr_ex_EW'] = float(np.corrcoef(wr2 - we, wm - we)[0, 1])
out['corr_n10_ex_EW'] = float(np.corrcoef(w10 - we, wm - we)[0, 1])
out['diff_vs_main_m'] = float(np.mean(wr2 - wm) * 100); out['t_diff_vs_main'] = L.tstat(wr2 - wm)
# 剔除 2013 / 只看 2013
D = pd.DatetimeIndex(wd[:-1])  # 窗起點
x = wr2 - we
out['t_ex2013'] = L.tstat(x[D.year < 2013]); out['t_only2013'] = L.tstat(x[D.year == 2013])
out['holdings'] = int(sum(len(p[1]) for p in pl2)); out['n_trades'] = len(pl2)
json.dump(out, open(HERE / f'rep_core_{nanmode}.json', 'w'), ensure_ascii=False, indent=1, default=float)
import pickle; pickle.dump({'navs': navs, 'ew': ew, 'pl2': pl2, 'pl2_10': pl2_10, 'plm': plm, 'plew': plew}, open(HERE / f'rep_core_{nanmode}.pkl', 'wb'))
for k, v in out.items(): print(k, v)
