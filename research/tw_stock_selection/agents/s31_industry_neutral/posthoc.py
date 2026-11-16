#!/usr/bin/env python3
"""S31 事後描述（PREREG 登記為描述、不參與判定；驗證期判定之後才跑）。輸出 results_posthoc.json。python3 -I -B posthoc.py
(1) 隨機對照：凍結規則改用「產業標籤隨機打散」（每個訊號列股池內產業代碼隨機重排、各產業家數不變），20 次（種子 3101～3120）；
    驗證窗（重新起跑，100 萬、位移 0）與全期連續 2019 起年化；凍結規則與現行落在第幾名。
(2) 凍結規則換掉了哪些現行交易、換進哪些（全期連續主跑，位移 0、100 萬）。"""
import sys, json, time, resource, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s31lib
from s31lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
assert (H / 'results.json').exists(), '先跑 validate.py'
FZ = json.loads((H / 'FROZEN.json').read_text())
RULE = FZ['frozen_rule']
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s31lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s31lib.surge(P)
s17lib.e50_fix_suspension(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
IND, inames = s31lib.industry_matrix(P, [x['k'] for x in S0['info']])
IL = s31lib.IndLookup()
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB, D19, DE = lastb('2022-01-01'), lastb('2019-01-01'), P.IDX[-1]
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
ratio = json.loads((H / 'results.json').read_text())['meta']['fresh_capital_scale']
out = {'note': '事後描述；隨機對照在 PREREG 登記為描述、不參與判定'}


def sim(S1, cap=1e6, fresh=False):
    if fresh:
        S1 = s31lib.fresh_from(S1, s31lib.VALID_FIRST_MONTH)
    S2, ekw = s10lib.variant_S(S1, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)


def run(S1):
    navf, lgf, _ = sim(S1, 1e6 / ratio, fresh=True); navc, lgc, _ = sim(S1)
    return dict(valid=stlib.cagr(navf, TB, DE), since2019=stlib.cagr(navc, D19, DE), maxshare_cont=s31lib.holdings_div(P, lgc, P.IDX[P.t0], DE, IL)['max_share_mean'],
                top5_cont=s17lib.top5_share(lgc, navc, 1e6)), lgc


SF, _ = s31lib.build_rule(P, U, RT, S0, IND, RULE)
bC, lgC = run(S0); bF, lgF = run(SF)
R = []
for sd in range(3101, 3121):
    INDR = s31lib.permute_ind(IND, U, RT, S0, sd)
    SR, _ = s31lib.build_rule(P, U, RT, S0, INDR, RULE)
    R.append(run(SR)[0])
    del SR, INDR
    gc.collect()
rr = {f: np.array([r[f] for r in R]) for f in R[0]}
out['random_industry_labels'] = dict(rule=RULE['id'], n=20, CUR=bC, FZ=bF,
                                     dist={f: dict(median=float(np.median(v)), p10=float(np.quantile(v, .1)), p90=float(np.quantile(v, .9)),
                                                   n_below_FZ=int((v < bF[f]).sum()), n_below_CUR=int((v < bC[f]).sum())) for f, v in rr.items()})
print('(1) 完成', f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)

kc = lambda lg: set(zip(lg['code'], lg['ent_t']))
cset, fset = kc(lgC), kc(lgF)
lost = lgC[[(c, e) not in fset for c, e in zip(lgC['code'], lgC['ent_t'])]].sort_values('pnl', ascending=False)
gain = lgF[[(c, e) not in cset for c, e in zip(lgF['code'], lgF['ent_t'])]].sort_values('pnl', ascending=False)
tl = lambda d: [dict(code=str(r.code), ent=str(pd.Timestamp(r.ent_date).date()), ind=IL(r.code, pd.Timestamp(r.sig_date)), ret=round(float(r.ret), 3), pnl=round(float(r.pnl))) for r in d.head(8).itertuples()]
out['trades_diff_cont'] = dict(n_only_CUR=int(len(lost)), pnl_only_CUR=float(lost['pnl'].sum()), top_only_CUR=tl(lost),
                               n_only_FZ=int(len(gain)), pnl_only_FZ=float(gain['pnl'].sum()), top_only_FZ=tl(gain), bottom_only_FZ=tl(gain.sort_values('pnl')))
# 產業別損益（全期連續主跑）
def by_ind(lg):
    g = pd.Series([IL(c, pd.Timestamp(d)) or '缺值' for c, d in zip(lg['code'], lg['sig_date'])], index=lg.index)
    t = lg.groupby(g)['pnl'].agg(['size', 'sum']).sort_values('sum', ascending=False)
    return {k: dict(n=int(r['size']), pnl=round(float(r['sum']))) for k, r in t.head(8).iterrows()}
out['pnl_by_industry_cont'] = dict(CUR=by_ind(lgC), FZ=by_ind(lgF))
out['meta'] = dict(runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:5000])
