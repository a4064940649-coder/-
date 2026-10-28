import sys, pickle, time; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
from ab import V2, avg_nav, SP
from multiprocessing import Pool
rf = pd.read_csv(K.O4 / 'rfix_cells.csv', dtype={'code': str})
# 檢查 R_FIX＝R_DEL＋45 格
diff = (K.YF.fillna(-9e9) != K.YD.fillna(-9e9))
print('R_FIX vs R_DEL diff cells', int(diff.values.sum()), 'existing overwritten', int((diff & K.YD.notna()).values.sum()))
YNO = K.YF.copy()
for r in rf.itertuples():
    p = pd.Period(r.ym, 'M')
    if r.ym[:4] == '2013' or (r.ym[:4] == '2014' and ((rf.code == r.code) & (rf.ym == str(p - 12))).any()):
        YNO.loc[p, r.code] = K.YD.loc[p, r.code]
print('NO2013 remaining diff', int((YNO.fillna(-9e9) != K.YD.fillna(-9e9)).values.sum()))
CFG = {
 'V4': dict(), 'SIG': dict(disp='sig'), 'RDEL': dict(rev='D'), 'NO2013': dict(rev=YNO), 'SPEC0': dict(fin2=False, forex=False),
 'O3REP': dict(rev='D', disp='sig', fin2=False, forex=False),
 'O3+forex': dict(rev='D', disp='sig', fin2=False, forex=True),
 'O3+forex+fin2': dict(rev='D', disp='sig', fin2=True, forex=True),
 'O3+forex+fin2+trade': dict(rev='D', disp='trade', fin2=True, forex=True),
 'B05': dict(bound='B05'), 'B10': dict(bound='B10'), 'BQ': dict(bound='BQ'),
}
ABCFG = {'AB': dict(), 'AB_B05': dict(bound='B05'), 'AB_B10': dict(bound='B10'), 'AB_BQ': dict(bound='BQ'), 'AB_RDEL': dict(rev='D')}
def plans_for(name):
    if name in CFG: return [K.build_plan(**CFG[name])]
    if name.endswith('_worst'):
        return [K.build_plan(unavail=V2.make_unavail('worst'), **ABCFG[name[:-6]])]
    return [K.build_plan(unavail=V2.make_unavail('mc', s), **ABCFG[name]) for s in range(40)]
def job(name):
    pls = plans_for(name)
    out = {}
    for cost in ['normal', 'stress', 'feefull']:
        b, s = K.rates('normal', fee=0.001425) if cost == 'feefull' else K.rates(cost)
        navs = []; sts = []
        for pl in pls:
            nav, st = K.run(pl, b, s); navs.append(nav); sts.append(st)
        nav = navs[0] if len(navs) == 1 else avg_nav(navs)
        out[cost] = (nav, sts[0])
    return name, pls, out
if __name__ == '__main__':
    names = list(CFG) + list(ABCFG) + ['AB_worst', 'AB_B10_worst']
    t0 = time.time()
    with Pool(4) as P:
        res = P.map(job, names, chunksize=1)
    print('done', time.time() - t0)
    R = {n: (pls, out) for n, pls, out in res}
    pickle.dump(R, open(SP + 'rep_o4_batch.pkl', 'wb'))
