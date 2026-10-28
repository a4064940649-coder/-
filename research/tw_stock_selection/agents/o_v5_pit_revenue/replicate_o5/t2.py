import sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import rcore5 as R
D = R.official_dates()
for name, rev in [('AB_V4', R.RAW), ('AB_V5', R.Rev('fv'))]:
    t=time.time(); navs=[]; ts=[]
    for s in range(40):
        pl = R.build_plan(D, rev, U=R.unavail_mc(s)); nav = R.run(pl, *R.rates()); navs.append(nav)
        ts.append(R.tstat(R.excess(nav, R.wins(pl))))
    nav = R.avg_nav(navs); W = R.wins(pl); ex = R.excess(nav, W)
    print(name, '%.0fs'%(time.time()-t), 'cagr %.4f t %.4f mdd %.4f mean %.4f sd %.4f'%(R.cagr(nav, W[0][0]), R.tstat(ex), R.mdd(nav, W[0][0]), ex.mean()*100, ex.std()*100))
    ts=np.array(ts); print(' seeds med %.3f min %.3f max %.3f p10 %.3f n<2.3 %d'%(np.median(ts), ts.min(), ts.max(), np.percentile(ts,10), (ts<2.3).sum()))
