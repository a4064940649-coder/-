import sys, time; sys.path.insert(0, '.')
t=time.time()
import rcore5 as R
print('load', time.time()-t, 'patch', R.NPATCH)
print('bad sha', R.check_freeze())
D = R.official_dates(); print(len(D), R.IDX[D[0]['i_d']], R.IDX[D[-1]['i_d']])
for name, rev in [('V4', R.RAW), ('V5', R.Rev('fv'))]:
    t=time.time(); pl = R.build_plan(D, rev); t2=time.time(); nav = R.run(pl, *R.rates()); t3=time.time()
    W = R.wins(pl); ex = R.excess(nav, W)
    print(name, 'plan %.1fs run %.1fs'%(t2-t, t3-t2), 'cagr %.4f t %.4f mdd %.4f n %d mean %.4f sd %.4f'%(R.cagr(nav, W[0][0]), R.tstat(ex), R.mdd(nav, W[0][0]), len(ex), ex.mean()*100, ex.std()*100))
    if name=='V4': p4=pl
    else: print('diff picks', sum(a['picks']!=b['picks'] for a,b in zip(p4,pl)))
