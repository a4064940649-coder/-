import sys, json, numpy as np
sys.path.insert(0, '.')
import engine as E
tmp = sys.argv[1]
off = json.load(open('/home/user/-/research/tw_stock_selection/agents/official_nav/out/res_official.json'))['strats']['E0|W10_70']
cases = [('2017-03','3481'),('2019-11','3437'),('2021-05','8016'),('2021-05','2012'),('2022-07','4746'),('2025-04','3231'),('2025-04','6869'),('2025-06','2480'),('2025-11','3443'),('2026-08','2303')]
for fix in (False, True):
    cfg = dict(E.CFG); cfg['nomove_fix'] = fix
    d = E.Data(tmp, cfg)
    print('== nomove_fix', fix)
    for pm, c in cases:
        t = d.di[off['sig_dates'][pm]]; j = d.ci[c]
        lma, f1 = E.make_signal(d, cfg, t, pm)
        rk = {x[0]: x for x in lma}
        k = d.rmi[pm]; s = d.surge[k, d.rci[c]] if c in d.rci else None
        w = d.adj[t-59:t+1, j]
        r20 = d.adj[t, j]/d.adj[t-20, j]-1
        liq = np.nansum(d.A[t-59:t+1, j])/60
        # 視窗內特殊格：除權息、前一天沒成交、無成交
        win = range(t-60, t+1)
        ex = [d.dates[i] for i in win if d.exm[i, j]]
        nt = [d.dates[i] for i in win if not d.traded[i, j]]
        print(pm, c, d.name[j], 'official_has', c in off['targets'][pm], 'r0_rank', rk.get(c, (0,0,0,None))[3], 'surge', None if s is None else round(float(s),3),
              'ret20 %.4f' % r20, 'close/ma60 %.4f' % (d.adj[t,j]/w.mean()), 'C', d.C[t,j], 'liq(萬) %.0f' % (liq/1e4), 'twse', bool(d.twse[t,j]), 'pres60', bool(d.present[t-59:t+1,j].all()),
              'ex', ex, 'notrade', nt[:5], 'top10', [x[0] for x in lma[:11]], 'offF1', off['targets'][pm])
