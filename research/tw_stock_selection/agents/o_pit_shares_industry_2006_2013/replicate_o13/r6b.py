import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib, numpy as np, pandas as pd
P = rlib.load_panel(); close = P['close_raw'].ffill(); S = rlib.survivors()
AR = rlib.anchors_raw(); AR = AR[AR.date.between('2006-01-01', '2013-12-31')]
tm = AR[AR.src == 'tejm'].set_index(['code', 'date']).shares
for advn in [120, 60]:
    AD = rlib.adv(advn)
    for other in ['gq', 'ga']:
        o = AR[AR.src == other].set_index(['code', 'date']).shares
        j = pd.concat([tm.rename('a'), o.rename('b')], axis=1, join='inner').reset_index()
        j['e'] = np.log(j.a / j.b).abs()
        for n in [100, 150]:
            tops = {d: set(AD.loc[d][S.loc[d]].nlargest(n).index) for d in j.date.unique()}
            jj = j[[c in tops[d] for c, d in zip(j.code, j.date)]]
            print('ADV%d top%d tejm vs %s med %.5f n %d' % (advn, n, other, jj.e.median(), len(jj)))
